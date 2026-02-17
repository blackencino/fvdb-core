# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: cross-leaf neighbor predicate -- cuTile vs PyTorch (CPU & GPU).

Measures wall-clock time for the hierarchical neighbor predicate (Decompose +
chained Gather through a two-level grid) at parameterized scale.

Three implementations:
  - PyTorch CPU (vectorized): broadcast + advanced indexing on CPU
  - PyTorch GPU (vectorized): same, on CUDA
  - cuTile (generated): DSL-emitted @ct.kernel via emit_runnable_kernel
"""

import importlib
import os
import time

import torch

import cuda.tile as ct

from fvdb_tile.prototype.dsl_to_cutile import emit_runnable_kernel
from fvdb_tile.prototype.types import ScalarType, Shape, Static, Type

ConstInt = ct.Constant[int]

_GEN_DIR = os.path.join(os.path.dirname(__file__), "_generated")

FACE_OFFSETS = torch.tensor(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=torch.int32,
)


# ---------------------------------------------------------------------------
# Grid builder (parameterized scale)
# ---------------------------------------------------------------------------


def build_grid(n_leaves, seed=42):
    """Build a two-level grid with n_leaves active lower nodes.

    Places leaves in a contiguous block within the (16,16,16) lower grid
    so that many voxels have cross-leaf neighbors.

    Returns: (lower_data, leaf_data, all_global_coords)
    """
    torch.manual_seed(seed)

    lower_data = torch.full((16, 16, 16), -1, dtype=torch.int32)

    # Place leaves in a contiguous 3D block for maximal boundary sharing
    leaf_blocks = []
    leaf_idx = 0
    active_lower = []
    for lx in range(16):
        for ly in range(16):
            for lz in range(16):
                if leaf_idx >= n_leaves:
                    break
                lower_data[lx, ly, lz] = leaf_idx
                active_lower.append((lx, ly, lz))
                leaf_idx += 1
            if leaf_idx >= n_leaves:
                break
        if leaf_idx >= n_leaves:
            break

    # Generate leaves with ~60% active voxels
    for _ in range(n_leaves):
        leaf = torch.full((8, 8, 8), -1, dtype=torch.int32)
        n_active = torch.randint(280, 340, (1,)).item()
        positions = torch.randperm(512)[:n_active]
        for idx, pos in enumerate(positions):
            vx = pos.item() // 64
            vy = (pos.item() // 8) % 8
            vz = pos.item() % 8
            leaf[vx, vy, vz] = len(leaf_blocks) * 1000 + idx
        leaf_blocks.append(leaf)

    leaf_data = torch.stack(leaf_blocks)

    # Collect all active global coords (vectorized per-leaf)
    all_coords_list = []
    for li, (lx, ly, lz) in enumerate(active_lower):
        active_voxels = torch.nonzero(leaf_data[li] >= 0).to(torch.int32)  # (M, 3)
        origin = torch.tensor([lx * 8, ly * 8, lz * 8], dtype=torch.int32)
        all_coords_list.append(active_voxels + origin)
    all_coords = torch.cat(all_coords_list)

    return lower_data, leaf_data, all_coords


# ---------------------------------------------------------------------------
# PyTorch (vectorized) -- used for both CPU and GPU
# ---------------------------------------------------------------------------


def pytorch_cross_leaf(lower_t, leaf_arr_t, coords_t, offsets_t):
    """Vectorized PyTorch: broadcast + hierarchical advanced indexing.

    Works on both CPU and CUDA tensors (all args must be on the same device).
    """
    K = leaf_arr_t.shape[0]

    # (N, 1, 3) + (1, 6, 3) -> (N, 6, 3)
    nbrs = coords_t.unsqueeze(1) + offsets_t.unsqueeze(0)

    # Decompose
    l1 = (nbrs >> 3) & 15  # (N, 6, 3)
    l0 = nbrs & 7  # (N, 6, 3)

    # Bounds check on lower-level coords
    l1_valid = (l1 >= 0).all(dim=2) & (l1 < 16).all(dim=2)  # (N, 6)
    l1_clamped = l1.clamp(0, 15)

    # Gather leaf_idx from lower: (N, 6)
    leaf_idx = lower_t[l1_clamped[..., 0], l1_clamped[..., 1], l1_clamped[..., 2]]
    valid_leaf = l1_valid & (leaf_idx >= 0) & (leaf_idx < K)
    safe_leaf_idx = leaf_idx.clamp(min=0, max=K - 1)

    # Gather voxel from leaf_arr: (N, 6)
    l0_clamped = l0.clamp(0, 7)
    voxel_val = leaf_arr_t[
        safe_leaf_idx, l0_clamped[..., 0], l0_clamped[..., 1], l0_clamped[..., 2]
    ]

    return valid_leaf & (voxel_val >= 0)


# ---------------------------------------------------------------------------
# cuTile (generated kernel)
# ---------------------------------------------------------------------------


def _compile_kernel(code, kernel_name):
    os.makedirs(_GEN_DIR, exist_ok=True)
    init_path = os.path.join(_GEN_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

    mod_name = f"_gen_{kernel_name}"
    filepath = os.path.join(_GEN_DIR, f"{mod_name}.py")
    with open(filepath, "w") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, kernel_name)


CROSS_LEAF_MAP = (
    'pred = Map(Input("offsets"), o => '
    "GE("
    "Gather("
    'Gather(Input("leaf_arr"), '
    'Gather(Input("lower"), '
    'field(Decompose(Add(Input("coord"), o), Const([3, 4])), "level_1"))), '
    'field(Decompose(Add(Input("coord"), o), Const([3, 4])), "level_0")), '
    "Const(0)))\n"
    "pred"
)


def build_cutile_launcher(K):
    """Emit and compile the cross-leaf kernel for K leaves."""
    input_types = {
        "coord": Type(Shape(Static(3)), ScalarType.I32),
        "offsets": Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
        "lower": Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        "leaf_arr": Type(
            Shape(Static(K)),
            Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
        ),
    }

    code, tile_size, map_len = emit_runnable_kernel(
        CROSS_LEAF_MAP,
        input_types,
        batch_input="coord",
        batch_dim=3,
        map_input="offsets",
        map_elem_rank=3,
        kernel_name="bench_cross_leaf",
    )

    kernel_fn = _compile_kernel(code, "bench_cross_leaf")
    return kernel_fn, tile_size, map_len


def cutile_cross_leaf(kernel_fn, coord_t, offsets_t, lower_t, leaf_arr_t, result_t, tile_size):
    """Launch the generated cross-leaf kernel."""
    N = coord_t.shape[0]
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        kernel_fn,
        (coord_t, offsets_t, lower_t, leaf_arr_t, result_t, tile_size),
    )


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def _time_fn(fn, warmup=3, repeats=50):
    """Time a function with warmup and repeats. Returns median time in us."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]


def bench(n_leaves=64):
    print(f"\n--- {n_leaves} leaves ---")

    lower_data, leaf_data, all_coords = build_grid(n_leaves)
    N = all_coords.shape[0]
    K = leaf_data.shape[0]

    # Count cross-leaf lookups (vectorized)
    all_nbrs = all_coords.unsqueeze(1) + FACE_OFFSETS.unsqueeze(0)  # (N, 6, 3)
    own_leaf = (all_coords >> 3) & 15  # (N, 3)
    nbr_leaf = (all_nbrs >> 3) & 15  # (N, 6, 3)
    n_cross = int((~(own_leaf.unsqueeze(1) == nbr_leaf).all(dim=2)).sum().item())

    print(f"Grid: {K} leaves, {N} active voxels, {N * 6} lookups, {n_cross} cross-leaf")

    # Reference: PyTorch CPU (vectorized)
    ref = pytorch_cross_leaf(lower_data, leaf_data, all_coords, FACE_OFFSETS)
    total_active = int(ref.sum().item())

    lower_t = lower_data.clone().cuda()
    leaf_arr_t = leaf_data.clone().cuda()
    coord_t = all_coords.clone().cuda()
    offsets_t = FACE_OFFSETS.clone().cuda()

    # Verify PyTorch GPU matches CPU
    pt_gpu = pytorch_cross_leaf(lower_t, leaf_arr_t, coord_t, offsets_t)
    torch.testing.assert_close(ref, pt_gpu.cpu(), atol=0, rtol=0)

    # Verify cuTile matches CPU
    kernel_fn, tile_size, map_len = build_cutile_launcher(K)
    result_t = torch.zeros(N, tile_size, dtype=torch.int32, device="cuda")
    cutile_cross_leaf(kernel_fn, coord_t, offsets_t, lower_t, leaf_arr_t, result_t, tile_size)
    ct_result = result_t.cpu()[:, :map_len].to(torch.bool)
    torch.testing.assert_close(ref, ct_result, atol=0, rtol=0)

    print(f"All agree: {total_active} active neighbors.\n")

    # Benchmark
    t_pt_cpu = _time_fn(
        lambda: pytorch_cross_leaf(lower_data, leaf_data, all_coords, FACE_OFFSETS),
        warmup=3,
        repeats=20,
    )
    t_pt_gpu = _time_fn(
        lambda: pytorch_cross_leaf(lower_t, leaf_arr_t, coord_t, offsets_t)
    )

    def _run_cutile():
        result_t.zero_()
        cutile_cross_leaf(kernel_fn, coord_t, offsets_t, lower_t, leaf_arr_t, result_t, tile_size)

    t_cutile = _time_fn(_run_cutile)

    print(f"{'Method':<25} {'Median (us)':>12} {'vs PyTorch GPU':>15}")
    print("-" * 55)
    print(f"{'PyTorch CPU (vectorized)':<25} {t_pt_cpu:>12.0f} {t_pt_cpu / t_pt_gpu:>14.1f}x")
    print(f"{'PyTorch GPU (vectorized)':<25} {t_pt_gpu:>12.1f} {'1.0x':>15}")
    print(f"{'cuTile (generated)':<25} {t_cutile:>12.1f} {t_pt_gpu / t_cutile:>14.1f}x")

    return {
        "n_leaves": n_leaves,
        "n_voxels": N,
        "n_lookups": N * 6,
        "n_cross_leaf": n_cross,
        "active_neighbors": total_active,
        "pytorch_cpu_us": t_pt_cpu,
        "pytorch_gpu_us": t_pt_gpu,
        "cutile_us": t_cutile,
    }


if __name__ == "__main__":
    print("=== Cross-leaf neighbor predicate benchmark ===")

    results = []
    for n_leaves in [4, 64, 256]:
        results.append(bench(n_leaves))

    print("\n\n=== Summary ===")
    print(
        f"{'Leaves':>7} {'Voxels':>8} {'Lookups':>10} "
        f"{'PT CPU (us)':>12} {'PT GPU (us)':>12} {'cuTile (us)':>12} {'CT/PT GPU':>10}"
    )
    print("-" * 80)
    for r in results:
        ct_vs_pt = r["pytorch_gpu_us"] / r["cutile_us"]
        print(
            f"{r['n_leaves']:>7} "
            f"{r['n_voxels']:>8} "
            f"{r['n_lookups']:>10} "
            f"{r['pytorch_cpu_us']:>12.0f} "
            f"{r['pytorch_gpu_us']:>12.1f} "
            f"{r['cutile_us']:>12.1f} "
            f"{ct_vs_pt:>9.1f}x"
        )
