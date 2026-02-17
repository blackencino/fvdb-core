# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end cuTile codegen test for the 3-level CIG ijk_to_index.

Fully fused: the root lookup (Find) is inside the cuTile kernel alongside
the 3-level masked chain. No torch barrier.
"""

import importlib
import math
import os

import numpy as np
import torch

import cuda.tile as ct

from fvdb_tile.prototype.cig import build_compressed_cig3
from fvdb_tile.prototype.dsl_to_cutile import emit_runnable_kernel
from fvdb_tile.tests.test_cig3 import cig3_ijk_to_index_numpy
from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Static, Type

_GEN_DIR = os.path.join(os.path.dirname(__file__), "_generated")

TILE = 256

CIG3_KERNEL_PROGRAM = """
parts = Decompose(Input("query"), Const([3, 4, 5]))
upper_idx = Find(Input("root_coords"), field(parts, "which_top"))
upper = masked(Gather(Input("upper_masks"), upper_idx), Gather(Input("upper_abs_prefix"), upper_idx))
lower_idx = Gather(upper, field(parts, "level_2"))
lower = masked(Gather(Input("lower_masks"), lower_idx), Gather(Input("lower_abs_prefix"), lower_idx))
leaf_idx = Gather(lower, field(parts, "level_1"))
leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_abs_prefix"), leaf_idx))
voxel_idx = Gather(leaf, field(parts, "level_0"))
voxel_idx
"""


def _make_cig3_input_types(n_upper: int):
    """Build input types with static R for the root_coords table."""
    return {
        "query": Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.I32)),
        "root_coords": Type(Shape(Static(n_upper)), Type(Shape(Static(3)), ScalarType.I32)),
        "upper_masks": Type(Shape(Dynamic()), Type(Shape(Static(512)), ScalarType.I64)),
        "upper_abs_prefix": Type(Shape(Dynamic()), Type(Shape(Static(512)), ScalarType.I32)),
        "lower_masks": Type(Shape(Dynamic()), Type(Shape(Static(64)), ScalarType.I64)),
        "lower_abs_prefix": Type(Shape(Dynamic()), Type(Shape(Static(64)), ScalarType.I32)),
        "leaf_masks": Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I64)),
        "leaf_abs_prefix": Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I32)),
    }


def _compile_kernel(code: str, kernel_name: str):
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


def _make_test_grid(n_voxels=500, seed=42):
    rng = np.random.RandomState(seed)
    coords_set = set()
    while len(coords_set) < n_voxels:
        batch = rng.randint(0, 4096, (n_voxels * 2, 3))
        for row in batch:
            coords_set.add(tuple(row))
            if len(coords_set) >= n_voxels:
                break
    ijk = torch.from_numpy(np.array(sorted(coords_set)[:n_voxels], dtype=np.int32))
    return build_compressed_cig3(ijk), ijk


def _make_mixed_queries(grid_coords, n_queries=2000, seed=99):
    rng = np.random.RandomState(seed)
    N = grid_coords.shape[0]
    n_hits = n_queries // 2
    hits = grid_coords[rng.choice(N, n_hits, replace=True)]
    randoms = torch.from_numpy(rng.randint(0, 8192, (n_queries - n_hits, 3)).astype(np.int32))
    return torch.cat([hits, randoms], dim=0)


def _emit_and_compile(cig, kernel_name):
    """Emit and compile a kernel for the given CIG (R = cig.n_upper)."""
    input_types = _make_cig3_input_types(cig.n_upper)
    code, _, _ = emit_runnable_kernel(
        CIG3_KERNEL_PROGRAM,
        input_types,
        kernel_name=kernel_name,
        tile_input="query",
        tile_input_rank=3,
        tile_size=TILE,
    )
    return code, _compile_kernel(code, kernel_name)


def _launch_kernel(kernel_fn, cig_cuda, queries_cuda):
    """Launch the fused 3-level CIG kernel (Find + masked chain)."""
    N = queries_cuda.shape[0]
    n_blocks = math.ceil(N / TILE)
    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device="cuda")
    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        kernel_fn,
        (
            queries_cuda, cig_cuda.root_coords,
            cig_cuda.upper_masks, cig_cuda.upper_abs_prefix.int(),
            cig_cuda.lower_masks, cig_cuda.lower_abs_prefix.int(),
            cig_cuda.leaf_masks, cig_cuda.leaf_abs_prefix.int(),
            result_t, TILE,
        ),
    )
    return result_t[:N]


# ---------------------------------------------------------------------------
# Test 1: Emit and inspect
# ---------------------------------------------------------------------------


def test_emit_cig3_kernel():
    cig, _ = _make_test_grid(n_voxels=200)
    code, _ = _emit_and_compile(cig, "gen_cig3_v11")

    n_lines = len(code.splitlines())
    print(f"--- Generated 3-level CIG kernel ({n_lines} lines) ---")
    for i, line in enumerate(code.splitlines()[:30]):
        print(f"  {i + 1:3d}| {line}")
    print(f"  ... ({n_lines} total lines)")

    assert "@ct.kernel" in code
    assert "ct.uint64" in code
    assert "Find" in code or "find_idx" in code or "linear scan" in code
    assert code.count("Masked gather (abs-prefix") == 3, "Should have 3 masked gather blocks"
    assert "ct.scatter(" in code

    print(f"  emit_cig3_kernel: {n_lines} lines, Find + 3 masked levels -- PASSED")
    return code


# ---------------------------------------------------------------------------
# Test 2: Single-upper fused pipeline vs numpy
# ---------------------------------------------------------------------------


def test_cig3_fused_single_upper():
    cig, grid_coords = _make_test_grid(n_voxels=1000)
    _, kernel_fn = _emit_and_compile(cig, "gen_cig3_fused1")

    queries = _make_mixed_queries(grid_coords, n_queries=3000)
    cig_cuda = cig.cuda()
    queries_cuda = queries.cuda().to(torch.int32)

    gpu_result = _launch_kernel(kernel_fn, cig_cuda, queries_cuda).cpu().numpy()
    ref_result = cig3_ijk_to_index_numpy(cig, queries.numpy())

    np.testing.assert_array_equal(gpu_result, ref_result)
    n_hits = int((gpu_result >= 0).sum())
    print(
        f"  cig3_fused_single_upper: {len(queries)} queries ({n_hits} hits), "
        f"{cig.n_upper} upper -- GPU == numpy ref -- PASSED"
    )


# ---------------------------------------------------------------------------
# Test 3: Multi-upper fused pipeline vs numpy
# ---------------------------------------------------------------------------


def test_cig3_fused_multi_upper():
    rng = np.random.RandomState(42)
    coords = []
    for base_x in [0, 4096, 8192]:
        for _ in range(200):
            coords.append([base_x + rng.randint(0, 4096), rng.randint(0, 4096), rng.randint(0, 4096)])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))
    cig = build_compressed_cig3(ijk)

    _, kernel_fn = _emit_and_compile(cig, "gen_cig3_fused3")

    queries = _make_mixed_queries(ijk, n_queries=2000)
    cig_cuda = cig.cuda()
    queries_cuda = queries.cuda().to(torch.int32)

    gpu_result = _launch_kernel(kernel_fn, cig_cuda, queries_cuda).cpu().numpy()
    ref_result = cig3_ijk_to_index_numpy(cig, queries.numpy())

    np.testing.assert_array_equal(gpu_result, ref_result)
    n_hits = int((gpu_result >= 0).sum())
    print(
        f"  cig3_fused_multi_upper: {len(queries)} queries ({n_hits} hits), "
        f"{cig.n_upper} upper nodes -- GPU == numpy ref -- PASSED"
    )


# =========================================================================

if __name__ == "__main__":
    print("=== 3-level CIG end-to-end codegen (fused Find + masked) ===")
    test_emit_cig3_kernel()
    print()
    test_cig3_fused_single_upper()
    print()
    test_cig3_fused_multi_upper()
    print("\nAll 3-level CIG end-to-end tests passed.")
