# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end cuTile codegen test for the 3-level CIG ijk_to_index.

Two-step pipeline:
  Step 1 (torch): root_lookup resolves which upper node each query belongs to.
  Step 2 (cuTile): fused 3-level masked chain (upper -> lower -> leaf -> voxel).

Validates the full path: DSL parse -> emit -> cuTile JIT -> GPU launch,
verified against the numpy reference from test_cig3.py.
"""

import importlib
import math
import os

import numpy as np
import torch

import cuda.tile as ct

from docs.wip.prototype.cig import build_compressed_cig3, root_lookup
from docs.wip.prototype.dsl_to_cutile import emit_runnable_kernel
from docs.wip.prototype.test_cig3 import cig3_ijk_to_index_numpy
from docs.wip.prototype.types import Dynamic, ScalarType, Shape, Static, Type

_GEN_DIR = os.path.join(os.path.dirname(__file__), "_generated")

TILE = 256


def _compile_kernel(code: str, kernel_name: str):
    """Write kernel code to a .py file and import the kernel function."""
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


# ---------------------------------------------------------------------------
# The DSL program for the 3-level masked chain (cuTile kernel)
# ---------------------------------------------------------------------------

CIG3_KERNEL_PROGRAM = """
parts = Decompose(Input("query"), Const([3, 4, 5]))
upper = masked(Gather(Input("upper_masks"), Input("upper_idx")), Gather(Input("upper_prefix"), Input("upper_idx")), Gather(Input("upper_offsets"), Input("upper_idx")))
lower_idx = Gather(upper, field(parts, "level_2"))
lower = masked(Gather(Input("lower_masks"), lower_idx), Gather(Input("lower_prefix"), lower_idx), Gather(Input("lower_offsets"), lower_idx))
leaf_idx = Gather(lower, field(parts, "level_1"))
leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_prefix"), leaf_idx), Gather(Input("leaf_offsets"), leaf_idx))
voxel_idx = Gather(leaf, field(parts, "level_0"))
voxel_idx
"""

CIG3_INPUT_TYPES = {
    "query": Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.I32)),
    "upper_idx": Type(Shape(Dynamic()), ScalarType.I32),
    "upper_masks": Type(Shape(Dynamic()), Type(Shape(Static(512)), ScalarType.I64)),
    "upper_prefix": Type(Shape(Dynamic()), Type(Shape(Static(512)), ScalarType.I32)),
    "upper_offsets": Type(Shape(Dynamic()), ScalarType.I32),
    "lower_masks": Type(Shape(Dynamic()), Type(Shape(Static(64)), ScalarType.I64)),
    "lower_prefix": Type(Shape(Dynamic()), Type(Shape(Static(64)), ScalarType.I32)),
    "lower_offsets": Type(Shape(Dynamic()), ScalarType.I32),
    "leaf_masks": Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I64)),
    "leaf_prefix": Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I32)),
    "leaf_offsets": Type(Shape(Dynamic()), ScalarType.I32),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_grid(n_voxels=500, seed=42):
    """Build a 3-level CompressedCIG from random coordinates in [0, 4096)^3."""
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
    """50% grid hits + 50% random."""
    rng = np.random.RandomState(seed)
    N = grid_coords.shape[0]
    n_hits = n_queries // 2
    hits = grid_coords[rng.choice(N, n_hits, replace=True)]
    randoms = torch.from_numpy(rng.randint(0, 8192, (n_queries - n_hits, 3)).astype(np.int32))
    return torch.cat([hits, randoms], dim=0)


# ---------------------------------------------------------------------------
# Test 1: Emit and inspect
# ---------------------------------------------------------------------------


def test_emit_cig3_kernel():
    """Emit the 3-level kernel and verify structural markers."""
    code, ts, ml = emit_runnable_kernel(
        CIG3_KERNEL_PROGRAM,
        CIG3_INPUT_TYPES,
        kernel_name="gen_cig3",
        tile_input="query",
        tile_input_rank=3,
        tile_size=TILE,
        tile_scalar_inputs=["upper_idx"],
    )

    n_lines = len(code.splitlines())
    print(f"--- Generated 3-level CIG kernel ({n_lines} lines) ---")
    for i, line in enumerate(code.splitlines()[:30]):
        print(f"  {i + 1:3d}| {line}")
    print(f"  ... ({n_lines} total lines)")

    assert "@ct.kernel" in code
    assert "def gen_cig3(" in code
    assert "ct.uint64" in code
    assert code.count("Masked gather (prefix-sum") == 3, "Should have 3 masked gather blocks"
    assert "32^3 node" in code, "Upper level should be 32^3"
    assert "16^3 node" in code, "Lower level should be 16^3"
    assert "8^3 node" in code, "Leaf level should be 8^3"
    assert "ct.scatter(" in code

    print(f"  emit_cig3_kernel: {n_lines} lines, 3 masked levels -- PASSED")
    return code


# ---------------------------------------------------------------------------
# Test 2: Full pipeline -- torch root + cuTile 3-level chain vs numpy
# ---------------------------------------------------------------------------


def test_cig3_pipeline():
    """Full 2-step pipeline vs numpy reference."""
    code, ts, _ = emit_runnable_kernel(
        CIG3_KERNEL_PROGRAM,
        CIG3_INPUT_TYPES,
        kernel_name="gen_cig3_pipeline",
        tile_input="query",
        tile_input_rank=3,
        tile_size=TILE,
        tile_scalar_inputs=["upper_idx"],
    )
    kernel_fn = _compile_kernel(code, "gen_cig3_pipeline")

    cig, grid_coords = _make_test_grid(n_voxels=1000)
    queries = _make_mixed_queries(grid_coords, n_queries=3000)

    cig_cuda = cig.cuda()
    queries_cuda = queries.cuda().to(torch.int32)
    N = queries_cuda.shape[0]
    n_blocks = math.ceil(N / TILE)

    # --- Step 1: torch root lookup ---
    upper_idx = root_lookup(cig_cuda.root_coords, queries_cuda)

    # --- Step 2: cuTile 3-level chain ---
    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device="cuda")
    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        kernel_fn,
        (
            queries_cuda, upper_idx,
            cig_cuda.upper_masks, cig_cuda.upper_prefix.int(), cig_cuda.upper_offsets.int(),
            cig_cuda.lower_masks, cig_cuda.lower_prefix.int(), cig_cuda.lower_offsets.int(),
            cig_cuda.leaf_masks, cig_cuda.leaf_prefix.int(), cig_cuda.leaf_offsets.int(),
            result_t, TILE,
        ),
    )
    gpu_result = result_t[:N].cpu().numpy()

    # --- Numpy reference ---
    ref_result = cig3_ijk_to_index_numpy(cig, queries.numpy())

    np.testing.assert_array_equal(gpu_result, ref_result)
    n_hits = int((gpu_result >= 0).sum())
    n_total = N
    print(
        f"  cig3_pipeline: {n_total} queries ({n_hits} hits), "
        f"{cig.n_upper} upper, {cig.n_lower} lower, {cig.n_leaves} leaves -- "
        f"GPU == numpy ref -- PASSED"
    )


# ---------------------------------------------------------------------------
# Test 3: Multi-upper pipeline (coordinates spanning multiple root entries)
# ---------------------------------------------------------------------------


def test_cig3_multi_upper():
    """3-level chain with multiple upper nodes (different which_top values)."""
    rng = np.random.RandomState(42)
    coords = []
    for base_x in [0, 4096, 8192]:
        for _ in range(200):
            coords.append([base_x + rng.randint(0, 4096), rng.randint(0, 4096), rng.randint(0, 4096)])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))
    cig = build_compressed_cig3(ijk)

    code, ts, _ = emit_runnable_kernel(
        CIG3_KERNEL_PROGRAM,
        CIG3_INPUT_TYPES,
        kernel_name="gen_cig3_multi",
        tile_input="query",
        tile_input_rank=3,
        tile_size=TILE,
        tile_scalar_inputs=["upper_idx"],
    )
    kernel_fn = _compile_kernel(code, "gen_cig3_multi")

    queries = _make_mixed_queries(ijk, n_queries=2000)
    cig_cuda = cig.cuda()
    queries_cuda = queries.cuda().to(torch.int32)
    N = queries_cuda.shape[0]
    n_blocks = math.ceil(N / TILE)

    # Step 1: torch root lookup
    upper_idx = root_lookup(cig_cuda.root_coords, queries_cuda)

    # Step 2: cuTile 3-level chain
    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device="cuda")
    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        kernel_fn,
        (
            queries_cuda, upper_idx,
            cig_cuda.upper_masks, cig_cuda.upper_prefix.int(), cig_cuda.upper_offsets.int(),
            cig_cuda.lower_masks, cig_cuda.lower_prefix.int(), cig_cuda.lower_offsets.int(),
            cig_cuda.leaf_masks, cig_cuda.leaf_prefix.int(), cig_cuda.leaf_offsets.int(),
            result_t, TILE,
        ),
    )
    gpu_result = result_t[:N].cpu().numpy()

    # Numpy reference
    ref_result = cig3_ijk_to_index_numpy(cig, queries.numpy())
    np.testing.assert_array_equal(gpu_result, ref_result)

    n_hits = int((gpu_result >= 0).sum())
    print(
        f"  cig3_multi_upper: {N} queries ({n_hits} hits), "
        f"{cig.n_upper} upper nodes, GPU == numpy ref -- PASSED"
    )


# =========================================================================

if __name__ == "__main__":
    print("=== 3-level CIG end-to-end codegen ===")
    test_emit_cig3_kernel()
    print()
    test_cig3_pipeline()
    print()
    test_cig3_multi_upper()
    print("\nAll 3-level CIG end-to-end tests passed.")
