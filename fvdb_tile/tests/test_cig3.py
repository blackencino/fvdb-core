# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the 3-level Compressed CIG with root lookup.

Verifies:
  1. build_compressed_cig3 produces correct structure
  2. root_lookup finds upper nodes correctly
  3. Full ijk_to_index chain via numpy reference
  4. DSL evaluator matches numpy reference
"""

import numpy as np
import torch

from fvdb_tile.prototype.cig import (
    CompressedCIG3,
    _masked_lookup_numpy,
    build_compressed_cig3,
    cig3_ijk_to_index_numpy,
    root_lookup,
)
from fvdb_tile.prototype.dsl_eval import run as dsl_run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Static, Type


# ---------------------------------------------------------------------------
# Test 1: Builder structure
# ---------------------------------------------------------------------------


def test_build_cig3():
    """Build a 3-level CIG and check structural invariants."""
    rng = np.random.RandomState(42)
    # Coordinates spanning multiple upper nodes (different which_top values)
    coords = []
    for _ in range(500):
        x = rng.randint(0, 4096)
        y = rng.randint(0, 4096)
        z = rng.randint(0, 4096)
        coords.append([x, y, z])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))

    cig = build_compressed_cig3(ijk)

    assert cig.root_coords.shape[1] == 3
    assert cig.n_upper == cig.root_coords.shape[0]
    assert cig.upper_masks.shape == (cig.n_upper, 512)
    assert cig.upper_abs_prefix.shape == (cig.n_upper, 512)
    assert cig.lower_masks.shape == (cig.n_lower, 64)
    assert cig.lower_abs_prefix.shape == (cig.n_lower, 64)
    assert cig.leaf_masks.shape == (cig.n_leaves, 8)
    assert cig.leaf_abs_prefix.shape == (cig.n_leaves, 8)

    print(
        f"  build_cig3: {cig.n_active} voxels, {cig.n_leaves} leaves, "
        f"{cig.n_lower} lower, {cig.n_upper} upper, "
        f"{cig.num_bytes / 1024:.1f} KB -- PASSED"
    )


# ---------------------------------------------------------------------------
# Test 2: Root lookup
# ---------------------------------------------------------------------------


def test_root_lookup():
    """Root linear scan finds correct upper node indices."""
    rng = np.random.RandomState(42)
    coords = []
    for _ in range(200):
        coords.append([rng.randint(0, 4096), rng.randint(0, 4096), rng.randint(0, 4096)])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))
    cig = build_compressed_cig3(ijk)

    # Query with grid coordinates (all should match)
    upper_idx = root_lookup(cig.root_coords, ijk)
    assert (upper_idx >= 0).all(), "All grid coords should find a root match"

    # Query with out-of-range coordinates (should return -1)
    bad_queries = torch.full((10, 3), 999999, dtype=torch.int32)
    bad_idx = root_lookup(cig.root_coords, bad_queries)
    assert (bad_idx == -1).all(), "Out-of-range should return -1"

    n_upper = cig.n_upper
    print(f"  root_lookup: {n_upper} upper nodes, all grid hits found, OOB returns -1 -- PASSED")


# ---------------------------------------------------------------------------
# Test 3: Numpy reference ijk_to_index
# ---------------------------------------------------------------------------


def test_cig3_numpy_reference():
    """Numpy reference produces correct results for known coordinates."""
    rng = np.random.RandomState(77)
    coords = []
    for _ in range(300):
        coords.append([rng.randint(0, 4096), rng.randint(0, 4096), rng.randint(0, 4096)])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))
    cig = build_compressed_cig3(ijk)

    query_np = ijk.numpy()
    result = cig3_ijk_to_index_numpy(cig, query_np)

    # All grid coordinates should be active (return >= 0)
    n_active = (result >= 0).sum()
    assert n_active == cig.n_active, f"Expected {cig.n_active} active, got {n_active}"

    # Indices should be unique and in [0, n_active)
    active_indices = result[result >= 0]
    assert len(set(active_indices)) == cig.n_active, "Indices should be unique"
    assert active_indices.min() >= 0
    assert active_indices.max() < cig.n_active

    # Random misses
    miss_coords = np.array([[4095, 4095, 4095]], dtype=np.int32)
    miss_result = cig3_ijk_to_index_numpy(cig, miss_coords)
    # This might or might not be in the grid -- just verify it's deterministic
    print(f"  cig3_numpy_ref: {cig.n_active} voxels, all found, indices unique -- PASSED")


# ---------------------------------------------------------------------------
# Test 4: DSL evaluator for one level
# ---------------------------------------------------------------------------

MASKED_CIG3_LEAF_PROGRAM = """
parts = Decompose(Input("query"), Const([3, 4, 5]))
leaf = masked(Gather(Input("leaf_masks"), Input("leaf_idx")), Gather(Input("leaf_abs_prefix"), Input("leaf_idx")))
voxel_idx = Gather(leaf, field(parts, "level_0"))
voxel_idx
"""


def test_cig3_dsl_leaf_level():
    """DSL evaluator handles a single masked leaf level correctly."""
    rng = np.random.RandomState(99)
    coords = []
    for _ in range(200):
        coords.append([rng.randint(0, 4096), rng.randint(0, 4096), rng.randint(0, 4096)])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))
    cig = build_compressed_cig3(ijk)

    # Pick a known active voxel and resolve its leaf manually
    query_np = ijk[0].numpy()
    fl = query_np & 7
    wt = query_np >> 12
    ul = (query_np >> 7) & 31
    ll = (query_np >> 3) & 15

    ref_result = cig3_ijk_to_index_numpy(cig, query_np.reshape(1, 3))
    assert ref_result[0] >= 0, "First grid coord should be active"

    flat_ul = int(ul[0]) * 1024 + int(ul[1]) * 32 + int(ul[2])
    flat_ll = int(ll[0]) * 256 + int(ll[1]) * 16 + int(ll[2])

    root_np = cig.root_coords.cpu().numpy()
    upper_idx = -1
    for r in range(root_np.shape[0]):
        if np.array_equal(root_np[r], wt):
            upper_idx = r
            break
    assert upper_idx >= 0

    lower_idx = _masked_lookup_numpy(
        cig.upper_masks.cpu().numpy()[upper_idx],
        cig.upper_abs_prefix.cpu().numpy()[upper_idx],
        flat_ul,
    )
    assert lower_idx >= 0

    leaf_idx = _masked_lookup_numpy(
        cig.lower_masks.cpu().numpy()[lower_idx],
        cig.lower_abs_prefix.cpu().numpy()[lower_idx],
        flat_ll,
    )
    assert leaf_idx >= 0

    query_val = Value(Type(Shape(Static(3)), ScalarType.I32), query_np)
    leaf_idx_val = Value(Type(Shape(), ScalarType.I32), np.int32(leaf_idx))
    masks_val = Value(
        Type(Shape(Static(cig.n_leaves)), Type(Shape(Static(8)), ScalarType.I64)),
        cig.leaf_masks.cpu().numpy(),
    )
    abs_prefix_val = Value(
        Type(Shape(Static(cig.n_leaves)), Type(Shape(Static(8)), ScalarType.I32)),
        cig.leaf_abs_prefix.cpu().numpy(),
    )

    _, result = dsl_run(MASKED_CIG3_LEAF_PROGRAM, {
        "query": query_val,
        "leaf_idx": leaf_idx_val,
        "leaf_masks": masks_val,
        "leaf_abs_prefix": abs_prefix_val,
    })

    dsl_idx = int(result.data)
    assert dsl_idx == ref_result[0], f"DSL {dsl_idx} != ref {ref_result[0]}"
    print(f"  cig3_dsl_leaf: voxel {query_np} -> idx {dsl_idx}, matches numpy ref -- PASSED")


# =========================================================================

if __name__ == "__main__":
    print("=== 3-level CIG tests ===")
    test_build_cig3()
    print()
    test_root_lookup()
    print()
    test_cig3_numpy_reference()
    print()
    test_cig3_dsl_leaf_level()
    print("\nAll 3-level CIG tests passed.")
