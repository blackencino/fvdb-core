# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for the Compact Index Grid (CIG) builder and ijk_to_index.

Tests:
  1. Builder produces correct structure (leaf counts, active counts, no gaps)
  2. ijk_to_index returns correct indices for active voxels
  3. ijk_to_index returns -1 for inactive / out-of-bounds coordinates
  4. PyTorch vectorized and numpy reference agree
  5. Roundtrip: build from coords, query those same coords, get valid indices
  6. Deduplication: duplicate input coords produce correct single-voxel entries
"""

import numpy as np
import torch

from fvdb_tile.prototype.cig import CIG, build_cig, cig_ijk_to_index, cig_ijk_to_index_numpy


def _make_coords(n_voxels=1000, seed=42):
    """Generate random voxel coordinates within a 2-level CIG range."""
    rng = np.random.RandomState(seed)
    # Coordinates in [0, 128) -- fits within 7 bits (3 for leaf + 4 for lower)
    coords = rng.randint(0, 128, (n_voxels, 3)).astype(np.int32)
    return torch.from_numpy(coords)


# ---------------------------------------------------------------------------
# Test 1: Builder structure
# ---------------------------------------------------------------------------


def test_builder_structure():
    """CIG builder produces valid lower/leaf_arr with correct counts."""
    coords = _make_coords(500, seed=10)
    cig = build_cig(coords)

    # Deduplicated count should be <= input count
    unique_coords = torch.unique(coords, dim=0)
    assert cig.n_active == unique_coords.shape[0], (
        f"n_active={cig.n_active}, expected {unique_coords.shape[0]}"
    )

    # Lower array shape
    assert cig.lower.shape == (16, 16, 16), f"lower shape: {cig.lower.shape}"

    # Leaf array shape
    assert cig.leaf_arr.shape[0] == cig.n_leaves
    assert cig.leaf_arr.shape[1:] == (8, 8, 8)

    # Number of active entries in lower == number of leaves
    n_active_lower = (cig.lower >= 0).sum().item()
    assert n_active_lower == cig.n_leaves, (
        f"Active lower entries={n_active_lower}, n_leaves={cig.n_leaves}"
    )

    # Leaf indices in lower should be contiguous [0, K)
    active_leaf_indices = cig.lower[cig.lower >= 0]
    assert active_leaf_indices.min().item() == 0
    assert active_leaf_indices.max().item() == cig.n_leaves - 1

    # Total active voxels in leaf_arr should match n_active
    n_active_in_leaves = (cig.leaf_arr >= 0).sum().item()
    assert n_active_in_leaves == cig.n_active, (
        f"Active in leaves={n_active_in_leaves}, n_active={cig.n_active}"
    )

    # Voxel indices should be contiguous [0, n_active)
    all_indices = cig.leaf_arr[cig.leaf_arr >= 0]
    assert all_indices.min().item() == 0
    assert all_indices.max().item() == cig.n_active - 1
    assert len(torch.unique(all_indices)) == cig.n_active, "Voxel indices should be unique"

    print(
        f"  builder_structure: {cig.n_active} voxels, {cig.n_leaves} leaves, "
        f"{cig.num_bytes} bytes -- PASSED"
    )


# ---------------------------------------------------------------------------
# Test 2: ijk_to_index roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip():
    """Query the same coords used to build the CIG -- all should return valid indices."""
    coords = _make_coords(800, seed=20)
    cig = build_cig(coords)

    # Query all unique input coordinates
    unique_coords = torch.unique(coords, dim=0)
    result = cig_ijk_to_index(cig, unique_coords)

    # All should be >= 0
    assert (result >= 0).all(), f"Some active coords got -1: {(result < 0).sum().item()} failures"

    # All indices should be in [0, n_active)
    assert result.max().item() < cig.n_active
    assert result.min().item() >= 0

    # All indices should be unique (bijective mapping)
    assert len(torch.unique(result)) == unique_coords.shape[0], "Indices should be unique per coord"

    print(f"  roundtrip: {unique_coords.shape[0]} coords all mapped correctly -- PASSED")


# ---------------------------------------------------------------------------
# Test 3: ijk_to_index returns -1 for inactive coords
# ---------------------------------------------------------------------------


def test_inactive_coords():
    """Coordinates not in the grid should return -1."""
    # Build a small grid with known coordinates
    coords = torch.tensor([[10, 20, 30], [10, 20, 31], [11, 20, 30]], dtype=torch.int32)
    cig = build_cig(coords)

    # Query coordinates that are definitely not in the grid
    bad_coords = torch.tensor(
        [
            [0, 0, 0],  # empty lower node
            [10, 20, 32],  # same lower node but different leaf-local
            [50, 50, 50],  # different lower node entirely
        ],
        dtype=torch.int32,
    )

    # Check that 10,20,32 is not the same leaf-local as any input
    # Input leaf-locals are (10&7, 20&7, 30&7)=(2,4,6), (2,4,7), (3,4,6)
    # Query 10,20,32 -> leaf-local (2,4,0) -- not in the grid
    result = cig_ijk_to_index(cig, bad_coords)
    assert (result == -1).all(), f"Expected all -1, got {result.tolist()}"

    print(f"  inactive_coords: 3 invalid queries all returned -1 -- PASSED")


# ---------------------------------------------------------------------------
# Test 4: PyTorch vs numpy agreement
# ---------------------------------------------------------------------------


def test_pytorch_vs_numpy():
    """PyTorch vectorized and numpy reference produce identical results."""
    coords = _make_coords(2000, seed=30)
    cig = build_cig(coords)

    # Mix of valid and invalid query coordinates
    rng = np.random.RandomState(99)
    query_np = rng.randint(-10, 140, (500, 3)).astype(np.int32)
    query_t = torch.from_numpy(query_np)

    result_pt = cig_ijk_to_index(cig, query_t).numpy()
    result_np = cig_ijk_to_index_numpy(cig, query_np)

    np.testing.assert_array_equal(result_pt, result_np)

    n_active = (result_pt >= 0).sum()
    n_inactive = (result_pt == -1).sum()
    print(
        f"  pytorch_vs_numpy: 500 queries, {n_active} hits, {n_inactive} misses, "
        f"results match -- PASSED"
    )


# ---------------------------------------------------------------------------
# Test 5: Deduplication
# ---------------------------------------------------------------------------


def test_deduplication():
    """Duplicate input coordinates produce a single voxel entry."""
    # 3 unique coords, each repeated multiple times
    coords = torch.tensor(
        [
            [10, 20, 30],
            [10, 20, 30],
            [10, 20, 30],
            [11, 21, 31],
            [11, 21, 31],
            [12, 22, 32],
        ],
        dtype=torch.int32,
    )
    cig = build_cig(coords)

    assert cig.n_active == 3, f"Expected 3 unique voxels, got {cig.n_active}"

    # All 3 unique coords should return valid indices
    unique = torch.tensor([[10, 20, 30], [11, 21, 31], [12, 22, 32]], dtype=torch.int32)
    result = cig_ijk_to_index(cig, unique)
    assert (result >= 0).all()
    assert len(torch.unique(result)) == 3

    print(f"  deduplication: 6 inputs -> 3 unique voxels -- PASSED")


# ---------------------------------------------------------------------------
# Test 6: Memory footprint
# ---------------------------------------------------------------------------


def test_memory_footprint():
    """CIG memory matches expected formula."""
    coords = _make_coords(5000, seed=40)
    cig = build_cig(coords)

    expected_bytes = 16 * 16 * 16 * 4 + cig.n_leaves * 8 * 8 * 8 * 4
    assert cig.num_bytes == expected_bytes, f"{cig.num_bytes} vs expected {expected_bytes}"

    print(
        f"  memory_footprint: {cig.n_leaves} leaves, "
        f"{cig.num_bytes:,} bytes ({cig.num_bytes / 1024:.1f} KB) -- PASSED"
    )


# =========================================================================

if __name__ == "__main__":
    print("=== CIG builder and ijk_to_index tests ===")
    test_builder_structure()
    test_roundtrip()
    test_inactive_coords()
    test_pytorch_vs_numpy()
    test_deduplication()
    test_memory_footprint()
    print("\nAll CIG tests passed.")
