# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
cuTile feasibility test: neighbor gather predicate on a (8,8,8) leaf.

Hand-written cuTile kernel implementing the DSL expression:

    Map(offsets, o => And(InBounds(Add(coord, o), 0, 8),
                          Gather(mask, Add(coord, o))))

For each active voxel, compute 6 face-neighbor coordinates, bounds-check,
and gather from the mask. Compare results against a numpy reference.

Key cuTile constraint: ct.load requires power-of-two tile dimensions, so
we use ct.gather throughout (which has no such restriction) and pad the
offset tile dimension to 8 (next power-of-two above 6).
"""

import numpy as np
import torch

import cuda.tile as ct

ConstInt = ct.Constant[int]

FACE_OFFSETS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# cuTile kernel: neighbor gather predicate
# ---------------------------------------------------------------------------

@ct.kernel
def neighbor_predicate_kernel(
    mask,
    active_coords,
    offsets,
    result,
    TILE_OFFSETS: ConstInt,
):
    """For one active voxel (bid), check which of 6 neighbors are active.

    mask:          (8, 8, 8) i32 -- leaf node data (-1 = inactive)
    active_coords: (N, 3) i32    -- coordinates of active voxels
    offsets:       (6, 3) i32    -- face-neighbor offsets
    result:        (N, TILE_OFFSETS) i32 -- output (padded; first 6 cols valid)
    TILE_OFFSETS:  8             -- next power-of-two >= 6
    """
    bid = ct.bid(0)

    # Load this voxel's 3 coordinate components via scalar gathers.
    ci = ct.gather(active_coords, (bid, 0))
    cj = ct.gather(active_coords, (bid, 1))
    ck = ct.gather(active_coords, (bid, 2))

    # Offset index tile: (8,) padded from 6. Indices 6,7 are OOB on the
    # (6,3) offsets array, so ct.gather returns padding_value=0 for them.
    idx = ct.arange(TILE_OFFSETS, dtype=ct.int32)

    # Gather offset components: each (TILE_OFFSETS,)
    oi = ct.gather(offsets, (idx, 0), check_bounds=True, padding_value=0)
    oj = ct.gather(offsets, (idx, 1), check_bounds=True, padding_value=0)
    ok = ct.gather(offsets, (idx, 2), check_bounds=True, padding_value=0)

    # Compute neighbor coordinates: scalar + (8,) -> (8,) via broadcast.
    ni = ci + oi
    nj = cj + oj
    nk = ck + ok

    # Gather from 3D mask with bounds checking.
    # OOB coordinates (from boundary voxels or padding slots) get -1.
    vals = ct.gather(mask, (ni, nj, nk), check_bounds=True, padding_value=-1)

    # Active if gathered value >= 0.
    is_active = ct.astype(vals >= 0, ct.int32)

    # Store the full padded row. Padding slots (6,7) will contain False
    # because their neighbor coords are (coord+0) which may look active,
    # but we only read the first 6 columns on the host side.
    ct.scatter(result, (bid, idx), is_active, check_bounds=True)


# ---------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------

TILE_OFFSETS = 8  # next power-of-two >= 6


def run_neighbor_predicate(mask_np, active_coords_np, offsets_np):
    """Launch the cuTile neighbor predicate kernel and return (N, 6) bool."""
    N = active_coords_np.shape[0]

    mask_t = torch.from_numpy(mask_np.copy()).cuda()
    coords_t = torch.from_numpy(active_coords_np.copy()).cuda()
    offsets_t = torch.from_numpy(offsets_np.copy()).cuda()
    result_t = torch.zeros(N, TILE_OFFSETS, dtype=torch.int32, device="cuda")

    grid = (N,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        neighbor_predicate_kernel,
        (mask_t, coords_t, offsets_t, result_t, TILE_OFFSETS),
    )

    # Only the first 6 columns are valid (rest is padding).
    return result_t.cpu().numpy()[:, :offsets_np.shape[0]].astype(bool)


# ---------------------------------------------------------------------------
# Numpy reference (same logic as the DSL evaluator)
# ---------------------------------------------------------------------------

def numpy_neighbor_predicate(mask_np, active_coords_np, offsets_np):
    """Brute-force numpy reference for the neighbor predicate."""
    N = active_coords_np.shape[0]
    N_OFFSETS = offsets_np.shape[0]
    result = np.zeros((N, N_OFFSETS), dtype=bool)

    for i in range(N):
        coord = active_coords_np[i]
        for j in range(N_OFFSETS):
            nbr = coord + offsets_np[j]
            in_bounds = np.all(nbr >= 0) and np.all(nbr < 8)
            if in_bounds and mask_np[nbr[0], nbr[1], nbr[2]] >= 0:
                result[i, j] = True
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_corner_voxel():
    """Corner voxel (0,0,0) has 3 of 6 neighbors out-of-bounds."""
    mask = np.ones((8, 8, 8), dtype=np.int32)
    coords = np.array([[0, 0, 0]], dtype=np.int32)

    cutile_result = run_neighbor_predicate(mask, coords, FACE_OFFSETS)
    numpy_result = numpy_neighbor_predicate(mask, coords, FACE_OFFSETS)

    np.testing.assert_array_equal(cutile_result, numpy_result)
    assert cutile_result.sum() == 3, f"Expected 3 active, got {cutile_result.sum()}"
    print("  corner_voxel: (0,0,0) has 3/6 active neighbors -- PASSED")


def test_center_voxel_all_active():
    """Center voxel (4,4,4) in a fully active leaf has all 6 neighbors."""
    mask = np.ones((8, 8, 8), dtype=np.int32)
    coords = np.array([[4, 4, 4]], dtype=np.int32)

    cutile_result = run_neighbor_predicate(mask, coords, FACE_OFFSETS)
    numpy_result = numpy_neighbor_predicate(mask, coords, FACE_OFFSETS)

    np.testing.assert_array_equal(cutile_result, numpy_result)
    assert cutile_result.sum() == 6, f"Expected 6 active, got {cutile_result.sum()}"
    print("  center_voxel: (4,4,4) has 6/6 active neighbors -- PASSED")


def test_all_inactive_neighbors():
    """Active voxel surrounded by inactive neighbors."""
    mask = np.full((8, 8, 8), -1, dtype=np.int32)
    mask[4, 4, 4] = 1
    coords = np.array([[4, 4, 4]], dtype=np.int32)

    cutile_result = run_neighbor_predicate(mask, coords, FACE_OFFSETS)
    numpy_result = numpy_neighbor_predicate(mask, coords, FACE_OFFSETS)

    np.testing.assert_array_equal(cutile_result, numpy_result)
    assert cutile_result.sum() == 0, f"Expected 0 active, got {cutile_result.sum()}"
    print("  all_inactive_neighbors: lone active voxel, 0/6 -- PASSED")


def test_neighbor_predicate():
    """Full neighbor predicate: cuTile vs numpy on random (8,8,8) leaf."""
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    mask_bool = leaf_data >= 0
    active_coords = np.argwhere(mask_bool).astype(np.int32)
    N = active_coords.shape[0]

    cutile_result = run_neighbor_predicate(leaf_data, active_coords, FACE_OFFSETS)
    numpy_result = numpy_neighbor_predicate(leaf_data, active_coords, FACE_OFFSETS)

    np.testing.assert_array_equal(cutile_result, numpy_result)

    total_active_nbrs = int(numpy_result.sum())
    print(
        f"  neighbor_predicate: {N} active voxels, "
        f"{total_active_nbrs} active neighbors -- PASSED"
    )


if __name__ == "__main__":
    print("=== cuTile neighbor gather predicate ===")
    test_corner_voxel()
    test_center_voxel_all_active()
    test_all_inactive_neighbors()
    test_neighbor_predicate()
    print("\nAll cuTile gather tests passed.")
