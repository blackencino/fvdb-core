# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Cross-leaf neighbor finding: the critical composability test.

Composes two patterns that individually work:
  - Neighbor offsets from test_neighbors.py (add offsets, filter active)
  - Hierarchical Decompose + chained Gather from test_two_level.py

The expression: for each active voxel with a global coordinate, compute 6
face-neighbor global coordinates, then use Decompose + chained Gather through
a two-level grid to check if each neighbor is active (potentially in a
different leaf node).

This is the hardest composability test: cross-boundary traversal through a
hierarchical sparse structure, expressed purely as a DSL string.
"""

import torch

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# ---------------------------------------------------------------------------
# DSL programs
# ---------------------------------------------------------------------------

# For a single global coordinate, check which of 6 face neighbors are active
# by traversing the two-level grid hierarchy.
#
# The chain for each neighbor coordinate:
#   1. Decompose into (level_0 = leaf-local, level_1 = lower-node)
#   2. Gather lower node using level_1 -> leaf_idx
#   3. Gather leaf block from leaf array using leaf_idx
#   4. Gather voxel value from leaf block using level_0
#   5. Check if value >= 0 (active)
#
# This is the cross-leaf version of:
#   GE(Gather(mask, Add(coord, o)), Const(0))
# but with the single Gather replaced by a 4-step hierarchical chain.

CROSS_LEAF_PREDICATE = """
nbr = Add(Input("coord"), Input("offset"))
parts = Decompose(nbr, Const([3, 4]))
leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
leaf_node = Gather(Input("leaf_arr"), leaf_idx)
voxel_val = Gather(leaf_node, field(parts, "level_0"))
is_active = GE(voxel_val, Const(0))
is_active
"""

# Full pipeline: for each active voxel, check all 6 neighbors via the
# hierarchical chain. Uses Each to iterate over active coords and Map to
# iterate over the 6 offsets.
CROSS_LEAF_FULL = """
nbr = Add(Input("coord"), Input("offset"))
parts = Decompose(nbr, Const([3, 4]))
leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
leaf_node = Gather(Input("leaf_arr"), leaf_idx)
voxel_val = Gather(leaf_node, field(parts, "level_0"))
is_active = GE(voxel_val, Const(0))
is_active
"""


# ---------------------------------------------------------------------------
# Test data builder
# ---------------------------------------------------------------------------

FACE_OFFSETS = torch.tensor(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=torch.int32,
)


def _build_grid(seed=42):
    """Build a two-level grid with known active voxels near leaf boundaries."""
    gen = torch.Generator().manual_seed(seed)

    # Lower level: (16, 16, 16) mapping to leaf indices
    lower_data = torch.full((16, 16, 16), -1, dtype=torch.int32)

    # Place a few adjacent lower nodes so cross-leaf neighbors exist
    active_lower = [(3, 4, 5), (3, 4, 6), (3, 5, 5), (4, 4, 5)]
    leaf_blocks = []
    for i, (lx, ly, lz) in enumerate(active_lower):
        lower_data[lx, ly, lz] = i
        leaf = torch.full((8, 8, 8), -1, dtype=torch.int32)
        n_active = torch.randint(100, 300, (1,), generator=gen).item()
        positions = torch.randperm(512, generator=gen)[:n_active]
        for idx, pos in enumerate(positions):
            pos_val = pos.item()
            vx = pos_val // 64
            vy = (pos_val // 8) % 8
            vz = pos_val % 8
            leaf[vx, vy, vz] = i * 1000 + idx
        leaf_blocks.append(leaf)

    leaf_data = torch.stack(leaf_blocks)
    return lower_data, leaf_data, active_lower


def _global_coord(lower_coord, leaf_coord):
    """Convert (lower_coord, leaf_coord) to global coordinate."""
    return torch.tensor([
        lower_coord[0] * 8 + leaf_coord[0],
        lower_coord[1] * 8 + leaf_coord[1],
        lower_coord[2] * 8 + leaf_coord[2],
    ], dtype=torch.int32)


def _reference_lookup(lower_data, leaf_data, global_coord):
    """Reference: hierarchical lookup for one global coordinate.

    Returns the voxel value, or -1 if any level is inactive or OOB.
    """
    ll = (global_coord >> 3) & 15
    vl = global_coord & 7

    # Check lower bounds
    if torch.any(ll < 0).item() or torch.any(ll >= 16).item():
        return -1

    leaf_idx = lower_data[ll[0].item(), ll[1].item(), ll[2].item()]
    if leaf_idx < 0 or leaf_idx >= leaf_data.shape[0]:
        return -1

    # Check leaf bounds
    if torch.any(vl < 0).item() or torch.any(vl >= 8).item():
        return -1

    return leaf_data[leaf_idx.item(), vl[0].item(), vl[1].item(), vl[2].item()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_coord_cross_leaf():
    """Single global coordinate through the hierarchical chain via DSL."""
    lower_data, leaf_data, active_lower = _build_grid()

    # Pick a coord in the first leaf, near the boundary
    lx, ly, lz = active_lower[0]
    leaf = leaf_data[0]
    active_leaf = torch.nonzero(leaf >= 0).to(torch.int32)
    # Pick a boundary voxel (leaf_coord component == 7) if available
    boundary = [c for c in active_leaf if c[2].item() == 7]
    if boundary:
        vl = boundary[0].to(torch.int32)
    else:
        vl = active_leaf[0]

    global_coord = _global_coord((lx, ly, lz), (vl[0].item(), vl[1].item(), vl[2].item()))

    # Run for each of 6 offsets individually
    lower_val = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data,
    )
    leaf_arr_val = Value(
        Type(
            Shape(Static(leaf_data.shape[0])),
            Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
        ),
        leaf_data,
    )

    dsl_results = []
    ref_results = []
    for off in FACE_OFFSETS:
        nbr_coord = global_coord + off
        coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), global_coord)
        offset_val = Value(Type(Shape(Static(3)), ScalarType.I32), off)

        _, result = run(CROSS_LEAF_PREDICATE, {
            "coord": coord_val,
            "offset": offset_val,
            "lower": lower_val,
            "leaf_arr": leaf_arr_val,
        })

        dsl_active = bool(result.data)
        dsl_results.append(dsl_active)

        ref_val = _reference_lookup(lower_data, leaf_data, nbr_coord)
        ref_results.append((ref_val >= 0).item() if isinstance(ref_val, torch.Tensor) else ref_val >= 0)

    assert dsl_results == ref_results, f"DSL: {dsl_results}, ref: {ref_results}"

    n_active = sum(ref_results)
    vl_list = [vl[0].item(), vl[1].item(), vl[2].item()]
    is_boundary = any(c == 7 or c == 0 for c in vl_list)
    print(
        f"  single_coord: global={global_coord}, leaf_local={vl}, "
        f"boundary={is_boundary}, {n_active}/6 active neighbors -- PASSED"
    )


def test_cross_boundary_neighbor():
    """Verify a neighbor that crosses a leaf boundary is correctly resolved."""
    lower_data, leaf_data, active_lower = _build_grid()

    # active_lower[0] = (3,4,5), active_lower[1] = (3,4,6)
    # A voxel at leaf_coord (x,y,7) in lower (3,4,5) has z-neighbor in lower (3,4,6)
    lx, ly, lz = active_lower[0]
    leaf = leaf_data[0]

    # Find a voxel with z=7 in the first leaf
    z7_mask = (leaf >= 0) & (torch.arange(8, device=leaf.device).reshape(1, 1, 8) == 7)
    z7_voxels = torch.nonzero(z7_mask)
    if len(z7_voxels) == 0:
        print("  cross_boundary: no z=7 voxels in first leaf -- SKIPPED")
        return

    vl = z7_voxels[0].to(torch.int32)
    global_coord = _global_coord((lx, ly, lz), (vl[0].item(), vl[1].item(), vl[2].item()))

    # The +z neighbor is at global_coord + (0,0,1)
    # This crosses into lower node (3,4,6), leaf_local (vl[0], vl[1], 0)
    nbr_global = global_coord + torch.tensor([0, 0, 1], dtype=torch.int32)
    nbr_ll = (nbr_global >> 3) & 15
    nbr_vl = nbr_global & 7

    # Verify the neighbor IS in a different lower node
    assert nbr_ll[2].item() != lz, f"Neighbor should cross leaf boundary: {nbr_ll}"

    # Run DSL
    lower_val = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data,
    )
    leaf_arr_val = Value(
        Type(
            Shape(Static(leaf_data.shape[0])),
            Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
        ),
        leaf_data,
    )
    coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), global_coord)
    offset_val = Value(Type(Shape(Static(3)), ScalarType.I32), torch.tensor([0, 0, 1], dtype=torch.int32))

    _, result = run(CROSS_LEAF_PREDICATE, {
        "coord": coord_val,
        "offset": offset_val,
        "lower": lower_val,
        "leaf_arr": leaf_arr_val,
    })

    dsl_active = bool(result.data)
    ref_val = _reference_lookup(lower_data, leaf_data, nbr_global)
    ref_active = (ref_val >= 0).item() if isinstance(ref_val, torch.Tensor) else ref_val >= 0

    assert dsl_active == ref_active, f"DSL: {dsl_active}, ref: {ref_active}"
    print(
        f"  cross_boundary: voxel {global_coord} -> +z neighbor {nbr_global}, "
        f"crosses from lower ({lx},{ly},{lz}) to ({nbr_ll[0].item()},{nbr_ll[1].item()},{nbr_ll[2].item()}), "
        f"active={dsl_active} -- PASSED"
    )


def test_batch_cross_leaf():
    """Batch of active voxels, all 6 neighbors, through the hierarchy."""
    lower_data, leaf_data, active_lower = _build_grid()

    lower_val = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data,
    )
    leaf_arr_val = Value(
        Type(
            Shape(Static(leaf_data.shape[0])),
            Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
        ),
        leaf_data,
    )

    # Collect all active global coords across all leaves
    all_global_coords = []
    for li, (lx, ly, lz) in enumerate(active_lower):
        leaf = leaf_data[li]
        active_leaf = torch.nonzero(leaf >= 0).to(torch.int32)
        for vl in active_leaf:
            gc = _global_coord((lx, ly, lz), (vl[0].item(), vl[1].item(), vl[2].item()))
            all_global_coords.append(gc)

    all_global_coords = torch.stack(all_global_coords)
    N = len(all_global_coords)

    # Test a subset (full batch would be slow through per-coord DSL evaluation)
    gen = torch.Generator().manual_seed(42)
    test_indices = torch.randperm(N, generator=gen)[: min(50, N)]

    n_cross = 0
    n_total = 0
    for idx in test_indices:
        gc = all_global_coords[idx]
        for off in FACE_OFFSETS:
            nbr = gc + off
            coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), gc)
            offset_val = Value(Type(Shape(Static(3)), ScalarType.I32), off)

            _, result = run(CROSS_LEAF_PREDICATE, {
                "coord": coord_val,
                "offset": offset_val,
                "lower": lower_val,
                "leaf_arr": leaf_arr_val,
            })

            dsl_active = bool(result.data)
            ref_val = _reference_lookup(lower_data, leaf_data, nbr)
            ref_active = (ref_val >= 0).item() if isinstance(ref_val, torch.Tensor) else ref_val >= 0

            assert dsl_active == ref_active, (
                f"Mismatch at {gc} + {off} = {nbr}: DSL={dsl_active}, ref={ref_active}"
            )

            # Check if this neighbor crosses a leaf boundary
            gc_lower = (gc >> 3) & 15
            nbr_lower = (nbr >> 3) & 15
            if not torch.equal(gc_lower, nbr_lower):
                n_cross += 1
            n_total += 1

    n_voxels = len(test_indices)
    print(
        f"  batch_cross_leaf: {n_voxels} voxels x 6 offsets = {n_total} lookups, "
        f"{n_cross} cross-leaf, all correct -- PASSED"
    )


# =========================================================================

if __name__ == "__main__":
    print("=== Cross-leaf neighbor finding ===")
    test_single_coord_cross_leaf()
    test_cross_boundary_neighbor()
    test_batch_cross_leaf()
    print("\nAll cross-leaf tests passed.")
