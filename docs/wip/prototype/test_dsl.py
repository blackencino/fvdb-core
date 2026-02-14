# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
DSL prototype tests: programs are strings, parsed into AST, type-checked,
and executed against numpy data.

Three programs:
  1. Where + Gather pipeline
  2. Neighbor finding (jagged emergence)
  3. Two-level hierarchical chain
"""

import numpy as np

from docs.wip.prototype.dsl_eval import run
from docs.wip.prototype.ops import Value
from docs.wip.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# =========================================================================
# Program 1: Where + Gather
# =========================================================================

WHERE_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
idx = Gather(Input("leaf"), active)
idx
"""


def test_where_program():
    """Parse + type-check + execute the Where pipeline from a string."""

    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    # ---- PROGRAM ----
    types, result = run(WHERE_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    print("Program 1 types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    assert types["mask"] == Type(Shape(Static(8), Static(8), Static(8)), ScalarType.BOOL)
    assert types["active"] == Type(Shape(Dynamic()), coord_type(3))
    assert types["idx"] == Type(Shape(Dynamic()), ScalarType.I32)

    # Data checks
    expected_count = np.sum(leaf_data >= 0)
    assert result.data.shape[0] == expected_count
    assert np.all(result.data >= 0)

    # Verify gathered values match manual lookup
    expected_coords = np.argwhere(leaf_data >= 0).astype(np.int32)
    expected_vals = leaf_data[tuple(expected_coords[:, i] for i in range(3))]
    np.testing.assert_array_equal(result.data, expected_vals)

    print(f"  -> {expected_count} active voxels, all values correct")


# =========================================================================
# Program 2: Neighbor finding
# =========================================================================

NEIGHBOR_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
nbrs = Each(active, a => Map(Input("offsets"), o => Add(a, o)))
filtered = Each(nbrs, cs => Gather(cs, Where(Map(cs, c => And(InBounds(c, Const(0), Const(8)), Gather(mask, c))))))
filtered
"""


def test_neighbor_program():
    """Neighbor finding as a DSL string. Verify jagged type emerges."""

    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    offsets_data = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int32,
    )
    offsets = Value(
        Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
        offsets_data,
    )

    # ---- PROGRAM ----
    types, result = run(NEIGHBOR_PROGRAM, {"leaf": leaf, "offsets": offsets})

    # ---- EXTRACTION ----
    print("Program 2 types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    # Key assertion: filtered should be (*) over (~) over (3) over i32
    assert result.type.iteration_shape == Shape(Dynamic())
    inner = result.type.element_type
    assert isinstance(inner, Type)
    assert isinstance(inner.iteration_shape.extents[0], Jagged), (
        f"Expected jagged inner, got {inner.iteration_shape.extents[0]!r}"
    )

    # Verify against brute-force reference
    mask_data = leaf_data >= 0
    active_coords = np.argwhere(mask_data).astype(np.int32)

    assert len(result.data) == len(active_coords)

    neighbor_counts = []
    for i, coord in enumerate(active_coords):
        nbrs = coord[np.newaxis, :] + offsets_data
        valid = [nb for nb in nbrs
                 if np.all(nb >= 0) and np.all(nb < 8) and mask_data[nb[0], nb[1], nb[2]]]
        ref = np.array(valid, dtype=np.int32).reshape(-1, 3)

        actual = result.data[i].data
        np.testing.assert_array_equal(actual, ref)
        neighbor_counts.append(len(ref))

    neighbor_counts = np.array(neighbor_counts)
    print(f"  -> {len(active_coords)} active, "
          f"neighbors min={neighbor_counts.min()} max={neighbor_counts.max()} "
          f"mean={neighbor_counts.mean():.1f}")
    assert neighbor_counts.min() != neighbor_counts.max(), "Expected jagged"


# =========================================================================
# Program 3: Two-level hierarchical chain
# =========================================================================

CHAIN_PROGRAM = """
parts = Decompose(Input("coord"), Const([3, 4]))
leaf_idx = Gather(Input("lower"), Field(parts, "level_1"))
leaf_node = Gather(Input("leaf_arr"), leaf_idx)
voxel_idx = Gather(leaf_node, Field(parts, "level_0"))
voxel_idx
"""


def test_chain_program():
    """Two-level hierarchical lookup as a DSL string.

    Uses a single lower node (no top-level indexing needed), so the chain is:
      Decompose -> Gather(lower, level_1) -> Gather(leaf_arr, leaf_idx) -> Gather(leaf_node, level_0)
    """

    # ---- SETUP ----
    np.random.seed(42)

    # Build a small two-level grid
    lower_data = np.full((16, 16, 16), -1, dtype=np.int32)
    leaf_blocks = []

    active_lower_coords = [(2, 3, 4), (5, 1, 7), (10, 10, 10)]
    for i, (lx, ly, lz) in enumerate(active_lower_coords):
        lower_data[lx, ly, lz] = i
        leaf = np.full((8, 8, 8), -1, dtype=np.int32)
        for v in range(50):
            vx, vy, vz = v // 16, (v // 4) % 4, v % 4
            if vx < 8 and vy < 8 and vz < 8:
                leaf[vx, vy, vz] = i * 100 + v
        leaf_blocks.append(leaf)

    leaf_data = np.stack(leaf_blocks)  # (3, 8, 8, 8)

    # A coord that maps to active_lower_coords[1]=(5,1,7), leaf voxel (2,1,3)
    global_coord = np.array([5 * 8 + 2, 1 * 8 + 1, 7 * 8 + 3], dtype=np.int32)

    lower_val = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data,
    )
    leaf_arr_val = Value(
        Type(Shape(Static(len(leaf_blocks))), Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32)),
        leaf_data,
    )
    coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), global_coord)

    # ---- PROGRAM ----
    types, result = run(CHAIN_PROGRAM, {
        "coord": coord_val,
        "lower": lower_val,
        "leaf_arr": leaf_arr_val,
    })

    # ---- EXTRACTION ----
    print("Program 3 types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    # Manual reference
    ll = (global_coord >> 3) & 15
    vl = global_coord & 7
    expected_leaf_idx = lower_data[ll[0], ll[1], ll[2]]
    assert expected_leaf_idx >= 0, "Expected active lower entry"
    expected_voxel_idx = leaf_data[expected_leaf_idx, vl[0], vl[1], vl[2]]
    assert expected_voxel_idx >= 0, "Expected active voxel"

    actual_voxel = int(result.data)
    assert actual_voxel == expected_voxel_idx, (
        f"Chain mismatch: {actual_voxel} vs {expected_voxel_idx}"
    )
    print(f"  coord {global_coord} -> voxel_idx {actual_voxel} (expected {expected_voxel_idx})")
    print(f"  -> chain fully executed from DSL string")


# =========================================================================

if __name__ == "__main__":
    print("=== Program 1: Where + Gather ===")
    test_where_program()

    print("\n=== Program 2: Neighbor finding ===")
    test_neighbor_program()

    print("\n=== Program 3: Two-level chain ===")
    test_chain_program()

    print("\nAll DSL tests passed.")
