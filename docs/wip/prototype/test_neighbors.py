# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Second proof: single-leaf neighbor finding.

Each test is structured in three phases:
  1. SETUP         -- create numpy arrays, wrap as typed Values (boilerplate)
  2. EXPRESSION    -- compose operations; this IS the algebra under test
  3. EXTRACTION    -- pull results back to numpy, compare against reference

The jagged extent kind (~) should emerge automatically from variable-length
filtering within Each.
"""

import numpy as np

from docs.wip.prototype.layouts import cut_by_size
from docs.wip.prototype.ops import Each, Gather, Map, Value, Where
from docs.wip.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


def _in_bounds(coord: np.ndarray, lo: int = 0, hi: int = 8) -> bool:
    """Check if all components of a coordinate are in [lo, hi)."""
    return bool(np.all(coord >= lo) and np.all(coord < hi))


def _make_filter(mask_data: np.ndarray):
    """Build the neighbor-filter function used in the expression phase.

    This is a scalar function: given a (6,) over (3,) i32 block of candidate
    coordinates, return the (~,) over (3,) i32 subset that are in-bounds and
    active. The variable-length output is what produces jaggedness.
    """

    def filter_active(coords6: Value) -> Value:
        all_coords = coords6.data
        keep = [c for c in all_coords if _in_bounds(c) and mask_data[c[0], c[1], c[2]]]
        data = np.stack(keep).astype(np.int32) if keep else np.zeros((0, 3), dtype=np.int32)
        return Value(Type(Shape(Dynamic()), coord_type(3)), data)

    return filter_active


# -- Face-neighbor offsets (constant, shared across tests) --
FACE_OFFSETS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int32,
)


def test_neighbor_types():
    """Verify type propagation through the neighbor-finding expression."""

    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    offsets_flat = Value.from_numpy(FACE_OFFSETS.ravel(), ScalarType.I32)
    offsets = Value(cut_by_size(3, offsets_flat.type), FACE_OFFSETS)

    # ---- EXPRESSION ----
    # Step 1: active voxel coordinates
    mask = Map(leaf, lambda x: x >= 0)                          # (8,8,8) bool
    active_ijk = Where(mask)                                    # (*) over (3) i32

    # Step 2: for each active voxel, 6 candidate neighbor coords
    neighbor_ijk = Each(active_ijk, lambda a: Value(            # (*) over (6) over (3) i32
        Type(Shape(Static(6)), coord_type(3)),
        a.data[np.newaxis, :] + FACE_OFFSETS,
    ))

    # Step 3: filter to in-bounds + active neighbors
    active_neighbors = Each(neighbor_ijk, _make_filter(mask.data))  # (*) over (~) over (3) i32

    # ---- EXTRACTION ----
    print(f"neighbor_ijk type:     {neighbor_ijk.type}")
    print(f"active_neighbors type: {active_neighbors.type}")

    assert active_ijk.type == Type(Shape(Dynamic()), coord_type(3))

    assert neighbor_ijk.type.iteration_shape == Shape(Dynamic())
    assert neighbor_ijk.type.element_type == Type(Shape(Static(6)), coord_type(3))

    assert active_neighbors.type.iteration_shape == Shape(Dynamic())
    inner = active_neighbors.type.element_type
    assert isinstance(inner, Type)
    assert isinstance(inner.iteration_shape.extents[0], (Dynamic, Jagged))
    assert inner.element_type == coord_type(3)


def test_neighbor_data():
    """Verify numerical correctness against brute-force reference."""

    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    # ---- EXPRESSION ----
    mask = Map(leaf, lambda x: x >= 0)
    active_ijk = Where(mask)
    mask_data = mask.data

    neighbor_ijk = Each(active_ijk, lambda a: Value(
        Type(Shape(Static(6)), coord_type(3)),
        a.data[np.newaxis, :] + FACE_OFFSETS,
    ))
    active_neighbors = Each(neighbor_ijk, _make_filter(mask_data))

    # ---- EXTRACTION ----
    # Brute-force reference
    active_coords = np.argwhere(leaf_data >= 0).astype(np.int32)
    reference = []
    for coord in active_coords:
        nbrs = coord[np.newaxis, :] + FACE_OFFSETS
        valid = [nb for nb in nbrs if _in_bounds(nb) and mask_data[nb[0], nb[1], nb[2]]]
        reference.append(np.array(valid, dtype=np.int32).reshape(-1, 3))

    # Compare element-by-element
    assert len(active_neighbors.data) == len(reference)
    neighbor_counts = []
    for i in range(len(reference)):
        expr_coords = active_neighbors.data[i].data
        np.testing.assert_array_equal(expr_coords, reference[i])
        neighbor_counts.append(len(reference[i]))

    neighbor_counts = np.array(neighbor_counts)
    print(f"Neighbor data OK: {len(reference)} active voxels, "
          f"neighbor counts min={neighbor_counts.min()} max={neighbor_counts.max()} "
          f"mean={neighbor_counts.mean():.1f}")
    assert neighbor_counts.min() != neighbor_counts.max(), "Expected jagged"


def test_jagged_type_emerges():
    """The key test: does ~ appear in the type automatically?"""

    # ---- SETUP ----
    np.random.seed(99)
    leaf_data = np.random.randint(-1, 4, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    # ---- EXPRESSION ----
    mask = Map(leaf, lambda x: x >= 0)
    active_ijk = Where(mask)

    neighbor_ijk = Each(active_ijk, lambda a: Value(
        Type(Shape(Static(6)), coord_type(3)),
        a.data[np.newaxis, :] + FACE_OFFSETS,
    ))
    result = Each(neighbor_ijk, _make_filter(mask.data))

    # ---- EXTRACTION ----
    inner_type = result.type.element_type
    assert isinstance(inner_type, Type), f"Expected nested Type, got {inner_type!r}"
    leading_extent = inner_type.iteration_shape.extents[0]
    assert isinstance(leading_extent, Jagged), (
        f"Expected jagged (~) inner extent, got {leading_extent!r}. "
        f"Full type: {result.type}"
    )
    print(f"Jagged type confirmed: {result.type}")


if __name__ == "__main__":
    test_neighbor_types()
    test_neighbor_data()
    test_jagged_type_emerges()
    print("\nAll test_neighbors tests passed.")
