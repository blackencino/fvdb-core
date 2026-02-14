# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Prototype v1: Indexed, Tuple, Struct, Flip.

Three scenarios exercising primitives that v0 did not touch:
  1. Multiple leaf nodes via Cut + Each (double nesting, jagged)
  2. Indexed as layout (type-check) then Gather (materialise)
  3. Struct + Flip for multi-feature voxels (array-of-structs)

Each test uses the SETUP / EXPRESSION / EXTRACTION phase markers.
"""

import numpy as np

from docs.wip.prototype.layouts import (
    cut_by_size,
    flip,
    indexed,
    reshape,
    struct_layout,
    StructElement,
)
from docs.wip.prototype.ops import (
    Each,
    FlipStruct,
    Gather,
    Map,
    StructValue,
    Value,
    Where,
)
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
# Scenario 1: Multiple leaf nodes via Cut + Each
# =========================================================================

def test_multi_leaf_cut():
    """Cut a flat array into K leaf nodes, reshape each to (8,8,8)."""

    # ---- SETUP ----
    np.random.seed(42)
    K = 5  # number of leaf nodes
    flat_data = np.random.randint(-1, 10, (K * 512,)).astype(np.int32)
    flat = Value.from_numpy(flat_data, ScalarType.I32)

    # ---- EXPRESSION ----
    # Layout: cut into K blocks of 512 (type-level only)
    leaves_type = cut_by_size(512, flat.type)
    # Materialise the cut by reshaping the numpy data
    leaves = Value(leaves_type, flat_data.reshape(K, 512))

    # Reshape each leaf's (512,) to (8,8,8) via Each
    leaves_3d = Each(leaves, lambda leaf: Value(
        reshape(leaf.type, (8, 8, 8)),
        leaf.data.reshape(8, 8, 8),
    ))

    # ---- EXTRACTION ----
    # Type: (K,) over (8,8,8) over i32
    assert leaves_3d.type.iteration_shape == Shape(Static(K))
    inner = leaves_3d.type.element_type
    assert isinstance(inner, Type)
    assert inner.iteration_shape == Shape(Static(8), Static(8), Static(8))
    assert inner.element_type == ScalarType.I32
    print(f"multi-leaf type: {leaves_3d.type}")


def test_multi_leaf_where():
    """Where over multiple leaves produces double-nested jagged."""

    # ---- SETUP ----
    np.random.seed(42)
    K = 5
    flat_data = np.random.randint(-1, 10, (K * 512,)).astype(np.int32)

    # ---- EXPRESSION ----
    leaves_type = cut_by_size(512, Value.from_numpy(flat_data, ScalarType.I32).type)
    leaves = Value(leaves_type, flat_data.reshape(K, 512))

    leaves_3d = Each(leaves, lambda leaf: Value(
        reshape(leaf.type, (8, 8, 8)),
        leaf.data.reshape(8, 8, 8),
    ))

    # For each leaf, find active voxel coordinates
    active_per_leaf = Each(leaves_3d, lambda leaf:
        Where(Map(leaf, lambda x: x >= 0))
    )

    # ---- EXTRACTION ----
    # Type: (K,) over (~) over (3) i32
    # (K,) because K is static; (~) because each leaf has a different
    # number of active voxels.
    print(f"active_per_leaf type: {active_per_leaf.type}")

    assert active_per_leaf.type.iteration_shape == Shape(Static(K))
    inner = active_per_leaf.type.element_type
    assert isinstance(inner, Type)
    assert isinstance(inner.iteration_shape.extents[0], Jagged), (
        f"Expected jagged inner extent, got {inner.iteration_shape.extents[0]!r}"
    )
    assert inner.element_type == coord_type(3)

    # Verify data: each leaf's active coords should match brute-force
    for i in range(K):
        leaf_data = flat_data[i * 512:(i + 1) * 512].reshape(8, 8, 8)
        expected = np.argwhere(leaf_data >= 0).astype(np.int32)
        actual = active_per_leaf.data[i].data
        np.testing.assert_array_equal(actual, expected)

    counts = [active_per_leaf.data[i].data.shape[0] for i in range(K)]
    print(f"  active counts per leaf: {counts}")
    assert len(set(counts)) > 1, "Expected varying counts (jagged)"


# =========================================================================
# Scenario 2: Indexed as layout then Gather
# =========================================================================

def test_indexed_type_then_gather():
    """Type-check Indexed as layout, then materialise with Gather."""

    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 8, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    max_idx = int(leaf_data.max()) + 1
    C = 4
    features_data = np.arange(max_idx * C, dtype=np.float32).reshape(max_idx, C)
    features = Value(
        Type(Shape(Dynamic()), Type(Shape(Static(C)), ScalarType.F32)),
        features_data,
    )

    # ---- EXPRESSION ----
    mask = Map(leaf, lambda x: x >= 0)
    active_ijk = Where(mask)
    active_idx = Gather(leaf, active_ijk)

    # Layout-only: type-check the indexed association (no data movement)
    predicted_type = indexed(active_idx.type, features.type)

    # Materialise: actually do the gather
    active_feat = Gather(features, active_idx)

    # ---- EXTRACTION ----
    # The layout prediction should match the gather result type exactly
    assert predicted_type == active_feat.type, (
        f"Layout predicted {predicted_type}, Gather produced {active_feat.type}"
    )
    print(f"indexed layout type: {predicted_type}")
    print(f"gather result type:  {active_feat.type}")
    assert predicted_type == Type(Shape(Dynamic()), Type(Shape(Static(C)), ScalarType.F32))

    # Verify data
    for i in range(len(active_idx.data)):
        idx = active_idx.data[i]
        np.testing.assert_array_equal(active_feat.data[i], features_data[idx])


# =========================================================================
# Scenario 3: Struct + Flip for multi-feature voxels
# =========================================================================

def test_struct_flip_types():
    """Type-level struct + flip produces the right array-of-structs type."""

    # ---- SETUP ----
    pos_ty = Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.F32))
    color_ty = Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.F32))
    dens_ty = Type(Shape(Dynamic()), ScalarType.F32)

    # ---- EXPRESSION ----
    sty = struct_layout(pos=pos_ty, color=color_ty, dens=dens_ty)
    flipped = flip(sty)

    # ---- EXTRACTION ----
    # struct_layout: (3,) over Struct({pos: ..., color: ..., dens: ...})
    print(f"struct type:  {sty}")
    assert sty.iteration_shape == Shape(Static(3))  # 3 fields

    # flip: (*) over Struct({pos: (3) f32, color: (3) f32, dens: f32})
    print(f"flipped type: {flipped}")
    assert flipped.iteration_shape == Shape(Dynamic())
    elem = flipped.element_type
    assert isinstance(elem, StructElement)
    field_names = [name for name, _ in elem.fields]
    assert field_names == ["pos", "color", "dens"]

    # Check each field's element type
    field_types = {name: ty for name, ty in elem.fields}
    assert field_types["pos"] == Type(Shape(Static(3)), ScalarType.F32)
    assert field_types["color"] == Type(Shape(Static(3)), ScalarType.F32)
    assert field_types["dens"] == ScalarType.F32


def test_struct_flip_data():
    """Full pipeline: leaf -> active indices -> gather features -> flip struct."""

    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 5, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    max_idx = int(leaf_data.max()) + 1
    pos_data = np.random.randn(max_idx, 3).astype(np.float32)
    color_data = np.random.randn(max_idx, 3).astype(np.float32)
    dens_data = np.random.randn(max_idx).astype(np.float32)

    positions = Value(
        Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.F32)), pos_data
    )
    colors = Value(
        Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.F32)), color_data
    )
    densities = Value(
        Type(Shape(Dynamic()), ScalarType.F32), dens_data
    )

    # ---- EXPRESSION ----
    # Step 1: active voxel linear indices
    mask = Map(leaf, lambda x: x >= 0)
    active_ijk = Where(mask)
    active_idx = Gather(leaf, active_ijk)

    # Step 2: gather each feature channel independently
    act_pos = Gather(positions, active_idx)      # (*) over (3) f32
    act_color = Gather(colors, active_idx)        # (*) over (3) f32
    act_dens = Gather(densities, active_idx)      # (*) over f32

    # Step 3: combine into array-of-structs via FlipStruct
    voxels = FlipStruct(pos=act_pos, color=act_color, dens=act_dens)

    # ---- EXTRACTION ----
    print(f"voxels type: {voxels.type}")

    # Type: (*) over Struct({pos: (3) f32, color: (3) f32, dens: f32})
    assert voxels.type.iteration_shape == Shape(Dynamic())
    elem = voxels.type.element_type
    assert isinstance(elem, StructElement)

    n_active = len(active_idx.data)
    assert len(voxels.data) == n_active

    # Verify each element's fields match the gathered data
    for i in range(n_active):
        s = voxels.data[i]
        assert isinstance(s, StructValue)
        idx = active_idx.data[i]
        np.testing.assert_array_equal(s.pos, pos_data[idx])
        np.testing.assert_array_equal(s.color, color_data[idx])
        np.testing.assert_almost_equal(s.dens, dens_data[idx])

    print(f"  {n_active} active voxels, each with pos/color/dens fields")
    print(f"  voxels[0]: {voxels.data[0]}")


# =========================================================================

if __name__ == "__main__":
    print("=== Scenario 1: Multiple leaf nodes ===")
    test_multi_leaf_cut()
    test_multi_leaf_where()

    print("\n=== Scenario 2: Indexed layout + Gather ===")
    test_indexed_type_then_gather()

    print("\n=== Scenario 3: Struct + Flip ===")
    test_struct_flip_types()
    test_struct_flip_data()

    print("\nAll test_indexed_flip tests passed.")
