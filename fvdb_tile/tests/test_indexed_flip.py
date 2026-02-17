# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Indexed, Tuple, Struct, Flip.

Tests are split into two groups:
  - ALGEBRA TESTS: EXPRESSION phase uses DSL strings (multi-leaf, indexed+gather)
  - API TESTS: test Python-level layout/type APIs directly (struct, flip,
    FlipStruct). These stay in Python because they test the API itself, not
    the algebra.
"""

import numpy as np

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.layouts import (
    cut_by_size,
    flip,
    indexed,
    reshape,
    struct_layout,
    StructElement,
)
from fvdb_tile.prototype.ops import FlipStruct, Gather, Map, StructValue, Value, Where
from fvdb_tile.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# =========================================================================
# ALGEBRA TESTS (DSL strings)
# =========================================================================

MULTI_LEAF_CUT_PROGRAM = """
leaves = cut(Input("flat"), Const(512))
leaves_3d = Each(leaves, leaf => reshape(leaf, Const([8, 8, 8])))
leaves_3d
"""

MULTI_LEAF_WHERE_PROGRAM = """
leaves = cut(Input("flat"), Const(512))
leaves_3d = Each(leaves, leaf => reshape(leaf, Const([8, 8, 8])))
active = Each(leaves_3d, leaf => Where(Map(leaf, x => GE(x, Const(0)))))
active
"""

INDEXED_GATHER_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
idx = Gather(Input("leaf"), active)
feat = Gather(Input("features"), idx)
feat
"""


def test_multi_leaf_cut():
    """Cut a flat array into K leaf nodes, reshape each to (8,8,8)."""

    # ---- SETUP ----
    np.random.seed(42)
    K = 5
    flat_data = np.random.randint(-1, 10, (K * 512,)).astype(np.int32)
    flat = Value.from_numpy(flat_data, ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(MULTI_LEAF_CUT_PROGRAM, {"flat": flat})

    # ---- EXTRACTION ----
    print(f"multi-leaf type: {result.type}")
    assert result.type.iteration_shape == Shape(Static(K))
    inner = result.type.element_type
    assert isinstance(inner, Type)
    assert inner.iteration_shape == Shape(Static(8), Static(8), Static(8))
    assert inner.element_type == ScalarType.I32


def test_multi_leaf_where():
    """Where over multiple leaves produces double-nested jagged."""

    # ---- SETUP ----
    np.random.seed(42)
    K = 5
    flat_data = np.random.randint(-1, 10, (K * 512,)).astype(np.int32)
    flat = Value.from_numpy(flat_data, ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(MULTI_LEAF_WHERE_PROGRAM, {"flat": flat})

    # ---- EXTRACTION ----
    print(f"active_per_leaf type: {result.type}")
    assert result.type.iteration_shape == Shape(Static(K))
    inner = result.type.element_type
    assert isinstance(inner, Type)
    assert isinstance(inner.iteration_shape.extents[0], Jagged)

    # Verify data per leaf
    for i in range(K):
        leaf_data = flat_data[i * 512:(i + 1) * 512].reshape(8, 8, 8)
        expected = np.argwhere(leaf_data >= 0).astype(np.int32)
        actual = result.data[i].data
        np.testing.assert_array_equal(actual, expected)

    counts = [result.data[i].data.shape[0] for i in range(K)]
    print(f"  active counts per leaf: {counts}")
    assert len(set(counts)) > 1, "Expected varying counts (jagged)"


def test_indexed_type_then_gather():
    """Type-check Indexed as layout, then materialise with Gather via DSL.

    Also verifies that the layout type prediction (Python API) matches the
    DSL Gather result type.
    """

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
    types, result = run(INDEXED_GATHER_PROGRAM, {"leaf": leaf, "features": features})

    # Layout-only type prediction (Python API, not DSL)
    predicted = indexed(types["idx"], features.type)

    # ---- EXTRACTION ----
    assert predicted == result.type, f"Layout: {predicted}, Gather: {result.type}"
    print(f"indexed layout type: {predicted}")
    print(f"gather result type:  {result.type}")

    # Data check
    expected_coords = np.argwhere(leaf_data >= 0).astype(np.int32)
    expected_idx = leaf_data[tuple(expected_coords[:, i] for i in range(3))]
    for i in range(len(expected_idx)):
        np.testing.assert_array_equal(result.data[i], features_data[expected_idx[i]])


# =========================================================================
# API TESTS (Python-level -- testing layout/type primitives directly)
# These stay in Python because they test the API itself, not the algebra.
# =========================================================================

def test_struct_flip_types():
    """Type-level struct + flip produces the right array-of-structs type."""

    # ---- SETUP ----
    pos_ty = Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.F32))
    color_ty = Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.F32))
    dens_ty = Type(Shape(Dynamic()), ScalarType.F32)

    # ---- EXPRESSION (Python API) ----
    sty = struct_layout(pos=pos_ty, color=color_ty, dens=dens_ty)
    flipped = flip(sty)

    # ---- EXTRACTION ----
    print(f"struct type:  {sty}")
    print(f"flipped type: {flipped}")
    assert flipped.iteration_shape == Shape(Dynamic())
    elem = flipped.element_type
    assert isinstance(elem, StructElement)
    field_names = [name for name, _ in elem.fields]
    assert field_names == ["pos", "color", "dens"]


def test_struct_flip_data():
    """Full pipeline: leaf -> active indices -> gather features -> flip struct.

    Uses DSL for the gather pipeline, Python FlipStruct for the struct
    composition (no DSL keyword for FlipStruct yet).
    """

    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 5, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)
    max_idx = int(leaf_data.max()) + 1
    pos_data = np.random.randn(max_idx, 3).astype(np.float32)
    color_data = np.random.randn(max_idx, 3).astype(np.float32)
    dens_data = np.random.randn(max_idx).astype(np.float32)

    positions = Value(Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.F32)), pos_data)
    colors = Value(Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.F32)), color_data)
    densities = Value(Type(Shape(Dynamic()), ScalarType.F32), dens_data)

    # ---- EXPRESSION (DSL for gathers, Python for FlipStruct) ----
    GATHER_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
idx = Gather(Input("leaf"), active)
idx
"""
    _, idx_result = run(GATHER_PROGRAM, {"leaf": leaf})
    _, act_pos = run('feat = Gather(Input("pos"), Input("idx"))\nfeat', {"pos": positions, "idx": idx_result})
    _, act_color = run('feat = Gather(Input("col"), Input("idx"))\nfeat', {"col": colors, "idx": idx_result})
    _, act_dens = run('feat = Gather(Input("den"), Input("idx"))\nfeat', {"den": densities, "idx": idx_result})

    # FlipStruct (Python API -- no DSL keyword yet)
    voxels = FlipStruct(pos=act_pos, color=act_color, dens=act_dens)

    # ---- EXTRACTION ----
    print(f"voxels type: {voxels.type}")
    n_active = len(idx_result.data)
    assert len(voxels.data) == n_active

    for i in range(n_active):
        s = voxels.data[i]
        assert isinstance(s, StructValue)
        idx = idx_result.data[i]
        np.testing.assert_array_equal(s.pos, pos_data[idx])
        np.testing.assert_array_equal(s.color, color_data[idx])
        np.testing.assert_almost_equal(s.dens, dens_data[idx])

    print(f"  {n_active} active voxels, each with pos/color/dens fields")


# =========================================================================

if __name__ == "__main__":
    print("=== Algebra: Multiple leaf nodes (DSL) ===")
    test_multi_leaf_cut()
    test_multi_leaf_where()

    print("\n=== Algebra: Indexed layout + Gather (DSL) ===")
    test_indexed_type_then_gather()

    print("\n=== API: Struct + Flip (Python) ===")
    test_struct_flip_types()
    test_struct_flip_data()

    print("\nAll test_indexed_flip tests passed.")
