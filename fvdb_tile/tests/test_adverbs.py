# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Tests for first-class function values, adverb composition, and leading-shape
theory.

Validates:
  1. Verbs as values (Over(Add, xs) via new AdverbApply + Apply path)
  2. EachRight / EachLeft with correct leading-shape semantics
  3. Nested adverbs: EachRight(EachLeft(f)) vs EachLeft(EachRight(f))
  4. Outer product as adverb composition (replaces broadcasting)
  5. Backward compatibility with existing DSL programs
"""

import numpy as np

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import (
    Dynamic,
    FnType,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
)


# ---------------------------------------------------------------------------
# 1. Over(Add, xs) via new code path
# ---------------------------------------------------------------------------

def test_over_add():
    """Over(Add, xs) reduces a vector by summation."""
    xs_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    xs = Value(Type(Shape(Static(5)), ScalarType.I32), xs_data)

    types, result = run("reduced = Over(Add, Input(\"xs\"))\nreduced", {"xs": xs})

    assert result.type == Type(Shape(), ScalarType.I32)
    assert int(result.data) == 15
    print("  Over(Add, [1..5]) = 15")


# ---------------------------------------------------------------------------
# 2. EachRight / EachLeft basic
# ---------------------------------------------------------------------------

def test_each_right_basic():
    """EachRight(Add, scalar, vector) adds scalar to each element."""
    x = Value(Type(Shape(), ScalarType.I32), np.int32(10))
    y_data = np.array([1, 2, 3], dtype=np.int32)
    y = Value(Type(Shape(Static(3)), ScalarType.I32), y_data)

    prog = "result = EachRight(Add, Input(\"x\"), Input(\"y\"))\nresult"
    types, result = run(prog, {"x": x, "y": y})

    assert result.type.iteration_shape == Shape(Static(3))
    np.testing.assert_array_equal(result.data, [11, 12, 13])
    print("  EachRight(Add, 10, [1,2,3]) = [11,12,13]")


def test_each_left_basic():
    """EachLeft(Add, vector, scalar) adds scalar to each element."""
    x_data = np.array([1, 2, 3], dtype=np.int32)
    x = Value(Type(Shape(Static(3)), ScalarType.I32), x_data)
    y = Value(Type(Shape(), ScalarType.I32), np.int32(10))

    prog = "result = EachLeft(Add, Input(\"x\"), Input(\"y\"))\nresult"
    types, result = run(prog, {"x": x, "y": y})

    assert result.type.iteration_shape == Shape(Static(3))
    np.testing.assert_array_equal(result.data, [11, 12, 13])
    print("  EachLeft(Add, [1,2,3], 10) = [11,12,13]")


# ---------------------------------------------------------------------------
# 3. Nested adverbs: the core test
# ---------------------------------------------------------------------------

def test_outer_product_each_right_each_left():
    """EachRight(EachLeft(Add))(x, y) produces S_y / (S_x / C).

    This is the outer-product pattern: for each element of y, add it
    to every element of x.

    x = [1, 2, 3]      -- (3,) / i32
    y = [10, 20]        -- (2,) / i32

    EachRight(EachLeft(Add))(x, y):
      For each y_elem in y:
        EachLeft(Add)(x, y_elem):
          For each x_elem in x: Add(x_elem, y_elem)

    y_elem=10: [1+10, 2+10, 3+10] = [11, 12, 13]
    y_elem=20: [1+20, 2+20, 3+20] = [21, 22, 23]

    Result: (2,) / (3,) / i32 = [[11,12,13], [21,22,23]]
    """
    x_data = np.array([1, 2, 3], dtype=np.int32)
    x = Value(Type(Shape(Static(3)), ScalarType.I32), x_data)
    y_data = np.array([10, 20], dtype=np.int32)
    y = Value(Type(Shape(Static(2)), ScalarType.I32), y_data)

    prog = "result = EachRight(EachLeft(Add), Input(\"x\"), Input(\"y\"))\nresult"
    types, result = run(prog, {"x": x, "y": y})

    # Type check: (2,) / (3,) / i32
    assert result.type.iteration_shape == Shape(Static(2)), f"got {result.type.iteration_shape}"
    inner = result.type.element_type
    assert isinstance(inner, Type), f"expected nested Type, got {inner!r}"
    assert inner.iteration_shape == Shape(Static(3)), f"inner shape {inner.iteration_shape}"

    # Data check
    expected = np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int32)
    np.testing.assert_array_equal(result.data, expected)
    print(f"  EachRight(EachLeft(Add))([1,2,3], [10,20]) = {result.data.tolist()}")
    print(f"  Type: {result.type}")


def test_outer_product_reversed():
    """EachLeft(EachRight(Add))(x, y) produces S_x / (S_y / C).

    Same data as above, but reversed nesting:
    x = [1, 2, 3], y = [10, 20]

    EachLeft(EachRight(Add))(x, y):
      For each x_elem in x:
        EachRight(Add)(x_elem, y):
          For each y_elem in y: Add(x_elem, y_elem)

    x_elem=1: [1+10, 1+20] = [11, 21]
    x_elem=2: [2+10, 2+20] = [12, 22]
    x_elem=3: [3+10, 3+20] = [13, 23]

    Result: (3,) / (2,) / i32 = [[11,21], [12,22], [13,23]]
    """
    x_data = np.array([1, 2, 3], dtype=np.int32)
    x = Value(Type(Shape(Static(3)), ScalarType.I32), x_data)
    y_data = np.array([10, 20], dtype=np.int32)
    y = Value(Type(Shape(Static(2)), ScalarType.I32), y_data)

    prog = "result = EachLeft(EachRight(Add), Input(\"x\"), Input(\"y\"))\nresult"
    types, result = run(prog, {"x": x, "y": y})

    # Type check: (3,) / (2,) / i32
    assert result.type.iteration_shape == Shape(Static(3)), f"got {result.type.iteration_shape}"
    inner = result.type.element_type
    assert isinstance(inner, Type), f"expected nested Type, got {inner!r}"
    assert inner.iteration_shape == Shape(Static(2)), f"inner shape {inner.iteration_shape}"

    # Data check
    expected = np.array([[11, 21], [12, 22], [13, 23]], dtype=np.int32)
    np.testing.assert_array_equal(result.data, expected)
    print(f"  EachLeft(EachRight(Add))([1,2,3], [10,20]) = {result.data.tolist()}")
    print(f"  Type: {result.type}")


# ---------------------------------------------------------------------------
# 4. Type inference for nested adverbs (no data, pure type check)
# ---------------------------------------------------------------------------

def test_nested_adverb_type_inference():
    """Verify type inference matches the leading-shape algebra."""
    from fvdb_tile.prototype.dsl_parse import parse

    prog = parse("result = EachRight(EachLeft(Add), Input(\"x\"), Input(\"y\"))\nresult")

    input_types = {
        "x": Type(Shape(Static(3), Static(4)), ScalarType.I32),  # (3, 4) / i32
        "y": Type(Shape(Static(5), Static(6), Static(7)), ScalarType.I32),  # (5, 6, 7) / i32
    }

    inferred = prog.infer_types(input_types)

    result_ty = inferred["result"]
    print(f"  x: (3, 4) / i32")
    print(f"  y: (5, 6, 7) / i32")
    print(f"  EachRight(EachLeft(Add))(x, y) type: {result_ty}")

    # Should be (5, 6, 7) / (3, 4) / i32
    assert result_ty.iteration_shape == Shape(Static(5), Static(6), Static(7))
    inner = result_ty.element_type
    assert isinstance(inner, Type)
    assert inner.iteration_shape == Shape(Static(3), Static(4))
    assert inner.element_type == ScalarType.I32


# ---------------------------------------------------------------------------
# 5. Adverb as let-bound value (partial application)
# ---------------------------------------------------------------------------

def test_let_bound_adverb():
    """A composed adverb can be let-bound and applied later."""
    x_data = np.array([1, 2], dtype=np.int32)
    x = Value(Type(Shape(Static(2)), ScalarType.I32), x_data)
    y_data = np.array([10, 20, 30], dtype=np.int32)
    y = Value(Type(Shape(Static(3)), ScalarType.I32), y_data)

    prog = """
f = EachRight(EachLeft(Add))
result = Apply(f, Input("x"), Input("y"))
result
"""
    types, result = run(prog, {"x": x, "y": y})

    expected = np.array([[11, 12], [21, 22], [31, 32]], dtype=np.int32)
    np.testing.assert_array_equal(result.data, expected)
    print(f"  f = EachRight(EachLeft(Add)); Apply(f, [1,2], [10,20,30]) = {result.data.tolist()}")


# ---------------------------------------------------------------------------
# 6. Backward compat: Over(Add, ...) in mesh centroid
# ---------------------------------------------------------------------------

CENTROID_PROGRAM = """
vertices = cut(Input("positions"), Const(3))
face_idx = cut(Input("faces"), Const(3))
centroids = Each(face_idx, f => Div(Over(Add, Map(f, i => Gather(vertices, i))), Const(3)))
centroids
"""

TETRA_POSITIONS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 1.0, 0.0],
    [0.5, 0.5, 1.0],
], dtype=np.float32)

TETRA_FACES = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3],
], dtype=np.int32)


def test_backward_compat_centroids():
    """Mesh centroid program uses Over(Add, ...) -- must still work."""
    positions = Value(
        Type(Shape(Static(TETRA_POSITIONS.size)), ScalarType.F32),
        TETRA_POSITIONS.ravel().astype(np.float32),
    )
    faces = Value(
        Type(Shape(Static(TETRA_FACES.size)), ScalarType.I32),
        TETRA_FACES.ravel().astype(np.int32),
    )

    types, result = run(CENTROID_PROGRAM, {"positions": positions, "faces": faces})

    assert result.type.iteration_shape == Shape(Static(4))

    for fi in range(4):
        actual = result.data[fi] if isinstance(result.data[fi], np.ndarray) else result.data[fi].data
        expected = TETRA_POSITIONS[TETRA_FACES[fi]].mean(axis=0)
        np.testing.assert_array_almost_equal(actual, expected)

    print(f"  Centroid backward compat: 4 faces verified")


# =========================================================================

if __name__ == "__main__":
    print("=== 1. Over(Add) via new path ===")
    test_over_add()

    print("\n=== 2. EachRight/EachLeft basic ===")
    test_each_right_basic()
    test_each_left_basic()

    print("\n=== 3. Nested adverbs: outer product ===")
    test_outer_product_each_right_each_left()
    test_outer_product_reversed()

    print("\n=== 4. Type inference for nested adverbs ===")
    test_nested_adverb_type_inference()

    print("\n=== 5. Let-bound adverb ===")
    test_let_bound_adverb()

    print("\n=== 6. Backward compat: centroids ===")
    test_backward_compat_centroids()

    print("\nAll test_adverbs tests passed.")
