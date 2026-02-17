# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Tests for advanced layout operations (fuse, flatten, permute) and EachBoth.

Validates:
  1. fuse: merge two outermost nesting levels
  2. flatten: merge all nesting levels
  3. permute: reorder axes within leading shape
  4. EachBoth: zip-iterate two values with matching leading shapes
  5. Round-trip: cut then fuse recovers original type
  6. Composition: outer product -> fuse
"""

import torch
import pytest

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
)


# ---------------------------------------------------------------------------
# 1. fuse
# ---------------------------------------------------------------------------

def test_fuse_basic():
    """fuse((5,) / (3,) / i32) = (5, 3) / i32."""
    data = torch.arange(15, dtype=torch.int32).reshape(5, 3)
    x = Value(Type(Shape(Static(5)), Type(Shape(Static(3)), ScalarType.I32)), data)

    types, result = run("y = fuse(Input(\"x\"))\ny", {"x": x})

    assert result.type == Type(Shape(Static(5), Static(3)), ScalarType.I32)
    torch.testing.assert_close(result.data, data, atol=0, rtol=0)
    print(f"  fuse((5,) / (3,) / i32) = {result.type}")


def test_fuse_multi_rank_inner():
    """fuse((2,) / (3, 4) / i32) = (2, 3, 4) / i32."""
    data = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
    x = Value(
        Type(Shape(Static(2)), Type(Shape(Static(3), Static(4)), ScalarType.I32)),
        data,
    )

    types, result = run("y = fuse(Input(\"x\"))\ny", {"x": x})

    assert result.type == Type(Shape(Static(2), Static(3), Static(4)), ScalarType.I32)
    torch.testing.assert_close(result.data, data, atol=0, rtol=0)
    print(f"  fuse((2,) / (3, 4) / i32) = {result.type}")


def test_fuse_rejects_jagged():
    """fuse must reject inner leading shape with Jagged extents."""
    jagged_type = Type(Shape(Static(3)), Type(Shape(Jagged()), ScalarType.I32))
    x = Value(jagged_type, [
        Value(Type(Shape(Dynamic()), ScalarType.I32), torch.tensor([1, 2], dtype=torch.int32)),
        Value(Type(Shape(Dynamic()), ScalarType.I32), torch.tensor([3], dtype=torch.int32)),
        Value(Type(Shape(Dynamic()), ScalarType.I32), torch.tensor([4, 5, 6], dtype=torch.int32)),
    ])

    with pytest.raises(TypeError, match="Jagged"):
        run("y = fuse(Input(\"x\"))\ny", {"x": x})
    print("  fuse correctly rejects jagged inner shape")


def test_fuse_rejects_scalar_element():
    """fuse requires nested type, not scalar element."""
    x = Value(Type(Shape(Static(5)), ScalarType.I32), torch.arange(5, dtype=torch.int32))

    with pytest.raises(TypeError, match="scalar element"):
        run("y = fuse(Input(\"x\"))\ny", {"x": x})
    print("  fuse correctly rejects scalar element type")


# ---------------------------------------------------------------------------
# 2. flatten
# ---------------------------------------------------------------------------

def test_flatten_triple_nested():
    """flatten((5,) / (3,) / (2,) / i32) = (5, 3, 2) / i32."""
    data = torch.arange(30, dtype=torch.int32).reshape(5, 3, 2)
    x = Value(
        Type(
            Shape(Static(5)),
            Type(Shape(Static(3)), Type(Shape(Static(2)), ScalarType.I32)),
        ),
        data,
    )

    types, result = run("y = flatten(Input(\"x\"))\ny", {"x": x})

    assert result.type == Type(Shape(Static(5), Static(3), Static(2)), ScalarType.I32)
    torch.testing.assert_close(result.data, data, atol=0, rtol=0)
    print(f"  flatten((5,) / (3,) / (2,) / i32) = {result.type}")


def test_flatten_already_flat():
    """flatten on already-flat type is a no-op."""
    data = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    x = Value(Type(Shape(Static(3), Static(4)), ScalarType.I32), data)

    types, result = run("y = flatten(Input(\"x\"))\ny", {"x": x})

    assert result.type == Type(Shape(Static(3), Static(4)), ScalarType.I32)
    torch.testing.assert_close(result.data, data, atol=0, rtol=0)
    print("  flatten on flat type is no-op")


# ---------------------------------------------------------------------------
# 3. permute
# ---------------------------------------------------------------------------

def test_permute_basic():
    """permute((3, 4, 5) / i32, [2, 0, 1]) = (5, 3, 4) / i32."""
    data = torch.arange(60, dtype=torch.int32).reshape(3, 4, 5)
    x = Value(Type(Shape(Static(3), Static(4), Static(5)), ScalarType.I32), data)

    types, result = run("y = permute(Input(\"x\"), [2, 0, 1])\ny", {"x": x})

    assert result.type == Type(Shape(Static(5), Static(3), Static(4)), ScalarType.I32)
    expected = data.permute(2, 0, 1)
    torch.testing.assert_close(result.data, expected, atol=0, rtol=0)
    print(f"  permute((3,4,5)/i32, [2,0,1]) = {result.type}")


def test_permute_preserves_element():
    """permute only affects leading shape, element type unchanged."""
    data = torch.arange(24, dtype=torch.int32).reshape(3, 4, 2)
    x = Value(
        Type(Shape(Static(3), Static(4)), Type(Shape(Static(2)), ScalarType.I32)),
        data,
    )

    types, result = run("y = permute(Input(\"x\"), [1, 0])\ny", {"x": x})

    assert result.type.iteration_shape == Shape(Static(4), Static(3))
    inner = result.type.element_type
    assert isinstance(inner, Type)
    assert inner == Type(Shape(Static(2)), ScalarType.I32)
    expected = data.permute(1, 0, 2)
    torch.testing.assert_close(result.data, expected, atol=0, rtol=0)
    print(f"  permute preserves element type: {result.type}")


# ---------------------------------------------------------------------------
# 4. EachBoth
# ---------------------------------------------------------------------------

def test_each_both_elementwise_add():
    """EachBoth(Add, x, y) with matching shapes: elementwise addition."""
    x_data = torch.tensor([1, 2, 3], dtype=torch.int32)
    y_data = torch.tensor([10, 20, 30], dtype=torch.int32)
    x = Value(Type(Shape(Static(3)), ScalarType.I32), x_data)
    y = Value(Type(Shape(Static(3)), ScalarType.I32), y_data)

    types, result = run(
        "result = EachBoth(Add, Input(\"x\"), Input(\"y\"))\nresult",
        {"x": x, "y": y},
    )

    assert result.type == Type(Shape(Static(3)), ScalarType.I32)
    torch.testing.assert_close(result.data, torch.tensor([11, 22, 33], dtype=torch.int32), atol=0, rtol=0)
    print(f"  EachBoth(Add, [1,2,3], [10,20,30]) = {result.data.tolist()}")


def test_each_both_nested_elements():
    """EachBoth with non-scalar elements: (3,) / (2,) i32."""
    x_data = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    y_data = torch.tensor([[10, 20], [30, 40], [50, 60]], dtype=torch.int32)
    inner_ty = Type(Shape(Static(2)), ScalarType.I32)
    x = Value(Type(Shape(Static(3)), inner_ty), x_data)
    y = Value(Type(Shape(Static(3)), inner_ty), y_data)

    types, result = run(
        "result = EachBoth(Add, Input(\"x\"), Input(\"y\"))\nresult",
        {"x": x, "y": y},
    )

    assert result.type.iteration_shape == Shape(Static(3))
    expected = x_data + y_data
    torch.testing.assert_close(result.data, expected, atol=0, rtol=0)
    print(f"  EachBoth(Add) on (3,)/(2,) i32: correct")


def test_each_both_type_mismatch():
    """EachBoth rejects mismatched leading shapes."""
    x = Value(Type(Shape(Static(3)), ScalarType.I32), torch.tensor([1, 2, 3], dtype=torch.int32))
    y = Value(Type(Shape(Static(5)), ScalarType.I32), torch.arange(5, dtype=torch.int32))

    with pytest.raises(TypeError, match="mismatch"):
        run(
            "result = EachBoth(Add, Input(\"x\"), Input(\"y\"))\nresult",
            {"x": x, "y": y},
        )
    print("  EachBoth correctly rejects mismatched shapes")


# ---------------------------------------------------------------------------
# 5. Round-trip: cut then fuse
# ---------------------------------------------------------------------------

def test_cut_fuse_roundtrip():
    """cut(-3, x) then fuse recovers the original type and data."""
    data = torch.arange(15, dtype=torch.int32)
    x = Value(Type(Shape(Static(15)), ScalarType.I32), data)

    prog = """
cut_x = cut(Input("x"), Const(3))
fused = fuse(cut_x)
fused
"""
    types, result = run(prog, {"x": x})

    assert result.type == Type(Shape(Static(5), Static(3)), ScalarType.I32)
    torch.testing.assert_close(result.data.flatten(), data, atol=0, rtol=0)
    print(f"  cut(-3) then fuse: (15,)/i32 -> (5,)/(3,)/i32 -> (5,3)/i32")


# ---------------------------------------------------------------------------
# 6. Composition: outer product -> fuse
# ---------------------------------------------------------------------------

def test_outer_product_then_fuse():
    """EachRight(EachLeft(Add)) then fuse merges the nesting."""
    x_data = torch.tensor([1, 2, 3], dtype=torch.int32)
    y_data = torch.tensor([10, 20], dtype=torch.int32)
    x = Value(Type(Shape(Static(3)), ScalarType.I32), x_data)
    y = Value(Type(Shape(Static(2)), ScalarType.I32), y_data)

    prog = """
outer = EachRight(EachLeft(Add), Input("x"), Input("y"))
flat = fuse(outer)
flat
"""
    types, result = run(prog, {"x": x, "y": y})

    # outer: (2,) / (3,) / i32  ->  fuse: (2, 3) / i32
    assert result.type == Type(Shape(Static(2), Static(3)), ScalarType.I32)
    expected = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    torch.testing.assert_close(result.data, expected, atol=0, rtol=0)
    print(f"  outer product then fuse: (2,)/(3,)/i32 -> (2,3)/i32")


def test_dot_product_via_each_both_over():
    """Dot product: EachBoth(Mul) then Over(Add).

    This is the inner loop of matrix multiply: sum(x_i * y_i).
    """
    x_data = torch.tensor([1, 2, 3], dtype=torch.int32)
    y_data = torch.tensor([4, 5, 6], dtype=torch.int32)
    x = Value(Type(Shape(Static(3)), ScalarType.I32), x_data)
    y = Value(Type(Shape(Static(3)), ScalarType.I32), y_data)

    prog = """
products = EachBoth(Mul, Input("x"), Input("y"))
dot = Over(Add, products)
dot
"""
    types, result = run(prog, {"x": x, "y": y})

    assert result.type == Type(Shape(), ScalarType.I32)
    expected = (x_data * y_data).sum().item()
    assert int(result.data) == expected
    print(f"  dot product via EachBoth(Mul)+Over(Add): [1,2,3].[4,5,6] = {int(result.data)}")


# =========================================================================

if __name__ == "__main__":
    print("=== 1. fuse ===")
    test_fuse_basic()
    test_fuse_multi_rank_inner()
    test_fuse_rejects_jagged()
    test_fuse_rejects_scalar_element()

    print("\n=== 2. flatten ===")
    test_flatten_triple_nested()
    test_flatten_already_flat()

    print("\n=== 3. permute ===")
    test_permute_basic()
    test_permute_preserves_element()

    print("\n=== 4. EachBoth ===")
    test_each_both_elementwise_add()
    test_each_both_nested_elements()
    test_each_both_type_mismatch()

    print("\n=== 5. Round-trip ===")
    test_cut_fuse_roundtrip()

    print("\n=== 6. Composition ===")
    test_outer_product_then_fuse()
    test_dot_product_via_each_both_over()

    print("\nAll test_layouts_advanced tests passed.")
