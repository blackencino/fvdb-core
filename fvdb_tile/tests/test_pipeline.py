# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Tests for barrier-aware DSL pipeline planning and GPU-backed collective dispatch.
"""

import numpy as np

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.dsl_pipeline import compile_source, plan_source
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Static, Type


def test_pipeline_partitions_collectives():
    source = """
a = Map(Input("x"), v => Add(v, Const(1)))
b = Sort(a)
c = Unique(b)
c
"""
    plan = plan_source(source)
    kinds = [seg.kind for seg in plan.segments]
    reasons = [seg.reason for seg in plan.segments]
    assert kinds == ["cutile", "collective", "collective"]
    assert reasons == ["pointwise_gather_segment", "sort_collective", "unique_collective"]
    assert plan.output == "c"


def test_pipeline_marks_nested_barrier():
    source = """
m = Map(Input("x"), v => GE(v, Const(0)))
w = Gather(Input("x"), Where(m))
w
"""
    plan = plan_source(source)
    kinds = [seg.kind for seg in plan.segments]
    assert kinds == ["cutile", "collective"]
    assert plan.segments[1].reason in ("where_dynamic_output", "contains_barrier_subgraph")


def test_pipeline_executable_matches_direct_run():
    source = """
a = Map(Input("x"), v => Add(v, Const(1)))
b = Sort(a)
c = Unique(b)
d = Map(c, v => Sub(v, Const(1)))
d
"""
    x_np = np.array([5, 3, 3, 2, 9, 2], dtype=np.int32)
    x_val = Value(Type(Shape(Static(x_np.shape[0])), ScalarType.I32), x_np.copy())

    direct_types, direct_out = run(source, {"x": x_val})
    exe = compile_source(source)
    result = exe.run({"x": x_val})

    np.testing.assert_array_equal(result.output.data, direct_out.data)
    assert result.output.type == direct_out.type
    assert direct_types["d"] == direct_out.type


def test_pipeline_executable_preserves_input_immutability():
    source = """
a = Sort(Input("x"))
b = Unique(a)
b
"""
    x_np = np.array([4, 1, 4, 2, 1], dtype=np.int32)
    before = x_np.copy()
    x_val = Value(Type(Shape(Static(x_np.shape[0])), ScalarType.I32), x_np)

    exe = compile_source(source)
    _ = exe.run({"x": x_val})

    np.testing.assert_array_equal(x_np, before)


# ---------------------------------------------------------------------------
# GPU-backed collective dispatch (torch ops via device parameter)
# ---------------------------------------------------------------------------


def test_pipeline_collective_where_matches_evaluator():
    """Where dispatched via torch.nonzero matches the pure numpy evaluator."""
    source = """
mask = Map(Input("x"), v => GE(v, Const(3)))
active = Where(mask)
active
"""
    x_np = np.array([5, 1, 3, 0, 7, 2, 4], dtype=np.int32)
    x_val = Value(Type(Shape(Static(x_np.shape[0])), ScalarType.I32), x_np.copy())

    _, direct_out = run(source, {"x": x_val})
    exe = compile_source(source)
    result_cpu = exe.run({"x": x_val}, device="cpu")

    np.testing.assert_array_equal(result_cpu.output.data, direct_out.data)
    assert isinstance(result_cpu.output.type.iteration_shape.extents[0], Dynamic)


def test_pipeline_collective_sort_unique_matches_evaluator():
    """Sort + Unique (the conv_grid dedup pattern) via torch matches evaluator."""
    source = """
sorted = Sort(Input("coords"))
unique = Unique(sorted)
unique
"""
    coords_np = np.array(
        [[2, 1, 0], [0, 0, 1], [2, 1, 0], [1, 3, 2], [0, 0, 1]],
        dtype=np.int32,
    )
    coords_val = Value(
        Type(Shape(Static(coords_np.shape[0])), Type(Shape(Static(3)), ScalarType.I32)),
        coords_np.copy(),
    )

    _, direct_out = run(source, {"coords": coords_val})
    exe = compile_source(source)
    result_cpu = exe.run({"coords": coords_val}, device="cpu")

    np.testing.assert_array_equal(result_cpu.output.data, direct_out.data)
    assert isinstance(result_cpu.output.type.iteration_shape.extents[0], Dynamic)


def test_pipeline_mixed_segments():
    """cutile -> collective -> cutile: full pipeline with intermediate wiring."""
    source = """
a = Map(Input("x"), v => Add(v, Const(1)))
b = Sort(a)
c = Unique(b)
d = Map(c, v => Sub(v, Const(1)))
d
"""
    x_np = np.array([5, 3, 3, 2, 9, 2], dtype=np.int32)
    x_val = Value(Type(Shape(Static(x_np.shape[0])), ScalarType.I32), x_np.copy())

    _, direct_out = run(source, {"x": x_val})
    exe = compile_source(source)

    # device=None uses evaluator for everything (correctness baseline)
    result_none = exe.run({"x": x_val}, device=None)
    np.testing.assert_array_equal(result_none.output.data, direct_out.data)

    # device="cpu" uses torch for collectives
    result_cpu = exe.run({"x": x_val}, device="cpu")
    np.testing.assert_array_equal(result_cpu.output.data, direct_out.data)


def test_pipeline_nested_barrier_gather_where():
    """Gather(x, Where(mask)) with Where nested inside a Gather node."""
    source = """
m = Map(Input("x"), v => GE(v, Const(0)))
w = Gather(Input("x"), Where(m))
w
"""
    x_np = np.array([3, -1, 5, -2, 7], dtype=np.int32)
    x_val = Value(Type(Shape(Static(x_np.shape[0])), ScalarType.I32), x_np.copy())

    _, direct_out = run(source, {"x": x_val})
    exe = compile_source(source)
    result_cpu = exe.run({"x": x_val}, device="cpu")

    np.testing.assert_array_equal(result_cpu.output.data, direct_out.data)
