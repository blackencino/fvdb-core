# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Tests for user-defined named functions in the DSL.

Covers: parsing, type inference, evaluation, and pipeline barrier
transparency for the (params) => body function definition syntax.
"""

import torch

from fvdb_tile.prototype.dsl_ast import FnCallNode, FnDefNode
from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.dsl_parse import parse
from fvdb_tile.prototype.dsl_pipeline import compile_source, plan_source
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Static, Type


def test_parse_fn_def():
    """Function definition parses to FnDefNode."""
    prog = parse("f = (x) => Add(x, Const(1))\nresult = f(Const(5))\nresult")
    assert len(prog.bindings) == 2
    name, node = prog.bindings[0]
    assert name == "f"
    assert isinstance(node, FnDefNode)
    assert node.params == ("x",)


def test_parse_fn_call():
    """Function call parses to FnCallNode."""
    prog = parse("f = (x) => Add(x, Const(1))\nresult = f(Const(5))\nresult")
    name, node = prog.bindings[1]
    assert name == "result"
    assert isinstance(node, FnCallNode)
    assert node.fn_name == "f"
    assert len(node.args) == 1


def test_parse_multi_param():
    """Multi-parameter function definition parses correctly."""
    prog = parse("g = (a, b, c) => Add(a, Add(b, c))\nresult = g(Const(1), Const(2), Const(3))\nresult")
    fn_node = prog.bindings[0][1]
    assert isinstance(fn_node, FnDefNode)
    assert fn_node.params == ("a", "b", "c")
    call_node = prog.bindings[1][1]
    assert isinstance(call_node, FnCallNode)
    assert len(call_node.args) == 3


def test_type_inference():
    """Function call infers the correct output type."""
    source = """\
f = (x) => Add(x, Const(1))
result = f(Const(5))
result
"""
    prog = parse(source)
    types = prog.infer_types({})
    assert types["result"].element_type == ScalarType.I32


def test_type_inference_with_inputs():
    """Function call works with typed inputs."""
    source = """\
double = (x) => Add(x, x)
result = double(Input("val"))
result
"""
    prog = parse(source)
    input_types = {"val": Type(Shape(Static(3)), ScalarType.I32)}
    types = prog.infer_types(input_types)
    assert types["result"] == Type(Shape(Static(3)), ScalarType.I32)


def test_eval_simple():
    """Function call evaluates to the correct result."""
    source = """\
f = (x) => Add(x, Const(1))
result = f(Const(5))
result
"""
    types, output = run(source, {})
    assert int(output.data) == 6


def test_eval_multi_param():
    """Multi-parameter function evaluates correctly."""
    source = """\
g = (a, b) => Add(a, b)
result = g(Const(3), Const(7))
result
"""
    types, output = run(source, {})
    assert int(output.data) == 10


def test_eval_with_tensor_input():
    """Function call works with tensor data from inputs."""
    source = """\
double = (x) => Add(x, x)
result = double(Input("vals"))
result
"""
    vals = torch.tensor([1, 2, 3], dtype=torch.int32)
    inputs = {"vals": Value(Type(Shape(Static(3)), ScalarType.I32), vals)}
    types, output = run(source, inputs)
    expected = torch.tensor([2, 4, 6], dtype=torch.int32)
    assert torch.equal(output.data, expected)


def test_reuse_function():
    """Same function called multiple times with different arguments."""
    source = """\
inc = (x) => Add(x, Const(1))
a = inc(Const(10))
b = inc(Const(20))
result = Add(a, b)
result
"""
    types, output = run(source, {})
    assert int(output.data) == 32  # (10+1) + (20+1)


def test_function_calling_function():
    """Function calls another user-defined function."""
    source = """\
inc = (x) => Add(x, Const(1))
double_inc = (x) => inc(inc(x))
result = double_inc(Const(5))
result
"""
    types, output = run(source, {})
    assert int(output.data) == 7  # 5 + 1 + 1


def test_pipeline_no_barrier():
    """FnCallNode with no barriers stays in the cutile segment."""
    source = """\
f = (x) => Add(x, Const(1))
result = f(Const(5))
result
"""
    plan = plan_source(source)
    assert len(plan.segments) == 1
    assert plan.segments[0].kind == "cutile"


def test_pipeline_barrier_in_body():
    """FnCallNode whose body contains a barrier is detected."""
    source = """\
f = (x) => Sort(x)
result = f(Input("data"))
result
"""
    plan = plan_source(source)
    has_collective = any(s.kind == "collective" for s in plan.segments)
    assert has_collective, "Sort inside function body should be detected as barrier"


def test_pipeline_execution():
    """Named function works through the full pipeline executor."""
    source = """\
inc = (x) => Add(x, Const(1))
result = inc(Input("val"))
result
"""
    pipeline = compile_source(source)
    val = torch.tensor(42, dtype=torch.int32)
    inputs = {"val": Value(Type(Shape(), ScalarType.I32), val)}
    result = pipeline.run(inputs)
    assert int(result.output.data) == 43


def test_gather_in_function():
    """Function body uses Gather -- a real DSL pattern."""
    source = """\
lookup = (arr, idx) => Gather(arr, idx)
result = lookup(Input("data"), Const(2))
result
"""
    data = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    inputs = {"data": Value(Type(Shape(Static(4)), ScalarType.I32), data)}
    types, output = run(source, inputs)
    assert int(output.data) == 30


def main():
    print("=" * 60)
    print("Named Functions: parse, type, eval, pipeline")
    print("=" * 60)

    test_parse_fn_def()
    print("  parse_fn_def -- PASSED")

    test_parse_fn_call()
    print("  parse_fn_call -- PASSED")

    test_parse_multi_param()
    print("  parse_multi_param -- PASSED")

    test_type_inference()
    print("  type_inference -- PASSED")

    test_type_inference_with_inputs()
    print("  type_inference_with_inputs -- PASSED")

    test_eval_simple()
    print("  eval_simple -- PASSED")

    test_eval_multi_param()
    print("  eval_multi_param -- PASSED")

    test_eval_with_tensor_input()
    print("  eval_with_tensor_input -- PASSED")

    test_reuse_function()
    print("  reuse_function -- PASSED")

    test_function_calling_function()
    print("  function_calling_function -- PASSED")

    test_pipeline_no_barrier()
    print("  pipeline_no_barrier -- PASSED")

    test_pipeline_barrier_in_body()
    print("  pipeline_barrier_in_body -- PASSED")

    test_pipeline_execution()
    print("  pipeline_execution -- PASSED")

    test_gather_in_function()
    print("  gather_in_function -- PASSED")

    print()


if __name__ == "__main__":
    main()
