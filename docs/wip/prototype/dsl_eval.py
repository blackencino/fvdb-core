# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Tree-walk evaluator for the micro DSL.

Two passes:
  1. Type-check: infer types for all bindings (no data).
  2. Execute: evaluate the AST against concrete numpy data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dsl_ast import (
    AddNode,
    AndNode,
    ConstNode,
    CutNode,
    DecomposeNode,
    EachNode,
    FieldNode,
    GatherNode,
    GENode,
    InBoundsNode,
    InputNode,
    MapNode,
    Morton3dNode,
    Node,
    NotNode,
    Program,
    RefNode,
    ReshapeNode,
    SubNode,
    WhereNode,
)
from .dsl_parse import parse
from .ops import (
    Value,
    StructValue,
    morton3d as np_morton3d,
)
from .types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# ---------------------------------------------------------------------------
# Evaluation environment
# ---------------------------------------------------------------------------

class EvalEnv:
    """Runtime environment: maps names to Values."""

    def __init__(self, inputs: dict[str, Value], bindings: dict[str, Value] | None = None):
        self.inputs = inputs
        self.bindings = bindings or {}
        self.locals: dict[str, Value] = {}  # bound variables from Each/Map

    def lookup(self, name: str) -> Value:
        if name in self.locals:
            return self.locals[name]
        if name in self.bindings:
            return self.bindings[name]
        raise NameError(f"Unbound name: {name!r}")

    def with_local(self, name: str, val: Value) -> EvalEnv:
        """Create a child env with an additional local binding."""
        child = EvalEnv(self.inputs, self.bindings)
        child.locals = {**self.locals, name: val}
        return child


# ---------------------------------------------------------------------------
# Node evaluator
# ---------------------------------------------------------------------------

def eval_node(node: Node, env: EvalEnv) -> Value:
    """Recursively evaluate an AST node to a concrete Value."""

    if isinstance(node, InputNode):
        if node.name not in env.inputs:
            raise NameError(f"Input {node.name!r} not provided")
        return env.inputs[node.name]

    if isinstance(node, ConstNode):
        if isinstance(node.value, int):
            return Value(Type(Shape(), node.stype), np.int32(node.value))
        elif isinstance(node.value, list):
            arr = np.array(node.value, dtype=np.int32)
            return Value(Type(Shape(Static(len(node.value))), node.stype), arr)
        elif isinstance(node.value, str):
            # String constants are labels, not data
            return Value(Type(Shape(), node.stype), node.value)
        raise TypeError(f"Cannot evaluate const: {node.value!r}")

    if isinstance(node, RefNode):
        return env.lookup(node.name)

    if isinstance(node, FieldNode):
        struct_val = eval_node(node.expr, env)
        # Infer the field type
        input_types = {k: v.type for k, v in env.inputs.items()}
        type_env = {k: v.type for k, v in {**env.bindings, **env.locals}.items()}
        field_type = node.infer_type(type_env, input_types)

        if isinstance(struct_val.data, StructValue):
            field_data = struct_val.data.fields[node.field_name]
            return Value(field_type, field_data)
        elif isinstance(struct_val.data, list):
            # List of StructValues -- extract field from each
            field_data = [sv.data.fields[node.field_name] if isinstance(sv.data, StructValue)
                          else sv.fields[node.field_name] for sv in struct_val.data]
            return Value(field_type, np.array(field_data, dtype=np.int32) if field_data else np.array([], dtype=np.int32))
        raise TypeError(f"Field access on non-struct: {type(struct_val.data)}")

    # -- Scalar primitives --

    if isinstance(node, AddNode):
        a = eval_node(node.a, env)
        b = eval_node(node.b, env)
        return Value(a.type, (a.data + b.data).astype(a.data.dtype))

    if isinstance(node, SubNode):
        a = eval_node(node.a, env)
        b = eval_node(node.b, env)
        return Value(a.type, (a.data - b.data).astype(a.data.dtype))

    if isinstance(node, GENode):
        a = eval_node(node.a, env)
        b = eval_node(node.b, env)
        b_val = b.data if isinstance(b.data, np.ndarray) else b.data
        result = a.data >= b_val
        return Value(Type(a.type.iteration_shape, ScalarType.BOOL), result)

    if isinstance(node, AndNode):
        a = eval_node(node.a, env)
        b = eval_node(node.b, env)
        return Value(Type(a.type.iteration_shape, ScalarType.BOOL), a.data & b.data)

    if isinstance(node, NotNode):
        a = eval_node(node.a, env)
        return Value(a.type, ~a.data)

    if isinstance(node, InBoundsNode):
        coord_val = eval_node(node.coord, env)
        lo_val = eval_node(node.lo, env)
        hi_val = eval_node(node.hi, env)
        lo = int(lo_val.data)
        hi = int(hi_val.data)
        c = coord_val.data
        result = bool(np.all(c >= lo) and np.all(c < hi))
        return Value(Type(Shape(), ScalarType.BOOL), np.bool_(result))

    # -- Structural operations --

    if isinstance(node, MapNode):
        input_val = eval_node(node.input, env)
        data = input_val.data

        if isinstance(data, np.ndarray):
            # Iterate over leading axis, apply body to each element
            n = data.shape[0]
            elem_type = input_val.type.element_type

            results = []
            for i in range(n):
                slice_data = data[i]
                if isinstance(elem_type, Type):
                    elem_val = Value(elem_type, slice_data)
                else:
                    elem_val = Value(Type(Shape(), elem_type), slice_data)

                child_env = env.with_local(node.var, elem_val)
                results.append(eval_node(node.body, child_env))

            # Stack results
            if results and isinstance(results[0].data, np.ndarray):
                stacked = np.stack([r.data for r in results])
            elif results and isinstance(results[0].data, (np.bool_, np.int32, np.float32, np.integer, np.floating)):
                stacked = np.array([r.data for r in results])
            else:
                stacked = np.array([r.data for r in results])

            from .ops import _numpy_dtype_to_stype
            result_stype = _numpy_dtype_to_stype(stacked.dtype)
            result_type = Type(input_val.type.iteration_shape, result_stype)
            return Value(result_type, stacked)

        elif isinstance(data, list):
            results = []
            for elem in data:
                child_env = env.with_local(node.var, elem)
                results.append(eval_node(node.body, child_env))
            return _collect_results(input_val.type, results)

        raise TypeError(f"Cannot Map over {type(data)}")

    if isinstance(node, EachNode):
        input_val = eval_node(node.input, env)
        data = input_val.data

        if isinstance(data, np.ndarray):
            n = data.shape[0]
            elem_type = input_val.type.element_type
            results = []
            for i in range(n):
                slice_data = data[i]
                if isinstance(elem_type, Type):
                    elem_val = Value(elem_type, slice_data)
                else:
                    elem_val = Value(Type(Shape(), elem_type), slice_data)
                child_env = env.with_local(node.var, elem_val)
                results.append(eval_node(node.body, child_env))
        elif isinstance(data, list):
            results = []
            for elem in data:
                child_env = env.with_local(node.var, elem)
                results.append(eval_node(node.body, child_env))
        else:
            raise TypeError(f"Cannot Each over {type(data)}")

        return _collect_results(input_val.type, results)

    if isinstance(node, WhereNode):
        input_val = eval_node(node.input, env)
        coords = np.argwhere(input_val.data).astype(np.int32)
        result_type = Type(Shape(Dynamic()), coord_type(input_val.type.rank))
        return Value(result_type, coords)

    if isinstance(node, GatherNode):
        target_val = eval_node(node.target, env)
        indexer_val = eval_node(node.indexer, env)
        return _gather(target_val, indexer_val)

    # -- Grid primitives --

    if isinstance(node, DecomposeNode):
        coord_val = eval_node(node.input, env)
        data = coord_val.data
        fields = {}
        shift = 0
        for i, bw in enumerate(node.bit_widths):
            mask = (1 << bw) - 1
            fields[f"level_{i}"] = ((data >> shift) & mask).astype(np.int32)
            shift += bw
        fields["which_top"] = (data >> shift).astype(np.int32)

        # Infer type
        input_types = {k: v.type for k, v in env.inputs.items()}
        type_env = {k: v.type for k, v in {**env.bindings, **env.locals}.items()}
        result_type = node.infer_type(type_env, input_types)

        return Value(result_type, StructValue(fields))

    if isinstance(node, Morton3dNode):
        coord_val = eval_node(node.input, env)
        result = np_morton3d(coord_val.data)
        return Value(Type(Shape(), ScalarType.I32), result)

    # -- Layout operations --

    if isinstance(node, CutNode):
        input_val = eval_node(node.input, env)
        from .layouts import cut_by_size
        new_type = cut_by_size(node.size, input_val.type)
        # Reshape the data to match
        data = input_val.data
        if isinstance(data, np.ndarray):
            n = data.shape[0] // node.size
            new_shape = (n, node.size) + data.shape[1:]
            data = data.reshape(new_shape)
        return Value(new_type, data)

    if isinstance(node, ReshapeNode):
        input_val = eval_node(node.input, env)
        from .layouts import reshape as reshape_layout
        new_type = reshape_layout(input_val.type, node.new_shape)
        data = input_val.data
        if isinstance(data, np.ndarray):
            data = data.reshape(node.new_shape)
        return Value(new_type, data)

    raise TypeError(f"Cannot evaluate node type: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gather(target: Value, indexer: Value) -> Value:
    """Execute a Gather: look up target at coordinates from indexer."""
    idx_elem = indexer.type.element_type

    def _sentinel(et):
        if et == ScalarType.BOOL:
            return np.bool_(False)
        elif et in (ScalarType.I32, ScalarType.I64):
            return np.int32(-1)
        return np.float32(0.0)

    # Special case 1: single-point lookup with vector coordinate.
    # (r,) integer indexer into rank-r target -> single element.
    if (isinstance(indexer.data, np.ndarray) and
            indexer.type.is_scalar_element and
            indexer.type.element_type in (ScalarType.I32, ScalarType.I64) and
            indexer.data.ndim == 1 and
            indexer.data.shape[0] == target.type.rank and
            target.type.rank > 0):
        coord = indexer.data
        et = target.type.element_type
        result_type = Type(Shape(), et) if isinstance(et, ScalarType) else et

        in_bounds = all(0 <= coord[i] < target.data.shape[i] for i in range(len(coord)))
        if not in_bounds:
            return Value(result_type, _sentinel(et) if isinstance(et, ScalarType) else np.int32(-1))

        result_data = target.data[tuple(coord)]
        return Value(result_type, result_data)

    # Special case 2: scalar integer indexing into rank-1 target.
    # Returns the element directly (unwrapped).
    if (indexer.type.rank == 0 and
            isinstance(indexer.type.element_type, ScalarType) and
            indexer.type.element_type in (ScalarType.I32, ScalarType.I64) and
            target.type.rank == 1):
        idx = int(indexer.data)
        et = target.type.element_type
        result_type = Type(Shape(), et) if isinstance(et, ScalarType) else et

        if idx < 0 or idx >= target.data.shape[0]:
            if isinstance(et, ScalarType):
                return Value(result_type, _sentinel(et))
            else:
                return Value(result_type, np.full(target.data.shape[1:], -1, dtype=np.int32))

        result_data = target.data[idx]
        return Value(result_type, result_data)

    from .layouts import indexed as indexed_layout
    result_type = indexed_layout(indexer.type, target.type)

    if isinstance(indexer.data, np.ndarray):
        if isinstance(idx_elem, ScalarType):
            result_data = target.data[indexer.data]
        elif isinstance(idx_elem, Type):
            coords = indexer.data
            if coords.ndim == 2:
                idx_tuple = tuple(coords[:, i] for i in range(coords.shape[1]))
                result_data = target.data[idx_tuple]
            elif coords.ndim == 1:
                result_data = target.data[tuple(coords)]
            else:
                result_data = target.data[tuple(coords)]
        else:
            raise TypeError(f"Unexpected indexer element: {idx_elem!r}")
        return Value(result_type, result_data)

    elif isinstance(indexer.data, list):
        results = [_gather(target, idx_val) for idx_val in indexer.data]
        return Value(result_type, results)

    raise TypeError(f"Unexpected indexer data: {type(indexer.data)}")


def _promote_dynamic_to_jagged(ty: Type) -> Type:
    """Promote Dynamic extents to Jagged in a type's iteration shape."""
    if ty.rank == 0:
        return ty
    new_extents = tuple(
        Jagged() if isinstance(e, Dynamic) else e
        for e in ty.iteration_shape.extents
    )
    if new_extents == ty.iteration_shape.extents:
        return ty
    return Type(Shape(*new_extents), ty.element_type)


def _collect_results(input_type: Type, results: list[Value]) -> Value:
    """Collect Each/Map results. Dynamic inner extents become Jagged."""
    if not results:
        raise TypeError("Empty result list")

    first_type = results[0].type
    outer_extent = input_type.iteration_shape.extents[0] if input_type.rank > 0 else Static(len(results))

    # Promote Dynamic -> Jagged: Each applies body independently per element.
    inner_type = _promote_dynamic_to_jagged(first_type) if isinstance(first_type, Type) else first_type

    # Check if all results have identical data shapes (for stacking).
    all_same_data = all(
        isinstance(r.data, np.ndarray) and r.data.shape == results[0].data.shape
        for r in results[1:]
    ) if isinstance(results[0].data, np.ndarray) else False

    if all_same_data and isinstance(results[0].data, np.ndarray):
        stacked = np.stack([r.data for r in results])
        result_type = Type(Shape(outer_extent), inner_type)
        return Value(result_type, stacked)
    else:
        result_type = Type(Shape(outer_extent), inner_type)
        return Value(result_type, results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(source: str, inputs: dict[str, Value]) -> tuple[dict[str, Type], Value]:
    """Parse, type-check, and execute a DSL program.

    Returns (inferred_types, output_value) where inferred_types maps each
    let-binding name to its type.
    """
    prog = parse(source)

    # Pass 1: type-check
    input_types = {name: val.type for name, val in inputs.items()}
    inferred = prog.infer_types(input_types)

    # Pass 2: execute
    env = EvalEnv(inputs)
    for name, node in prog.bindings:
        val = eval_node(node, env)
        env.bindings[name] = val

    output_val = env.bindings[prog.output]
    return inferred, output_val
