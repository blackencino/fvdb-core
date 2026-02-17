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
    CountNode,
    CutNode,
    DecomposeNode,
    DivNode,
    EachLeftNode,
    EachNode,
    EachRightNode,
    FieldNode,
    FindNode,
    GatherNode,
    GENode,
    InBoundsNode,
    InputNode,
    MapNode,
    MaskedNode,
    Morton3dNode,
    MulNode,
    Node,
    NotNode,
    OverNode,
    PriorNode,
    Program,
    RefNode,
    ReshapeNode,
    ScanNode,
    SortNode,
    SubNode,
    UniqueNode,
    WhereNode,
)
from .dsl_parse import parse
from .layouts import MaskedElement
from .ops import StructValue, Value
from .ops import morton3d as np_morton3d
from .types import Dynamic, Jagged, ScalarType, Shape, Static, Type, coord_type


# ---------------------------------------------------------------------------
# Masked value -- carries bitmask + offset for masked Gather
# ---------------------------------------------------------------------------


class MaskedValue:
    """Runtime representation of a masked layout: bitmask + absolute prefix.

    The absolute prefix folds the base offset into the cumulative popcount:
    abs_prefix[word] = node_offset + cum_popc_before_word.
    Query is just abs_prefix[word] + partial_popcount. Two arrays, two gathers.

    Works for any mask size: 8 words (leaf 8^3), 64 words (lower 16^3),
    512 words (upper 32^3), or any other width.
    """

    def __init__(self, mask_data, abs_prefix_data):
        self.mask_data = mask_data  # numpy array (W,) i64 -- packed u64 mask
        self.abs_prefix_data = abs_prefix_data  # numpy array (W,) i32 -- absolute prefix

    def lookup(self, coord):
        """Check bitmask and compute dense index for a 3D coordinate.

        Args:
            coord: (3,) i32 array -- node-local coordinate

        Returns:
            int: abs_prefix[word] + partial_popcount if active, else -1
        """
        # Guard: if abs_prefix[0] is negative, this masked value was
        # constructed from sentinel data (the parent returned -1).
        if len(self.abs_prefix_data) > 0 and int(self.abs_prefix_data[0]) < 0:
            return np.int64(-1)

        n_words = len(self.mask_data)
        bits_per_word = 64
        total_bits = n_words * bits_per_word
        axis_size = round(total_bits ** (1 / 3))
        flat_idx = int(coord[0]) * axis_size * axis_size + int(coord[1]) * axis_size + int(coord[2])
        word_idx = flat_idx >> 6
        bit_pos = flat_idx & 63

        if word_idx < 0 or word_idx >= n_words:
            return np.int64(-1)

        word = int(self.mask_data[word_idx])
        if not ((word >> bit_pos) & 1):
            return np.int64(-1)

        # Absolute prefix already includes the base offset
        abs_cum = int(self.abs_prefix_data[word_idx])
        partial_mask = word & ((1 << bit_pos) - 1)
        partial = bin(partial_mask & 0xFFFFFFFFFFFFFFFF).count("1")

        return np.int64(abs_cum + partial)

    def __repr__(self):
        n_active = sum(bin(int(w) & 0xFFFFFFFFFFFFFFFF).count("1") for w in self.mask_data)
        total = len(self.mask_data) * 64
        return f"MaskedValue(active={n_active}/{total})"

# ---------------------------------------------------------------------------
# Evaluation environment
# ---------------------------------------------------------------------------


class EvalEnv:
    """Runtime environment: maps names to Values.

    Optional hooks dict maps Node subclasses to callables
    ``(node, env) -> Value`` that override the default evaluator for
    those node types.  Used by the pipeline executor to intercept
    collective operations (Where, Sort, Unique) with torch-backed
    implementations while evaluating the rest of the AST normally.
    """

    def __init__(
        self,
        inputs: dict[str, Value],
        bindings: dict[str, Value] | None = None,
        hooks: dict[type, Any] | None = None,
    ):
        self.inputs = inputs
        self.bindings = bindings or {}
        self.locals: dict[str, Value] = {}  # bound variables from Each/Map
        self.hooks: dict[type, Any] = hooks or {}

    def lookup(self, name: str) -> Value:
        if name in self.locals:
            return self.locals[name]
        if name in self.bindings:
            return self.bindings[name]
        raise NameError(f"Unbound name: {name!r}")

    def with_local(self, name: str, val: Value) -> EvalEnv:
        """Create a child env with an additional local binding."""
        child = EvalEnv(self.inputs, self.bindings, self.hooks)
        child.locals = {**self.locals, name: val}
        return child


# ---------------------------------------------------------------------------
# Node evaluator
# ---------------------------------------------------------------------------


def eval_node(node: Node, env: EvalEnv) -> Value:
    """Recursively evaluate an AST node to a concrete Value.

    If ``env.hooks`` contains an entry for the node's type, that hook
    is called instead of the default evaluation logic.  Hooks receive
    ``(node, env)`` and must return a ``Value``.
    """
    hook = env.hooks.get(type(node))
    if hook is not None:
        return hook(node, env)

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
            field_data = [
                sv.data.fields[node.field_name] if isinstance(sv.data, StructValue) else sv.fields[node.field_name]
                for sv in struct_val.data
            ]
            return Value(
                field_type, np.array(field_data, dtype=np.int32) if field_data else np.array([], dtype=np.int32)
            )
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

            # Determine result element type from body results, not numpy dtype.
            first_result_type = results[0].type

            if all(isinstance(r.data, np.ndarray) for r in results):
                stacked = np.stack([r.data for r in results])
            elif all(isinstance(r.data, (np.bool_, np.integer, np.floating)) for r in results):
                stacked = np.array([r.data for r in results])
            else:
                stacked = np.array([r.data for r in results])

            # Use body's result type as element type (preserves nesting info)
            if first_result_type.rank == 0:
                result_type = Type(input_val.type.iteration_shape, first_result_type.element_type)
            else:
                result_type = Type(input_val.type.iteration_shape, first_result_type)
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

    if isinstance(node, SortNode):
        input_val = eval_node(node.input, env)
        data = input_val.data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Sort requires ndarray data, got {type(data)}")
        sorted_data = _sort_leading_axis(data)
        return Value(input_val.type, sorted_data)

    if isinstance(node, UniqueNode):
        input_val = eval_node(node.input, env)
        data = input_val.data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Unique requires ndarray data, got {type(data)}")
        unique_data = _unique_leading_axis(data)

        # Preserve element type, but output length is data-dependent.
        in_ty = input_val.type
        tail = in_ty.iteration_shape.extents[1:]
        result_type = Type(Shape(Dynamic(), *tail), in_ty.element_type)
        return Value(result_type, unique_data)

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

    if isinstance(node, FindNode):
        table_val = eval_node(node.table, env)
        key_val = eval_node(node.key, env)
        table_data = table_val.data  # (R, K) numpy array
        key_data = key_val.data  # (K,) numpy array
        R = table_data.shape[0]
        for r in range(R):
            if np.array_equal(table_data[r], key_data):
                return Value(Type(Shape(), ScalarType.I32), np.int32(r))
        return Value(Type(Shape(), ScalarType.I32), np.int32(-1))

    if isinstance(node, MaskedNode):
        mask_val = eval_node(node.mask, env)
        abs_prefix_val = eval_node(node.abs_prefix, env)
        input_types = {k: v.type for k, v in env.inputs.items()}
        type_env = {k: v.type for k, v in {**env.bindings, **env.locals}.items()}
        result_type = node.infer_type(type_env, input_types)
        return Value(result_type, MaskedValue(mask_val.data, abs_prefix_val.data))

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

    # -- Adverbs --

    if isinstance(node, OverNode):
        input_val = eval_node(node.input, env)
        verb_fn = _resolve_verb(node.verb)
        data = input_val.data
        if isinstance(data, np.ndarray):
            # Reduce over all elements of the iteration space
            # For multi-rank, this reduces the full array to the element shape
            et = input_val.type.element_type
            if isinstance(et, Type):
                # Element is itself an array -- reduce leading axis
                result_data = data[0].copy()
                for i in range(1, data.shape[0]):
                    result_data = verb_fn(result_data, data[i])
                result_type = et
            else:
                # Scalar element -- reduce everything
                result_data = data.flat[0]
                for i in range(1, data.size):
                    result_data = verb_fn(result_data, data.flat[i])
                result_type = Type(Shape(), et)
            return Value(result_type, result_data)
        elif isinstance(data, list):
            # List of Values -- fold
            acc = data[0]
            for v in data[1:]:
                acc_data = verb_fn(acc.data, v.data)
                acc = Value(acc.type, acc_data)
            return acc
        raise TypeError(f"Cannot Over {type(data)}")

    if isinstance(node, ScanNode):
        input_val = eval_node(node.input, env)
        verb_fn = _resolve_verb(node.verb)
        data = input_val.data
        if isinstance(data, np.ndarray):
            result = np.empty_like(data)
            result[0] = data[0]
            for i in range(1, data.shape[0]):
                result[i] = verb_fn(result[i - 1], data[i])
            return Value(input_val.type, result)
        raise TypeError(f"Cannot Scan {type(data)}")

    if isinstance(node, EachRightNode):
        left_val = eval_node(node.left, env)
        right_val = eval_node(node.right, env)
        verb_fn = _resolve_verb(node.verb)
        data = right_val.data
        if isinstance(data, np.ndarray):
            results = []
            for i in range(data.shape[0]):
                r = verb_fn(left_val.data, data[i])
                results.append(r)
            stacked = np.stack(results) if isinstance(results[0], np.ndarray) else np.array(results)
            from .ops import _numpy_dtype_to_stype

            result_stype = _numpy_dtype_to_stype(stacked.dtype)
            return Value(Type(right_val.type.iteration_shape, result_stype), stacked)
        raise TypeError(f"Cannot EachRight over {type(data)}")

    if isinstance(node, EachLeftNode):
        left_val = eval_node(node.left, env)
        right_val = eval_node(node.right, env)
        verb_fn = _resolve_verb(node.verb)
        data = left_val.data
        if isinstance(data, np.ndarray):
            results = []
            for i in range(data.shape[0]):
                r = verb_fn(data[i], right_val.data)
                results.append(r)
            stacked = np.stack(results) if isinstance(results[0], np.ndarray) else np.array(results)
            from .ops import _numpy_dtype_to_stype

            result_stype = _numpy_dtype_to_stype(stacked.dtype)
            return Value(Type(left_val.type.iteration_shape, result_stype), stacked)
        raise TypeError(f"Cannot EachLeft over {type(data)}")

    if isinstance(node, PriorNode):
        input_val = eval_node(node.input, env)
        verb_fn = _resolve_verb(node.verb)
        data = input_val.data
        if isinstance(data, np.ndarray):
            results = []
            for i in range(1, data.shape[0]):
                results.append(verb_fn(data[i], data[i - 1]))
            stacked = np.stack(results) if isinstance(results[0], np.ndarray) else np.array(results)
            lead = input_val.type.iteration_shape.extents[0]
            new_lead = Static(lead.n - 1) if isinstance(lead, Static) else lead
            from .ops import _numpy_dtype_to_stype

            result_stype = _numpy_dtype_to_stype(stacked.dtype)
            return Value(Type(Shape(new_lead), result_stype), stacked)
        raise TypeError(f"Cannot Prior over {type(data)}")

    # -- Additional scalar primitives --

    if isinstance(node, DivNode):
        a = eval_node(node.a, env)
        b = eval_node(node.b, env)
        b_data = b.data if isinstance(b.data, np.ndarray) else float(b.data)
        result = (a.data / b_data).astype(np.float32)
        a_ty = a.type
        if a_ty.rank == 0:
            return Value(Type(Shape(), ScalarType.F32), result)
        return Value(Type(a_ty.iteration_shape, ScalarType.F32), result)

    if isinstance(node, MulNode):
        a = eval_node(node.a, env)
        b = eval_node(node.b, env)
        b_data = b.data if isinstance(b.data, np.ndarray) else b.data
        return Value(a.type, (a.data * b_data).astype(a.data.dtype))

    if isinstance(node, CountNode):
        input_val = eval_node(node.input, env)
        if isinstance(input_val.data, np.ndarray):
            count = input_val.data.shape[0]
        elif isinstance(input_val.data, list):
            count = len(input_val.data)
        else:
            count = 1
        return Value(Type(Shape(), ScalarType.I32), np.int32(count))

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

    # Masked target: bitmask check + popcount.
    if isinstance(target.data, MaskedValue):
        coord = indexer.data
        result = target.data.lookup(coord)
        return Value(Type(Shape(), ScalarType.I64), result)

    # Special case 1: single-point lookup with vector coordinate.
    # (r,) integer indexer into rank-r target -> single element.
    if (
        isinstance(indexer.data, np.ndarray)
        and indexer.type.is_scalar_element
        and indexer.type.element_type in (ScalarType.I32, ScalarType.I64)
        and indexer.data.ndim == 1
        and indexer.data.shape[0] == target.type.rank
        and target.type.rank > 0
    ):
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
    if (
        indexer.type.rank == 0
        and isinstance(indexer.type.element_type, ScalarType)
        and indexer.type.element_type in (ScalarType.I32, ScalarType.I64)
        and target.type.rank == 1
    ):
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


def _resolve_verb(name: str):
    """Map a verb name to a binary numpy function."""
    verbs = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b,
        "Div": lambda a, b: a / b,
        "Min": lambda a, b: np.minimum(a, b),
        "Max": lambda a, b: np.maximum(a, b),
        "And": lambda a, b: a & b,
        "Or": lambda a, b: a | b,
    }
    if name not in verbs:
        raise TypeError(f"Unknown verb: {name!r}")
    return verbs[name]


def _sort_leading_axis(data: np.ndarray) -> np.ndarray:
    """Stable sort along the leading axis with immutable value semantics."""
    if data.ndim == 1:
        return np.sort(data, kind="stable")
    rows = data.reshape(data.shape[0], -1)
    # np.lexsort uses last key as primary; reverse to sort by column order.
    order = np.lexsort(tuple(rows[:, i] for i in range(rows.shape[1] - 1, -1, -1)))
    return data[order].copy()


def _unique_leading_axis(data: np.ndarray) -> np.ndarray:
    """Deduplicate along leading axis with immutable value semantics."""
    if data.ndim == 1:
        return np.unique(data)
    rows = data.reshape(data.shape[0], -1)
    _, first_idx = np.unique(rows, axis=0, return_index=True)
    # Keep first occurrence order to preserve value-semantic predictability.
    first_idx_sorted = np.sort(first_idx)
    return data[first_idx_sorted].copy()


def _promote_dynamic_to_jagged(ty: Type) -> Type:
    """Promote Dynamic extents to Jagged in a type's iteration shape."""
    if ty.rank == 0:
        return ty
    new_extents = tuple(Jagged() if isinstance(e, Dynamic) else e for e in ty.iteration_shape.extents)
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
    all_same_data = (
        all(isinstance(r.data, np.ndarray) and r.data.shape == results[0].data.shape for r in results[1:])
        if isinstance(results[0].data, np.ndarray)
        else False
    )

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
