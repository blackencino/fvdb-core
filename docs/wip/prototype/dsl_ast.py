# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
AST node classes for the micro DSL.

Each node can infer its output type from an environment mapping bound variable
names to types. No data, no execution -- pure type-level reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from .layouts import (
    StructElement,
    cut_by_size,
    flip as flip_layout,
    indexed as indexed_layout,
    reshape as reshape_layout,
    struct_layout,
)
from .types import (
    Dynamic,
    Extent,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)

# Type environment: maps bound variable names -> Type
Env = dict[str, Type]

# Input declarations: maps input names -> Type
InputDecls = dict[str, Type]


def _promote_dynamic_to_jagged(ty: Type) -> Type:
    """Promote Dynamic extents to Jagged in a type's iteration shape.

    Used by Each: since the body runs independently per element, any Dynamic
    extent in the result could resolve to a different value for each element,
    making it jagged by definition. Static extents are preserved.
    """
    if ty.rank == 0:
        return ty
    new_extents = tuple(
        Jagged() if isinstance(e, Dynamic) else e
        for e in ty.iteration_shape.extents
    )
    if new_extents == ty.iteration_shape.extents:
        return ty
    return Type(Shape(*new_extents), ty.element_type)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Node:
    """Base class for all AST nodes."""

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        raise NotImplementedError(f"{type(self).__name__}.infer_type")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(...)"


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputNode(Node):
    """Reference to an externally-provided named input."""
    name: str

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        if self.name not in inputs:
            raise TypeError(f"Unknown input: {self.name!r}")
        return inputs[self.name]

    def __repr__(self) -> str:
        return f'Input("{self.name}")'


@dataclass(frozen=True)
class ConstNode(Node):
    """A literal constant value."""
    value: Any
    stype: ScalarType

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        if isinstance(self.value, (int, float, bool)):
            return Type(Shape(), self.stype)
        elif isinstance(self.value, list):
            return Type(Shape(Static(len(self.value))), self.stype)
        raise TypeError(f"Cannot infer type for const value: {self.value!r}")

    def __repr__(self) -> str:
        return f"Const({self.value})"


@dataclass(frozen=True)
class RefNode(Node):
    """Reference to a bound variable (from Each/Map binding)."""
    name: str

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        if self.name not in env:
            raise TypeError(f"Unbound variable: {self.name!r}")
        return env[self.name]

    def __repr__(self) -> str:
        return f"Ref({self.name})"


# ---------------------------------------------------------------------------
# Struct field access
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldNode(Node):
    """Project a named field from a struct-typed expression."""
    expr: Node
    field_name: str

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        ty = self.expr.infer_type(env, inputs)
        elem = ty.element_type
        if isinstance(elem, StructElement):
            for name, field_ty in elem.fields:
                if name == self.field_name:
                    # Preserve the outer iteration shape, use the field
                    # type as the new element type.
                    if isinstance(field_ty, Type):
                        return Type(ty.iteration_shape, field_ty) if ty.rank > 0 else field_ty
                    else:
                        return Type(ty.iteration_shape, field_ty)
            raise TypeError(f"Struct has no field {self.field_name!r}")
        raise TypeError(f"Field access requires struct element, got {elem!r}")

    def __repr__(self) -> str:
        return f"Field({self.expr}, {self.field_name!r})"


# ---------------------------------------------------------------------------
# Scalar / vector primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AddNode(Node):
    a: Node
    b: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        ta = self.a.infer_type(env, inputs)
        tb = self.b.infer_type(env, inputs)
        # Scalar + scalar, or vector + vector (same shape)
        return ta  # result has same type as inputs

    def __repr__(self) -> str:
        return f"Add({self.a}, {self.b})"


@dataclass(frozen=True)
class SubNode(Node):
    a: Node
    b: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return self.a.infer_type(env, inputs)

    def __repr__(self) -> str:
        return f"Sub({self.a}, {self.b})"


@dataclass(frozen=True)
class GENode(Node):
    """Greater-than-or-equal comparison."""
    a: Node
    b: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        ta = self.a.infer_type(env, inputs)
        return Type(ta.iteration_shape, ScalarType.BOOL)

    def __repr__(self) -> str:
        return f"GE({self.a}, {self.b})"


@dataclass(frozen=True)
class AndNode(Node):
    a: Node
    b: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        ta = self.a.infer_type(env, inputs)
        return Type(ta.iteration_shape, ScalarType.BOOL)

    def __repr__(self) -> str:
        return f"And({self.a}, {self.b})"


@dataclass(frozen=True)
class NotNode(Node):
    a: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return self.a.infer_type(env, inputs)

    def __repr__(self) -> str:
        return f"Not({self.a})"


@dataclass(frozen=True)
class InBoundsNode(Node):
    """Check if all components of a coordinate are in [lo, hi)."""
    coord: Node
    lo: Node
    hi: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return Type(Shape(), ScalarType.BOOL)

    def __repr__(self) -> str:
        return f"InBounds({self.coord}, {self.lo}, {self.hi})"


# ---------------------------------------------------------------------------
# Structural operations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MapNode(Node):
    """Map(input, var => body): apply body to each element."""
    input: Node
    var: str
    body: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        # Bind var to the element type
        if isinstance(input_ty.element_type, Type):
            elem_ty = input_ty.element_type
        else:
            elem_ty = Type(Shape(), input_ty.element_type)
        inner_env = {**env, self.var: elem_ty}
        body_ty = self.body.infer_type(inner_env, inputs)
        # Map preserves iteration shape, body result becomes new element
        if body_ty.rank == 0:
            return Type(input_ty.iteration_shape, body_ty.element_type)
        else:
            return Type(input_ty.iteration_shape, body_ty)

    def __repr__(self) -> str:
        return f"Map({self.input}, {self.var} => {self.body})"


@dataclass(frozen=True)
class EachNode(Node):
    """Each(input, var => body): apply body per outer element, may nest."""
    input: Node
    var: str
    body: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        # Bind var to the element type
        if isinstance(input_ty.element_type, Type):
            elem_ty = input_ty.element_type
        else:
            elem_ty = Type(Shape(), input_ty.element_type)
        inner_env = {**env, self.var: elem_ty}
        body_ty = self.body.infer_type(inner_env, inputs)
        # Each applies body independently per element. Any Dynamic extent
        # in the body's result becomes Jagged, because the extent may
        # resolve to a different value for each element (the definition
        # of jagged). Static extents are preserved -- they're guaranteed
        # uniform across elements.
        body_ty = _promote_dynamic_to_jagged(body_ty)
        return Type(input_ty.iteration_shape, body_ty)

    def __repr__(self) -> str:
        return f"Each({self.input}, {self.var} => {self.body})"


@dataclass(frozen=True)
class WhereNode(Node):
    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        if input_ty.element_type != ScalarType.BOOL:
            # Check if it's a Type wrapping bool
            if isinstance(input_ty.element_type, Type) and input_ty.element_type.element_type == ScalarType.BOOL:
                pass
            else:
                raise TypeError(f"Where requires bool element, got {input_ty.element_type!r}")
        return Type(Shape(Dynamic()), coord_type(input_ty.rank))

    def __repr__(self) -> str:
        return f"Where({self.input})"


@dataclass(frozen=True)
class GatherNode(Node):
    target: Node
    indexer: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        target_ty = self.target.infer_type(env, inputs)
        indexer_ty = self.indexer.infer_type(env, inputs)

        # Special case 1: single-point lookup with a vector coordinate.
        # If the indexer is (r,) integer and the target is rank r, treat the
        # entire indexer as one coordinate, returning the target's element type.
        if (indexer_ty.is_scalar_element and
                indexer_ty.element_type in (ScalarType.I32, ScalarType.I64) and
                indexer_ty.rank == 1 and target_ty.rank > 0):
            lead = indexer_ty.iteration_shape.extents[0]
            if isinstance(lead, Static) and lead.n == target_ty.rank:
                et = target_ty.element_type
                return Type(Shape(), et) if isinstance(et, ScalarType) else et

        # Special case 2: scalar integer indexing into rank-1 target.
        # Returns the element type directly (unwrapped, not () over E).
        if (indexer_ty.rank == 0 and
                isinstance(indexer_ty.element_type, ScalarType) and
                indexer_ty.element_type in (ScalarType.I32, ScalarType.I64) and
                target_ty.rank == 1):
            et = target_ty.element_type
            return Type(Shape(), et) if isinstance(et, ScalarType) else et

        return indexed_layout(indexer_ty, target_ty)

    def __repr__(self) -> str:
        return f"Gather({self.target}, {self.indexer})"


# ---------------------------------------------------------------------------
# Grid primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecomposeNode(Node):
    input: Node
    bit_widths: list[int]

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        # Build struct type with level_0, level_1, ..., which_top
        field_types = {}
        for i in range(len(self.bit_widths)):
            field_types[f"level_{i}"] = input_ty
        field_types["which_top"] = input_ty
        sty = struct_layout(**field_types)
        return flip_layout(sty)

    def __repr__(self) -> str:
        return f"Decompose({self.input}, {self.bit_widths})"


@dataclass(frozen=True)
class Morton3dNode(Node):
    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        # (3,) i32 -> scalar i32, or (*,3) i32 -> (*,) i32
        input_ty = self.input.infer_type(env, inputs)
        return Type(Shape(), ScalarType.I32)

    def __repr__(self) -> str:
        return f"Morton3d({self.input})"


# ---------------------------------------------------------------------------
# Layout operations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CutNode(Node):
    input: Node
    size: int

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return cut_by_size(self.size, self.input.infer_type(env, inputs))

    def __repr__(self) -> str:
        return f"Cut({self.input}, {self.size})"


@dataclass(frozen=True)
class ReshapeNode(Node):
    input: Node
    new_shape: tuple

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return reshape_layout(self.input.infer_type(env, inputs), self.new_shape)

    def __repr__(self) -> str:
        return f"Reshape({self.input}, {self.new_shape})"


# ---------------------------------------------------------------------------
# Adverbs (K-style)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OverNode(Node):
    """Over(f, xs): reduce the full iteration space with a dyadic verb.

    Result is the element type (iteration space fully consumed).
    For rank > 1, f must be commutative+associative.
    f is a named verb (e.g. "Add", "Mul") resolved at evaluation time.
    """
    verb: str
    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        et = input_ty.element_type
        if isinstance(et, Type):
            return et
        return Type(Shape(), et)

    def __repr__(self) -> str:
        return f"Over({self.verb}, {self.input})"


@dataclass(frozen=True)
class ScanNode(Node):
    """Scan(f, xs): running accumulation. Rank 1 only.

    Same shape as input; element i is f applied cumulatively to elements 0..i.
    """
    verb: str
    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        if input_ty.rank != 1:
            raise TypeError(f"Scan requires rank 1, got rank {input_ty.rank}")
        return input_ty

    def __repr__(self) -> str:
        return f"Scan({self.verb}, {self.input})"


@dataclass(frozen=True)
class EachRightNode(Node):
    """EachRight(f, x, ys): for each y in ys, compute f(x, y). x is fixed."""
    verb: str
    left: Node
    right: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        right_ty = self.right.infer_type(env, inputs)
        # Result has the iteration shape of ys, element type from f
        left_ty = self.left.infer_type(env, inputs)
        # For arithmetic verbs, result element = left element type
        et = left_ty.element_type if isinstance(left_ty.element_type, ScalarType) else left_ty
        if isinstance(right_ty.element_type, Type):
            et = right_ty.element_type.element_type if isinstance(right_ty.element_type.element_type, ScalarType) else right_ty.element_type
        return Type(right_ty.iteration_shape, et)

    def __repr__(self) -> str:
        return f"EachRight({self.verb}, {self.left}, {self.right})"


@dataclass(frozen=True)
class EachLeftNode(Node):
    """EachLeft(f, xs, y): for each x in xs, compute f(x, y). y is fixed."""
    verb: str
    left: Node
    right: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        left_ty = self.left.infer_type(env, inputs)
        et = left_ty.element_type
        if isinstance(et, Type):
            et = et.element_type if isinstance(et.element_type, ScalarType) else et
        return Type(left_ty.iteration_shape, et)

    def __repr__(self) -> str:
        return f"EachLeft({self.verb}, {self.left}, {self.right})"


@dataclass(frozen=True)
class PriorNode(Node):
    """Prior(f, xs): apply f to adjacent pairs. Rank 1 only.

    Result length = input length - 1.
    """
    verb: str
    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        if input_ty.rank != 1:
            raise TypeError(f"Prior requires rank 1, got rank {input_ty.rank}")
        lead = input_ty.iteration_shape.extents[0]
        if isinstance(lead, Static):
            new_lead = Static(lead.n - 1)
        else:
            new_lead = lead  # Dynamic stays Dynamic
        return Type(Shape(new_lead), input_ty.element_type)

    def __repr__(self) -> str:
        return f"Prior({self.verb}, {self.input})"


# ---------------------------------------------------------------------------
# Additional scalar primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DivNode(Node):
    a: Node
    b: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        ta = self.a.infer_type(env, inputs)
        # Division produces float
        if ta.rank == 0:
            return Type(Shape(), ScalarType.F32)
        return Type(ta.iteration_shape, ScalarType.F32)

    def __repr__(self) -> str:
        return f"Div({self.a}, {self.b})"


@dataclass(frozen=True)
class MulNode(Node):
    a: Node
    b: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return self.a.infer_type(env, inputs)

    def __repr__(self) -> str:
        return f"Mul({self.a}, {self.b})"


@dataclass(frozen=True)
class CountNode(Node):
    """Count(xs): length of the iteration space. Returns scalar i32."""
    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return Type(Shape(), ScalarType.I32)

    def __repr__(self) -> str:
        return f"Count({self.input})"


# ---------------------------------------------------------------------------
# Program: a sequence of let-bindings + an output
# ---------------------------------------------------------------------------

@dataclass
class Program:
    """A sequence of named bindings and a final output expression."""
    bindings: list[tuple[str, Node]]  # [(name, expr), ...]
    output: str  # name of the output binding

    def infer_types(self, inputs: InputDecls) -> dict[str, Type]:
        """Type-check all bindings, returning a map of name -> inferred type."""
        env: Env = {}
        result = {}
        for name, node in self.bindings:
            ty = node.infer_type(env, inputs)
            env[name] = ty
            result[name] = ty
        return result

    def output_type(self, inputs: InputDecls) -> Type:
        types = self.infer_types(inputs)
        if self.output not in types:
            raise TypeError(f"Output {self.output!r} not found in bindings")
        return types[self.output]
