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
    MaskedElement,
    StructElement,
    cut_by_size,
    flatten as flatten_layout,
    flip as flip_layout,
    fuse as fuse_layout,
    indexed as indexed_layout,
    masked_layout,
    permute as permute_layout,
    reshape as reshape_layout,
    struct_layout,
)
from .types import (
    Dynamic,
    Extent,
    FnType,
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
        return f"field({self.expr}, {self.field_name!r})"


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
class SortNode(Node):
    """Sort(xs): stable ascending sort over the leading iteration axis.

    For scalar elements this is the usual rank-1 sort.
    For non-scalar elements (e.g. coordinate vectors), rows are sorted
    lexicographically over their flattened element values.
    """

    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        if input_ty.rank == 0:
            raise TypeError("Sort requires rank >= 1")
        return input_ty

    def __repr__(self) -> str:
        return f"Sort({self.input})"


@dataclass(frozen=True)
class UniqueNode(Node):
    """Unique(xs): deduplicate elements along the leading iteration axis.

    Output length is data-dependent; leading extent becomes Dynamic.
    Element type is preserved.
    """

    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        if input_ty.rank == 0:
            raise TypeError("Unique requires rank >= 1")
        tail = input_ty.iteration_shape.extents[1:]
        return Type(Shape(Dynamic(), *tail), input_ty.element_type)

    def __repr__(self) -> str:
        return f"Unique({self.input})"


@dataclass(frozen=True)
class GatherNode(Node):
    target: Node
    indexer: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        target_ty = self.target.infer_type(env, inputs)
        indexer_ty = self.indexer.infer_type(env, inputs)

        # Masked target: Gather(masked_layout, coord) -> element type.
        # The coord must match the masked layout's iteration shape rank.
        # Returns the masked element type (i64 dense index, or -1 sentinel).
        if isinstance(target_ty.element_type, MaskedElement):
            me = target_ty.element_type
            return Type(Shape(), me.element_type)

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
class FindNode(Node):
    """Find(table, key): linear search for a matching row in a small table.

    table: (R, K) -- a small array of K-dimensional keys
    key: (K,) -- the key to search for

    Returns the row index (i32 scalar) of the first match, or -1 if not found.
    General-purpose search primitive, not CIG-specific.
    """

    table: Node
    key: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return Type(Shape(), ScalarType.I32)

    def __repr__(self) -> str:
        return f"Find({self.table}, {self.key})"


@dataclass(frozen=True)
class Morton3dNode(Node):
    """Morton3d: unsigned morton encoding for non-negative coordinates.

    (3,) i32 -> () i32, or (*,3) i32 -> (*,) i32.
    Used for indexing into morton-linearized arrays.
    For signed coordinates, use Morton3dSigned.
    """

    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        # (3,) i32 -> scalar i32, or (*,3) i32 -> (*,) i32
        input_ty = self.input.infer_type(env, inputs)
        return Type(Shape(), ScalarType.I32)

    def __repr__(self) -> str:
        return f"Morton3d({self.input})"


@dataclass(frozen=True)
class Morton3dSignedNode(Node):
    """Morton3dSigned: signed morton encoding for arbitrary i32 coordinates.

    Offsets each axis by 2^20 to handle negative values.
    (3,) i32 -> () i64, or (*,) / (3,) i32 -> (*,) i64.
    21 bits per axis x 3 = 63 bits, fits in i64.
    """

    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        # Batched: (*,) / (3,) i32 -> (*,) / i64
        if isinstance(input_ty.element_type, Type):
            return Type(input_ty.iteration_shape, ScalarType.I64)
        # Scalar: (3,) i32 -> () / i64
        return Type(Shape(), ScalarType.I64)

    def __repr__(self) -> str:
        return f"Morton3dSigned({self.input})"


@dataclass(frozen=True)
class MortonDecode3dNode(Node):
    """Decode morton codes back to signed 3D coordinates.

    Inverse of Morton3d.  Extracts per-axis bits and subtracts the
    2^20 offset to restore signed i32 coordinates.
    """

    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        input_ty = self.input.infer_type(env, inputs)
        coord_elem = Type(Shape(Static(3)), ScalarType.I32)
        # Batched: (*,) / i64 -> (*,) / (3,) i32
        if input_ty.rank > 0:
            return Type(input_ty.iteration_shape, coord_elem)
        # Scalar: () / i64 -> (3,) i32
        return coord_elem

    def __repr__(self) -> str:
        return f"MortonDecode3d({self.input})"


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
        return f"cut({self.input}, {self.size})"


@dataclass(frozen=True)
class ReshapeNode(Node):
    input: Node
    new_shape: tuple

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return reshape_layout(self.input.infer_type(env, inputs), self.new_shape)

    def __repr__(self) -> str:
        return f"reshape({self.input}, {self.new_shape})"


@dataclass(frozen=True)
class FuseNode(Node):
    """fuse(x): merge the two outermost nesting levels (layout, free)."""
    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return fuse_layout(self.input.infer_type(env, inputs))

    def __repr__(self) -> str:
        return f"fuse({self.input})"


@dataclass(frozen=True)
class FlattenNode(Node):
    """flatten(x): merge ALL nesting levels into one (layout, free)."""
    input: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return flatten_layout(self.input.infer_type(env, inputs))

    def __repr__(self) -> str:
        return f"flatten({self.input})"


@dataclass(frozen=True)
class PermuteNode(Node):
    """permute(x, order): reorder axes within the leading shape (layout, free)."""
    input: Node
    order: tuple

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        return permute_layout(self.input.infer_type(env, inputs), self.order)

    def __repr__(self) -> str:
        return f"permute({self.input}, {self.order})"


@dataclass(frozen=True)
class MaskedNode(Node):
    """masked(mask_expr, abs_prefix_expr): layout (lowercase, free).

    Constructs a masked layout from a bitmask and an absolute prefix-sum
    array. Access via Gather computes bitmask check + prefix lookup +
    partial popcount.

    Physical storage per node:
      mask:       (W,) i64  -- W packed u64 words (W * 64 = total positions)
      abs_prefix: (W,) i32  -- absolute index: node_offset + cum_popc_before_word

    The base offset is folded into the prefix at build time, so the query
    is just abs_prefix[word] + partial_popcount. Two gathers per level.

    Works for any node shape: W=8 for 8x8x8, W=64 for 16^3, W=512 for 32^3.
    """

    mask: Node
    abs_prefix: Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        mask_ty = self.mask.infer_type(env, inputs)
        return Type(Shape(), MaskedElement(Shape(Static(8), Static(8), Static(8)), ScalarType.I64))

    def __repr__(self) -> str:
        return f"masked({self.mask}, {self.abs_prefix})"


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
# Functions as values: VerbRef, AdverbApply, Apply
# ---------------------------------------------------------------------------

# The set of adverb names recognised by AdverbApplyNode.
_ADVERB_NAMES = {"Over", "Scan", "EachRight", "EachLeft", "EachBoth", "Prior", "Each"}

# Arity of the function produced by each adverb.
_ADVERB_ARITY = {
    "Over": 1,      # Over(f): monadic -- consumes leading shape
    "Scan": 1,      # Scan(f): monadic -- rank-1 running accumulation
    "Prior": 1,     # Prior(f): monadic -- rank-1 adjacent pairs
    "Each": 1,      # Each(f): monadic -- iterate leading shape
    "EachRight": 2, # EachRight(f): dyadic -- x whole, iterate y
    "EachLeft": 2,  # EachLeft(f): dyadic -- iterate x, y whole
    "EachBoth": 2,  # EachBoth(f): dyadic -- zip x and y, iterate in lockstep
}


@dataclass(frozen=True)
class VerbRefNode(Node):
    """Reference to a built-in verb (Add, Sub, etc.) as a function value.

    Evaluates to a FnValue at runtime. Type is () / FnType(arity, name).
    """
    name: str

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        from .ops import VERBS
        if self.name not in VERBS:
            raise TypeError(f"Unknown verb: {self.name!r}")
        v = VERBS[self.name]
        return Type(Shape(), FnType(v.arity, self.name))

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class AdverbApplyNode(Node):
    """Apply an adverb to a function, producing a new function.

    EachLeft(Add)  ->  AdverbApplyNode("EachLeft", VerbRefNode("Add"))

    The result is a function value; no data is consumed.
    """
    adverb: str  # one of _ADVERB_NAMES
    fn: Node     # must evaluate to a FnValue

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        if self.adverb not in _ADVERB_NAMES:
            raise TypeError(f"Unknown adverb: {self.adverb!r}")
        result_arity = _ADVERB_ARITY[self.adverb]
        return Type(Shape(), FnType(result_arity, f"{self.adverb}(...)"))

    def __repr__(self) -> str:
        return f"{self.adverb}({self.fn})"


@dataclass(frozen=True)
class ApplyNode(Node):
    """Apply a function value to data arguments.

    Apply(EachLeft(Add), x, y)  ->  ApplyNode(AdverbApplyNode(...), [x, y])
    Over(Add, xs)               ->  ApplyNode(AdverbApplyNode("Over", VerbRefNode("Add")), [xs])

    The fn node must evaluate to a FnValue. Type inference resolves the
    function structurally: it pattern-matches on the fn node to compute
    the output type from the argument types.
    """
    fn: Node
    args: tuple  # tuple of Node

    def infer_type(self, env: Env, inputs: InputDecls) -> Type:
        arg_types = [a.infer_type(env, inputs) for a in self.args]
        return _infer_apply_type(self.fn, arg_types, env, inputs)

    def __repr__(self) -> str:
        arg_strs = ", ".join(repr(a) for a in self.args)
        return f"Apply({self.fn}, {arg_strs})"


def _as_element_type(ty: Type):
    """Convert a Type to an ElementType suitable for nesting.

    When a function returns () / scalar (rank-0 with scalar element), the
    result should nest as just the ScalarType, not as a rank-0 Type wrapper.
    This keeps types clean: (5,) / i32 instead of (5,) / (() / i32).

    When a function returns something with rank > 0, it nests as a Type.
    """
    if ty.rank == 0 and isinstance(ty.element_type, ScalarType):
        return ty.element_type  # unwrap: () / i32 -> i32
    return ty  # keep as Type: (3, 4) / i32 stays as-is


def _infer_apply_type(fn_node: Node, arg_types: list[Type], env: Env, inputs: InputDecls) -> Type:
    """Structurally infer the return type of applying fn_node to arg_types.

    This walks the fn_node AST to determine what the composed function
    does to the argument types. No runtime data needed.
    """
    # Case 1: VerbRefNode -- apply a built-in verb directly
    if isinstance(fn_node, VerbRefNode):
        from .ops import VERBS
        verb = VERBS[fn_node.name]
        return verb.type_fn(*arg_types)

    # Case 2: AdverbApplyNode -- apply an adverb-wrapped function
    if isinstance(fn_node, AdverbApplyNode):
        adverb = fn_node.adverb
        inner_fn = fn_node.fn

        if adverb == "Over":
            # Over(f)(xs: S / E) -> E
            xs_ty = arg_types[0]
            et = xs_ty.element_type
            return et if isinstance(et, Type) else Type(Shape(), et)

        if adverb == "Scan":
            # Scan(f)(xs: (D,) / E) -> (D,) / E  -- rank 1 only
            xs_ty = arg_types[0]
            if xs_ty.rank != 1:
                raise TypeError(f"Scan requires rank-1 leading shape, got rank {xs_ty.rank}")
            return xs_ty

        if adverb == "Prior":
            # Prior(f)(xs: (D,) / E) -> (D-1,) / E  -- rank 1 only
            xs_ty = arg_types[0]
            if xs_ty.rank != 1:
                raise TypeError(f"Prior requires rank-1 leading shape, got rank {xs_ty.rank}")
            lead = xs_ty.iteration_shape.extents[0]
            new_lead = Static(lead.n - 1) if isinstance(lead, Static) else lead
            return Type(Shape(new_lead), xs_ty.element_type)

        if adverb == "Each":
            # Each(f)(xs: S / E) -> S / f(E)
            xs_ty = arg_types[0]
            elem_ty = xs_ty.element_type
            elem_as_type = elem_ty if isinstance(elem_ty, Type) else Type(Shape(), elem_ty)
            result_inner = _infer_apply_type(inner_fn, [elem_as_type], env, inputs)
            result_elem = _as_element_type(result_inner)
            if isinstance(result_elem, Type):
                result_elem = _promote_dynamic_to_jagged(result_elem)
            return Type(xs_ty.iteration_shape, result_elem)

        if adverb == "EachRight":
            # EachRight(f)(x: T, y: S_y / E_y) -> S_y / f(T, E_y)
            x_ty, y_ty = arg_types[0], arg_types[1]
            y_elem = y_ty.element_type
            y_elem_as_type = y_elem if isinstance(y_elem, Type) else Type(Shape(), y_elem)
            result_inner = _infer_apply_type(inner_fn, [x_ty, y_elem_as_type], env, inputs)
            result_elem = _as_element_type(result_inner)
            if isinstance(result_elem, Type):
                result_elem = _promote_dynamic_to_jagged(result_elem)
            return Type(y_ty.iteration_shape, result_elem)

        if adverb == "EachLeft":
            # EachLeft(f)(x: S_x / E_x, y: T) -> S_x / f(E_x, T)
            x_ty, y_ty = arg_types[0], arg_types[1]
            x_elem = x_ty.element_type
            x_elem_as_type = x_elem if isinstance(x_elem, Type) else Type(Shape(), x_elem)
            result_inner = _infer_apply_type(inner_fn, [x_elem_as_type, y_ty], env, inputs)
            result_elem = _as_element_type(result_inner)
            if isinstance(result_elem, Type):
                result_elem = _promote_dynamic_to_jagged(result_elem)
            return Type(x_ty.iteration_shape, result_elem)

        if adverb == "EachBoth":
            # EachBoth(f)(x: S / A, y: S / B) -> S / f(A, B)
            # Leading shapes must be compatible.
            x_ty, y_ty = arg_types[0], arg_types[1]
            resolved_shape = x_ty.iteration_shape.resolve(y_ty.iteration_shape)
            x_elem = x_ty.element_type
            y_elem = y_ty.element_type
            x_elem_as_type = x_elem if isinstance(x_elem, Type) else Type(Shape(), x_elem)
            y_elem_as_type = y_elem if isinstance(y_elem, Type) else Type(Shape(), y_elem)
            result_inner = _infer_apply_type(inner_fn, [x_elem_as_type, y_elem_as_type], env, inputs)
            result_elem = _as_element_type(result_inner)
            if isinstance(result_elem, Type):
                result_elem = _promote_dynamic_to_jagged(result_elem)
            return Type(resolved_shape, result_elem)

        raise TypeError(f"Unknown adverb in type inference: {adverb!r}")

    # Case 3: RefNode -- a let-bound function value.
    # Look up the original expression node from the program's bindings
    # and recurse structurally.
    if isinstance(fn_node, RefNode):
        # The env maps names -> Types, but we need the original AST node
        # to do structural inference. We look it up from the enclosing
        # Program's bindings, which ApplyNode.infer_type passes through
        # the env via a special key.
        original_node = env.get(f"__ast__{fn_node.name}")
        if original_node is not None:
            return _infer_apply_type(original_node, arg_types, env, inputs)
        # Fallback: check if the type is a FnType
        fn_ty = fn_node.infer_type(env, inputs)
        if isinstance(fn_ty.element_type, FnType):
            raise TypeError(
                f"Cannot infer Apply result type for let-bound function {fn_node.name!r} "
                f"(AST node not found in env for structural inference)"
            )

    raise TypeError(f"Cannot infer Apply result type for fn node: {type(fn_node).__name__}")


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
            # Store the AST node so that ApplyNode can do structural
            # type inference on let-bound function values.
            env[f"__ast__{name}"] = node
            result[name] = ty
        return result

    def output_type(self, inputs: InputDecls) -> Type:
        types = self.infer_types(inputs)
        if self.output not in types:
            raise TypeError(f"Output {self.output!r} not found in bindings")
        return types[self.output]
