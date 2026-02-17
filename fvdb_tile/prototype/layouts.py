# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Layout wrappers -- pure type-level transformations, no data movement.
"""

from __future__ import annotations

import functools
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from .types import (
    Dynamic,
    ElementType,
    Extent,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# ---------------------------------------------------------------------------
# Cut variants
# ---------------------------------------------------------------------------

def cut_by_size(size: int, ty: Type) -> Type:
    """Split the leading axis into chunks of static *size*.

    Outer extent becomes Dynamic (how-many-chunks depends on data).
    Inner extent becomes Static(size).
    Requires leading axis to be evenly divisible if static.
    """
    if ty.rank == 0:
        raise TypeError("Cannot cut a rank-0 type")

    leading = ty.iteration_shape.extents[0]
    remaining = ty.iteration_shape.extents[1:]

    if isinstance(leading, Static):
        if leading.n % size != 0:
            raise TypeError(
                f"Leading static extent {leading.n} not divisible by {size}"
            )
        outer = Static(leading.n // size)
    else:
        outer = Dynamic()

    inner_shape = Shape(Static(size), *remaining)
    inner_type = Type(inner_shape, ty.element_type)
    return Type(Shape(outer), inner_type)


def cut_by_count(count: int, ty: Type) -> Type:
    """Split the leading axis into exactly *count* equal pieces.

    Outer extent becomes Static(count).
    Inner extent becomes Static(D/count) if leading is static.
    """
    if ty.rank == 0:
        raise TypeError("Cannot cut a rank-0 type")

    leading = ty.iteration_shape.extents[0]
    remaining = ty.iteration_shape.extents[1:]

    if isinstance(leading, Static):
        if leading.n % count != 0:
            raise TypeError(
                f"Leading static extent {leading.n} not divisible by {count}"
            )
        inner_extent = Static(leading.n // count)
    else:
        inner_extent = Dynamic()

    inner_shape = Shape(inner_extent, *remaining)
    inner_type = Type(inner_shape, ty.element_type)
    return Type(Shape(Static(count)), inner_type)


def cut_by_offsets(ty: Type) -> Type:
    """Split the leading axis by an offsets array (jagged cut).

    Outer extent becomes Dynamic (segment count).
    Inner extent becomes Jagged (varies per element).
    """
    if ty.rank == 0:
        raise TypeError("Cannot cut a rank-0 type")

    remaining = ty.iteration_shape.extents[1:]
    inner_shape = Shape(Jagged(), *remaining)
    inner_type = Type(inner_shape, ty.element_type)
    return Type(Shape(Dynamic()), inner_type)


# Alias
jagged = cut_by_offsets


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------

def reshape(ty: Type, new_extents: tuple[int | str, ...]) -> Type:
    """Reinterpret the iteration shape. Product must match if both static."""

    def _parse(e) -> Extent:
        if isinstance(e, int):
            if e < 0:
                return Dynamic()
            return Static(e)
        if e == "*":
            return Dynamic()
        if e == "~":
            return Jagged()
        raise ValueError(f"Unknown extent: {e!r}")

    new_shape = Shape(*[_parse(e) for e in new_extents])

    old_size = ty.iteration_shape.static_size()
    new_size = new_shape.static_size()
    if old_size is not None and new_size is not None and old_size != new_size:
        raise TypeError(
            f"Reshape product mismatch: {old_size} vs {new_size}"
        )

    return Type(new_shape, ty.element_type)


# ---------------------------------------------------------------------------
# Fuse -- merge the two outermost nesting levels
# ---------------------------------------------------------------------------

def fuse(ty: Type) -> Type:
    """Merge the two outermost nesting levels into one.

    fuse(S_1 / S_2 / E) = (S_1 ++ S_2) / E

    This is the inverse of cut. No data movement -- the physical storage
    is already shaped as the concatenation of the two shapes.

    Constraint: the inner leading shape must not contain Jagged extents
    (jagged means "varies per parent" and cannot be fused into a uniform axis).
    Dynamic extents are fine (uniform, just unknown at compile time).
    """
    if not isinstance(ty.element_type, Type):
        raise TypeError("fuse requires nested type (S1 / S2 / E), got scalar element")
    inner = ty.element_type
    for e in inner.iteration_shape.extents:
        if isinstance(e, Jagged):
            raise TypeError(
                f"fuse: inner leading shape contains Jagged extent -- "
                f"cannot fuse jagged nesting into a uniform shape"
            )
    merged = Shape(*ty.iteration_shape.extents, *inner.iteration_shape.extents)
    return Type(merged, inner.element_type)


def flatten(ty: Type) -> Type:
    """Recursively fuse ALL nesting levels into a single leading shape.

    flatten(S_1 / S_2 / ... / S_n / scalar) = (S_1 ++ S_2 ++ ... ++ S_n) / scalar

    Equivalent to repeated fuse until the element type is no longer a Type.
    Same jagged constraint as fuse at each level.
    """
    result = ty
    while isinstance(result.element_type, Type):
        result = fuse(result)
    return result


# ---------------------------------------------------------------------------
# Permute -- reorder axes within the leading shape
# ---------------------------------------------------------------------------

def permute(ty: Type, order: tuple[int, ...]) -> Type:
    """Reorder axes within the leading shape.

    permute((a, b, c) / E, (2, 0, 1)) = (c, a, b) / E

    Does NOT cross nesting boundaries. To permute across levels,
    fuse first, then permute, then cut to re-establish the boundary.
    """
    if len(order) != ty.rank:
        raise TypeError(
            f"permute order length {len(order)} does not match rank {ty.rank}"
        )
    if sorted(order) != list(range(ty.rank)):
        raise TypeError(
            f"permute order must be a permutation of 0..{ty.rank - 1}, got {order}"
        )
    new_extents = tuple(ty.iteration_shape.extents[i] for i in order)
    return Type(Shape(*new_extents), ty.element_type)


# ---------------------------------------------------------------------------
# Indexed
# ---------------------------------------------------------------------------

def indexed(indexer_ty: Type, target_ty: Type) -> Type:
    """Type of an indexed layout: indexer provides coordinates into target.

    Constraint: indexer's element must be a coordinate matching target's
    iteration rank. A scalar i32 element indexes rank-1 targets.
    An (r,) i32 element indexes rank-r targets.
    """
    target_rank = target_ty.rank

    # Determine the indexer's coordinate rank
    idx_elem = indexer_ty.element_type
    if isinstance(idx_elem, ScalarType):
        # Scalar integer -> rank-1 index
        if idx_elem not in (ScalarType.I32, ScalarType.I64):
            raise TypeError(f"Indexer element must be integer, got {idx_elem}")
        coord_rank = 1
    elif isinstance(idx_elem, Type):
        # (r,) i32 -> rank-r index
        if not idx_elem.is_scalar_element:
            raise TypeError("Indexer element must be a flat integer vector")
        if idx_elem.element_type not in (ScalarType.I32, ScalarType.I64):
            raise TypeError(
                f"Indexer element must be integer, got {idx_elem.element_type}"
            )
        if idx_elem.rank != 1:
            raise TypeError(
                f"Indexer element must be rank-1, got rank {idx_elem.rank}"
            )
        lead = idx_elem.iteration_shape.extents[0]
        if isinstance(lead, Static):
            coord_rank = lead.n
        else:
            coord_rank = None  # dynamic, can't check at type level

    if coord_rank is not None and coord_rank != target_rank:
        raise TypeError(
            f"Indexer coordinate rank {coord_rank} does not match "
            f"target iteration rank {target_rank}"
        )

    return Type(indexer_ty.iteration_shape, target_ty.element_type)


# ---------------------------------------------------------------------------
# Tuple / Struct
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TupleElement:
    """Element type for a generic (heterogeneous) tuple."""
    children: tuple[ElementType, ...]

    def __repr__(self) -> str:
        inner = ", ".join(repr(c) for c in self.children)
        return f"Tuple({inner})"


@dataclass(frozen=True)
class StructElement:
    """Element type for a named struct."""
    fields: tuple[tuple[str, ElementType], ...]

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}: {v!r}" for k, v in self.fields)
        return f"Struct({{{inner}}})"


@dataclass(frozen=True)
class MaskedElement:
    """Element type for a masked (sparse-occupancy) layout.

    Represents a fixed-shape iteration space where only some positions have
    data, identified by a bitmask. The dense index of an active position is
    computed via popcount of the mask bits below that position.

    Physical storage: a packed bitmask + a base offset into a flat data array.
    This is the sparse-occupancy counterpart of jagged (variable-length).
    """

    mask_shape: Shape  # shape of the iteration space (e.g., (8,8,8))
    element_type: "ElementType"  # type of each active element

    def __repr__(self) -> str:
        return f"Masked({self.mask_shape}, {self.element_type!r})"


def masked_layout(mask_shape: Shape, elem_type: "ElementType") -> Type:
    """Masked layout: fixed-shape space with sparse occupancy.

    The mask_shape defines the logical iteration space (e.g., (8,8,8) for a
    leaf block). Only positions where the bitmask is set have data. Access
    requires a bitmask check + popcount to compute the dense index.

    This is a layout (lowercase, free): it describes how to interpret a
    bitmask + flat data array as a sparse block. No data is moved.

    The resulting Type has rank 0 (the masked layout wraps the full space
    into a single element that can be Gathered into with a coordinate).
    """
    return Type(Shape(), MaskedElement(mask_shape, elem_type))


def tuple_layout(*children: Type) -> Type:
    """Generic tuple: rank-1, length = number of children, heterogeneous."""
    elem = TupleElement(tuple(c for c in children))
    return Type(Shape(Static(len(children))), elem)


def struct_layout(**fields: Type) -> Type:
    """Named struct: rank-1, length = number of fields, named heterogeneous."""
    elem = StructElement(tuple((k, v) for k, v in fields.items()))
    return Type(Shape(Static(len(fields))), elem)


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------

def flip(ty: Type) -> Type:
    """Transpose tuple/struct of arrays into array of tuples/structs.

    Precondition: all children must have compatible iteration spaces.
    """
    elem = ty.element_type

    if isinstance(elem, TupleElement):
        children_types = elem.children
    elif isinstance(elem, StructElement):
        children_types = [et for _, et in elem.fields]
    else:
        raise TypeError(f"flip requires a tuple or struct element type, got {elem!r}")

    # All children must be Type (not bare ScalarType)
    for i, child in enumerate(children_types):
        if not isinstance(child, Type):
            raise TypeError(
                f"flip: child {i} is a scalar type {child!r}, not an iterable"
            )

    # Resolve compatible iteration shapes
    resolved = children_types[0].iteration_shape
    for child in children_types[1:]:
        resolved = resolved.resolve(child.iteration_shape)

    # Build the flipped element type
    if isinstance(elem, TupleElement):
        flipped_elem = TupleElement(tuple(c.element_type for c in children_types))
    else:
        flipped_elem = StructElement(
            tuple((k, c.element_type) for (k, _), c in zip(elem.fields, children_types))
        )

    return Type(resolved, flipped_elem)
