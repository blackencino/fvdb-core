# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Operations and adverbs -- type rules + numpy execution.

Unlike layouts, operations do computational work: they allocate new storage
and produce new Values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .layouts import flip as flip_layout
from .layouts import indexed as indexed_layout
from .layouts import struct_layout, StructElement
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
    tensor_type,
)


# ---------------------------------------------------------------------------
# Value -- concrete data with an attached logical type
# ---------------------------------------------------------------------------

@dataclass
class Value:
    """A piece of concrete data carrying its logical type.

    For flat tensors: data is a numpy array whose shape matches the
    iteration shape (all static extents resolved).

    For nested types (element is a Type): data is a numpy array if the
    nesting is regular (all inner shapes are the same static size), or a
    list[Value] if the nesting involves jagged/dynamic inner shapes.
    """

    type: Type
    data: Any  # np.ndarray | list[Value]

    @staticmethod
    def from_numpy(arr: np.ndarray, stype: ScalarType) -> Value:
        """Wrap a numpy array as a Value with a fully-static tensor type."""
        shape = Shape(*[Static(d) for d in arr.shape])
        ty = Type(shape, stype)
        return Value(ty, arr)

    def __repr__(self) -> str:
        if isinstance(self.data, np.ndarray):
            return f"Value({self.type}, shape={self.data.shape})"
        elif isinstance(self.data, list):
            return f"Value({self.type}, [{len(self.data)} elements])"
        return f"Value({self.type}, {type(self.data).__name__})"


# ---------------------------------------------------------------------------
# Scalar function helpers
# ---------------------------------------------------------------------------

# Map from ScalarType to numpy dtype
_STYPE_TO_DTYPE = {
    ScalarType.F32: np.float32,
    ScalarType.F16: np.float16,
    ScalarType.I32: np.int32,
    ScalarType.I64: np.int64,
    ScalarType.BOOL: np.bool_,
}

_DTYPE_TO_STYPE = {v: k for k, v in _STYPE_TO_DTYPE.items()}


def _numpy_dtype_to_stype(dtype: np.dtype) -> ScalarType:
    dtype = np.dtype(dtype)
    for np_dt, st in _DTYPE_TO_STYPE.items():
        if dtype == np.dtype(np_dt):
            return st
    raise TypeError(f"No ScalarType for numpy dtype {dtype}")


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def map_typecheck(input_type: Type, result_stype: ScalarType) -> Type:
    """Map preserves iteration shape, transforms element type."""
    if not input_type.is_scalar_element:
        raise TypeError(
            f"Map requires scalar element type, got {input_type.element_type!r}"
        )
    return Type(input_type.iteration_shape, result_stype)


def Map(val: Value, fn: Callable) -> Value:
    """Apply a scalar function elementwise over the full iteration space."""
    result_data = fn(val.data)
    result_stype = _numpy_dtype_to_stype(result_data.dtype)
    result_type = map_typecheck(val.type, result_stype)
    return Value(result_type, result_data)


# ---------------------------------------------------------------------------
# Where
# ---------------------------------------------------------------------------

def where_typecheck(input_type: Type) -> Type:
    """Where: (S) over bool -> (*) over (rank,) i32."""
    if not input_type.is_scalar_element:
        raise TypeError(f"Where requires scalar element, got {input_type.element_type!r}")
    if input_type.element_type != ScalarType.BOOL:
        raise TypeError(f"Where requires bool element, got {input_type.element_type!r}")
    rank = input_type.rank
    return Type(Shape(Dynamic()), coord_type(rank))


def Where(val: Value) -> Value:
    """Coordinates of truthy elements."""
    result_type = where_typecheck(val.type)
    coords = np.argwhere(val.data).astype(np.int32)
    # coords has shape (N, rank)
    return Value(result_type, coords)


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------

def gather_typecheck(target_type: Type, indexer_type: Type) -> Type:
    """Gather: materialise an Indexed layout."""
    return indexed_layout(indexer_type, target_type)


def Gather(target: Value, indexer: Value) -> Value:
    """Materialise: look up target at coordinates given by indexer."""
    result_type = gather_typecheck(target.type, indexer.type)

    target_data = target.data
    idx_elem = indexer.type.element_type

    if isinstance(indexer.data, np.ndarray):
        if isinstance(idx_elem, ScalarType):
            # Scalar index into rank-1 target
            result_data = target_data[indexer.data]
        elif isinstance(idx_elem, Type):
            # (r,) i32 coordinate vectors
            coords = indexer.data  # shape (N, r)
            if coords.ndim == 2:
                idx_tuple = tuple(coords[:, i] for i in range(coords.shape[1]))
                result_data = target_data[idx_tuple]
            else:
                # Single coordinate
                result_data = target_data[tuple(coords)]
        else:
            raise TypeError(f"Unexpected indexer element type: {idx_elem!r}")
    elif isinstance(indexer.data, list):
        # List of Value -- gather element-by-element
        results = []
        for idx_val in indexer.data:
            results.append(Gather(target, idx_val))
        result_data = results
        return Value(result_type, result_data)
    else:
        raise TypeError(f"Unexpected indexer data type: {type(indexer.data)}")

    return Value(result_type, result_data)


# ---------------------------------------------------------------------------
# Each
# ---------------------------------------------------------------------------

def each_typecheck(input_type: Type, fn_result_type: ElementType) -> Type:
    """Each: apply function per element, preserving outer iteration.

    If fn returns an iterable (Type), the result nests.
    If fn returns a scalar, the result is a new tensor type.
    """
    return Type(input_type.iteration_shape, fn_result_type)


def Each(val: Value, fn: Callable[[Value], Value]) -> Value:
    """Apply fn to each element of the outer iteration space.

    Elements are extracted along the leading axis. fn receives a Value
    and must return a Value. Results are collected; if they have varying
    shapes, the result is jagged.
    """
    if isinstance(val.data, np.ndarray):
        n = val.data.shape[0]
        # Determine inner element type from val.type
        inner_elem = val.type.element_type

        results = []
        for i in range(n):
            slice_data = val.data[i]
            if isinstance(inner_elem, Type):
                elem_val = Value(inner_elem, slice_data)
            else:
                # Scalar element -- wrap as a 0-d value
                elem_val = Value(Type(Shape(), inner_elem), slice_data)
            results.append(fn(elem_val))

    elif isinstance(val.data, list):
        results = [fn(elem) for elem in val.data]
    else:
        raise TypeError(f"Cannot iterate over {type(val.data)}")

    if not results:
        # Empty -- infer type from input
        raise TypeError("Each over empty iterable: cannot infer result type")

    # Determine result type from first element
    first_type = results[0].type

    # Check if all results have the same shape -- both at the type level
    # and at the data level (a Dynamic extent may resolve to different
    # concrete sizes across elements).
    all_same_type = all(r.type.iteration_shape == first_type.iteration_shape for r in results[1:])
    all_same_data_shape = all(
        isinstance(r.data, np.ndarray) and r.data.shape == results[0].data.shape
        for r in results[1:]
    ) if isinstance(results[0].data, np.ndarray) else False

    all_same = all_same_type and all_same_data_shape

    # Build the outer iteration shape
    outer_extent = val.type.iteration_shape.extents[0] if val.type.rank > 0 else Static(len(results))

    if all_same and isinstance(results[0].data, np.ndarray):
        # Regular: stack into a single array
        stacked = np.stack([r.data for r in results])
        result_type = Type(Shape(outer_extent), first_type)
        return Value(result_type, stacked)
    else:
        # Jagged or heterogeneous: keep as list of Values.
        # Mark the inner leading extent as jagged since sizes vary.
        inner_type = first_type
        if inner_type.rank > 0:
            jagged_shape = Shape(Jagged(), *inner_type.iteration_shape.extents[1:])
            inner_type = Type(jagged_shape, inner_type.element_type)

        result_type = Type(Shape(outer_extent), inner_type)
        return Value(result_type, results)


# ---------------------------------------------------------------------------
# FlipStruct -- data-level struct-of-arrays -> array-of-structs
# ---------------------------------------------------------------------------

@dataclass
class StructValue:
    """A single struct element: named fields, each a scalar or small array."""

    fields: dict[str, Any]  # name -> numpy scalar or array

    def __getattr__(self, name: str) -> Any:
        if name == "fields":
            return object.__getattribute__(self, "fields")
        flds = object.__getattribute__(self, "fields")
        if name in flds:
            return flds[name]
        raise AttributeError(f"StructValue has no field '{name}'")

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.fields.items())
        return f"StructValue({inner})"


def FlipStruct(**fields: Value) -> Value:
    """Build an array-of-structs Value from named arrays with compatible shapes.

    This is the data-level counterpart of the type-level flip(struct(...)):
    it takes separate named arrays (struct-of-arrays) and produces a single
    Value whose type is the flipped struct and whose data can be accessed
    per-element as structs.

    All field Values must have the same outer iteration length.
    """
    if not fields:
        raise TypeError("FlipStruct requires at least one field")

    names = list(fields.keys())
    values = list(fields.values())

    # Type-level: build struct then flip
    sty = struct_layout(**{k: v.type for k, v in fields.items()})
    flipped_type = flip_layout(sty)

    # Data-level: determine outer length and zip element-wise
    first = values[0]
    if isinstance(first.data, np.ndarray):
        n = first.data.shape[0]
    elif isinstance(first.data, list):
        n = len(first.data)
    else:
        raise TypeError(f"Cannot determine length of {type(first.data)}")

    # Verify all fields have the same outer length
    for name, val in fields.items():
        if isinstance(val.data, np.ndarray):
            length = val.data.shape[0]
        elif isinstance(val.data, list):
            length = len(val.data)
        else:
            raise TypeError(f"Cannot determine length of field '{name}'")
        if length != n:
            raise TypeError(
                f"Field '{name}' has length {length}, expected {n}"
            )

    # Build per-element structs
    elements = []
    for i in range(n):
        elem_fields = {}
        for name, val in fields.items():
            if isinstance(val.data, np.ndarray):
                elem_fields[name] = val.data[i]
            else:
                elem_fields[name] = val.data[i]
        elements.append(StructValue(elem_fields))

    return Value(flipped_type, elements)


# ---------------------------------------------------------------------------
# Decompose -- bitfield coordinate split for hierarchical grids
# ---------------------------------------------------------------------------

def Decompose(coord: Value, bit_widths: list[int]) -> Value:
    """Split a global coordinate into sub-coordinates via bitfield extraction.

    bit_widths is leaf-first: [leaf_bits, lower_bits, ...].
    For bit_widths = [3, 4] (leaf=8, lower=16):
      - leaf_local   = coord & 7           -- bottom 3 bits
      - lower_local  = (coord >> 3) & 15   -- next 4 bits
      - which_lower  = coord >> 7          -- remaining bits

    Input:  (3,) i32  (a single 3D coordinate)
    Output: StructValue with fields 'level_0', 'level_1', ..., 'which_top',
            each a (3,) i32 sub-coordinate.

    Also works batched: (*,r) i32 -> struct of (*,r) i32 arrays.
    """
    data = coord.data  # (3,) or (N, 3)

    fields = {}
    shift = 0
    for i, bw in enumerate(bit_widths):
        mask = (1 << bw) - 1
        fields[f"level_{i}"] = ((data >> shift) & mask).astype(np.int32)
        shift += bw
    fields["which_top"] = (data >> shift).astype(np.int32)

    # Build type: struct of sub-coord types matching input's element shape
    field_types = {}
    for name in fields:
        field_types[name] = coord.type

    sty = struct_layout(**field_types)
    flipped = flip_layout(sty)

    # Return as StructValue
    return Value(flipped, StructValue(fields))


# ---------------------------------------------------------------------------
# Morton curve primitives
# ---------------------------------------------------------------------------

def _part1by2(x: np.ndarray) -> np.ndarray:
    """Spread bits of x so that there are two zero bits between each."""
    x = x.astype(np.int64)
    x = x & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8))  & 0x100F00F00F00F00F
    x = (x | (x << 4))  & 0x10C30C30C30C30C3
    x = (x | (x << 2))  & 0x1249249249249249
    return x


def _compact1by2(x: np.ndarray) -> np.ndarray:
    """Inverse of _part1by2: extract every third bit."""
    x = x.astype(np.int64)
    x = x & 0x1249249249249249
    x = (x | (x >> 2))  & 0x10C30C30C30C30C3
    x = (x | (x >> 4))  & 0x100F00F00F00F00F
    x = (x | (x >> 8))  & 0x1F0000FF0000FF
    x = (x | (x >> 16)) & 0x1F00000000FFFF
    x = (x | (x >> 32)) & 0x1FFFFF
    return x.astype(np.int32)


def morton3d(coord: np.ndarray) -> np.ndarray:
    """Encode (3,) i32 or (N, 3) i32 coordinates to morton indices.

    Returns i32 scalar or (N,) i32 array.
    """
    if coord.ndim == 1:
        return (_part1by2(coord[0]) | (_part1by2(coord[1]) << 1) | (_part1by2(coord[2]) << 2)).astype(np.int32)
    else:
        return (_part1by2(coord[:, 0]) | (_part1by2(coord[:, 1]) << 1) | (_part1by2(coord[:, 2]) << 2)).astype(np.int32)


def inv_morton3d(code: np.ndarray) -> np.ndarray:
    """Decode morton index back to (3,) i32 or (N, 3) i32 coordinates."""
    code = code.astype(np.int64)
    if code.ndim == 0:
        x = _compact1by2(code)
        y = _compact1by2(code >> 1)
        z = _compact1by2(code >> 2)
        return np.array([x, y, z], dtype=np.int32)
    else:
        x = _compact1by2(code)
        y = _compact1by2(code >> 1)
        z = _compact1by2(code >> 2)
        return np.stack([x, y, z], axis=1).astype(np.int32)
