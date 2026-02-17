# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Operations and adverbs -- type rules + torch execution.

Unlike layouts, operations do computational work: they allocate new storage
and produce new Values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .layouts import flip as flip_layout
from .layouts import indexed as indexed_layout
from .layouts import struct_layout, StructElement
from .types import (
    Dynamic,
    ElementType,
    Extent,
    FnType,
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

    For flat tensors: data is a torch tensor whose shape matches the
    iteration shape (all static extents resolved).

    For nested types (element is a Type): data is a torch tensor if the
    nesting is regular (all inner shapes are the same static size), or a
    list[Value] if the nesting involves jagged/dynamic inner shapes.
    """

    type: Type
    data: Any  # torch.Tensor | list[Value]

    @staticmethod
    def from_tensor(arr: torch.Tensor, stype: ScalarType) -> Value:
        """Wrap a torch tensor as a Value with a fully-static tensor type."""
        shape = Shape(*[Static(d) for d in arr.shape])
        ty = Type(shape, stype)
        return Value(ty, arr)

    # Backward-compatible alias
    from_numpy = from_tensor

    def __repr__(self) -> str:
        if isinstance(self.data, torch.Tensor):
            return f"Value({self.type}, shape={tuple(self.data.shape)})"
        elif isinstance(self.data, list):
            return f"Value({self.type}, [{len(self.data)} elements])"
        return f"Value({self.type}, {type(self.data).__name__})"


# ---------------------------------------------------------------------------
# Scalar function helpers
# ---------------------------------------------------------------------------

# Map from ScalarType to torch dtype
_STYPE_TO_DTYPE = {
    ScalarType.F32: torch.float32,
    ScalarType.F16: torch.float16,
    ScalarType.I32: torch.int32,
    ScalarType.I64: torch.int64,
    ScalarType.BOOL: torch.bool,
}

_DTYPE_TO_STYPE = {v: k for k, v in _STYPE_TO_DTYPE.items()}


def _torch_dtype_to_stype(dtype: torch.dtype) -> ScalarType:
    if dtype in _DTYPE_TO_STYPE:
        return _DTYPE_TO_STYPE[dtype]
    raise TypeError(f"No ScalarType for torch dtype {dtype}")


# Backward-compatible alias
_numpy_dtype_to_stype = _torch_dtype_to_stype


# ---------------------------------------------------------------------------
# FnValue -- first-class function values
# ---------------------------------------------------------------------------

@dataclass
class FnValue:
    """Runtime representation of a function (verb or composed adverb).

    A FnValue carries:
      - arity: 1 (monadic) or 2 (dyadic)
      - apply_fn: callable that takes Value args and returns a Value
      - type_fn: callable that takes Type args and returns a Type
                 (for type inference without data)
      - name: for debugging and display
    """

    arity: int
    apply_fn: Callable  # (Value, ...) -> Value
    type_fn: Callable  # (Type, ...) -> Type
    name: str = ""

    def __repr__(self) -> str:
        return f"FnValue({self.name}/{self.arity})"


def _verb_apply_add(a: Value, b: Value) -> Value:
    return Value(a.type, (a.data + b.data).to(a.data.dtype))

def _verb_apply_sub(a: Value, b: Value) -> Value:
    return Value(a.type, (a.data - b.data).to(a.data.dtype))

def _verb_apply_mul(a: Value, b: Value) -> Value:
    return Value(a.type, (a.data * b.data).to(a.data.dtype))

def _verb_apply_div(a: Value, b: Value) -> Value:
    b_data = b.data if isinstance(b.data, torch.Tensor) else float(b.data)
    result = (a.data / b_data).to(torch.float32)
    if a.type.rank == 0:
        return Value(Type(Shape(), ScalarType.F32), result)
    return Value(Type(a.type.iteration_shape, ScalarType.F32), result)

def _verb_apply_min(a: Value, b: Value) -> Value:
    return Value(a.type, torch.minimum(a.data, b.data))

def _verb_apply_max(a: Value, b: Value) -> Value:
    return Value(a.type, torch.maximum(a.data, b.data))

def _verb_apply_and(a: Value, b: Value) -> Value:
    return Value(a.type, a.data & b.data)

def _verb_apply_or(a: Value, b: Value) -> Value:
    return Value(a.type, a.data | b.data)


def _verb_type_preserving(ta: Type, tb: Type) -> Type:
    """Type rule for verbs that preserve the left argument's type (Add, Sub, Mul, etc.)."""
    return ta

def _verb_type_div(ta: Type, tb: Type) -> Type:
    """Type rule for Div: produces f32."""
    if ta.rank == 0:
        return Type(Shape(), ScalarType.F32)
    return Type(ta.iteration_shape, ScalarType.F32)

def _verb_type_bool(ta: Type, tb: Type) -> Type:
    """Type rule for comparison verbs: preserves shape, produces bool."""
    return Type(ta.iteration_shape, ScalarType.BOOL) if ta.rank > 0 else Type(Shape(), ScalarType.BOOL)


VERBS: dict[str, FnValue] = {
    "Add": FnValue(2, _verb_apply_add, _verb_type_preserving, "Add"),
    "Sub": FnValue(2, _verb_apply_sub, _verb_type_preserving, "Sub"),
    "Mul": FnValue(2, _verb_apply_mul, _verb_type_preserving, "Mul"),
    "Div": FnValue(2, _verb_apply_div, _verb_type_div, "Div"),
    "Min": FnValue(2, _verb_apply_min, _verb_type_preserving, "Min"),
    "Max": FnValue(2, _verb_apply_max, _verb_type_preserving, "Max"),
    "And": FnValue(2, _verb_apply_and, _verb_type_preserving, "And"),
    "Or":  FnValue(2, _verb_apply_or,  _verb_type_preserving, "Or"),
}


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
    result_stype = _torch_dtype_to_stype(result_data.dtype)
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
    coords = torch.nonzero(val.data).to(torch.int32)
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

    if isinstance(indexer.data, torch.Tensor):
        if isinstance(idx_elem, ScalarType):
            result_data = target_data[indexer.data.long()]
        elif isinstance(idx_elem, Type):
            coords = indexer.data
            if coords.ndim == 2:
                idx_tuple = tuple(coords[:, i].long() for i in range(coords.shape[1]))
                result_data = target_data[idx_tuple]
            else:
                result_data = target_data[tuple(coords.long())]
        else:
            raise TypeError(f"Unexpected indexer element type: {idx_elem!r}")
    elif isinstance(indexer.data, list):
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


def each_typecheck(input_type: Type, fn_result_type: ElementType) -> Type:
    """Each: apply function per element, preserving outer iteration.

    If fn returns an iterable (Type), the result nests. Dynamic extents in
    the result are promoted to Jagged (the body runs independently per
    element, so inner sizes may vary).
    """
    if isinstance(fn_result_type, Type):
        fn_result_type = _promote_dynamic_to_jagged(fn_result_type)
    return Type(input_type.iteration_shape, fn_result_type)


def Each(val: Value, fn: Callable[[Value], Value]) -> Value:
    """Apply fn to each element of the outer iteration space.

    Elements are extracted along the leading axis. fn receives a Value
    and must return a Value. Results are collected; if they have varying
    shapes, the result is jagged.
    """
    if isinstance(val.data, torch.Tensor):
        n = val.data.shape[0]
        inner_elem = val.type.element_type

        results = []
        for i in range(n):
            slice_data = val.data[i]
            if isinstance(inner_elem, Type):
                elem_val = Value(inner_elem, slice_data)
            else:
                elem_val = Value(Type(Shape(), inner_elem), slice_data)
            results.append(fn(elem_val))

    elif isinstance(val.data, list):
        results = [fn(elem) for elem in val.data]
    else:
        raise TypeError(f"Cannot iterate over {type(val.data)}")

    if not results:
        raise TypeError("Each over empty iterable: cannot infer result type")

    first_type = results[0].type
    inner_type = _promote_dynamic_to_jagged(first_type) if isinstance(first_type, Type) else first_type

    all_same_data_shape = all(
        isinstance(r.data, torch.Tensor) and r.data.shape == results[0].data.shape
        for r in results[1:]
    ) if isinstance(results[0].data, torch.Tensor) else False

    outer_extent = val.type.iteration_shape.extents[0] if val.type.rank > 0 else Static(len(results))

    if all_same_data_shape and isinstance(results[0].data, torch.Tensor):
        stacked = torch.stack([r.data for r in results])
        result_type = Type(Shape(outer_extent), inner_type)
        return Value(result_type, stacked)
    else:
        result_type = Type(Shape(outer_extent), inner_type)
        return Value(result_type, results)


# ---------------------------------------------------------------------------
# FlipStruct -- data-level struct-of-arrays -> array-of-structs
# ---------------------------------------------------------------------------

@dataclass
class StructValue:
    """A single struct element: named fields, each a scalar or small tensor."""

    fields: dict[str, Any]  # name -> torch tensor (scalar or array)

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

    sty = struct_layout(**{k: v.type for k, v in fields.items()})
    flipped_type = flip_layout(sty)

    first = values[0]
    if isinstance(first.data, torch.Tensor):
        n = first.data.shape[0]
    elif isinstance(first.data, list):
        n = len(first.data)
    else:
        raise TypeError(f"Cannot determine length of {type(first.data)}")

    for name, val in fields.items():
        if isinstance(val.data, torch.Tensor):
            length = val.data.shape[0]
        elif isinstance(val.data, list):
            length = len(val.data)
        else:
            raise TypeError(f"Cannot determine length of field '{name}'")
        if length != n:
            raise TypeError(
                f"Field '{name}' has length {length}, expected {n}"
            )

    elements = []
    for i in range(n):
        elem_fields = {}
        for name, val in fields.items():
            if isinstance(val.data, torch.Tensor):
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
    data = coord.data

    fields = {}
    shift = 0
    for i, bw in enumerate(bit_widths):
        mask = (1 << bw) - 1
        fields[f"level_{i}"] = ((data >> shift) & mask).to(torch.int32)
        shift += bw
    fields["which_top"] = (data >> shift).to(torch.int32)

    field_types = {}
    for name in fields:
        field_types[name] = coord.type

    sty = struct_layout(**field_types)
    flipped = flip_layout(sty)

    return Value(flipped, StructValue(fields))


# ---------------------------------------------------------------------------
# Morton curve primitives
# ---------------------------------------------------------------------------

def _part1by2(x: torch.Tensor) -> torch.Tensor:
    """Spread bits of x so that there are two zero bits between each."""
    x = x.to(torch.int64)
    x = x & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8))  & 0x100F00F00F00F00F
    x = (x | (x << 4))  & 0x10C30C30C30C30C3
    x = (x | (x << 2))  & 0x1249249249249249
    return x


def _compact1by2(x: torch.Tensor) -> torch.Tensor:
    """Inverse of _part1by2: extract every third bit."""
    x = x.to(torch.int64)
    x = x & 0x1249249249249249
    x = (x | (x >> 2))  & 0x10C30C30C30C30C3
    x = (x | (x >> 4))  & 0x100F00F00F00F00F
    x = (x | (x >> 8))  & 0x1F0000FF0000FF
    x = (x | (x >> 16)) & 0x1F00000000FFFF
    x = (x | (x >> 32)) & 0x1FFFFF
    return x.to(torch.int32)


_MORTON_OFFSET = 1 << 20  # 1,048,576 -- shift signed coords to non-negative


def morton3d(coord: torch.Tensor) -> torch.Tensor:
    """Encode (3,) i32 or (N, 3) i32 coordinates to morton indices.

    Returns i32 scalar or (N,) i32 tensor.

    .. deprecated:: Use :func:`morton3d_signed` for new code.
    """
    if coord.ndim == 1:
        return (_part1by2(coord[0]) | (_part1by2(coord[1]) << 1) | (_part1by2(coord[2]) << 2)).to(torch.int32)
    else:
        return (_part1by2(coord[:, 0]) | (_part1by2(coord[:, 1]) << 1) | (_part1by2(coord[:, 2]) << 2)).to(torch.int32)


def morton3d_signed(coord: torch.Tensor) -> torch.Tensor:
    """Encode signed (3,) i32 or (N, 3) i32 coordinates to morton codes.

    Offsets each axis by 2^20 so that the range [-2^20, 2^20) maps to
    [0, 2^21).  21 bits per axis x 3 axes = 63 bits, fits in i64.

    Returns i64 scalar or (N,) i64 tensor.
    """
    c = coord.to(torch.int64) + _MORTON_OFFSET
    if c.ndim == 1:
        return (_part1by2(c[0]) | (_part1by2(c[1]) << 1) | (_part1by2(c[2]) << 2)).to(torch.int64)
    else:
        return (_part1by2(c[:, 0]) | (_part1by2(c[:, 1]) << 1) | (_part1by2(c[:, 2]) << 2)).to(torch.int64)


def morton3d_decode(codes: torch.Tensor) -> torch.Tensor:
    """Decode morton codes back to signed (3,) i32 or (N, 3) i32 coordinates.

    Inverse of :func:`morton3d_signed`.  Extracts per-axis bits via
    ``_compact1by2`` and subtracts the 2^20 offset to restore sign.
    """
    codes_i64 = codes.to(torch.int64)
    x = _compact1by2(codes_i64) - _MORTON_OFFSET
    y = _compact1by2(codes_i64 >> 1) - _MORTON_OFFSET
    z = _compact1by2(codes_i64 >> 2) - _MORTON_OFFSET
    if codes.ndim == 0:
        return torch.stack([x, y, z]).to(torch.int32)
    return torch.stack([x, y, z], dim=-1).to(torch.int32)


# ---------------------------------------------------------------------------
# Hierarchical key -- CIG-compatible voxel ordering
# ---------------------------------------------------------------------------


def hierarchical_key(coord: torch.Tensor, bit_widths: list[int]) -> torch.Tensor:
    """Compute a CIG-compatible hierarchical sort key for 3D coordinates.

    Unlike morton encoding (which interleaves all coordinate bits globally),
    the hierarchical key concatenates per-level row-major linear indices from
    outermost (MSB) to innermost (LSB).  This matches the CIG builder's
    tree-traversal order: iterate each node's children in row-major order,
    recurse.

    Each level uses ``3 * bit_width`` bits for its linear index
    (``x * dim^2 + y * dim + z`` where ``dim = 2^bit_width``).

    Key layout (LSB to MSB)::

        [level_0 bits] [level_1 bits] ... [level_N-1 bits] [root bits]

    For bit_widths=[3,4,5]: level_0 uses 9 bits, level_1 uses 12, level_2
    uses 15, root gets the rest.  Total non-root: 36 bits.

    Args:
        coord: (3,) i32 or (N, 3) i32 -- voxel coordinates.
        bit_widths: leaf-first list, e.g. [3, 4, 5] for 8^3 / 16^3 / 32^3.

    Returns:
        i64 scalar or (N,) i64 tensor -- hierarchical sort key.
    """
    c = coord.to(torch.int64)
    single = c.ndim == 1
    if single:
        c = c.reshape(1, 3)

    key = torch.zeros(c.shape[0], dtype=torch.int64, device=c.device)
    coord_shift = 0
    key_shift = 0

    for bw in bit_widths:
        dim = 1 << bw
        mask = dim - 1
        lx = (c[:, 0] >> coord_shift) & mask
        ly = (c[:, 1] >> coord_shift) & mask
        lz = (c[:, 2] >> coord_shift) & mask
        linear = (lx * dim * dim + ly * dim + lz).to(torch.int64)
        key |= linear << key_shift
        coord_shift += bw
        key_shift += 3 * bw

    rx = c[:, 0] >> coord_shift
    ry = c[:, 1] >> coord_shift
    rz = c[:, 2] >> coord_shift
    root_linear = (rx * (1 << 20) + ry * (1 << 10) + rz).to(torch.int64)
    key |= root_linear << key_shift

    return key[0] if single else key


def hierarchical_key_decode(key: torch.Tensor, bit_widths: list[int]) -> torch.Tensor:
    """Decode a hierarchical sort key back to 3D coordinates.

    Inverse of :func:`hierarchical_key`.

    Args:
        key: i64 scalar or (N,) i64 tensor -- hierarchical sort keys.
        bit_widths: same list used for encoding, e.g. [3, 4, 5].

    Returns:
        (3,) i32 or (N, 3) i32 -- reconstructed coordinates.
    """
    k = torch.atleast_1d(key).to(torch.int64)
    single = key.ndim == 0

    coord = torch.zeros((k.shape[0], 3), dtype=torch.int64, device=k.device)
    key_shift = 0
    coord_shift = 0

    for bw in bit_widths:
        dim = 1 << bw
        n_bits = 3 * bw
        level_mask = (1 << n_bits) - 1
        linear = (k >> key_shift) & level_mask

        lz = linear % dim
        ly = (linear // dim) % dim
        lx = linear // (dim * dim)

        coord[:, 0] |= lx << coord_shift
        coord[:, 1] |= ly << coord_shift
        coord[:, 2] |= lz << coord_shift

        key_shift += n_bits
        coord_shift += bw

    root_linear = k >> key_shift
    rz = root_linear & ((1 << 10) - 1)
    ry = (root_linear >> 10) & ((1 << 10) - 1)
    rx = root_linear >> 20

    coord[:, 0] |= rx << coord_shift
    coord[:, 1] |= ry << coord_shift
    coord[:, 2] |= rz << coord_shift

    result = coord.to(torch.int32)
    if single:
        return result[0]
    return result
