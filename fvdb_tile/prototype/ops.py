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

from .layouts import StructElement
from .layouts import flip as flip_layout
from .layouts import indexed as indexed_layout
from .layouts import struct_layout
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


def _verb_apply_xor(a: Value, b: Value) -> Value:
    return Value(a.type, a.data ^ b.data)


def _verb_apply_shift_left(a: Value, b: Value) -> Value:
    return Value(a.type, (a.data << b.data).to(a.data.dtype))


def _verb_apply_shift_right(a: Value, b: Value) -> Value:
    return Value(a.type, (a.data >> b.data).to(a.data.dtype))


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
    "Or": FnValue(2, _verb_apply_or, _verb_type_preserving, "Or"),
    "Xor": FnValue(2, _verb_apply_xor, _verb_type_preserving, "Xor"),
    "ShiftLeft": FnValue(2, _verb_apply_shift_left, _verb_type_preserving, "ShiftLeft"),
    "ShiftRight": FnValue(2, _verb_apply_shift_right, _verb_type_preserving, "ShiftRight"),
}


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


def map_typecheck(input_type: Type, result_stype: ScalarType) -> Type:
    """Map preserves iteration shape, transforms element type."""
    if not input_type.is_scalar_element:
        raise TypeError(f"Map requires scalar element type, got {input_type.element_type!r}")
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
    new_extents = tuple(Jagged() if isinstance(e, Dynamic) else e for e in ty.iteration_shape.extents)
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

    all_same_data_shape = (
        all(isinstance(r.data, torch.Tensor) and r.data.shape == results[0].data.shape for r in results[1:])
        if isinstance(results[0].data, torch.Tensor)
        else False
    )

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
            raise TypeError(f"Field '{name}' has length {length}, expected {n}")

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
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def _compact1by2(x: torch.Tensor) -> torch.Tensor:
    """Inverse of _part1by2: extract every third bit."""
    x = x.to(torch.int64)
    x = x & 0x1249249249249249
    x = (x | (x >> 2)) & 0x10C30C30C30C30C3
    x = (x | (x >> 4)) & 0x100F00F00F00F00F
    x = (x | (x >> 8)) & 0x1F0000FF0000FF
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


def _hkey_params(bit_widths: list[int]) -> tuple[int, int]:
    """Compute (offset, root_bits_per_axis) for signed hierarchical keys.

    The offset shifts all coordinates to non-negative before encoding.
    root_bits_per_axis is sized so the total key fits in 63 bits (positive
    i64), ensuring correct sort ordering.
    """
    tree_key_bits = sum(3 * bw for bw in bit_widths)
    remaining = 63 - tree_key_bits
    root_bits_per_axis = remaining // 3
    total_bits_per_axis = sum(bit_widths) + root_bits_per_axis
    offset = 1 << (total_bits_per_axis - 1)
    return offset, root_bits_per_axis


def hierarchical_key(coord: torch.Tensor, bit_widths: list[int]) -> torch.Tensor:
    """Compute a CIG-compatible hierarchical sort key for 3D coordinates.

    Unlike morton encoding (which interleaves all coordinate bits globally),
    the hierarchical key concatenates per-level row-major linear indices from
    outermost (MSB) to innermost (LSB).  This matches the CIG builder's
    tree-traversal order: iterate each node's children in row-major order,
    recurse.

    Handles signed coordinates by adding a fixed offset to make all values
    non-negative before encoding.  The offset is derived from the bit-widths
    to guarantee the total key fits in 63 bits (positive i64).

    Each level uses ``3 * bit_width`` bits for its linear index
    (``x * dim^2 + y * dim + z`` where ``dim = 2^bit_width``).

    Key layout (LSB to MSB)::

        [level_0 bits] [level_1 bits] ... [level_N-1 bits] [root bits]

    For bit_widths=[3,4,5]: level_0 uses 9 bits, level_1 uses 12, level_2
    uses 15, root uses 27 bits (9 per axis).  Total: 63 bits.

    Args:
        coord: (3,) i32 or (N, 3) i32 -- voxel coordinates (signed ok).
        bit_widths: leaf-first list, e.g. [3, 4, 5] for 8^3 / 16^3 / 32^3.

    Returns:
        i64 scalar or (N,) i64 tensor -- hierarchical sort key (non-negative).
    """
    offset, root_bits = _hkey_params(bit_widths)

    c = coord.to(torch.int64) + offset
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

    root_mask = (1 << root_bits) - 1
    rx = (c[:, 0] >> coord_shift) & root_mask
    ry = (c[:, 1] >> coord_shift) & root_mask
    rz = (c[:, 2] >> coord_shift) & root_mask
    root_linear = (rx * (1 << (2 * root_bits)) + ry * (1 << root_bits) + rz).to(torch.int64)
    key |= root_linear << key_shift

    return key[0] if single else key


# ---------------------------------------------------------------------------
# Coordinate primitives for topology operations
# ---------------------------------------------------------------------------


def expand_offsets(coords: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Broadcast-add coordinates with offsets and flatten.

    This is the leading-shape equivalent of::

        fuse(EachLeft(EachRight(EachBoth(Add)), coords, offsets))

    For each coordinate in ``coords`` and each offset in ``offsets``,
    produce ``coord + offset``, then flatten the (N, K, 3) result into
    (N*K, 3).

    Args:
        coords: (N, 3) i32 active coordinates.
        offsets: (K, 3) i32 kernel offsets.

    Returns:
        (N*K, 3) i32 expanded coordinates.
    """
    return (coords.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)


def stride_filter(coords: torch.Tensor, stride: tuple[int, int, int]) -> torch.Tensor:
    """Filter coordinates by stride divisibility and divide.

    Keeps only coordinates where all components are divisible by the
    corresponding stride component, then integer-divides.  For stride
    ``(1, 1, 1)`` this is a no-op (returns input unchanged).

    In DSL terms::

        divisible = Map(coords, c => All(Eq(Mod(c, stride), 0)))
        filtered  = Gather(coords, Where(divisible))
        result    = Map(filtered, c => FloorDiv(c, stride))

    Args:
        coords: (N, 3) i32 coordinates.
        stride: (sx, sy, sz) stride values (all positive).

    Returns:
        (M, 3) i32 filtered and divided coordinates.
    """
    if stride == (1, 1, 1):
        return coords
    dev = coords.device
    stride_t = torch.tensor(stride, dtype=torch.int32, device=dev)
    divisible = (coords % stride_t == 0).all(dim=1)
    result = coords[divisible]
    if result.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=dev)
    return result // stride_t


def dedup_coords(coords: torch.Tensor, bit_widths: list[int]) -> torch.Tensor:
    """Deduplicate (N, 3) i32 coords via hierarchical key sort + unique.

    Computes a hierarchical sort key for each coordinate, sorts, removes
    duplicates, and returns unique coordinates in CIG-compatible
    tree-traversal order.

    This is the key deduplication primitive for topology operations
    (conv_grid, dilated_grid, etc.).  It composes::

        keys    = HierarchicalKey(coords, bit_widths)
        sorted  = Sort(keys)
        unique  = Unique(sorted)
        result  = HierarchicalKeyDecode(unique, bit_widths)

    Args:
        coords: (N, 3) i32 coordinates (may contain duplicates).
        bit_widths: CIG level bit-widths, leaf-first.  Default [3, 4, 5].

    Returns:
        (M, 3) i32 unique coordinates sorted by hierarchical key.
    """
    if coords.shape[0] == 0:
        return coords
    keys = hierarchical_key(coords.to(torch.int32), bit_widths)
    keys_t = keys.to(torch.int64).to(coords.device)
    sorted_keys, sort_idx = torch.sort(keys_t, stable=True)
    sorted_coords = coords[sort_idx]
    keep = torch.ones(sorted_keys.shape[0], dtype=torch.bool, device=coords.device)
    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return sorted_coords[keep]


def hierarchical_key_decode(key: torch.Tensor, bit_widths: list[int]) -> torch.Tensor:
    """Decode a hierarchical sort key back to 3D coordinates.

    Inverse of :func:`hierarchical_key`.  Handles signed coordinates by
    removing the offset that was added during encoding.

    Args:
        key: i64 scalar or (N,) i64 tensor -- hierarchical sort keys.
        bit_widths: same list used for encoding, e.g. [3, 4, 5].

    Returns:
        (3,) i32 or (N, 3) i32 -- reconstructed signed coordinates.
    """
    offset, root_bits = _hkey_params(bit_widths)

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

    root_mask = (1 << root_bits) - 1
    root_linear = k >> key_shift
    rz = root_linear & root_mask
    ry = (root_linear >> root_bits) & root_mask
    rx = (root_linear >> (2 * root_bits)) & root_mask

    coord[:, 0] |= rx << coord_shift
    coord[:, 1] |= ry << coord_shift
    coord[:, 2] |= rz << coord_shift

    result = (coord - offset).to(torch.int32)
    if single:
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Hash map -- open-addressing with MurmurHash3 finalizer
# ---------------------------------------------------------------------------
#
# Pure-tensor hash map: the "map" is just a flat i64 key array with a
# sentinel protocol.  Values are separate tensors indexed by the same slot.
# This is an index protocol, not a container type.
#
# The hash function is an implementation detail -- not exposed in the DSL.
# Deterministic within a build+lookup pair; no ordering guarantees.
# ---------------------------------------------------------------------------

# Sentinel values.  0xFFFF_FFFF_FFFF_FFFF in unsigned u64, which is -1
# in signed i64 (two's complement).  Torch uses signed int64.
HASH_MAP_EMPTY_KEY: int = -1
HASH_MAP_NO_SLOT: int = -1
HASH_MAP_LOAD_FACTOR: int = 4


def _murmurhash3_fmix64(h: int) -> int:
    """MurmurHash3 64-bit finalizer.

    Standard GPU hash table hash (used by NVIDIA cuCollections/cuco).
    Five operations: xor-shift, multiply, xor-shift, multiply, xor-shift.
    Formally proven avalanche -- every output bit depends on every input bit.
    """
    mask64 = 0xFFFF_FFFF_FFFF_FFFF
    h = (h ^ (h >> 33)) & mask64
    h = (h * 0xFF51AFD7ED558CCD) & mask64
    h = (h ^ (h >> 33)) & mask64
    h = (h * 0xC4CEB9FE1A85EC53) & mask64
    h = (h ^ (h >> 33)) & mask64
    return h


def hash_map_compute_storage_size(item_count: int) -> int:
    """Compute power-of-two storage size with load factor headroom."""
    if item_count == 0:
        return 0
    size = 1
    while size < item_count:
        size *= 2
    size *= HASH_MAP_LOAD_FACTOR
    return size


def hash_map_build(keys: torch.Tensor) -> torch.Tensor:
    """Build hash map from i64 keys.  Returns the key array (the map).

    Sequential reference implementation (no CAS needed on CPU).
    The key array length IS the storage_size.  Sentinel-filled.

    Args:
        keys: (N,) i64 tensor of keys.  Must not contain EMPTY_KEY.

    Returns:
        (storage_size,) i64 key array.
    """
    n = keys.shape[0]
    storage_size = hash_map_compute_storage_size(n)
    if storage_size == 0:
        return torch.empty(0, dtype=torch.int64, device=keys.device)

    key_arr = torch.full(
        (storage_size,), HASH_MAP_EMPTY_KEY, dtype=torch.int64, device=keys.device
    )
    mask = storage_size - 1

    for i in range(n):
        k = int(keys[i].item())
        assert k != HASH_MAP_EMPTY_KEY, "Key must not be the sentinel value"
        # Hash using unsigned interpretation for the hash function
        k_unsigned = k & 0xFFFF_FFFF_FFFF_FFFF
        slot = _murmurhash3_fmix64(k_unsigned) & mask
        while True:
            existing = int(key_arr[slot].item())
            if existing == HASH_MAP_EMPTY_KEY:
                key_arr[slot] = k
                break
            elif existing == k:
                break  # duplicate key -- keep first
            else:
                slot = (slot + 1) & mask

    return key_arr


def hash_map_lookup(key_arr: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Look up keys in a hash map.  Returns slot indices.

    Args:
        key_arr: (storage_size,) i64 key array from hash_map_build.
        queries: (M,) i64 query keys.

    Returns:
        (M,) i64 tensor of slot indices.  HASH_MAP_NO_SLOT for misses.
    """
    storage_size = key_arr.shape[0]
    if storage_size == 0:
        return torch.full(
            (queries.shape[0],), HASH_MAP_NO_SLOT, dtype=torch.int64, device=queries.device
        )

    mask = storage_size - 1
    result = torch.full((queries.shape[0],), HASH_MAP_NO_SLOT, dtype=torch.int64, device=queries.device)

    for i in range(queries.shape[0]):
        k = int(queries[i].item())
        assert k != HASH_MAP_EMPTY_KEY, "Query must not be the sentinel value"
        k_unsigned = k & 0xFFFF_FFFF_FFFF_FFFF
        slot = _murmurhash3_fmix64(k_unsigned) & mask
        for _ in range(storage_size):
            existing = int(key_arr[slot].item())
            if existing == k:
                result[i] = slot
                break
            elif existing == HASH_MAP_EMPTY_KEY:
                break  # not found
            else:
                slot = (slot + 1) & mask

    return result


def hash_map_scatter_reduce(
    key_arr: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    reduce_fn: str = "or",
) -> torch.Tensor:
    """Scatter values into hash map slots, combining with a reduce function.

    Args:
        key_arr: (storage_size,) i64 key array from hash_map_build.
        keys: (N,) i64 keys to scatter.
        values: (N, ...) values to scatter.
        reduce_fn: "or", "add", "max" -- combining function for collisions.

    Returns:
        (storage_size, ...) value array with reduced values at each slot.
    """
    storage_size = key_arr.shape[0]
    value_shape = values.shape[1:]
    result = torch.zeros((storage_size, *value_shape), dtype=values.dtype, device=values.device)

    slots = hash_map_lookup(key_arr, keys)

    for i in range(keys.shape[0]):
        slot = int(slots[i].item())
        if slot == HASH_MAP_NO_SLOT:
            continue
        if reduce_fn == "or":
            result[slot] = result[slot] | values[i]
        elif reduce_fn == "add":
            result[slot] = result[slot] + values[i]
        elif reduce_fn == "max":
            result[slot] = torch.maximum(result[slot], values[i])
        else:
            raise ValueError(f"Unknown reduce_fn: {reduce_fn}")

    return result


# ---------------------------------------------------------------------------
# Leaf mask operations -- 8x8x8 bitmask primitives
# ---------------------------------------------------------------------------
#
# The 8x8x8 leaf mask is the atomic unit of the sparse grid.
# Layout: (L, 8) i64  --  8 u64 words per leaf, one per x-plane.
#   word_index = x
#   bit_position = y*8 + z      (within word)
#   flat_in_leaf = x*64 + y*8 + z
#
# These operations are fully vectorized over L leaves (no Python loops
# except over the small fixed set of boundary cases and the 8 x-planes
# -- never over L or N).
# ---------------------------------------------------------------------------


def _to_signed_i64(val: int) -> int:
    """Convert an unsigned 64-bit bitmask to signed int64 for torch."""
    val = val & 0xFFFF_FFFF_FFFF_FFFF
    if val >= (1 << 63):
        val -= 1 << 64
    return val


def shift_leaf_masks(
    masks: torch.Tensor,
    ox: int,
    oy: int,
    oz: int,
) -> list[tuple[torch.Tensor, tuple[int, int, int]]]:
    """Shift (L, 8) i64 leaf masks by voxel offset (ox, oy, oz).

    Returns a list of (shifted_masks, leaf_delta) pairs -- one for
    each target leaf that receives contributions.  A single offset
    produces up to 2^3 = 8 target leaves (boundary crossings in each
    axis), but most are empty for small offsets.

    Each shifted_masks is (L, 8) i64, same shape as input.  Leaves
    with no bits set for a given target are all-zero (harmless for OR).

    The leaf_delta is (dx, dy, dz) in leaf coordinates -- the offset
    of the target leaf relative to the source leaf.

    Fully vectorized over all L leaves.  The only iteration is over
    the small fixed set of boundary cases (at most 8) and the 8
    x-planes -- never over L.
    """
    device = masks.device
    L = masks.shape[0]
    results: list[tuple[torch.Tensor, tuple[int, int, int]]] = []

    # Per axis, determine which leaf deltas are possible and which
    # source positions (0..7) map to each delta.
    def _axis_cases(o: int) -> list[tuple[int, int, int]]:
        """Return list of (leaf_delta, src_lo, src_hi) for offset o."""
        cases = []
        lo0 = max(0, -o)
        hi0 = min(7, 7 - o)
        if lo0 <= hi0:
            cases.append((0, lo0, hi0))
        if o > 0:
            lo1 = max(0, 8 - o)
            if lo1 <= 7:
                cases.append((1, lo1, 7))
        if o < 0:
            hi_m1 = min(7, -o - 1)
            if 0 <= hi_m1:
                cases.append((-1, 0, hi_m1))
        return cases

    x_cases = _axis_cases(ox)
    y_cases = _axis_cases(oy)
    z_cases = _axis_cases(oz)

    for dx, x_lo, x_hi in x_cases:
        for dy, y_lo, y_hi in y_cases:
            for dz, z_lo, z_hi in z_cases:
                # Build a 64-bit constant mask selecting valid (y,z) source
                # positions for this boundary case.
                src_yz_mask = 0
                for y in range(y_lo, y_hi + 1):
                    for z in range(z_lo, z_hi + 1):
                        src_yz_mask |= 1 << (y * 8 + z)

                # Local shift after subtracting the leaf delta
                local_ox = ox - dx * 8
                local_oy = oy - dy * 8
                local_oz = oz - dz * 8
                bit_shift = local_oy * 8 + local_oz

                # Convert mask to signed i64 for torch tensor operations
                src_yz_mask_signed = _to_signed_i64(src_yz_mask)

                # Build shifted mask: iterate over the at-most-8 source
                # x-planes (word indices), vectorized over L leaves.
                out = torch.zeros(L, 8, dtype=torch.int64, device=device)
                for src_x in range(x_lo, x_hi + 1):
                    dst_x = src_x + local_ox
                    if dst_x < 0 or dst_x >= 8:
                        continue
                    word = masks[:, src_x] & src_yz_mask_signed
                    if bit_shift > 0:
                        out[:, dst_x] |= word << bit_shift
                    elif bit_shift < 0:
                        # Logical right shift: arithmetic shift + mask to
                        # clear sign-extended bits.
                        n = -bit_shift
                        shifted = word >> n
                        clear_mask = _to_signed_i64((1 << (64 - n)) - 1)
                        out[:, dst_x] |= shifted & clear_mask
                    else:
                        out[:, dst_x] |= word

                if out.any():
                    results.append((out, (dx, dy, dz)))

    return results


def mask_to_coords(
    masks: torch.Tensor,
    leaf_coords: torch.Tensor,
) -> torch.Tensor:
    """Extract active voxel coordinates from (L, 8) i64 leaf masks.

    Args:
        masks: (L, 8) i64 -- 8 u64 words per leaf.
        leaf_coords: (L, 3) i32 -- leaf coordinates (voxel >> 3).

    Returns:
        (M, 3) i32 -- global voxel coordinates of all set bits.
    """
    L = masks.shape[0]
    device = masks.device

    if L == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=device)

    # Expand masks to (L, 8, 64) bool via bit testing -- fully vectorized.
    bit_indices = torch.arange(64, dtype=torch.int64, device=device)
    expanded = (masks.unsqueeze(2) >> bit_indices.unsqueeze(0).unsqueeze(0)) & 1
    active = expanded.to(torch.bool).nonzero(as_tuple=False)  # (M, 3)

    if active.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=device)

    leaf_idx = active[:, 0]
    x_local = active[:, 1]          # word index = x
    bit_pos = active[:, 2]
    y_local = bit_pos >> 3          # bit_pos // 8
    z_local = bit_pos & 7           # bit_pos % 8

    base = leaf_coords[leaf_idx].to(torch.int32) * 8
    local = torch.stack([x_local, y_local, z_local], dim=1).to(torch.int32)

    return base + local
