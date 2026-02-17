# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
DSL-to-cuTile emitter.

Produces a complete, compilable @ct.kernel with grid parallelism, per-axis
decomposition, and ct.scatter output via emit_runnable_kernel().

Key insight: multi-component values (like 3D coordinates) must be decomposed
into per-axis tiles for ct.gather index tuples. The emitter tracks whether
each value is a "scalar", "tile", or "decomposed" (list of per-axis names).
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from typing import Any, Union

from .dsl_ast import (
    AddNode,
    AndNode,
    ConstNode,
    CountNode,
    DecomposeNode,
    DivNode,
    FieldNode,
    FindNode,
    GatherNode,
    GENode,
    HierarchicalKeyDecodeNode,
    HierarchicalKeyNode,
    InBoundsNode,
    InputNode,
    MapNode,
    MaskedNode,
    Morton3dNode,
    Morton3dSignedNode,
    MortonDecode3dNode,
    MulNode,
    Node,
    NotNode,
    Program,
    RefNode,
    SubNode,
)
from .dsl_parse import parse
from .types import ScalarType, Shape, Static, Type


# ---------------------------------------------------------------------------
# Value representations in the emitter
# ---------------------------------------------------------------------------

# A value in the emitter is one of:
#   str                    -- a single variable name (scalar or 1D tile)
#   list[str]              -- per-axis decomposed (e.g. 3 variables for a 3D coord)
#   dict[str, list[str]]   -- struct: field name -> per-axis decomposed
#                             (produced by DecomposeNode, consumed by FieldNode)
#   dict with "__masked__" -- masked layout sentinel (produced by MaskedNode,
#                             consumed by GatherNode for popcount emission)
EmitVal = Union[str, list[str], dict]


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


_MORTON_OFFSET = 1 << 20


def _emit_part1by2(ctx: EmitCtx, x_var: str, dtype: str = "ct.int64") -> str:
    """Emit inline _part1by2 bit-spread: space x so 2 zero bits separate each.

    Produces a single output variable holding the spread result in i64.
    """
    ctx.emit(f"# _part1by2 bit-spread")
    v = ctx.fresh("pb")
    ctx.emit(f"{v} = ct.astype({x_var}, {dtype}) & {dtype}(0x1FFFFF)")
    for shift, mask in [
        (32, 0x1F00000000FFFF),
        (16, 0x1F0000FF0000FF),
        (8, 0x100F00F00F00F00F),
        (4, 0x10C30C30C30C30C3),
        (2, 0x1249249249249249),
    ]:
        prev = v
        v = ctx.fresh("pb")
        ctx.emit(f"{v} = ({prev} | ({prev} << {dtype}({shift}))) & {dtype}({hex(mask)})")
    return v


def _emit_compact1by2(ctx: EmitCtx, x_var: str, dtype: str = "ct.int64") -> str:
    """Emit inline _compact1by2 bit-extract: inverse of _part1by2.

    Extracts every third bit from x_var. Produces i64 result.
    """
    ctx.emit(f"# _compact1by2 bit-extract")
    v = ctx.fresh("cb")
    ctx.emit(f"{v} = ct.astype({x_var}, {dtype}) & {dtype}({hex(0x1249249249249249)})")
    for shift, mask in [
        (2, 0x10C30C30C30C30C3),
        (4, 0x100F00F00F00F00F),
        (8, 0x1F0000FF0000FF),
        (16, 0x1F00000000FFFF),
        (32, 0x1FFFFF),
    ]:
        prev = v
        v = ctx.fresh("cb")
        ctx.emit(f"{v} = ({prev} | ({prev} >> {dtype}({shift}))) & {dtype}({hex(mask)})")
    return v


# ---------------------------------------------------------------------------
# Emitter context
# ---------------------------------------------------------------------------

@dataclass
class EmitCtx:
    """Tracks state during code emission."""

    indent: int = 1
    var_counter: int = 0
    lines: list[str] = field(default_factory=list)
    input_params: dict[str, str] = field(default_factory=dict)
    bindings: dict[str, EmitVal] = field(default_factory=dict)
    locals: dict[str, EmitVal] = field(default_factory=dict)
    hamming_constants_emitted: bool = False

    def fresh(self, prefix: str = "t") -> str:
        self.var_counter += 1
        return f"{prefix}_{self.var_counter}"

    def emit(self, line: str):
        self.lines.append("    " * self.indent + line)

    def lookup(self, name: str) -> EmitVal:
        if name in self.locals:
            return self.locals[name]
        if name in self.bindings:
            return self.bindings[name]
        raise NameError(f"Unresolved name in emitter: {name!r}")


# ---------------------------------------------------------------------------
# Masked-Gather emission: u64 popcount chain
# ---------------------------------------------------------------------------


def _emit_masked_gather_prefix(ctx: EmitCtx, masked_val: dict, coord: EmitVal) -> str:
    """Emit absolute-prefix masked gather: 2 gathers + 1 popcount, any mask size.

    Uses pre-computed absolute prefix sums (base offset folded in) for O(1)
    lookup regardless of mask width.  Result = abs_prefix[word] + partial.

    Args:
        ctx: Emitter context.
        masked_val: Masked sentinel dict with keys:
            mask_arr   -- cuTile param name for (K, W) i64 mask array
            prefix_arr -- cuTile param name for (K, W) i32 abs prefix array
            node_idx   -- emitted variable holding the gathered node index
            bit_width  -- int, bits per axis (3 for 8^3, 4 for 16^3, 5 for 32^3)
        coord: Per-axis decomposed coordinate [x, y, z].

    Returns:
        Variable name holding the result (i32 index or -1).
    """
    if not isinstance(coord, list) or len(coord) != 3:
        raise TypeError(f"Masked gather requires 3D decomposed coord, got {type(coord)}")

    mask_arr = masked_val["mask_arr"]
    prefix_arr = masked_val["prefix_arr"]
    node_idx = masked_val["node_idx"]
    bw = masked_val["bit_width"]
    l0_x, l0_y, l0_z = coord

    # Flat-index strides depend on the node shape (2^bw per axis)
    axis_size = 1 << bw
    stride_y = axis_size
    stride_x = axis_size * axis_size

    ctx.emit(f"# --- Masked gather (abs-prefix, {axis_size}^3 node) ---")

    # Flat bit index
    bit_idx = ctx.fresh("bit_idx")
    ctx.emit(f"{bit_idx} = {l0_x} * {stride_x} + {l0_y} * {stride_y} + {l0_z}")

    # Word index and bit position (u64 words)
    word_idx = ctx.fresh("word_idx")
    n_words = (axis_size ** 3) // 64
    word_mask = n_words - 1
    ctx.emit(f"{word_idx} = ({bit_idx} >> 6) & {word_mask}")
    bit_pos = ctx.fresh("bit_pos")
    ctx.emit(f"{bit_pos} = ct.astype({bit_idx} & 63, ct.uint64)")

    # Gather 1: target mask word
    tgt_word = ctx.fresh("tgt_word")
    ctx.emit(
        f"{tgt_word} = ct.astype("
        f"ct.gather({mask_arr}, ({node_idx}, {word_idx}), check_bounds=True, padding_value=0), ct.uint64)"
    )

    # Bitmask check: is this position active?
    is_active_u = ctx.fresh("is_active_u")
    is_active = ctx.fresh("is_active")
    ctx.emit(f"{is_active_u} = ({tgt_word} >> {bit_pos}) & ct.uint64(1)")
    ctx.emit(f"{is_active} = ct.astype({is_active_u}, ct.int32)")

    # Gather 2: absolute prefix (includes base offset)
    abs_popc = ctx.fresh("abs_popc")
    ctx.emit(f"{abs_popc} = ct.gather({prefix_arr}, ({node_idx}, {word_idx}), check_bounds=True, padding_value=0)")

    # Partial word popcount: count bits below bit_pos in the target word
    partial_mask = ctx.fresh("pmask")
    ctx.emit(f"{partial_mask} = {tgt_word} & ((ct.uint64(1) << {bit_pos}) - ct.uint64(1))")

    # Hamming weight of partial word (single u64 popcount)
    if not ctx.hamming_constants_emitted:
        ctx.emit("m1_u64 = ct.uint64(0x5555555555555555)")
        ctx.emit("m2_u64 = ct.uint64(0x3333333333333333)")
        ctx.emit("m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)")
        ctx.emit("h01_u64 = ct.uint64(0x0101010101010101)")
        ctx.hamming_constants_emitted = True
    v1 = ctx.fresh("pc")
    ctx.emit(f"{v1} = {partial_mask} - (({partial_mask} >> ct.uint64(1)) & m1_u64)")
    v2 = ctx.fresh("pc")
    ctx.emit(f"{v2} = ({v1} & m2_u64) + (({v1} >> ct.uint64(2)) & m2_u64)")
    v3 = ctx.fresh("pc")
    ctx.emit(f"{v3} = ({v2} + ({v2} >> ct.uint64(4))) & m4_u64")
    partial_popc = ctx.fresh("pc")
    ctx.emit(f"{partial_popc} = ct.astype(({v3} * h01_u64) >> ct.uint64(56), ct.int32)")

    # Result: active -> abs_prefix + partial, inactive -> -1
    result = ctx.fresh("masked_idx")
    ctx.emit(f"{result} = ({abs_popc} + {partial_popc}) * {is_active} + (-1) * (1 - {is_active})")

    return result


# ---------------------------------------------------------------------------
# Axis-decomposed emitter (produces runnable @ct.kernel)
# ---------------------------------------------------------------------------


def _emit_decomposed(node: Node, ctx: EmitCtx, input_types: dict[str, Type]) -> EmitVal:
    """Emit cuTile code with axis decomposition. Returns EmitVal."""

    if isinstance(node, InputNode):
        # Check if this input has been decomposed into per-axis variables
        if node.name in ctx.locals and ctx.locals[node.name] != "__decomposed__":
            return ctx.locals[node.name]
        if node.name in ctx.bindings and ctx.bindings[node.name] != "__decomposed__":
            return ctx.bindings[node.name]
        if node.name not in ctx.input_params:
            raise NameError(f"Input {node.name!r} not declared")
        param = ctx.input_params[node.name]
        if param == "__decomposed__":
            raise NameError(f"Input {node.name!r} was decomposed but not found in bindings")
        return param

    if isinstance(node, RefNode):
        return ctx.lookup(node.name)

    if isinstance(node, ConstNode):
        if isinstance(node.value, int):
            return repr(node.value)
        if isinstance(node.value, list):
            v = ctx.fresh("const")
            ctx.emit(f"{v} = {node.value!r}")
            return v
        if isinstance(node.value, str):
            return repr(node.value)
        raise TypeError(f"Cannot emit const: {node.value!r}")

    if isinstance(node, AddNode):
        a = _emit_decomposed(node.a, ctx, input_types)
        b = _emit_decomposed(node.b, ctx, input_types)
        if isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), f"Axis count mismatch: {len(a)} vs {len(b)}"
            result = []
            for i, (ai, bi) in enumerate(zip(a, b)):
                v = ctx.fresh("add")
                ctx.emit(f"{v} = {ai} + {bi}")
                result.append(v)
            return result
        if isinstance(a, list):
            result = []
            for i, ai in enumerate(a):
                v = ctx.fresh("add")
                ctx.emit(f"{v} = {ai} + {b}")
                result.append(v)
            return result
        if isinstance(b, list):
            result = []
            for i, bi in enumerate(b):
                v = ctx.fresh("add")
                ctx.emit(f"{v} = {a} + {bi}")
                result.append(v)
            return result
        v = ctx.fresh("add")
        ctx.emit(f"{v} = {a} + {b}")
        return v

    if isinstance(node, SubNode):
        a = _emit_decomposed(node.a, ctx, input_types)
        b = _emit_decomposed(node.b, ctx, input_types)
        if isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), f"Axis count mismatch: {len(a)} vs {len(b)}"
            result = []
            for i, (ai, bi) in enumerate(zip(a, b)):
                v = ctx.fresh("sub")
                ctx.emit(f"{v} = {ai} - {bi}")
                result.append(v)
            return result
        if isinstance(a, list):
            result = []
            for i, ai in enumerate(a):
                v = ctx.fresh("sub")
                ctx.emit(f"{v} = {ai} - {b}")
                result.append(v)
            return result
        if isinstance(b, list):
            result = []
            for i, bi in enumerate(b):
                v = ctx.fresh("sub")
                ctx.emit(f"{v} = {a} - {bi}")
                result.append(v)
            return result
        v = ctx.fresh("sub")
        ctx.emit(f"{v} = {a} - {b}")
        return v

    if isinstance(node, GENode):
        a = _emit_decomposed(node.a, ctx, input_types)
        b = _emit_decomposed(node.b, ctx, input_types)
        a_str = a if isinstance(a, str) else a[0]
        b_str = b if isinstance(b, str) else b[0]
        v = ctx.fresh("ge")
        ctx.emit(f"{v} = {a_str} >= {b_str}")
        return v

    if isinstance(node, AndNode):
        a = _emit_decomposed(node.a, ctx, input_types)
        b = _emit_decomposed(node.b, ctx, input_types)
        a_str = a if isinstance(a, str) else a[0]
        b_str = b if isinstance(b, str) else b[0]
        v = ctx.fresh("and")
        ctx.emit(f"{v} = {a_str} & {b_str}")
        return v

    # -----------------------------------------------------------------
    # Pattern recognition: Decompose -> per-level bit extractions
    # -----------------------------------------------------------------
    if isinstance(node, DecomposeNode):
        coord = _emit_decomposed(node.input, ctx, input_types)
        if not isinstance(coord, list):
            raise TypeError(f"Decompose expects decomposed coord (list), got {type(coord).__name__}")
        ctx.emit(f"# Decompose: bit_widths={node.bit_widths}")
        result: dict[str, list[str]] = {}
        shift = 0
        for i, bw in enumerate(node.bit_widths):
            mask = (1 << bw) - 1
            field_name = f"level_{i}"
            axes = []
            for ax_var in coord:
                v = ctx.fresh(f"d{i}")
                if shift == 0:
                    ctx.emit(f"{v} = {ax_var} & {mask}")
                else:
                    ctx.emit(f"{v} = ({ax_var} >> {shift}) & {mask}")
                axes.append(v)
            result[field_name] = axes
            shift += bw
        top_axes = []
        for ax_var in coord:
            v = ctx.fresh("dt")
            ctx.emit(f"{v} = {ax_var} >> {shift}")
            top_axes.append(v)
        result["which_top"] = top_axes
        return result

    # -----------------------------------------------------------------
    # Pattern recognition: field projection from struct
    # -----------------------------------------------------------------
    if isinstance(node, FieldNode):
        struct_val = _emit_decomposed(node.expr, ctx, input_types)
        if isinstance(struct_val, dict):
            if node.field_name not in struct_val:
                raise TypeError(f"Struct has no field {node.field_name!r}")
            return struct_val[node.field_name]
        raise TypeError(f"Field access requires struct (dict), got {type(struct_val).__name__}")

    # -----------------------------------------------------------------
    # Find: linear scan of a small table for a matching row
    # -----------------------------------------------------------------
    if isinstance(node, FindNode):
        table_val = _emit_decomposed(node.table, ctx, input_types)
        key_val = _emit_decomposed(node.key, ctx, input_types)

        table_str = table_val if isinstance(table_val, str) else table_val[0]
        if not isinstance(key_val, list):
            raise TypeError(f"Find key must be decomposed (list), got {type(key_val).__name__}")

        # Determine R (table size) from the input type
        table_name = node.table.name if isinstance(node.table, InputNode) else None
        R = None
        if table_name and table_name in input_types:
            ty = input_types[table_name]
            if ty.rank > 0 and isinstance(ty.iteration_shape.extents[0], Static):
                R = ty.iteration_shape.extents[0].n
        if R is None:
            raise TypeError("Find requires a table with Static leading extent (R must be known at emit time)")

        K = len(key_val)
        ctx.emit(f"# --- Find: linear scan of {R} entries ---")

        result_var = ctx.fresh("find_idx")
        ctx.emit(f"{result_var} = -1")

        for r in range(R):
            # Gather each axis of the table at row r
            axis_matches = []
            for ax in range(K):
                rc = ctx.fresh("rc")
                ctx.emit(f"{rc} = ct.gather({table_str}, ({r}, {ax}), check_bounds=True, padding_value=-9999)")
                m = ctx.fresh("fm")
                ctx.emit(f"{m} = ct.astype({key_val[ax]} == {rc}, ct.int32)")
                axis_matches.append(m)

            # All axes must match
            match_var = axis_matches[0]
            for am in axis_matches[1:]:
                combined = ctx.fresh("fm")
                ctx.emit(f"{combined} = {match_var} & {am}")
                match_var = combined

            # Conditional select: if match, set result to r
            new_result = ctx.fresh("find_idx")
            ctx.emit(f"{new_result} = {result_var} * (1 - {match_var}) + {r} * {match_var}")
            result_var = new_result

        return result_var

    # -----------------------------------------------------------------
    # Masked layout: deferred computation (no code emitted yet)
    # -----------------------------------------------------------------
    if isinstance(node, MaskedNode):
        # Pattern-match: masked(Gather(Input("masks"), idx),
        #                       Gather(Input("abs_prefix"), idx))
        # Extract the underlying array names and the shared node index.
        # The actual popcount code is emitted when a GatherNode targets this.
        if not isinstance(node.mask, GatherNode) or not isinstance(node.mask.target, InputNode):
            raise TypeError("masked() mask argument must be Gather(Input(...), idx)")
        if not isinstance(node.abs_prefix, GatherNode) or not isinstance(node.abs_prefix.target, InputNode):
            raise TypeError("masked() abs_prefix argument must be Gather(Input(...), idx)")

        mask_arr_name = ctx.input_params[node.mask.target.name]
        prefix_arr_name = ctx.input_params[node.abs_prefix.target.name]
        node_idx_val = _emit_decomposed(node.mask.indexer, ctx, input_types)
        node_idx_str = node_idx_val if isinstance(node_idx_val, str) else node_idx_val[0]

        # Determine bit_width from the mask array's type (W words -> axis = (W*64)^(1/3))
        mask_input_name = node.mask.target.name
        if mask_input_name in input_types:
            mask_type = input_types[mask_input_name]
            if isinstance(mask_type.element_type, Type):
                w_extent = mask_type.element_type.iteration_shape.extents[0]
                if isinstance(w_extent, Static):
                    total_bits = w_extent.n * 64
                    axis_size = round(total_bits ** (1 / 3))
                    bit_width = axis_size.bit_length() - 1
                else:
                    bit_width = 3
            else:
                bit_width = 3
        else:
            bit_width = 3

        return {
            "__masked__": True,
            "mask_arr": mask_arr_name,
            "prefix_arr": prefix_arr_name,
            "node_idx": node_idx_str,
            "bit_width": bit_width,
        }

    # -----------------------------------------------------------------
    # Gather -- with masked-target and chain flattening (idiom detection)
    # -----------------------------------------------------------------
    if isinstance(node, GatherNode):
        # Check for masked target (via RefNode -> masked EmitVal).
        # Must come before chained-gather check since masked is detected
        # at the EmitVal level, not the AST level.
        if isinstance(node.target, RefNode):
            val = ctx.lookup(node.target.name)
            if isinstance(val, dict) and val.get("__masked__"):
                indexer = _emit_decomposed(node.indexer, ctx, input_types)
                return _emit_masked_gather_prefix(ctx, val, indexer)

        # Detect chained gathers: Gather(Gather(source, idx1), idx2).
        # Fuse into a single ct.gather(source, (idx1..., idx2...)).
        # This is the key idiom for hierarchical traversal: the two-step
        # "look up leaf block, then index into it" becomes one 4D gather.
        if isinstance(node.target, GatherNode):
            inner = node.target
            source = _emit_decomposed(inner.target, ctx, input_types)
            inner_idx = _emit_decomposed(inner.indexer, ctx, input_types)
            outer_idx = _emit_decomposed(node.indexer, ctx, input_types)

            source_str = source if isinstance(source, str) else source[0]
            all_indices: list[str] = []
            if isinstance(inner_idx, list):
                all_indices.extend(inner_idx)
            else:
                all_indices.append(inner_idx)
            if isinstance(outer_idx, list):
                all_indices.extend(outer_idx)
            else:
                all_indices.append(outer_idx)

            v = ctx.fresh("gath")
            idx_tuple = "(" + ", ".join(all_indices) + ")"
            ctx.emit(f"# Fused chained gather: {len(all_indices)}D index")
            ctx.emit(f"{v} = ct.gather({source_str}, {idx_tuple}, check_bounds=True, padding_value=-1)")
            return v

        # Non-chained gather (existing path)
        target = _emit_decomposed(node.target, ctx, input_types)
        indexer = _emit_decomposed(node.indexer, ctx, input_types)
        target_str = target if isinstance(target, str) else target[0]
        v = ctx.fresh("gath")
        if isinstance(indexer, list):
            idx_tuple = "(" + ", ".join(indexer) + ")"
            ctx.emit(f"{v} = ct.gather({target_str}, {idx_tuple}, check_bounds=True, padding_value=-1)")
        else:
            ctx.emit(f"{v} = ct.gather({target_str}, {indexer}, check_bounds=True, padding_value=-1)")
        return v

    if isinstance(node, MapNode):
        input_val = _emit_decomposed(node.input, ctx, input_types)
        old_locals = ctx.locals.copy()
        ctx.emit(f"# Map: {node.var} => ...")
        ctx.locals[node.var] = input_val
        result = _emit_decomposed(node.body, ctx, input_types)
        ctx.locals = old_locals
        return result

    # -----------------------------------------------------------------
    # MulNode: axis-decomposed multiplication
    # -----------------------------------------------------------------
    if isinstance(node, MulNode):
        a = _emit_decomposed(node.a, ctx, input_types)
        b = _emit_decomposed(node.b, ctx, input_types)
        if isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), f"Axis count mismatch: {len(a)} vs {len(b)}"
            result = []
            for ai, bi in zip(a, b):
                v = ctx.fresh("mul")
                ctx.emit(f"{v} = {ai} * {bi}")
                result.append(v)
            return result
        if isinstance(a, list):
            result = []
            for ai in a:
                v = ctx.fresh("mul")
                ctx.emit(f"{v} = {ai} * {b}")
                result.append(v)
            return result
        if isinstance(b, list):
            result = []
            for bi in b:
                v = ctx.fresh("mul")
                ctx.emit(f"{v} = {a} * {bi}")
                result.append(v)
            return result
        v = ctx.fresh("mul")
        ctx.emit(f"{v} = {a} * {b}")
        return v

    # -----------------------------------------------------------------
    # DivNode: axis-decomposed division (result is f32)
    # -----------------------------------------------------------------
    if isinstance(node, DivNode):
        a = _emit_decomposed(node.a, ctx, input_types)
        b = _emit_decomposed(node.b, ctx, input_types)
        if isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), f"Axis count mismatch: {len(a)} vs {len(b)}"
            result = []
            for ai, bi in zip(a, b):
                v = ctx.fresh("div")
                ctx.emit(f"{v} = ct.astype({ai}, ct.float32) / ct.astype({bi}, ct.float32)")
                result.append(v)
            return result
        if isinstance(a, list):
            result = []
            for ai in a:
                v = ctx.fresh("div")
                ctx.emit(f"{v} = ct.astype({ai}, ct.float32) / ct.astype({b}, ct.float32)")
                result.append(v)
            return result
        if isinstance(b, list):
            result = []
            for bi in b:
                v = ctx.fresh("div")
                ctx.emit(f"{v} = ct.astype({a}, ct.float32) / ct.astype({bi}, ct.float32)")
                result.append(v)
            return result
        v = ctx.fresh("div")
        ctx.emit(f"{v} = ct.astype({a}, ct.float32) / ct.astype({b}, ct.float32)")
        return v

    # -----------------------------------------------------------------
    # NotNode: bitwise NOT
    # -----------------------------------------------------------------
    if isinstance(node, NotNode):
        a = _emit_decomposed(node.a, ctx, input_types)
        if isinstance(a, list):
            result = []
            for ai in a:
                v = ctx.fresh("not")
                ctx.emit(f"{v} = ~{ai}")
                result.append(v)
            return result
        v = ctx.fresh("not")
        ctx.emit(f"{v} = ~{a}")
        return v

    # -----------------------------------------------------------------
    # InBoundsNode: per-axis bounds check  (lo <= x_i < hi), AND'd
    # -----------------------------------------------------------------
    if isinstance(node, InBoundsNode):
        coord = _emit_decomposed(node.coord, ctx, input_types)
        lo = _emit_decomposed(node.lo, ctx, input_types)
        hi = _emit_decomposed(node.hi, ctx, input_types)
        lo_str = lo if isinstance(lo, str) else lo[0]
        hi_str = hi if isinstance(hi, str) else hi[0]
        ctx.emit("# InBounds: per-axis range check")
        if isinstance(coord, list):
            checks = []
            for ax_var in coord:
                c = ctx.fresh("ib")
                ctx.emit(f"{c} = ct.astype(({ax_var} >= {lo_str}) & ({ax_var} < {hi_str}), ct.int32)")
                checks.append(c)
            result_var = checks[0]
            for c in checks[1:]:
                r = ctx.fresh("ib")
                ctx.emit(f"{r} = {result_var} & {c}")
                result_var = r
            return result_var
        v = ctx.fresh("ib")
        ctx.emit(f"{v} = ct.astype(({coord} >= {lo_str}) & ({coord} < {hi_str}), ct.int32)")
        return v

    # -----------------------------------------------------------------
    # CountNode: static count from the type's leading extent
    # -----------------------------------------------------------------
    if isinstance(node, CountNode):
        input_node = node.input
        if isinstance(input_node, InputNode) and input_node.name in input_types:
            ty = input_types[input_node.name]
            if ty.rank > 0 and isinstance(ty.iteration_shape.extents[0], Static):
                return repr(ty.iteration_shape.extents[0].n)
        if isinstance(input_node, RefNode):
            name = input_node.name
            if name in input_types:
                ty = input_types[name]
                if ty.rank > 0 and isinstance(ty.iteration_shape.extents[0], Static):
                    return repr(ty.iteration_shape.extents[0].n)
        raise TypeError("CountNode requires input with Static leading extent at emit time")

    # -----------------------------------------------------------------
    # Morton3dNode: unsigned 3D morton encoding via _part1by2
    # -----------------------------------------------------------------
    if isinstance(node, Morton3dNode):
        coord = _emit_decomposed(node.input, ctx, input_types)
        if not isinstance(coord, list) or len(coord) != 3:
            raise TypeError(f"Morton3d requires 3D decomposed coord, got {type(coord)}")
        ctx.emit("# Morton3d: bit-interleave 3D coord")
        px = _emit_part1by2(ctx, coord[0])
        py = _emit_part1by2(ctx, coord[1])
        pz = _emit_part1by2(ctx, coord[2])
        v = ctx.fresh("morton")
        ctx.emit(f"{v} = ct.astype({px} | ({py} << ct.int64(1)) | ({pz} << ct.int64(2)), ct.int32)")
        return v

    # -----------------------------------------------------------------
    # Morton3dSignedNode: signed 3D morton encoding (offset + _part1by2)
    # -----------------------------------------------------------------
    if isinstance(node, Morton3dSignedNode):
        coord = _emit_decomposed(node.input, ctx, input_types)
        if not isinstance(coord, list) or len(coord) != 3:
            raise TypeError(f"Morton3dSigned requires 3D decomposed coord, got {type(coord)}")
        ctx.emit(f"# Morton3dSigned: offset by {_MORTON_OFFSET} then bit-interleave")
        offset_axes = []
        for ax_var in coord:
            v = ctx.fresh("moff")
            ctx.emit(f"{v} = ct.astype({ax_var}, ct.int64) + ct.int64({_MORTON_OFFSET})")
            offset_axes.append(v)
        px = _emit_part1by2(ctx, offset_axes[0])
        py = _emit_part1by2(ctx, offset_axes[1])
        pz = _emit_part1by2(ctx, offset_axes[2])
        v = ctx.fresh("morton_s")
        ctx.emit(f"{v} = {px} | ({py} << ct.int64(1)) | ({pz} << ct.int64(2))")
        return v

    # -----------------------------------------------------------------
    # MortonDecode3dNode: decode morton codes to signed 3D coords
    # -----------------------------------------------------------------
    if isinstance(node, MortonDecode3dNode):
        code_val = _emit_decomposed(node.input, ctx, input_types)
        code_str = code_val if isinstance(code_val, str) else code_val[0]
        ctx.emit("# MortonDecode3d: extract per-axis bits and restore sign")
        code_i64 = ctx.fresh("mc")
        ctx.emit(f"{code_i64} = ct.astype({code_str}, ct.int64)")
        shifted_y = ctx.fresh("mc")
        ctx.emit(f"{shifted_y} = {code_i64} >> ct.int64(1)")
        shifted_z = ctx.fresh("mc")
        ctx.emit(f"{shifted_z} = {code_i64} >> ct.int64(2)")
        rx = _emit_compact1by2(ctx, code_i64)
        ry = _emit_compact1by2(ctx, shifted_y)
        rz = _emit_compact1by2(ctx, shifted_z)
        axes = []
        for raw in [rx, ry, rz]:
            v = ctx.fresh("md")
            ctx.emit(f"{v} = ct.astype({raw} - ct.int64({_MORTON_OFFSET}), ct.int32)")
            axes.append(v)
        return axes

    # -----------------------------------------------------------------
    # HierarchicalKeyNode: CIG-compatible hierarchical sort key
    # -----------------------------------------------------------------
    if isinstance(node, HierarchicalKeyNode):
        coord = _emit_decomposed(node.input, ctx, input_types)
        if not isinstance(coord, list) or len(coord) != 3:
            raise TypeError(f"HierarchicalKey requires 3D decomposed coord, got {type(coord)}")
        bit_widths = node.bit_widths
        ctx.emit(f"# HierarchicalKey: bit_widths={bit_widths}")
        cx = ctx.fresh("hc")
        cy = ctx.fresh("hc")
        cz = ctx.fresh("hc")
        ctx.emit(f"{cx} = ct.astype({coord[0]}, ct.int64)")
        ctx.emit(f"{cy} = ct.astype({coord[1]}, ct.int64)")
        ctx.emit(f"{cz} = ct.astype({coord[2]}, ct.int64)")
        key_var = ctx.fresh("hk")
        ctx.emit(f"{key_var} = ct.int64(0)")
        coord_shift = 0
        key_shift = 0
        for bw in bit_widths:
            dim = 1 << bw
            mask = dim - 1
            lx = ctx.fresh("lx")
            ly = ctx.fresh("ly")
            lz = ctx.fresh("lz")
            if coord_shift == 0:
                ctx.emit(f"{lx} = {cx} & ct.int64({mask})")
                ctx.emit(f"{ly} = {cy} & ct.int64({mask})")
                ctx.emit(f"{lz} = {cz} & ct.int64({mask})")
            else:
                ctx.emit(f"{lx} = ({cx} >> ct.int64({coord_shift})) & ct.int64({mask})")
                ctx.emit(f"{ly} = ({cy} >> ct.int64({coord_shift})) & ct.int64({mask})")
                ctx.emit(f"{lz} = ({cz} >> ct.int64({coord_shift})) & ct.int64({mask})")
            lin = ctx.fresh("lin")
            ctx.emit(f"{lin} = {lx} * ct.int64({dim * dim}) + {ly} * ct.int64({dim}) + {lz}")
            new_key = ctx.fresh("hk")
            if key_shift == 0:
                ctx.emit(f"{new_key} = {key_var} | {lin}")
            else:
                ctx.emit(f"{new_key} = {key_var} | ({lin} << ct.int64({key_shift}))")
            key_var = new_key
            coord_shift += bw
            key_shift += 3 * bw
        rx = ctx.fresh("rx")
        ry = ctx.fresh("ry")
        rz = ctx.fresh("rz")
        ctx.emit(f"{rx} = {cx} >> ct.int64({coord_shift})")
        ctx.emit(f"{ry} = {cy} >> ct.int64({coord_shift})")
        ctx.emit(f"{rz} = {cz} >> ct.int64({coord_shift})")
        root_lin = ctx.fresh("rlin")
        ctx.emit(f"{root_lin} = {rx} * ct.int64({1 << 20}) + {ry} * ct.int64({1 << 10}) + {rz}")
        final_key = ctx.fresh("hk")
        if key_shift == 0:
            ctx.emit(f"{final_key} = {key_var} | {root_lin}")
        else:
            ctx.emit(f"{final_key} = {key_var} | ({root_lin} << ct.int64({key_shift}))")
        return final_key

    # -----------------------------------------------------------------
    # HierarchicalKeyDecodeNode: decode hierarchical key to 3D coords
    # -----------------------------------------------------------------
    if isinstance(node, HierarchicalKeyDecodeNode):
        key_val = _emit_decomposed(node.input, ctx, input_types)
        key_str = key_val if isinstance(key_val, str) else key_val[0]
        bit_widths = node.bit_widths
        ctx.emit(f"# HierarchicalKeyDecode: bit_widths={bit_widths}")
        k = ctx.fresh("dk")
        ctx.emit(f"{k} = ct.astype({key_str}, ct.int64)")
        ax = ctx.fresh("dx")
        ay = ctx.fresh("dy")
        az = ctx.fresh("dz")
        ctx.emit(f"{ax} = ct.int64(0)")
        ctx.emit(f"{ay} = ct.int64(0)")
        ctx.emit(f"{az} = ct.int64(0)")
        key_shift = 0
        coord_shift = 0
        for bw in bit_widths:
            dim = 1 << bw
            n_bits = 3 * bw
            level_mask = (1 << n_bits) - 1
            lin = ctx.fresh("dlin")
            if key_shift == 0:
                ctx.emit(f"{lin} = {k} & ct.int64({level_mask})")
            else:
                ctx.emit(f"{lin} = ({k} >> ct.int64({key_shift})) & ct.int64({level_mask})")
            lz = ctx.fresh("dlz")
            ly = ctx.fresh("dly")
            lx = ctx.fresh("dlx")
            ctx.emit(f"{lz} = {lin} % ct.int64({dim})")
            ctx.emit(f"{ly} = ({lin} / ct.int64({dim})) % ct.int64({dim})")
            ctx.emit(f"{lx} = {lin} / ct.int64({dim * dim})")
            new_ax = ctx.fresh("dx")
            new_ay = ctx.fresh("dy")
            new_az = ctx.fresh("dz")
            if coord_shift == 0:
                ctx.emit(f"{new_ax} = {ax} | {lx}")
                ctx.emit(f"{new_ay} = {ay} | {ly}")
                ctx.emit(f"{new_az} = {az} | {lz}")
            else:
                ctx.emit(f"{new_ax} = {ax} | ({lx} << ct.int64({coord_shift}))")
                ctx.emit(f"{new_ay} = {ay} | ({ly} << ct.int64({coord_shift}))")
                ctx.emit(f"{new_az} = {az} | ({lz} << ct.int64({coord_shift}))")
            ax, ay, az = new_ax, new_ay, new_az
            key_shift += n_bits
            coord_shift += bw
        root_lin = ctx.fresh("drlin")
        ctx.emit(f"{root_lin} = {k} >> ct.int64({key_shift})")
        drz = ctx.fresh("drz")
        dry = ctx.fresh("dry")
        drx = ctx.fresh("drx")
        ctx.emit(f"{drz} = {root_lin} & ct.int64({(1 << 10) - 1})")
        ctx.emit(f"{dry} = ({root_lin} >> ct.int64(10)) & ct.int64({(1 << 10) - 1})")
        ctx.emit(f"{drx} = {root_lin} >> ct.int64(20)")
        final_x = ctx.fresh("fx")
        final_y = ctx.fresh("fy")
        final_z = ctx.fresh("fz")
        if coord_shift == 0:
            ctx.emit(f"{final_x} = ct.astype({ax} | {drx}, ct.int32)")
            ctx.emit(f"{final_y} = ct.astype({ay} | {dry}, ct.int32)")
            ctx.emit(f"{final_z} = ct.astype({az} | {drz}, ct.int32)")
        else:
            ctx.emit(f"{final_x} = ct.astype({ax} | ({drx} << ct.int64({coord_shift})), ct.int32)")
            ctx.emit(f"{final_y} = ct.astype({ay} | ({dry} << ct.int64({coord_shift})), ct.int32)")
            ctx.emit(f"{final_z} = ct.astype({az} | ({drz} << ct.int64({coord_shift})), ct.int32)")
        return [final_x, final_y, final_z]

    raise TypeError(f"Unsupported node for decomposed emission: {type(node).__name__}")


def emit_runnable_kernel(
    source: str,
    input_types: dict[str, Type],
    kernel_name: str = "generated_kernel",
    # --- Pattern 1: batch + Map (existing) ---
    batch_input: str | None = None,
    batch_dim: int = 0,
    map_input: str | None = None,
    map_elem_rank: int = 0,
    # --- Pattern 2: tile-parallel (new) ---
    tile_input: str | None = None,
    tile_input_rank: int = 0,
    tile_size: int = 256,
    tile_scalar_inputs: list[str] | None = None,
) -> tuple[str, int, int]:
    """Emit a complete, compilable @ct.kernel from a DSL program.

    Two patterns are supported:

    **Batch + Map** (set batch_input and map_input): one block per batch
    element, one tile per inner Map. Used for neighbor predicates.

    **Tile-parallel** (set tile_input): flat parallelism where each tile
    element processes one query from an (N, K) input array.  Used for
    CIG ijk_to_index and similar pointwise-over-queries kernels.

    Returns:
        (code, tile_size, map_len) where map_len is 0 for tile-parallel.
    """
    prog = parse(source)
    ctx = EmitCtx()

    # Build kernel parameter names
    param_map = {}
    for name in input_types:
        param_map[name] = name + "_arr"
    ctx.input_params = param_map

    # ------------------------------------------------------------------
    # Pattern 2: tile-parallel (flat query_idx = bid * TILE + arange)
    # ------------------------------------------------------------------
    if tile_input is not None:
        ctx.emit("bid = ct.bid(0)")
        idx_var = ctx.fresh("idx")
        ctx.emit(f"{idx_var} = ct.arange({tile_size}, dtype=ct.int32)")
        query_idx = ctx.fresh("qidx")
        ctx.emit(f"{query_idx} = bid * {tile_size} + {idx_var}")
        ctx.emit("")

        # Decompose the tile input into per-axis tile gathers
        tile_param = param_map[tile_input]
        if tile_input_rank == 0:
            # Scalar 1D array: single gather at query_idx
            v = ctx.fresh("qi")
            ctx.emit(f"{v} = ct.gather({tile_param}, {query_idx}, check_bounds=True, padding_value=0)")
            ctx.input_params[tile_input] = "__decomposed__"
            ctx.locals[tile_input] = v
            ctx.bindings[tile_input] = v
        else:
            tile_axes = []
            for ax in range(tile_input_rank):
                v = ctx.fresh("qi")
                ctx.emit(f"{v} = ct.gather({tile_param}, ({query_idx}, {ax}), check_bounds=True, padding_value=0)")
                tile_axes.append(v)
            ctx.input_params[tile_input] = "__decomposed__"
            ctx.locals[tile_input] = tile_axes  # type: ignore[assignment]
            ctx.bindings[tile_input] = tile_axes  # type: ignore[assignment]
        ctx.emit("")

        # Pre-gather explicitly listed scalar inputs at query_idx.
        # These are 1D per-query inputs (e.g. upper_idx from torch root lookup)
        # that should be gathered once at query_idx, not used as raw arrays.
        for name in (tile_scalar_inputs or []):
            p = param_map[name]
            v = ctx.fresh("ti")
            ctx.emit(f"{v} = ct.gather({p}, {query_idx}, check_bounds=True, padding_value=-1)")
            ctx.input_params[name] = "__decomposed__"
            ctx.bindings[name] = v
        ctx.emit("")

        # Emit the program body
        for name, node in prog.bindings:
            val = _emit_decomposed(node, ctx, input_types)
            ctx.bindings[name] = val

        # Get the output variable
        output_val = ctx.bindings.get(prog.output)
        if output_val is None:
            raise TypeError(f"Output {prog.output!r} not found")
        output_str = output_val if isinstance(output_val, str) else output_val[0]

        # Emit type cast and 1D scatter
        result_var = ctx.fresh("out")
        ctx.emit("")
        ctx.emit(f"{result_var} = ct.astype({output_str}, ct.int32)")
        ctx.emit(f"ct.scatter(result_arr, {query_idx}, {result_var}, check_bounds=True)")

        # Assemble kernel source
        all_params = [f"{name}_arr" for name in input_types]
        all_params.append("result_arr")
        all_params.append("TILE: ct.Constant[int]")
        params_str = ", ".join(all_params)
        body_lines = "\n".join(ctx.lines)

        code = f"""\
import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: {source.strip().replace(chr(10), chr(10) + '# ')}
# Tile input: {tile_input} (rank={tile_input_rank}), TILE={tile_size}

@ct.kernel
def {kernel_name}({params_str}):
{body_lines}
"""
        return code, tile_size, 0

    # ------------------------------------------------------------------
    # Pattern 1: batch + Map (existing)
    # ------------------------------------------------------------------
    if batch_input is None or map_input is None:
        raise TypeError("Either tile_input or both batch_input and map_input must be provided")

    # Determine tile sizes
    map_type = input_types[map_input]
    if isinstance(map_type.iteration_shape.extents[0], Static):
        map_len = map_type.iteration_shape.extents[0].n
    else:
        raise TypeError("Map input must have static leading extent")
    tile_size = _next_pow2(map_len)

    # Emit kernel body
    ctx.emit("bid = ct.bid(0)")
    ctx.emit("")

    # Decompose the batch input into per-axis scalar gathers
    batch_param = param_map[batch_input]
    batch_axes = []
    for ax in range(batch_dim):
        v = ctx.fresh("bc")
        ctx.emit(f"{v} = ct.gather({batch_param}, (bid, {ax}))")
        batch_axes.append(v)
    ctx.input_params[batch_input] = "__decomposed__"
    ctx.locals[batch_input] = batch_axes  # type: ignore[assignment]
    ctx.bindings[batch_input] = batch_axes  # type: ignore[assignment]
    ctx.emit("")

    # Create arange tile for the Map dimension
    idx_var = ctx.fresh("idx")
    ctx.emit(f"{idx_var} = ct.arange({tile_size}, dtype=ct.int32)")

    # Decompose the Map input into per-axis tile gathers
    map_param = param_map[map_input]
    map_axes = []
    for ax in range(map_elem_rank):
        v = ctx.fresh("mi")
        ctx.emit(f"{v} = ct.gather({map_param}, ({idx_var}, {ax}), check_bounds=True, padding_value=0)")
        map_axes.append(v)
    ctx.input_params[map_input] = "__decomposed__"
    ctx.locals[map_input] = map_axes  # type: ignore[assignment]
    ctx.bindings[map_input] = map_axes  # type: ignore[assignment]
    ctx.emit("")

    # Emit the program body with decomposition
    for name, node in prog.bindings:
        val = _emit_decomposed(node, ctx, input_types)
        ctx.bindings[name] = val

    # Get the output variable
    output_val = ctx.bindings.get(prog.output)
    if output_val is None:
        raise TypeError(f"Output {prog.output!r} not found")
    output_str = output_val if isinstance(output_val, str) else output_val[0]

    # Emit type cast and scatter
    result_var = ctx.fresh("out")
    ctx.emit("")
    ctx.emit(f"{result_var} = ct.astype({output_str}, ct.int32)")
    ctx.emit(f"ct.scatter(result_arr, (bid, {idx_var}), {result_var}, check_bounds=True)")

    # Assemble the full kernel source
    all_params = [f"{name}_arr" for name in input_types]
    all_params.append("result_arr")
    all_params.append("TILE: ct.Constant[int]")
    params_str = ", ".join(all_params)

    body_lines = "\n".join(ctx.lines)

    code = f"""\
import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (batch + Map) ---
# Source: {source.strip().replace(chr(10), chr(10) + '# ')}
# Batch input: {batch_input} (dim={batch_dim}), Map input: {map_input} (elem_rank={map_elem_rank})
# Tile size: {tile_size} (next power-of-two >= {map_len})

@ct.kernel
def {kernel_name}({params_str}):
{body_lines}
"""

    return code, tile_size, map_len
