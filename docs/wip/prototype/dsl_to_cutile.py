# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
DSL-to-cuTile emitter.

Two levels:
  1. emit_program()  -- proof-of-concept text emitter (v4 original)
  2. emit_runnable_kernel() -- produces a complete, compilable @ct.kernel
     with grid parallelism, per-axis decomposition, and ct.scatter output.
     Used by test_cutile_e2e.py to close the DSL -> GPU execution loop.

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
    DecomposeNode,
    FieldNode,
    GatherNode,
    GENode,
    InBoundsNode,
    InputNode,
    MapNode,
    MaskedNode,
    Node,
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
# Node emitters (v4 original -- text output, no decomposition)
# ---------------------------------------------------------------------------

def emit_node(node: Node, ctx: EmitCtx) -> str:
    """Emit cuTile code for a single AST node. Returns variable name (str)."""

    if isinstance(node, InputNode):
        if node.name not in ctx.input_params:
            raise NameError(f"Input {node.name!r} not declared")
        return ctx.input_params[node.name]

    if isinstance(node, RefNode):
        val = ctx.lookup(node.name)
        if isinstance(val, list):
            return val[0]
        return val

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
        a = emit_node(node.a, ctx)
        b = emit_node(node.b, ctx)
        v = ctx.fresh("add")
        ctx.emit(f"{v} = {a} + {b}")
        return v

    if isinstance(node, SubNode):
        a = emit_node(node.a, ctx)
        b = emit_node(node.b, ctx)
        v = ctx.fresh("sub")
        ctx.emit(f"{v} = {a} - {b}")
        return v

    if isinstance(node, GENode):
        a = emit_node(node.a, ctx)
        b = emit_node(node.b, ctx)
        v = ctx.fresh("ge")
        ctx.emit(f"{v} = {a} >= {b}")
        return v

    if isinstance(node, AndNode):
        a = emit_node(node.a, ctx)
        b = emit_node(node.b, ctx)
        v = ctx.fresh("and")
        ctx.emit(f"{v} = {a} & {b}")
        return v

    if isinstance(node, InBoundsNode):
        coord = emit_node(node.coord, ctx)
        lo = emit_node(node.lo, ctx)
        hi = emit_node(node.hi, ctx)
        v = ctx.fresh("ib")
        ctx.emit(f"# InBounds: fuse with gather in opt pass")
        ctx.emit(f"{v}_lo = {coord} >= {lo}")
        ctx.emit(f"{v}_hi = {coord} < {hi}")
        ctx.emit(f"{v} = {v}_lo & {v}_hi")
        return v

    if isinstance(node, GatherNode):
        target = emit_node(node.target, ctx)
        indexer = emit_node(node.indexer, ctx)
        v = ctx.fresh("gath")
        ctx.emit(f"{v} = ct.gather({target}, {indexer}, check_bounds=True, padding_value=-1)")
        return v

    if isinstance(node, MapNode):
        input_var = emit_node(node.input, ctx)
        old_locals = ctx.locals.copy()
        ctx.emit(f"# Map: {node.var} => ... (tile-level elementwise)")
        ctx.locals[node.var] = input_var
        result = emit_node(node.body, ctx)
        ctx.locals = old_locals
        return result

    raise TypeError(f"Unsupported node for cuTile emission: {type(node).__name__}")


def emit_program(source: str, input_map: dict[str, str], kernel_name: str = "generated_kernel") -> tuple[str, str]:
    """Parse a DSL program and emit cuTile kernel source code (text only, v4 original)."""
    prog = parse(source)
    ctx = EmitCtx()
    ctx.input_params = dict(input_map)
    for name, node in prog.bindings:
        var = emit_node(node, ctx)
        ctx.bindings[name] = var
    output_var = ctx.bindings.get(prog.output, prog.output)
    param_names = list(input_map.values())
    params_str = ", ".join(param_names)
    body = "\n".join(ctx.lines)
    code = textwrap.dedent(f"""\
        import cuda.tile as ct

        ConstInt = ct.Constant[int]

        # --- Generated from DSL ---
        # Source:
        #   {source.strip().replace(chr(10), chr(10) + '#   ')}
        #
        # Input mapping: {input_map}

        @ct.kernel
        def {kernel_name}({params_str}):
        {body}
            # output: {output_var}
    """)
    return code, output_var


# ---------------------------------------------------------------------------
# Masked-Gather emission: u64 popcount chain
# ---------------------------------------------------------------------------


def _emit_masked_gather_prefix(ctx: EmitCtx, masked_val: dict, coord: EmitVal) -> str:
    """Emit prefix-sum masked gather: 2 gathers + 1 popcount, any mask size.

    Uses pre-computed prefix-sum popcounts for O(1) lookup regardless of
    the number of mask words. Works for 8x8x8 leaf (8 words), 16x16x16
    lower (64 words), 32x32x32 upper (512 words), or any other shape.

    Args:
        ctx: Emitter context.
        masked_val: Masked sentinel dict with keys:
            mask_arr   -- cuTile param name for (K, W) i64 mask array
            prefix_arr -- cuTile param name for (K, W) i32 prefix-sum array
            node_idx   -- emitted variable holding the gathered node index
            base       -- emitted variable holding the gathered base offset
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
    base_var = masked_val["base"]
    bw = masked_val["bit_width"]
    l0_x, l0_y, l0_z = coord

    # Flat-index strides depend on the node shape (2^bw per axis)
    axis_size = 1 << bw
    stride_y = axis_size
    stride_x = axis_size * axis_size

    ctx.emit(f"# --- Masked gather (prefix-sum, {axis_size}^3 node) ---")

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

    # Gather the target mask word (1 gather)
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

    # Gather cumulative popcount from prefix-sum array (1 gather)
    cum_popc = ctx.fresh("cum_popc")
    ctx.emit(f"{cum_popc} = ct.gather({prefix_arr}, ({node_idx}, {word_idx}), check_bounds=True, padding_value=0)")

    # Partial word popcount: count bits below bit_pos in the target word
    partial_mask = ctx.fresh("pmask")
    ctx.emit(f"{partial_mask} = {tgt_word} & ((ct.uint64(1) << {bit_pos}) - ct.uint64(1))")

    # Hamming weight of partial word (single u64 popcount)
    ctx.emit("m1_u64 = ct.uint64(0x5555555555555555)")
    ctx.emit("m2_u64 = ct.uint64(0x3333333333333333)")
    ctx.emit("m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)")
    ctx.emit("h01_u64 = ct.uint64(0x0101010101010101)")
    v1 = ctx.fresh("pc")
    ctx.emit(f"{v1} = {partial_mask} - (({partial_mask} >> ct.uint64(1)) & m1_u64)")
    v2 = ctx.fresh("pc")
    ctx.emit(f"{v2} = ({v1} & m2_u64) + (({v1} >> ct.uint64(2)) & m2_u64)")
    v3 = ctx.fresh("pc")
    ctx.emit(f"{v3} = ({v2} + ({v2} >> ct.uint64(4))) & m4_u64")
    partial_popc = ctx.fresh("pc")
    ctx.emit(f"{partial_popc} = ct.astype(({v3} * h01_u64) >> ct.uint64(56), ct.int32)")

    # Total = base + prefix cumulative + partial
    total_popc = ctx.fresh("total_popc")
    ctx.emit(f"{total_popc} = {cum_popc} + {partial_popc}")

    # Result: active -> base + total, inactive -> -1
    result = ctx.fresh("masked_idx")
    ctx.emit(f"{result} = ({base_var} + {total_popc}) * {is_active} + (-1) * (1 - {is_active})")

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
    # Masked layout: deferred computation (no code emitted yet)
    # -----------------------------------------------------------------
    if isinstance(node, MaskedNode):
        # Pattern-match: masked(Gather(Input("masks"), idx),
        #                       Gather(Input("prefix"), idx),
        #                       Gather(Input("offsets"), idx))
        # Extract the underlying array names and the shared node index.
        # The actual popcount code is emitted when a GatherNode targets this.
        if not isinstance(node.mask, GatherNode) or not isinstance(node.mask.target, InputNode):
            raise TypeError("masked() mask argument must be Gather(Input(...), idx)")
        if not isinstance(node.prefix, GatherNode) or not isinstance(node.prefix.target, InputNode):
            raise TypeError("masked() prefix argument must be Gather(Input(...), idx)")
        if not isinstance(node.offset, GatherNode) or not isinstance(node.offset.target, InputNode):
            raise TypeError("masked() offset argument must be Gather(Input(...), idx)")

        mask_arr_name = ctx.input_params[node.mask.target.name]
        prefix_arr_name = ctx.input_params[node.prefix.target.name]
        node_idx_val = _emit_decomposed(node.mask.indexer, ctx, input_types)
        node_idx_str = node_idx_val if isinstance(node_idx_val, str) else node_idx_val[0]

        offset_arr_name = ctx.input_params[node.offset.target.name]

        # Emit the base offset gather (we need the scalar value at runtime)
        base_var = ctx.fresh("mbase")
        ctx.emit(f"{base_var} = ct.gather({offset_arr_name}, {node_idx_str}, check_bounds=True, padding_value=0)")

        # Determine bit_width from the mask array's type (W words -> axis = (W*64)^(1/3))
        mask_input_name = node.mask.target.name
        if mask_input_name in input_types:
            mask_type = input_types[mask_input_name]
            # mask_type is (*) over (W,) i64 -- extract W from the element shape
            if isinstance(mask_type.element_type, Type):
                w_extent = mask_type.element_type.iteration_shape.extents[0]
                if isinstance(w_extent, Static):
                    total_bits = w_extent.n * 64
                    axis_size = round(total_bits ** (1 / 3))
                    bit_width = axis_size.bit_length() - 1
                else:
                    bit_width = 3  # default to leaf
            else:
                bit_width = 3
        else:
            bit_width = 3

        return {
            "__masked__": True,
            "mask_arr": mask_arr_name,
            "prefix_arr": prefix_arr_name,
            "node_idx": node_idx_str,
            "base": base_var,
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
