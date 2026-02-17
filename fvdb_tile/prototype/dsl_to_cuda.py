# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# DSL status: core (last-resort CUDA C++ emitter)
"""
DSL-to-CUDA C++ emitter (last-resort backend).

Generates CUDA C++ source from DSL AST nodes, compiled in-memory via NVRTC.
No files on disk.  This backend exists ONLY for operations that cuTile cannot
express (atomics, inline hash probing, fine-grained control flow).

Backend preference order: cuTile > torch GPU > CUDA/NVRTC.
This emitter should only be reached when the first two options are insufficient.

The generated code uses grid-stride loops as the fundamental parallelism
pattern.  Each/Map lower to grid-stride iteration; the body is emitted
as inline C++ inside the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dsl_ast import (
    AddNode,
    AndNode,
    BitXorNode,
    ConstNode,
    DivNode,
    EachNode,
    GatherNode,
    GENode,
    InputNode,
    MapNode,
    MulNode,
    Node,
    NotNode,
    OverNode,
    RefNode,
    ShiftLeftNode,
    ShiftRightNode,
    SubNode,
)
from .dsl_parse import parse
from .types import Dynamic, ScalarType, Shape, Static, Type


# ---------------------------------------------------------------------------
# C++ type mapping
# ---------------------------------------------------------------------------

_STYPE_TO_CTYPE = {
    ScalarType.I32: "int32_t",
    ScalarType.I64: "int64_t",
    ScalarType.F32: "float",
    ScalarType.BOOL: "int32_t",
}


def _ctype(stype: ScalarType) -> str:
    return _STYPE_TO_CTYPE.get(stype, "int32_t")


# ---------------------------------------------------------------------------
# Emitter context
# ---------------------------------------------------------------------------


@dataclass
class CudaEmitCtx:
    """Tracks state during CUDA C++ code emission."""

    indent: int = 2
    var_counter: int = 0
    lines: list[str] = field(default_factory=list)
    input_params: dict[str, str] = field(default_factory=dict)
    bindings: dict[str, object] = field(default_factory=dict)
    locals: dict[str, object] = field(default_factory=dict)
    idx_var: str = "idx"

    def fresh(self, prefix: str = "t") -> str:
        self.var_counter += 1
        return f"{prefix}_{self.var_counter}"

    def emit(self, line: str):
        self.lines.append("    " * self.indent + line)

    def lookup(self, name: str):
        if name in self.locals:
            return self.locals[name]
        if name in self.bindings:
            return self.bindings[name]
        raise NameError(f"Unresolved name in CUDA emitter: {name!r}")


# ---------------------------------------------------------------------------
# Value representations
# ---------------------------------------------------------------------------

# Same convention as cuTile emitter:
#   str        -- a single C++ variable name
#   list[str]  -- per-axis decomposed (e.g. 3 variables for a 3D coord)
CudaEmitVal = str | list[str]


# ---------------------------------------------------------------------------
# AST-to-C++ emission
# ---------------------------------------------------------------------------


def _emit_node(node: Node, ctx: CudaEmitCtx, input_types: dict[str, Type]) -> CudaEmitVal:
    """Recursively emit CUDA C++ code for an AST node."""

    if isinstance(node, InputNode):
        if node.name in ctx.locals:
            return ctx.locals[node.name]
        if node.name in ctx.bindings:
            return ctx.bindings[node.name]
        if node.name not in ctx.input_params:
            raise NameError(f"Input {node.name!r} not declared")
        return ctx.input_params[node.name]

    if isinstance(node, RefNode):
        return ctx.lookup(node.name)

    if isinstance(node, ConstNode):
        if isinstance(node.value, int):
            return str(node.value)
        if isinstance(node.value, list):
            raise TypeError("List constants not yet supported in CUDA emitter")
        return str(node.value)

    # --- Binary arithmetic / bitwise ops ---

    _BINARY_OPS = {
        AddNode: "+",
        SubNode: "-",
        MulNode: "*",
        DivNode: "/",
        ShiftLeftNode: "<<",
        ShiftRightNode: ">>",
        BitXorNode: "^",
        GENode: ">=",
        AndNode: "&",
    }

    for node_type, op in _BINARY_OPS.items():
        if isinstance(node, node_type):
            a = _emit_node(node.a, ctx, input_types)
            b = _emit_node(node.b, ctx, input_types)
            if isinstance(a, list) and isinstance(b, list):
                result = []
                for ai, bi in zip(a, b):
                    v = ctx.fresh("bin")
                    ctx.emit(f"auto {v} = {ai} {op} {bi};")
                    result.append(v)
                return result
            if isinstance(a, list):
                result = []
                for ai in a:
                    v = ctx.fresh("bin")
                    ctx.emit(f"auto {v} = {ai} {op} {b};")
                    result.append(v)
                return result
            if isinstance(b, list):
                result = []
                for bi in b:
                    v = ctx.fresh("bin")
                    ctx.emit(f"auto {v} = {a} {op} {bi};")
                    result.append(v)
                return result
            v = ctx.fresh("bin")
            ctx.emit(f"auto {v} = {a} {op} {b};")
            return v

    if isinstance(node, NotNode):
        a = _emit_node(node.a, ctx, input_types)
        v = ctx.fresh("not")
        ctx.emit(f"auto {v} = !{a};")
        return v

    # --- Gather: array indexing ---

    if isinstance(node, GatherNode):
        target = _emit_node(node.target, ctx, input_types)
        indexer = _emit_node(node.indexer, ctx, input_types)
        target_str = target if isinstance(target, str) else target[0]
        if isinstance(indexer, list):
            raise TypeError("Multi-dimensional CUDA gather not yet implemented")
        v = ctx.fresh("gath")
        ctx.emit(f"auto {v} = {target_str}[{indexer}];")
        return v

    # --- Map / Each: bind var, emit body ---

    if isinstance(node, (MapNode, EachNode)):
        input_val = _emit_node(node.input, ctx, input_types)
        old_locals = ctx.locals.copy()
        ctx.emit(f"// {'Map' if isinstance(node, MapNode) else 'Each'}: {node.var} => ...")
        ctx.locals[node.var] = input_val
        result = _emit_node(node.body, ctx, input_types)
        ctx.locals = old_locals
        return result

    # --- Over: inline reduction ---

    if isinstance(node, OverNode):
        input_val = _emit_node(node.input, ctx, input_types)
        # Adverb-to-CUDA-pattern mapping for Over (last-resort backend):
        #   Over(Add, [a,b,c]) -> a + b + c
        #   Over(Mul, [a,b,c]) -> a * b * c
        #   Over(Or,  [a,b,c]) -> a | b | c   (bitwise OR, for bitmask accumulation)
        #   Over(Max, [a,b,c]) -> max(max(a,b), c)
        _VERB_OPS = {"Add": "+", "Mul": "*", "Or": "|"}
        _VERB_FUNCS = {"Max": "max", "Min": "min"}
        if isinstance(input_val, list):
            if node.verb in _VERB_OPS:
                op = _VERB_OPS[node.verb]
                acc = input_val[0]
                for v in input_val[1:]:
                    new_acc = ctx.fresh("ov")
                    ctx.emit(f"auto {new_acc} = {acc} {op} {v};")
                    acc = new_acc
                return acc
            if node.verb in _VERB_FUNCS:
                fn = _VERB_FUNCS[node.verb]
                acc = input_val[0]
                for v in input_val[1:]:
                    new_acc = ctx.fresh("ov")
                    ctx.emit(f"auto {new_acc} = {fn}({acc}, {v});")
                    acc = new_acc
                return acc
        raise TypeError(
            f"Over({node.verb}) in CUDA emitter requires decomposed (list) input"
        )

    raise TypeError(f"Unsupported node for CUDA emission: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Top-level kernel emitter
# ---------------------------------------------------------------------------


_BLOCK_SIZE = 256


def emit_cuda_kernel(
    source: str,
    input_types: dict[str, Type],
    kernel_name: str = "generated_cuda_kernel",
    tile_input: str | None = None,
    tile_input_rank: int = 0,
) -> str:
    """Emit a complete CUDA C++ kernel from a DSL program.

    Uses grid-stride loop parallelism.  Each thread processes one element
    from the tile_input array.

    Args:
        source: DSL program string.
        input_types: named input types.
        kernel_name: name for the generated __global__ function.
        tile_input: which input to parallelise over.
        tile_input_rank: element rank of the tile input (0=scalar, 3=vec3).

    Returns:
        CUDA C++ source string ready for NVRTC compilation.
    """
    prog = parse(source)
    ctx = CudaEmitCtx()

    # Determine element scalar type for the tile input
    tile_stype = ScalarType.I32
    if tile_input and tile_input in input_types:
        ty = input_types[tile_input]
        et = ty.element_type
        if isinstance(et, Type):
            tile_stype = et.element_type if isinstance(et.element_type, ScalarType) else ScalarType.I32
        elif isinstance(et, ScalarType):
            tile_stype = et

    # Build parameter declarations
    param_decls = []
    for name, ty in input_types.items():
        et = ty.element_type
        stype = et.element_type if isinstance(et, Type) and isinstance(et.element_type, ScalarType) else (et if isinstance(et, ScalarType) else ScalarType.I32)
        ctype = _ctype(stype)
        param_decls.append(f"const {ctype}* __restrict__ d_{name}")
        ctx.input_params[name] = f"d_{name}"
    param_decls.append("int32_t* __restrict__ d_result")
    param_decls.append("size_t N")

    # Open grid-stride loop
    ctx.emit(f"for (size_t {ctx.idx_var} = threadIdx.x + (size_t)blockIdx.x * blockDim.x;")
    ctx.emit(f"     {ctx.idx_var} < N;")
    ctx.emit(f"     {ctx.idx_var} += (size_t)blockDim.x * gridDim.x)")
    ctx.emit("{")
    ctx.indent += 1

    # Load tile input
    if tile_input:
        tile_param = ctx.input_params[tile_input]
        if tile_input_rank == 0:
            v = ctx.fresh("qi")
            ctype = _ctype(tile_stype)
            ctx.emit(f"{ctype} {v} = {tile_param}[{ctx.idx_var}];")
            ctx.bindings[tile_input] = v
        else:
            axes = []
            ctype = _ctype(tile_stype)
            for ax in range(tile_input_rank):
                v = ctx.fresh("qi")
                ctx.emit(f"{ctype} {v} = {tile_param}[{ctx.idx_var} * {tile_input_rank} + {ax}];")
                axes.append(v)
            ctx.bindings[tile_input] = axes

    # Emit program body
    for name, node in prog.bindings:
        val = _emit_node(node, ctx, input_types)
        ctx.bindings[name] = val

    # Store output
    output_val = ctx.bindings.get(prog.output)
    if output_val is None:
        raise TypeError(f"Output {prog.output!r} not found")
    output_str = output_val if isinstance(output_val, str) else output_val[0]
    ctx.emit(f"d_result[{ctx.idx_var}] = (int32_t)({output_str});")

    # Close grid-stride loop
    ctx.indent -= 1
    ctx.emit("}")

    # Assemble kernel
    params_str = ",\n    ".join(param_decls)
    body_lines = "\n".join(ctx.lines)

    code = f"""\
// No standard library headers in NVRTC -- use built-in types only.
typedef int                  int32_t;
typedef long long            int64_t;
typedef unsigned long long   uint64_t;

extern "C" __global__ void {kernel_name}(
    {params_str}
) {{
{body_lines}
}}
"""
    return code
