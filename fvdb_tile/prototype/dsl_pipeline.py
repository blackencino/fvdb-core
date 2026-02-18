# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Barrier-aware pipeline planner and executor for the prototype DSL.

This module provides:
  - planning: partition top-level bindings into execution segments
  - AST normalization: rewrite parser adverb patterns to canonical nodes
  - execution: run planned segments through a single immutable pipeline API

Segment kinds (strict preference order: cuTile > torch > CUDA):

  "cutile"     -- tile-parallel pointwise/gather work fused into a single
                  @ct.kernel launch via dsl_to_cutile.  Preferred backend.
  "collective" -- operations requiring cross-thread coordination (Sort,
                  Unique, Where, Over, HashMapBuild/Lookup, DilateLeafMasks,
                  MaskToCoords).  GPU-accelerated via torch ops or CUDA hooks.
  "cuda"       -- last-resort backend for operations requiring atomics or
                  control flow that cuTile cannot express.  Compiled via
                  dsl_to_cuda + NVRTC (in-memory, no files on disk).

When ``device`` is set on ``PipelineExecutable.run()``, cutile segments
compile to cuTile kernels, collective segments dispatch to torch/CUDA ops
via hooks, and cuda segments compile to CUDA C++ kernels.  When ``device``
is None, the pure torch evaluator handles everything (correctness ref).
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import os
from dataclasses import dataclass

import torch

from .dsl_ast import (
    AdverbApplyNode,
    AllNode,
    ApplyNode,
    BitXorNode,
    DilateLeafMasksNode,
    EachNode,
    EqNode,
    FloorDivNode,
    HashMapBuildNode,
    HashMapLookupNode,
    HashMapOccupiedNode,
    InputNode,
    MapNode,
    MaskToCoordsNode,
    ModNode,
    Node,
    OverNode,
    Program,
    RefNode,
    ShiftLeafMaskNode,
    ShiftLeftNode,
    ShiftRightNode,
    SortNode,
    UniqueNode,
    VerbRefNode,
    WhereNode,
)
from .dsl_eval import EvalEnv, eval_node
from .dsl_parse import parse
from .ops import HASH_MAP_EMPTY_KEY, Value, hash_map_build, hash_map_lookup
from .types import Dynamic, ScalarType, Shape, Static, Type, coord_type


@dataclass(frozen=True)
class PlannedBinding:
    name: str
    node: Node


@dataclass(frozen=True)
class PipelineSegment:
    kind: str  # "cutile" (kernel-fusible) | "collective" (torch GPU barrier)
    reason: str
    bindings: tuple[PlannedBinding, ...]


@dataclass(frozen=True)
class PipelinePlan:
    segments: tuple[PipelineSegment, ...]
    output: str


@dataclass(frozen=True)
class PipelineRunResult:
    bindings: dict[str, Value]
    output_name: str
    output: Value


@dataclass(frozen=True)
class PipelineExecutable:
    plan: PipelinePlan
    program: Program

    def run(self, inputs: dict[str, Value], device: str | None = None) -> PipelineRunResult:
        """Execute the pipeline.

        Args:
            inputs: named input Values (torch tensors).
            device: torch device string.  ``"cpu"`` dispatches collectives to
                torch CPU ops (cutile segments use evaluator).  ``"cuda"``
                compiles cutile segments to cuTile GPU kernels AND dispatches
                collectives to torch GPU ops.  ``None`` (default) uses the
                pure torch evaluator for everything.
        """
        frozen_inputs = {name: _clone_value(val) for name, val in inputs.items()}

        input_types = {name: val.type for name, val in frozen_inputs.items()}
        all_types = self.program.infer_types(input_types)

        hooks = _make_collective_hooks(device) if device is not None else {}
        env = EvalEnv(frozen_inputs, hooks=hooks)

        for segment in self.plan.segments:
            if segment.kind == "cutile" and device == "cuda":
                try:
                    _run_cutile_segment(segment, env, input_types, all_types, device)
                except Exception:
                    # cuTile compilation is an optimisation; fall back to the
                    # evaluator (torch ops are device-agnostic and work on CUDA).
                    for binding in segment.bindings:
                        env.bindings[binding.name] = eval_node(binding.node, env)
            elif segment.kind == "cuda" and device == "cuda":
                _run_cuda_segment(segment, env, input_types, all_types, device)
            else:
                for binding in segment.bindings:
                    env.bindings[binding.name] = eval_node(binding.node, env)

        if self.plan.output not in env.bindings:
            raise TypeError(f"Output binding {self.plan.output!r} was not produced")
        return PipelineRunResult(
            bindings=dict(env.bindings),
            output_name=self.plan.output,
            output=env.bindings[self.plan.output],
        )


# ---------------------------------------------------------------------------
# CuTile segment compilation and execution
# ---------------------------------------------------------------------------

_GEN_DIR = os.path.join(os.path.dirname(__file__), "_generated")
_KERNEL_CACHE: dict[str, object] = {}


def _collect_refs(node: Node, bound: set[str] | None = None) -> set[str]:
    """Collect all RefNode names reachable from a node, excluding lambda-bound variables."""
    if bound is None:
        bound = set()
    refs: set[str] = set()
    if isinstance(node, RefNode) and node.name not in bound:
        refs.add(node.name)
    if isinstance(node, (MapNode, EachNode)):
        inner_bound = bound | {node.var}
        refs |= _collect_refs(node.input, bound)
        refs |= _collect_refs(node.body, inner_bound)
        return refs
    for child in vars(node).values():
        if isinstance(child, Node):
            refs |= _collect_refs(child, bound)
        elif isinstance(child, list):
            for elem in child:
                if isinstance(elem, Node):
                    refs |= _collect_refs(elem, bound)
    return refs


def _segment_external_refs(segment: PipelineSegment) -> set[str]:
    """Find names referenced by a segment that are not defined within it."""
    defined = {b.name for b in segment.bindings}
    all_refs: set[str] = set()
    for binding in segment.bindings:
        all_refs |= _collect_refs(binding.node)
    return all_refs - defined


def _rewrite_refs_to_inputs(node: Node, external_names: set[str]) -> Node:
    """Rewrite RefNode(name) -> InputNode(name) for external names."""
    if isinstance(node, RefNode) and node.name in external_names:
        return InputNode(name=node.name)

    changes = {}
    for field_obj in dataclasses.fields(node):
        val = getattr(node, field_obj.name)
        if isinstance(val, Node):
            new_val = _rewrite_refs_to_inputs(val, external_names)
            if new_val is not val:
                changes[field_obj.name] = new_val
        elif isinstance(val, list):
            new_list = []
            changed = False
            for elem in val:
                if isinstance(elem, Node):
                    new_elem = _rewrite_refs_to_inputs(elem, external_names)
                    new_list.append(new_elem)
                    if new_elem is not elem:
                        changed = True
                else:
                    new_list.append(elem)
            if changed:
                changes[field_obj.name] = new_list

    if changes:
        return dataclasses.replace(node, **changes)
    return node


def _determine_tile_input(segment_input_types: dict[str, Type]) -> tuple[str, int]:
    """Pick the tile input and its element rank for tile-parallel emission."""
    best_dynamic = None
    best_static = None
    best_static_n = -1

    for name, ty in segment_input_types.items():
        if ty.rank == 0:
            continue
        extent = ty.iteration_shape.extents[0]
        elem_rank = ty.element_type.rank if isinstance(ty.element_type, Type) else 0
        if isinstance(extent, Dynamic):
            if best_dynamic is None:
                best_dynamic = (name, elem_rank)
        elif isinstance(extent, Static):
            if extent.n > best_static_n:
                best_static = (name, elem_rank)
                best_static_n = extent.n

    if best_dynamic is not None:
        return best_dynamic
    if best_static is not None:
        return best_static
    raise TypeError("No suitable tile input found in segment input types")


def _compile_kernel(code: str, kernel_name: str):
    """Write kernel code to a .py file and import the function (cached)."""
    if kernel_name in _KERNEL_CACHE:
        return _KERNEL_CACHE[kernel_name]

    os.makedirs(_GEN_DIR, exist_ok=True)
    init_path = os.path.join(_GEN_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

    mod_name = f"_gen_pipe_{kernel_name}"
    filepath = os.path.join(_GEN_DIR, f"{mod_name}.py")
    with open(filepath, "w") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, kernel_name)
    _KERNEL_CACHE[kernel_name] = fn
    return fn


def _node_to_source(node: Node) -> str:
    """Serialize an AST node to a parseable DSL source string."""
    if isinstance(node, InputNode):
        return f'Input("{node.name}")'
    if isinstance(node, RefNode):
        return node.name
    from .dsl_ast import (
        AddNode,
        AdverbApplyNode,
        AllNode,
        AndNode,
        ApplyNode,
        ConstNode,
        CountNode,
        CutNode,
        DecomposeNode,
        DivNode,
        EachLeftNode,
        EachRightNode,
        EqNode,
        FieldNode,
        FindNode,
        FlattenNode,
        FloorDivNode,
        FuseNode,
        GatherNode,
        GENode,
        HierarchicalKeyDecodeNode,
        HierarchicalKeyNode,
        InBoundsNode,
        MaskedNode,
        ModNode,
        Morton3dNode,
        Morton3dSignedNode,
        MortonDecode3dNode,
        MulNode,
        NotNode,
        OverNode,
        PermuteNode,
        PriorNode,
        ReshapeNode,
        ScanNode,
        SubNode,
        VerbRefNode,
    )

    if isinstance(node, ConstNode):
        return f"Const({node.value!r})"
    if isinstance(node, FieldNode):
        return f"field({_node_to_source(node.expr)}, {node.field_name!r})"
    if isinstance(node, AddNode):
        return f"Add({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, SubNode):
        return f"Sub({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, GENode):
        return f"GE({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, AndNode):
        return f"And({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, NotNode):
        return f"Not({_node_to_source(node.a)})"
    if isinstance(node, InBoundsNode):
        return f"InBounds({_node_to_source(node.coord)}, {_node_to_source(node.lo)}, {_node_to_source(node.hi)})"
    if isinstance(node, MapNode):
        return f"Map({_node_to_source(node.input)}, {node.var} => {_node_to_source(node.body)})"
    if isinstance(node, EachNode):
        return f"Each({_node_to_source(node.input)}, {node.var} => {_node_to_source(node.body)})"
    if isinstance(node, WhereNode):
        return f"Where({_node_to_source(node.input)})"
    if isinstance(node, SortNode):
        return f"Sort({_node_to_source(node.input)})"
    if isinstance(node, UniqueNode):
        return f"Unique({_node_to_source(node.input)})"
    if isinstance(node, GatherNode):
        return f"Gather({_node_to_source(node.target)}, {_node_to_source(node.indexer)})"
    if isinstance(node, DecomposeNode):
        return f"Decompose({_node_to_source(node.input)}, Const({node.bit_widths!r}))"
    if isinstance(node, FindNode):
        return f"Find({_node_to_source(node.table)}, {_node_to_source(node.key)})"
    if isinstance(node, MaskedNode):
        return f"masked({_node_to_source(node.mask)}, {_node_to_source(node.abs_prefix)})"
    if isinstance(node, DivNode):
        return f"Div({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, MulNode):
        return f"Mul({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, CountNode):
        return f"Count({_node_to_source(node.input)})"
    if isinstance(node, OverNode):
        return f"Over({node.verb}, {_node_to_source(node.input)})"
    if isinstance(node, Morton3dNode):
        return f"Morton3d({_node_to_source(node.input)})"
    if isinstance(node, Morton3dSignedNode):
        return f"Morton3dSigned({_node_to_source(node.input)})"
    if isinstance(node, MortonDecode3dNode):
        return f"MortonDecode3d({_node_to_source(node.input)})"
    if isinstance(node, HierarchicalKeyNode):
        return f"HierarchicalKey({_node_to_source(node.input)}, Const({node.bit_widths!r}))"
    if isinstance(node, HierarchicalKeyDecodeNode):
        return f"HierarchicalKeyDecode({_node_to_source(node.input)}, Const({node.bit_widths!r}))"
    if isinstance(node, FuseNode):
        return f"fuse({_node_to_source(node.input)})"
    if isinstance(node, FlattenNode):
        return f"flatten({_node_to_source(node.input)})"
    if isinstance(node, ReshapeNode):
        return f"reshape({_node_to_source(node.input)}, Const({list(node.new_shape)!r}))"
    if isinstance(node, VerbRefNode):
        return node.name
    if isinstance(node, AdverbApplyNode):
        return f"{node.adverb}({_node_to_source(node.fn)})"
    if isinstance(node, ApplyNode):
        arg_strs = ", ".join(_node_to_source(a) for a in node.args)
        if isinstance(node.fn, AdverbApplyNode):
            inner_src = _node_to_source(node.fn.fn)
            return f"{node.fn.adverb}({inner_src}, {arg_strs})"
        fn_src = _node_to_source(node.fn)
        return f"Apply({fn_src}, {arg_strs})"
    if isinstance(node, ModNode):
        return f"Mod({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, EqNode):
        return f"Eq({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, FloorDivNode):
        return f"FloorDiv({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, AllNode):
        return f"All({_node_to_source(node.input)})"
    if isinstance(node, ShiftLeftNode):
        return f"ShiftLeft({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, ShiftRightNode):
        return f"ShiftRight({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, BitXorNode):
        return f"BitXor({_node_to_source(node.a)}, {_node_to_source(node.b)})"
    if isinstance(node, HashMapBuildNode):
        return f"HashMapBuild({_node_to_source(node.keys)})"
    if isinstance(node, HashMapLookupNode):
        return f"HashMapLookup({_node_to_source(node.key_arr)}, {_node_to_source(node.queries)})"
    if isinstance(node, HashMapOccupiedNode):
        return f"HashMapOccupied({_node_to_source(node.key_arr)})"
    if isinstance(node, DilateLeafMasksNode):
        return (
            f"DilateLeafMasks({_node_to_source(node.leaf_masks)}, "
            f"{_node_to_source(node.leaf_coords)}, {_node_to_source(node.offsets)}, "
            f"{_node_to_source(node.hash_map_keys)})"
        )
    raise TypeError(f"Cannot serialize {type(node).__name__} to DSL source")


def _run_cutile_segment(
    segment: PipelineSegment,
    env: EvalEnv,
    original_input_types: dict[str, Type],
    all_types: dict[str, Type],
    device: str,
):
    """Compile a cutile segment to a cuTile kernel, launch it, store result."""
    from .dsl_to_cutile import emit_runnable_kernel

    external_names = _segment_external_refs(segment)

    segment_input_types: dict[str, Type] = {}
    for name in external_names:
        if name in original_input_types:
            segment_input_types[name] = original_input_types[name]
        elif name in all_types:
            segment_input_types[name] = all_types[name]
        elif name in env.bindings:
            segment_input_types[name] = env.bindings[name].type
        else:
            raise TypeError(f"Cannot resolve type for external ref {name!r}")

    for binding in segment.bindings:
        for name in _collect_input_names(binding.node):
            if name not in segment_input_types and name in original_input_types:
                segment_input_types[name] = original_input_types[name]

    rewritten_bindings = []
    for binding in segment.bindings:
        new_node = _rewrite_refs_to_inputs(binding.node, external_names)
        rewritten_bindings.append((binding.name, new_node))

    last_name = segment.bindings[-1].name

    lines = []
    for name, node in rewritten_bindings:
        lines.append(f"{name} = {_node_to_source(node)}")
    lines.append(last_name)
    source = "\n".join(lines)

    tile_input, tile_input_rank = _determine_tile_input(segment_input_types)

    import hashlib

    seg_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    kernel_name = f"seg_{last_name}_{seg_hash}"

    TILE = 256
    code, tile_size, _ = emit_runnable_kernel(
        source,
        segment_input_types,
        kernel_name=kernel_name,
        tile_input=tile_input,
        tile_input_rank=tile_input_rank,
        tile_size=TILE,
    )

    kernel_fn = _compile_kernel(code, kernel_name)

    torch_device = torch.device(device)
    input_tensors = {}
    tile_N = 0
    for name, ty in segment_input_types.items():
        if name in env.inputs:
            data = env.inputs[name].data
        elif name in env.bindings:
            data = env.bindings[name].data
        else:
            raise TypeError(f"No data for input {name!r}")
        if isinstance(data, torch.Tensor):
            t = data.to(torch_device)
        else:
            t = torch.as_tensor(data).to(torch_device)
        input_tensors[name] = t
        if name == tile_input:
            tile_N = t.shape[0]

    n_blocks = math.ceil(tile_N / TILE)
    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device=torch_device)

    import cuda.tile as ct

    launch_args = []
    for name in segment_input_types:
        launch_args.append(input_tensors[name])
    launch_args.append(result_t)
    launch_args.append(TILE)

    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        kernel_fn,
        tuple(launch_args),
    )

    result_trimmed = result_t[:tile_N].to(torch.int32)
    result_type = all_types.get(last_name)
    if result_type is None:
        result_type = Type(Shape(Dynamic()), ScalarType.I32)
    env.bindings[last_name] = Value(result_type, result_trimmed)

    for binding in segment.bindings[:-1]:
        if binding.name not in env.bindings:
            env.bindings[binding.name] = eval_node(binding.node, env)


def _run_cuda_segment(
    segment: PipelineSegment,
    env: EvalEnv,
    original_input_types: dict[str, Type],
    all_types: dict[str, Type],
    device: str,
):
    """Compile a CUDA segment to a CUDA C++ kernel, launch it, store result.

    Last-resort backend for operations requiring atomics or control flow that
    cuTile cannot express.  Preference order: cuTile > torch > CUDA.
    """
    from .cuda_launch import compile_and_get_function, launch_kernel
    from .dsl_to_cuda import emit_cuda_kernel, _BLOCK_SIZE

    external_names = _segment_external_refs(segment)

    segment_input_types: dict[str, Type] = {}
    for name in external_names:
        if name in original_input_types:
            segment_input_types[name] = original_input_types[name]
        elif name in all_types:
            segment_input_types[name] = all_types[name]
        elif name in env.bindings:
            segment_input_types[name] = env.bindings[name].type
        else:
            raise TypeError(f"Cannot resolve type for external ref {name!r}")

    for binding in segment.bindings:
        for name in _collect_input_names(binding.node):
            if name not in segment_input_types and name in original_input_types:
                segment_input_types[name] = original_input_types[name]

    rewritten_bindings = []
    for binding in segment.bindings:
        new_node = _rewrite_refs_to_inputs(binding.node, external_names)
        rewritten_bindings.append((binding.name, new_node))

    last_name = segment.bindings[-1].name

    lines = []
    for name, node in rewritten_bindings:
        lines.append(f"{name} = {_node_to_source(node)}")
    lines.append(last_name)
    source = "\n".join(lines)

    tile_input, tile_input_rank = _determine_tile_input(segment_input_types)

    import hashlib

    seg_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    kernel_name = f"cuda_seg_{last_name}_{seg_hash}"

    code = emit_cuda_kernel(
        source,
        segment_input_types,
        kernel_name=kernel_name,
        tile_input=tile_input,
        tile_input_rank=tile_input_rank,
    )

    func = compile_and_get_function(code, kernel_name)

    torch_device = torch.device(device)
    input_tensors = {}
    tile_N = 0
    for name, ty in segment_input_types.items():
        if name in env.inputs:
            data = env.inputs[name].data
        elif name in env.bindings:
            data = env.bindings[name].data
        else:
            raise TypeError(f"No data for input {name!r}")
        if isinstance(data, torch.Tensor):
            t = data.to(torch_device)
        else:
            t = torch.as_tensor(data).to(torch_device)
        input_tensors[name] = t
        if name == tile_input:
            tile_N = t.shape[0]

    result_t = torch.full((tile_N,), -1, dtype=torch.int32, device=torch_device)

    launch_args = []
    for name in segment_input_types:
        launch_args.append(input_tensors[name])
    launch_args.append(result_t)
    launch_args.append(tile_N)

    grid = (max(1, (tile_N + _BLOCK_SIZE - 1) // _BLOCK_SIZE),)
    block = (_BLOCK_SIZE,)
    launch_kernel(func, grid, block, launch_args)

    result_type = all_types.get(last_name)
    if result_type is None:
        result_type = Type(Shape(Dynamic()), ScalarType.I32)
    env.bindings[last_name] = Value(result_type, result_t)

    for binding in segment.bindings[:-1]:
        if binding.name not in env.bindings:
            env.bindings[binding.name] = eval_node(binding.node, env)


def _collect_input_names(node: Node) -> set[str]:
    """Collect all InputNode names reachable from a node."""
    names: set[str] = set()
    if isinstance(node, InputNode):
        names.add(node.name)
    for child in vars(node).values():
        if isinstance(child, Node):
            names |= _collect_input_names(child)
        elif isinstance(child, list):
            for elem in child:
                if isinstance(elem, Node):
                    names |= _collect_input_names(elem)
    return names


# ---------------------------------------------------------------------------
# Torch-backed collective implementations
# ---------------------------------------------------------------------------


def _torch_where(node: WhereNode, env: EvalEnv) -> Value:
    """Where via torch.nonzero -- returns (*, r) i32 coordinates."""
    input_val = eval_node(node.input, EvalEnv(env.inputs, env.bindings))
    data = input_val.data
    if isinstance(data, torch.Tensor):
        coords = torch.nonzero(data).to(dtype=torch.int32)
    else:
        coords = torch.nonzero(torch.as_tensor(data)).to(dtype=torch.int32)
    result_type = Type(Shape(Dynamic()), coord_type(input_val.type.rank))
    return Value(result_type, coords)


def _torch_sort(node: SortNode, env: EvalEnv) -> Value:
    """Sort via torch -- stable ascending sort over leading axis."""
    input_val = eval_node(node.input, EvalEnv(env.inputs, env.bindings))
    data = input_val.data
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Sort requires tensor data, got {type(data)}")
    t = data.clone()
    if t.ndim == 1:
        sorted_t, _ = torch.sort(t, stable=True)
    else:
        rows = t.reshape(t.shape[0], -1)
        keys = tuple(rows[:, i] for i in range(rows.shape[1] - 1, -1, -1))
        order = torch.argsort(torch.stack(keys).T.contiguous()[:, -1], stable=True)
        for k in reversed(keys[:-1]):
            order = order[torch.argsort(k[order], stable=True)]
        sorted_t = t[order]
    return Value(input_val.type, sorted_t)


def _torch_unique(node: UniqueNode, env: EvalEnv) -> Value:
    """Unique via torch.unique -- deduplicate along leading axis."""
    input_val = eval_node(node.input, EvalEnv(env.inputs, env.bindings))
    data = input_val.data
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Unique requires tensor data, got {type(data)}")
    t = data.clone()
    if t.ndim == 1:
        unique_t = torch.unique(t, sorted=True)
    else:
        unique_t = torch.unique(t, dim=0, sorted=True)
    in_ty = input_val.type
    tail = in_ty.iteration_shape.extents[1:]
    result_type = Type(Shape(Dynamic(), *tail), in_ty.element_type)
    return Value(result_type, unique_t)


def _make_collective_hooks(device: str) -> dict:
    """Build a hooks dict that intercepts collective nodes with torch ops.

    Results are kept on the target device throughout pipeline execution --
    no GPU->CPU round-trips between segments.
    """
    torch_device = torch.device(device)

    def where_hook(node, env):
        val = _torch_where(node, env)
        return Value(val.type, val.data.to(torch_device))

    def sort_hook(node, env):
        val = _torch_sort(node, env)
        return Value(val.type, val.data.to(torch_device))

    def unique_hook(node, env):
        val = _torch_unique(node, env)
        return Value(val.type, val.data.to(torch_device))

    def hashmap_build_hook(node, env):
        keys_val = eval_node(node.keys, EvalEnv(env.inputs, env.bindings))
        data = keys_val.data
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"HashMapBuild requires tensor data, got {type(data)}")
        result_type = Type(Shape(Dynamic()), ScalarType.I64)
        if torch_device.type != "cpu":
            data_gpu = data.to(device=torch_device, dtype=torch.int64)
            from .hashmap_cuda import gpu_hash_map_build

            key_arr, _slots = gpu_hash_map_build(data_gpu)
            return Value(result_type, key_arr)
        else:
            key_arr = hash_map_build(data)
            return Value(result_type, key_arr)

    def hashmap_lookup_hook(node, env):
        key_arr_val = eval_node(node.key_arr, EvalEnv(env.inputs, env.bindings))
        queries_val = eval_node(node.queries, EvalEnv(env.inputs, env.bindings))
        key_arr_data = key_arr_val.data
        queries_data = queries_val.data
        if not isinstance(key_arr_data, torch.Tensor) or not isinstance(queries_data, torch.Tensor):
            raise TypeError("HashMapLookup requires tensor data")
        query_ty = node.queries.infer_type(
            {k: v.type for k, v in {**env.bindings, **getattr(env, "locals", {})}.items()},
            {k: v.type for k, v in env.inputs.items()},
        )
        result_type = Type(query_ty.iteration_shape, ScalarType.I64)
        if torch_device.type != "cpu":
            ka_gpu = key_arr_data.to(device=torch_device, dtype=torch.int64)
            q_gpu = queries_data.to(device=torch_device, dtype=torch.int64)
            from .hashmap_cuda import gpu_hash_map_lookup

            slots = gpu_hash_map_lookup(ka_gpu, q_gpu)
            return Value(result_type, slots)
        else:
            slots = hash_map_lookup(key_arr_data, queries_data)
            return Value(result_type, slots)

    # Adverb-to-GPU-pattern mapping for Over (torch collectives, preference #2).
    # Each verb maps to a torch reduction op.  For verbs not listed here,
    # Over falls through to the evaluator (CPU).
    #
    # Mapping table (see also the CUDA emitter for last-resort atomics):
    #   Over(Add)  -> torch.sum     | ct.sum (cuTile)    | atomicAdd (CUDA)
    #   Over(Mul)  -> torch.prod    | (not available)    | (not available)
    #   Over(Max)  -> torch.amax    | ct.maximum (cuTile)| atomicMax (CUDA)
    #   Over(Min)  -> torch.amin    | ct.minimum (cuTile)| (not available)
    #   Over(Or)   -> bitwise_or    | (not available)    | atomicOr (CUDA)
    #   Scan(Add)  -> torch.cumsum  | (not available)    | CUB prefix sum
    _OVER_TORCH_OPS = {
        "Add": lambda t, d: torch.sum(t, dim=0).to(d),
        "Mul": lambda t, d: torch.prod(t, dim=0).to(d),
        "Max": lambda t, d: (t.amax(dim=0) if t.ndim > 1 else t.max()).to(d),
        "Min": lambda t, d: (t.amin(dim=0) if t.ndim > 1 else t.min()).to(d),
        "Or": lambda t, d: _reduce_bitwise_or(t).to(d),
    }

    def _reduce_bitwise_or(t: torch.Tensor) -> torch.Tensor:
        acc = t[0]
        for i in range(1, t.shape[0]):
            acc = torch.bitwise_or(acc, t[i])
        return acc

    def over_hook(node, env):
        input_val = eval_node(node.input, EvalEnv(env.inputs, env.bindings))
        data = input_val.data
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Over requires tensor data, got {type(data)}")
        t = data.to(torch_device)
        op = _OVER_TORCH_OPS.get(node.verb)
        if op is None:
            raise TypeError(f"Over({node.verb}) not supported as GPU collective")
        result_data = op(t, torch_device)
        et = input_val.type.element_type
        result_type = et if isinstance(et, Type) else Type(Shape(), et)
        return Value(result_type, result_data)

    def dilate_leaf_masks_hook(node, env):
        leaf_masks_val = eval_node(node.leaf_masks, EvalEnv(env.inputs, env.bindings))
        leaf_coords_val = eval_node(node.leaf_coords, EvalEnv(env.inputs, env.bindings))
        offsets_val = eval_node(node.offsets, EvalEnv(env.inputs, env.bindings))
        hash_map_keys_val = eval_node(node.hash_map_keys, EvalEnv(env.inputs, env.bindings))
        ka = hash_map_keys_val.data
        ss = ka.shape[0]
        result_type = Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I64))
        if torch_device.type != "cpu":
            from .hashmap_cuda import gpu_conv_grid_dilate

            lm = leaf_masks_val.data.to(torch_device)
            lc = leaf_coords_val.data.to(torch_device)
            offs = offsets_val.data.to(device=torch_device, dtype=torch.int32)
            ka = ka.to(torch_device)
            output_masks = gpu_conv_grid_dilate(lm, lc, offs, ka, ss)
            return Value(result_type, output_masks)
        else:
            return eval_node(node, EvalEnv(env.inputs, env.bindings))

    def hashmap_occupied_hook(node, env):
        key_arr_val = eval_node(node.key_arr, EvalEnv(env.inputs, env.bindings))
        ka = key_arr_val.data
        if not isinstance(ka, torch.Tensor):
            raise TypeError(f"HashMapOccupied requires tensor data, got {type(ka)}")
        ka = ka.to(torch_device)
        occupied = torch.nonzero(ka != HASH_MAP_EMPTY_KEY, as_tuple=False).squeeze(1)
        return Value(Type(Shape(Dynamic()), ScalarType.I64), occupied.to(torch.int64))

    return {
        WhereNode: where_hook,
        SortNode: sort_hook,
        UniqueNode: unique_hook,
        OverNode: over_hook,
        DilateLeafMasksNode: dilate_leaf_masks_hook,
        HashMapBuildNode: hashmap_build_hook,
        HashMapLookupNode: hashmap_lookup_hook,
        HashMapOccupiedNode: hashmap_occupied_hook,
    }


# ---------------------------------------------------------------------------
# Barrier detection and planning
# ---------------------------------------------------------------------------


_BARRIER_NODE_TYPES = (
    WhereNode, SortNode, UniqueNode, OverNode,
    HashMapBuildNode, HashMapLookupNode, HashMapOccupiedNode,
    DilateLeafMasksNode,
    ShiftLeafMaskNode, MaskToCoordsNode,
)

# Barrier types that can be emitted inline when nested inside a Map/Each body
# (they operate on per-element tile-sized data, not global collectives).
_CUTILE_EMITTABLE_IN_BODY = (OverNode,)


def _contains_barrier(node: Node) -> bool:
    return _check_barriers(node, in_map_body=False)


def _check_barriers(node: Node, in_map_body: bool) -> bool:
    if isinstance(node, _BARRIER_NODE_TYPES):
        if in_map_body and isinstance(node, _CUTILE_EMITTABLE_IN_BODY):
            pass  # safe for cuTile: tile-level reduction inside per-element body
        else:
            return True
    # Adverb data applications (EachLeft/EachRight with data args) create new
    # iteration dimensions, so they are barriers -- same as Where or Unique.
    if (
        isinstance(node, ApplyNode)
        and isinstance(node.fn, AdverbApplyNode)
        and node.fn.adverb in ("EachLeft", "EachRight")
        and len(node.args) > 0
        and not in_map_body
    ):
        return True
    for name, child in vars(node).items():
        child_in_body = in_map_body or (isinstance(node, (MapNode, EachNode)) and name == "body")
        if isinstance(child, Node):
            if _check_barriers(child, child_in_body):
                return True
        elif isinstance(child, list):
            for elem in child:
                if isinstance(elem, Node) and _check_barriers(elem, child_in_body):
                    return True
    return False


def _barrier_reason(node: Node) -> str:
    if isinstance(node, WhereNode):
        return "where_dynamic_output"
    if isinstance(node, SortNode):
        return "sort_collective"
    if isinstance(node, UniqueNode):
        return "unique_collective"
    if isinstance(node, OverNode):
        return "reduction_barrier"
    if isinstance(node, HashMapBuildNode):
        return "hashmap_build_collective"
    if isinstance(node, HashMapLookupNode):
        return "hashmap_lookup_collective"
    if isinstance(node, HashMapOccupiedNode):
        return "hashmap_occupied_collective"
    if isinstance(node, DilateLeafMasksNode):
        return "dilate_leaf_masks_cuda"
    if isinstance(node, ShiftLeafMaskNode):
        return "shift_leaf_mask_collective"
    if isinstance(node, MaskToCoordsNode):
        return "mask_to_coords_collective"
    if (
        isinstance(node, ApplyNode)
        and isinstance(node.fn, AdverbApplyNode)
        and node.fn.adverb in ("EachLeft", "EachRight")
    ):
        return "adverb_data_application"
    return "contains_barrier_subgraph"


def _normalize_adverb_node(node: Node) -> Node:
    """Rewrite Apply(AdverbApply("Over", VerbRef(v)), [input]) -> OverNode(v, input).

    The parser creates the composed form for Over(Add, xs).  The pipeline
    planner and hooks expect the canonical OverNode.  This normalization
    bridges the gap so barrier detection and GPU dispatch work correctly.
    """
    import dataclasses as _dc

    if (
        isinstance(node, ApplyNode)
        and isinstance(node.fn, AdverbApplyNode)
        and node.fn.adverb == "Over"
        and isinstance(node.fn.fn, VerbRefNode)
        and len(node.args) == 1
    ):
        return OverNode(verb=node.fn.fn.name, input=_normalize_adverb_node(node.args[0]))

    changes = {}
    for f in _dc.fields(node):
        val = getattr(node, f.name)
        if isinstance(val, Node):
            new_val = _normalize_adverb_node(val)
            if new_val is not val:
                changes[f.name] = new_val
        elif isinstance(val, (list, tuple)):
            new_list = []
            changed = False
            for elem in val:
                if isinstance(elem, Node):
                    new_elem = _normalize_adverb_node(elem)
                    new_list.append(new_elem)
                    if new_elem is not elem:
                        changed = True
                else:
                    new_list.append(elem)
            if changed:
                changes[f.name] = type(val)(new_list)
    if changes:
        return _dc.replace(node, **changes)
    return node


def _normalize_adverbs(program: Program) -> Program:
    """Normalize adverb patterns in a program before planning."""
    new_bindings = []
    for name, node in program.bindings:
        new_bindings.append((name, _normalize_adverb_node(node)))
    return Program(bindings=tuple(new_bindings), output=program.output)


def plan_program(program: Program) -> PipelinePlan:
    segments: list[PipelineSegment] = []
    pending_cutile: list[PlannedBinding] = []

    def flush_cutile():
        if pending_cutile:
            segments.append(
                PipelineSegment(
                    kind="cutile",
                    reason="pointwise_gather_segment",
                    bindings=tuple(pending_cutile),
                )
            )
            pending_cutile.clear()

    for name, node in program.bindings:
        binding = PlannedBinding(name=name, node=node)
        if _contains_barrier(node):
            flush_cutile()
            segments.append(
                PipelineSegment(
                    kind="collective",
                    reason=_barrier_reason(node),
                    bindings=(binding,),
                )
            )
        else:
            pending_cutile.append(binding)

    flush_cutile()
    return PipelinePlan(segments=tuple(segments), output=program.output)


def plan_source(source: str) -> PipelinePlan:
    return plan_program(_normalize_adverbs(parse(source)))


def compile_program(program: Program, dialects=None) -> PipelineExecutable:
    if dialects:
        from .dsl_lower import lower_program
        program = lower_program(program, dialects)
    program = _normalize_adverbs(program)
    return PipelineExecutable(plan=plan_program(program), program=program)


def compile_source(source: str, dialects=None) -> PipelineExecutable:
    return compile_program(parse(source), dialects=dialects)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clone_value(val: Value) -> Value:
    return Value(type=val.type, data=_clone_data(val.data))


def _clone_data(data):
    if isinstance(data, torch.Tensor):
        return data.clone()
    if isinstance(data, list):
        cloned = []
        for elem in data:
            if isinstance(elem, Value):
                cloned.append(_clone_value(elem))
            else:
                cloned.append(_clone_data(elem))
        return cloned
    return data
