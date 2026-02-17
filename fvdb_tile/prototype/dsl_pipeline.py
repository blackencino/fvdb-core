# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Barrier-aware pipeline planner and executor for the prototype DSL.

This module provides:
  - planning: partition top-level bindings into execution segments
  - execution: run planned segments through a single immutable pipeline API

Segment kinds:

  "cutile"     -- tile-parallel pointwise/gather work that can be fused into a
                  single @ct.kernel launch via dsl_to_cutile.emit_runnable_kernel.
  "collective" -- operations requiring cross-thread coordination (Sort, Unique,
                  Where, Over).  These are GPU-accelerated via torch ops (e.g.
                  torch.sort, torch.unique) and require a synchronization barrier
                  between kernel launches.

Both segment kinds target GPU execution.  When ``device`` is set on
``PipelineExecutable.run()``, cutile segments compile to cuTile kernels
and collective segments dispatch to torch ops.  When ``device`` is None,
the pure numpy evaluator handles everything (correctness reference).
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from .dsl_ast import (
    InputNode,
    Node,
    OverNode,
    Program,
    RefNode,
    SortNode,
    UniqueNode,
    WhereNode,
)
from .dsl_eval import EvalEnv, eval_node
from .dsl_parse import parse
from .ops import Value
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
            inputs: named input Values (numpy arrays).
            device: torch device string.  ``"cpu"`` dispatches collectives to
                torch CPU ops (cutile segments use evaluator).  ``"cuda"``
                compiles cutile segments to cuTile GPU kernels AND dispatches
                collectives to torch GPU ops.  ``None`` (default) uses the
                pure numpy evaluator for everything.
        """
        frozen_inputs = {name: _clone_value(val) for name, val in inputs.items()}

        input_types = {name: val.type for name, val in frozen_inputs.items()}
        all_types = self.program.infer_types(input_types)

        hooks = _make_collective_hooks(device) if device is not None else {}
        env = EvalEnv(frozen_inputs, hooks=hooks)

        for segment in self.plan.segments:
            if segment.kind == "cutile" and device == "cuda":
                _run_cutile_segment(segment, env, input_types, all_types, device)
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


def _collect_refs(node: Node) -> set[str]:
    """Collect all RefNode names reachable from a node."""
    refs: set[str] = set()
    if isinstance(node, RefNode):
        refs.add(node.name)
    for child in vars(node).values():
        if isinstance(child, Node):
            refs |= _collect_refs(child)
        elif isinstance(child, list):
            for elem in child:
                if isinstance(elem, Node):
                    refs |= _collect_refs(elem)
    return refs


def _segment_external_refs(segment: PipelineSegment) -> set[str]:
    """Find names referenced by a segment that are not defined within it."""
    defined = {b.name for b in segment.bindings}
    all_refs: set[str] = set()
    for binding in segment.bindings:
        all_refs |= _collect_refs(binding.node)
    return all_refs - defined


def _rewrite_refs_to_inputs(node: Node, external_names: set[str]) -> Node:
    """Rewrite RefNode(name) -> InputNode(name) for external names.

    Returns a new AST tree (nodes are frozen dataclasses).
    """
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
    """Pick the tile input and its element rank for tile-parallel emission.

    Prefers the first input with Dynamic leading extent.  Falls back to
    the input with the largest Static leading extent.  Returns
    (input_name, element_rank).
    """
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

    # Build the segment's input types: original inputs + types of external refs
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

    # Also include original program inputs that are directly referenced via InputNode
    for binding in segment.bindings:
        for name in _collect_input_names(binding.node):
            if name not in segment_input_types and name in original_input_types:
                segment_input_types[name] = original_input_types[name]

    # Rewrite external RefNodes to InputNodes
    rewritten_bindings = []
    for binding in segment.bindings:
        new_node = _rewrite_refs_to_inputs(binding.node, external_names)
        rewritten_bindings.append((binding.name, new_node))

    last_name = segment.bindings[-1].name

    # Serialize to DSL source
    lines = []
    for name, node in rewritten_bindings:
        lines.append(f"{name} = {repr(node)}")
    lines.append(last_name)
    source = "\n".join(lines)

    # Determine parallelism
    tile_input, tile_input_rank = _determine_tile_input(segment_input_types)

    # Generate a unique kernel name from the segment content
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

    # Prepare input tensors on device
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
        t = torch.from_numpy(np.asarray(data)).to(torch_device)
        input_tensors[name] = t
        if name == tile_input:
            tile_N = t.shape[0]

    n_blocks = math.ceil(tile_N / TILE)
    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device=torch_device)

    # Build launch args in the order emit_runnable_kernel expects
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

    # Extract result and store in env
    result_np = result_t[:tile_N].cpu().numpy().astype(np.int32)
    result_type = all_types.get(last_name)
    if result_type is None:
        result_type = Type(Shape(Dynamic()), ScalarType.I32)
    env.bindings[last_name] = Value(result_type, result_np)

    # For multi-binding segments, evaluate intermediate bindings via evaluator
    # so they're available if referenced later (rare case).
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
    t = torch.from_numpy(np.asarray(input_val.data))
    device = t.device
    coords = torch.nonzero(t).to(dtype=torch.int32, device=device)
    result_type = Type(Shape(Dynamic()), coord_type(input_val.type.rank))
    return Value(result_type, coords.numpy())


def _torch_sort(node: SortNode, env: EvalEnv) -> Value:
    """Sort via torch -- stable ascending sort over leading axis."""
    input_val = eval_node(node.input, EvalEnv(env.inputs, env.bindings))
    data = input_val.data
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Sort requires ndarray data, got {type(data)}")
    t = torch.from_numpy(data.copy())
    if t.ndim == 1:
        sorted_t, _ = torch.sort(t, stable=True)
    else:
        rows = t.reshape(t.shape[0], -1)
        keys = tuple(rows[:, i] for i in range(rows.shape[1] - 1, -1, -1))
        order = torch.argsort(torch.stack(keys).T.contiguous()[:, -1], stable=True)
        for k in reversed(keys[:-1]):
            order = order[torch.argsort(k[order], stable=True)]
        sorted_t = t[order]
    return Value(input_val.type, sorted_t.numpy())


def _torch_unique(node: UniqueNode, env: EvalEnv) -> Value:
    """Unique via torch.unique -- deduplicate along leading axis."""
    input_val = eval_node(node.input, EvalEnv(env.inputs, env.bindings))
    data = input_val.data
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Unique requires ndarray data, got {type(data)}")
    t = torch.from_numpy(data.copy())
    if t.ndim == 1:
        unique_t = torch.unique(t, sorted=True)
    else:
        unique_t = torch.unique(t, dim=0, sorted=True)
    in_ty = input_val.type
    tail = in_ty.iteration_shape.extents[1:]
    result_type = Type(Shape(Dynamic(), *tail), in_ty.element_type)
    return Value(result_type, unique_t.numpy())


def _make_collective_hooks(device: str) -> dict:
    """Build a hooks dict that intercepts collective nodes with torch ops."""
    torch_device = torch.device(device)

    def where_hook(node, env):
        val = _torch_where(node, env)
        if torch_device.type != "cpu":
            t = torch.from_numpy(val.data).to(torch_device)
            return Value(val.type, t.cpu().numpy())
        return val

    def sort_hook(node, env):
        val = _torch_sort(node, env)
        if torch_device.type != "cpu":
            t = torch.from_numpy(val.data).to(torch_device)
            sorted_t, _ = torch.sort(t, stable=True, dim=0)
            return Value(val.type, sorted_t.cpu().numpy())
        return val

    def unique_hook(node, env):
        val = _torch_unique(node, env)
        if torch_device.type != "cpu":
            t = torch.from_numpy(val.data).to(torch_device)
            if t.ndim == 1:
                unique_t = torch.unique(t, sorted=True)
            else:
                unique_t = torch.unique(t, dim=0, sorted=True)
            in_ty = val.type
            return Value(in_ty, unique_t.cpu().numpy())
        return val

    return {
        WhereNode: where_hook,
        SortNode: sort_hook,
        UniqueNode: unique_hook,
    }


# ---------------------------------------------------------------------------
# Barrier detection and planning
# ---------------------------------------------------------------------------


_BARRIER_NODE_TYPES = (WhereNode, SortNode, UniqueNode, OverNode)


def _contains_barrier(node: Node) -> bool:
    if isinstance(node, _BARRIER_NODE_TYPES):
        return True
    for child in vars(node).values():
        if isinstance(child, Node) and _contains_barrier(child):
            return True
        if isinstance(child, list):
            for elem in child:
                if isinstance(elem, Node) and _contains_barrier(elem):
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
    return "contains_barrier_subgraph"


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
    return plan_program(parse(source))


def compile_program(program: Program) -> PipelineExecutable:
    return PipelineExecutable(plan=plan_program(program), program=program)


def compile_source(source: str) -> PipelineExecutable:
    return compile_program(parse(source))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clone_value(val: Value) -> Value:
    return Value(type=val.type, data=_clone_data(val.data))


def _clone_data(data):
    if isinstance(data, np.ndarray):
        return data.copy()
    if isinstance(data, list):
        cloned = []
        for elem in data:
            if isinstance(elem, Value):
                cloned.append(_clone_value(elem))
            else:
                cloned.append(_clone_data(elem))
        return cloned
    return data
