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

Both segment kinds target GPU execution.  Cutile segments currently evaluate
through the pure DSL evaluator as a correctness reference.  Collective segments
dispatch to torch ops for GPU acceleration when ``device`` is specified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dsl_ast import (
    Node,
    OverNode,
    Program,
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
            device: torch device for collective ops.  ``"cpu"`` dispatches
                collectives to torch CPU ops, ``"cuda"`` to torch GPU ops,
                and ``None`` (default) uses the pure numpy evaluator for
                everything (the original behaviour, useful as a correctness
                reference).
        """
        frozen_inputs = {name: _clone_value(val) for name, val in inputs.items()}

        input_types = {name: val.type for name, val in frozen_inputs.items()}
        self.program.infer_types(input_types)

        hooks = _make_collective_hooks(device) if device is not None else {}
        env = EvalEnv(frozen_inputs, hooks=hooks)

        for segment in self.plan.segments:
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
    """Build a hooks dict that intercepts collective nodes with torch ops.

    The hooks evaluate the collective node's children via the normal
    evaluator (without hooks, to avoid infinite recursion), then dispatch
    the collective itself to a torch op on the given device.
    """
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
