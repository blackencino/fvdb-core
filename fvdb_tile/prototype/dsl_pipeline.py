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

Both segment kinds target GPU execution.  The prototype evaluator executes all
segments through the pure DSL evaluator as a correctness reference; the segment
boundaries it identifies are the plan for backend dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    def run(self, inputs: dict[str, Value]) -> PipelineRunResult:
        # Clone input data to enforce value-semantic boundary guarantees.
        frozen_inputs = {name: _clone_value(val) for name, val in inputs.items()}

        # Type-check prior to execution.
        input_types = {name: val.type for name, val in frozen_inputs.items()}
        self.program.infer_types(input_types)

        env = EvalEnv(frozen_inputs)
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
