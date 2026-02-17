# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Dialect-based lowering pass for the DSL.

A **Dialect** is a named collection of lowering rules. Each rule maps a
high-level AST node type to a rewrite function that produces a sequence of
lower-level let-bindings.

The lowering pass walks a Program's bindings in order. When a binding's
node type matches a rule in any active dialect, the single binding is
replaced by the substitute bindings from the rule. The pass repeats
until no more rewrites apply (supporting multi-level lowering).

This runs between parsing and pipeline planning::

    parse(source) -> Program
    lower_program(program, dialects) -> Program  (lowered)
    plan_program(lowered) -> PipelinePlan
    execute(plan, inputs) -> result

The planner only ever sees the lower-level primitive nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .dsl_ast import Node, Program, RefNode


# ---------------------------------------------------------------------------
# Fresh name generator
# ---------------------------------------------------------------------------


class _FreshNames:
    """Generate unique binding names for lowering expansions."""

    def __init__(self, prefix: str = "_lower"):
        self._counter = 0
        self._prefix = prefix

    def __call__(self, hint: str = "") -> str:
        name = f"{self._prefix}_{hint}_{self._counter}" if hint else f"{self._prefix}_{self._counter}"
        self._counter += 1
        return name


# ---------------------------------------------------------------------------
# Dialect and lowering rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoweringRule:
    """Rewrite a high-level node into a sequence of let-bindings.

    The rewrite function receives:
      - node: the high-level AST node to lower
      - name: the original binding name (the final substitute binding
        should use this name so downstream references remain valid)
      - fresh: a callable that generates fresh binding names

    It returns a list of (name, node) pairs that replace the original
    binding.  The last binding in the list MUST use the original ``name``
    so that references from later bindings resolve correctly.
    """

    node_type: type
    rewrite: Callable[[Node, str, _FreshNames], list[tuple[str, Node]]]


@dataclass
class Dialect:
    """A named collection of lowering rules."""

    name: str
    rules: dict[type, LoweringRule] = field(default_factory=dict)

    def add_rule(self, node_type: type, rewrite: Callable) -> None:
        self.rules[node_type] = LoweringRule(node_type=node_type, rewrite=rewrite)


# ---------------------------------------------------------------------------
# Lowering pass
# ---------------------------------------------------------------------------


def _collect_rules(dialects: list[Dialect]) -> dict[type, LoweringRule]:
    """Merge rules from all dialects into a single lookup dict.

    Later dialects override earlier ones for the same node type.
    """
    merged: dict[type, LoweringRule] = {}
    for dialect in dialects:
        for node_type, rule in dialect.rules.items():
            merged[node_type] = rule
    return merged


def lower_program(program: Program, dialects: list[Dialect], max_passes: int = 10) -> Program:
    """Rewrite all dialect nodes into lower-level compositions.

    Walks bindings in order.  For each binding whose node type has a
    matching rule, replace that binding with the substitute bindings
    from the rule.  Repeat until no more rewrites apply.

    Args:
        program: the parsed Program.
        dialects: list of Dialect instances with lowering rules.
        max_passes: safety limit on rewrite iterations.

    Returns:
        A new Program with dialect nodes replaced by their lowered forms.
    """
    rules = _collect_rules(dialects)
    if not rules:
        return program

    fresh = _FreshNames()

    bindings = list(program.bindings)

    for _pass_idx in range(max_passes):
        changed = False
        new_bindings: list[tuple[str, Node]] = []

        for name, node in bindings:
            rule = rules.get(type(node))
            if rule is not None:
                substitutes = rule.rewrite(node, name, fresh)
                if substitutes[-1][0] != name:
                    raise ValueError(
                        f"Lowering rule for {type(node).__name__} must produce "
                        f"a final binding named {name!r}, got {substitutes[-1][0]!r}"
                    )
                new_bindings.extend(substitutes)
                changed = True
            else:
                new_bindings.append((name, node))

        bindings = new_bindings
        if not changed:
            break

    return Program(bindings=bindings, output=program.output)
