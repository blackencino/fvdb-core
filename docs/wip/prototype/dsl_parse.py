# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Recursive-descent parser for the micro DSL.

Parses a string program into an AST (dsl_ast.Program).

Grammar:
    program   := (assignment NL)* result_name
    assignment := NAME '=' expr
    expr      := call | ref | literal
    call      := BUILTIN '(' arglist ')'
    arglist   := arg (',' arg)*
    arg       := NAME '=>' expr      # binding (for Each/Map)
                | expr
    ref       := NAME
    literal   := INT | '-' INT | '[' INT (',' INT)* ']' | STRING
"""

from __future__ import annotations

import re
from typing import Any

from .dsl_ast import (
    AddNode,
    AndNode,
    ConstNode,
    CountNode,
    CutNode,
    DecomposeNode,
    DivNode,
    EachLeftNode,
    EachNode,
    EachRightNode,
    FieldNode,
    GatherNode,
    GENode,
    InBoundsNode,
    InputNode,
    MapNode,
    MaskedNode,
    Morton3dNode,
    MulNode,
    Node,
    NotNode,
    OverNode,
    PriorNode,
    Program,
    RefNode,
    ReshapeNode,
    ScanNode,
    SubNode,
    WhereNode,
)
from .types import ScalarType

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Token patterns (order matters: longer matches first)
_TOKEN_RE = re.compile(
    r"""
    (?P<STRING>"[^"]*")       |  # quoted string
    (?P<ARROW>=>)              |  # binding arrow
    (?P<ASSIGN>=)              |  # assignment
    (?P<LPAREN>\()             |  # left paren
    (?P<RPAREN>\))             |  # right paren
    (?P<LBRACKET>\[)           |  # left bracket
    (?P<RBRACKET>\])           |  # right bracket
    (?P<COMMA>,)               |  # comma
    (?P<NEWLINE>\n)            |  # newline
    (?P<INT>-?\d+)             |  # integer literal (possibly negative)
    (?P<NAME>[A-Za-z_]\w*)     |  # identifier or keyword
    (?P<WS>[ \t\r]+)           |  # whitespace (skip)
    (?P<COMMENT>\#[^\n]*)         # comment (skip)
    """,
    re.VERBOSE,
)

# Operations (PascalCase) -- computational work, moves/creates data.
_BUILTINS = {
    "Map",
    "Each",
    "Where",
    "Gather",
    "Over",
    "Scan",
    "EachRight",
    "EachLeft",
    "Prior",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "GE",
    "LE",
    "And",
    "Not",
    "InBounds",
    "Count",
    "Decompose",
    "Morton3d",
    "Input",
    "Const",
}

# Layouts (lowercase) -- zero-cost type reinterpretation, no data movement.
_LAYOUTS = {
    "cut",
    "reshape",
    "field",
    "masked",
}


def _tokenize(source: str) -> list[tuple[str, str]]:
    tokens = []
    for m in _TOKEN_RE.finditer(source):
        kind = m.lastgroup
        value = m.group()
        if kind in ("WS", "COMMENT"):
            continue
        tokens.append((kind, value))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0
        # Track let-bindings so RefNode can resolve to prior bindings
        self.bindings: dict[str, Node] = {}

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str) -> str:
        tok = self.advance()
        if tok[0] != kind:
            raise SyntaxError(f"Expected {kind}, got {tok}")
        return tok[1]

    def peek_kind(self, kind: str) -> bool:
        tok = self.peek()
        if tok is None:
            return False
        return tok[0] == kind

    def peek_not_kind(self, kind: str) -> bool:
        tok = self.peek()
        if tok is None:
            return False
        return tok[0] != kind

    def skip_newlines(self):
        while self.peek_kind("NEWLINE"):
            self.advance()

    # -- Top-level --

    def parse_program(self) -> Program:
        bindings = []
        self.skip_newlines()

        while self.pos < len(self.tokens):
            # Look ahead: is this `NAME = expr` or just `NAME` (output)?
            if self._is_assignment():
                name, expr = self.parse_assignment()
                bindings.append((name, expr))
                self.bindings[name] = expr
                self.skip_newlines()
            else:
                break

        # Remaining token should be the output name
        if self.peek_kind("NAME"):
            output = self.advance()[1]
        elif bindings:
            output = bindings[-1][0]
        else:
            raise SyntaxError("Empty program")

        self.skip_newlines()
        return Program(bindings, output)

    def _is_assignment(self) -> bool:
        """Look ahead to see if current position is NAME = expr."""
        if self.pos + 1 >= len(self.tokens):
            return False
        return self.tokens[self.pos][0] == "NAME" and self.tokens[self.pos + 1][0] == "ASSIGN"

    def parse_assignment(self) -> tuple[str, Node]:
        name = self.expect("NAME")
        self.expect("ASSIGN")
        expr = self.parse_expr()
        return name, expr

    # -- Expressions --

    def parse_expr(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of input")

        kind, value = tok

        if kind == "NAME" and (value in _BUILTINS or value in _LAYOUTS):
            # Only parse as a call if followed by '('
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == "LPAREN":
                return self.parse_call()
            else:
                # Bare builtin name (e.g. Add as a verb argument to Over)
                name = self.advance()[1]
                return RefNode(name)
        elif kind == "NAME":
            name = self.advance()[1]
            if name in self.bindings:
                return RefNode(name)
            return RefNode(name)
        elif kind == "INT":
            return self.parse_int_literal()
        elif kind == "STRING":
            return self.parse_string_literal()
        elif kind == "LBRACKET":
            return self.parse_list_literal()
        else:
            raise SyntaxError(f"Unexpected token: {tok}")

    def parse_int_literal(self) -> ConstNode:
        value = int(self.advance()[1])
        return ConstNode(value, ScalarType.I32)

    def parse_string_literal(self) -> ConstNode:
        raw = self.advance()[1]
        return ConstNode(raw.strip('"'), ScalarType.I32)  # strings are just labels

    def parse_list_literal(self) -> ConstNode:
        self.expect("LBRACKET")
        values = [int(self.expect("INT"))]
        while self.peek_kind("COMMA"):
            self.advance()
            values.append(int(self.expect("INT")))
        self.expect("RBRACKET")
        return ConstNode(values, ScalarType.I32)

    # -- Calls --

    def parse_call(self) -> Node:
        name = self.advance()[1]
        self.expect("LPAREN")
        args = self.parse_arglist()
        self.expect("RPAREN")
        return self._build_call(name, args)

    def parse_arglist(self) -> list:
        """Parse comma-separated arguments. Each arg may be a binding (name => expr) or an expr."""
        args = []
        if self.peek_not_kind("RPAREN"):
            args.append(self.parse_arg())
            while self.peek_kind("COMMA"):
                self.advance()
                args.append(self.parse_arg())
        return args

    def parse_arg(self):
        """Parse one argument: either `name => expr` (binding) or `expr`."""
        # Look ahead for binding: NAME =>
        if self.peek_kind("NAME") and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == "ARROW":
            name = self.advance()[1]
            self.advance()  # consume =>
            body = self.parse_expr()
            return ("binding", name, body)
        return ("expr", self.parse_expr())

    def _build_call(self, name: str, args: list) -> Node:
        """Construct an AST node from a builtin call name and parsed args."""

        def _expr(arg):
            """Extract the Node from a parsed arg."""
            if arg[0] == "expr":
                return arg[1]
            raise SyntaxError(f"Expected expression, got binding in {name}")

        def _binding(arg):
            """Extract (var_name, body_node) from a binding arg."""
            if arg[0] == "binding":
                return arg[1], arg[2]
            raise SyntaxError(f"Expected binding (name => expr), got expression in {name}")

        if name == "Input":
            label = _expr(args[0])
            if isinstance(label, ConstNode) and isinstance(label.value, str):
                return InputNode(label.value)
            raise SyntaxError(f"Input expects a string argument, got {label}")

        if name == "Const":
            node = _expr(args[0])
            if isinstance(node, ConstNode):
                return node
            raise SyntaxError(f"Const expects a literal, got {node}")

        if name == "Map":
            input_node = _expr(args[0])
            var, body = _binding(args[1])
            return MapNode(input_node, var, body)

        if name == "Each":
            input_node = _expr(args[0])
            var, body = _binding(args[1])
            return EachNode(input_node, var, body)

        if name == "Where":
            return WhereNode(_expr(args[0]))

        if name == "Gather":
            return GatherNode(_expr(args[0]), _expr(args[1]))

        if name == "Add":
            return AddNode(_expr(args[0]), _expr(args[1]))

        if name == "Sub":
            return SubNode(_expr(args[0]), _expr(args[1]))

        if name == "GE":
            return GENode(_expr(args[0]), _expr(args[1]))

        if name == "And":
            return AndNode(_expr(args[0]), _expr(args[1]))

        if name == "Not":
            return NotNode(_expr(args[0]))

        if name == "InBounds":
            return InBoundsNode(_expr(args[0]), _expr(args[1]), _expr(args[2]))

        if name == "Decompose":
            input_node = _expr(args[0])
            bw_node = _expr(args[1])
            if isinstance(bw_node, ConstNode) and isinstance(bw_node.value, list):
                return DecomposeNode(input_node, bw_node.value)
            raise SyntaxError(f"Decompose expects list of bit_widths, got {bw_node}")

        if name == "Morton3d":
            return Morton3dNode(_expr(args[0]))

        if name == "field":
            expr_node = _expr(args[0])
            field_node = _expr(args[1])
            if isinstance(field_node, ConstNode) and isinstance(field_node.value, str):
                return FieldNode(expr_node, field_node.value)
            raise SyntaxError(f"field expects string field name, got {field_node}")

        if name == "cut":
            input_node = _expr(args[0])
            size_node = _expr(args[1])
            if isinstance(size_node, ConstNode) and isinstance(size_node.value, int):
                return CutNode(input_node, size_node.value)
            raise SyntaxError(f"cut expects integer size, got {size_node}")

        if name == "reshape":
            input_node = _expr(args[0])
            shape_node = _expr(args[1])
            if isinstance(shape_node, ConstNode) and isinstance(shape_node.value, list):
                return ReshapeNode(input_node, tuple(shape_node.value))
            raise SyntaxError(f"reshape expects list shape, got {shape_node}")

        if name == "masked":
            return MaskedNode(_expr(args[0]), _expr(args[1]), _expr(args[2]))

        # -- Adverbs --

        if name == "Over":
            # Over(VerbName, xs) -- VerbName is a bare name, not a call
            verb_arg = _expr(args[0])
            if isinstance(verb_arg, RefNode):
                verb_name = verb_arg.name
            elif isinstance(verb_arg, ConstNode) and isinstance(verb_arg.value, str):
                verb_name = verb_arg.value
            else:
                raise SyntaxError(f"Over expects a verb name, got {verb_arg}")
            return OverNode(verb_name, _expr(args[1]))

        if name == "Scan":
            verb_arg = _expr(args[0])
            verb_name = verb_arg.name if isinstance(verb_arg, RefNode) else str(verb_arg)
            return ScanNode(verb_name, _expr(args[1]))

        if name == "EachRight":
            verb_arg = _expr(args[0])
            verb_name = verb_arg.name if isinstance(verb_arg, RefNode) else str(verb_arg)
            return EachRightNode(verb_name, _expr(args[1]), _expr(args[2]))

        if name == "EachLeft":
            verb_arg = _expr(args[0])
            verb_name = verb_arg.name if isinstance(verb_arg, RefNode) else str(verb_arg)
            return EachLeftNode(verb_name, _expr(args[1]), _expr(args[2]))

        if name == "Prior":
            verb_arg = _expr(args[0])
            verb_name = verb_arg.name if isinstance(verb_arg, RefNode) else str(verb_arg)
            return PriorNode(verb_name, _expr(args[1]))

        # -- Scalar primitives --

        if name == "Div":
            return DivNode(_expr(args[0]), _expr(args[1]))

        if name == "Mul":
            return MulNode(_expr(args[0]), _expr(args[1]))

        if name == "Count":
            return CountNode(_expr(args[0]))

        raise SyntaxError(f"Unknown builtin: {name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(source: str) -> Program:
    """Parse a DSL program string into an AST."""
    tokens = _tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()
