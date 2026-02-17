# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Type lattice for the layout algebra prototype.

Extent kinds (static < dynamic < jagged), Shape, ScalarType, Type.
"""

from __future__ import annotations

import enum
import functools
from dataclasses import dataclass
from typing import Union


# ---------------------------------------------------------------------------
# Extent kinds
# ---------------------------------------------------------------------------

class Extent:
    """Base class for axis extent kinds."""

    def resolve(self, other: Extent) -> Extent:
        """Resolve two extents per the compatibility table.

        static(n) + static(n) = static(n)   (mismatch -> TypeError)
        static(n) + dynamic   = static(n)
        dynamic   + dynamic   = dynamic
        jagged    + jagged    = jagged       (offsets must agree at runtime)
        uniform   + jagged    = TypeError
        """
        return _resolve_extents(self, other)

    def is_compatible(self, other: Extent) -> bool:
        try:
            self.resolve(other)
            return True
        except TypeError:
            return False


@dataclass(frozen=True)
class Static(Extent):
    n: int

    def __repr__(self) -> str:
        return str(self.n)


@dataclass(frozen=True)
class Dynamic(Extent):
    def __repr__(self) -> str:
        return "*"


@dataclass(frozen=True)
class Jagged(Extent):
    def __repr__(self) -> str:
        return "~"


def _resolve_extents(a: Extent, b: Extent) -> Extent:
    # Same type shortcuts
    if isinstance(a, Static) and isinstance(b, Static):
        if a.n != b.n:
            raise TypeError(f"Static extent mismatch: {a.n} vs {b.n}")
        return a
    if isinstance(a, Dynamic) and isinstance(b, Dynamic):
        return Dynamic()
    if isinstance(a, Jagged) and isinstance(b, Jagged):
        return Jagged()

    # Static + Dynamic -> Static (order-independent)
    if isinstance(a, Static) and isinstance(b, Dynamic):
        return a
    if isinstance(a, Dynamic) and isinstance(b, Static):
        return b

    # Any mix of uniform (static/dynamic) with jagged -> error
    raise TypeError(
        f"Incompatible extent kinds: {a!r} vs {b!r} "
        "(cannot mix uniform and jagged)"
    )


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    extents: tuple[Extent, ...]

    def __init__(self, *extents: Extent):
        object.__setattr__(self, "extents", tuple(extents))

    @property
    def rank(self) -> int:
        return len(self.extents)

    def resolve(self, other: Shape) -> Shape:
        if self.rank != other.rank:
            raise TypeError(
                f"Shape rank mismatch: {self.rank} vs {other.rank}"
            )
        resolved = tuple(a.resolve(b) for a, b in zip(self.extents, other.extents))
        return Shape(*resolved)

    def is_compatible(self, other: Shape) -> bool:
        try:
            self.resolve(other)
            return True
        except TypeError:
            return False

    def static_size(self) -> int | None:
        """Return total element count if all extents are static, else None."""
        prod = 1
        for e in self.extents:
            if not isinstance(e, Static):
                return None
            prod *= e.n
        return prod

    def __repr__(self) -> str:
        inner = ", ".join(repr(e) for e in self.extents)
        return f"({inner})"


# ---------------------------------------------------------------------------
# Scalar types
# ---------------------------------------------------------------------------

class ScalarType(enum.Enum):
    F32 = "f32"
    F16 = "f16"
    I32 = "i32"
    I64 = "i64"
    BOOL = "bool"

    def __repr__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Type  (iteration_shape, element_type)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Function type (for first-class function values)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FnType:
    """Type of a function value (verb or composed adverb).

    Carried in a Type with rank-0 leading shape: () / FnType(...).
    FnType is polymorphic -- it does not carry input/output type signatures.
    The output type is determined at application time from the concrete
    input types, via the FnValue's type_fn callable.
    """

    arity: int  # 1 (monadic) or 2 (dyadic)
    name: str = ""  # for debugging/display

    def __repr__(self) -> str:
        label = self.name or "fn"
        return f"Fn({label}/{self.arity})"


# Element type is a ScalarType (leaf), another Type (nested iterable),
# or a FnType (first-class function value).
ElementType = Union[ScalarType, "Type", FnType]


@dataclass(frozen=True)
class Type:
    """Logical type: iteration shape + element type."""

    iteration_shape: Shape
    element_type: ElementType

    @property
    def rank(self) -> int:
        return self.iteration_shape.rank

    @property
    def is_scalar_element(self) -> bool:
        return isinstance(self.element_type, ScalarType)

    def __repr__(self) -> str:
        return f"{self.iteration_shape} over {self.element_type!r}"


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def tensor_type(*extents: int | str, stype: ScalarType = ScalarType.F32) -> Type:
    """Build a tensor type from integer extents (static) or '*'/'~' (dynamic/jagged)."""

    def _parse_extent(e) -> Extent:
        if isinstance(e, int):
            return Static(e)
        if e == "*":
            return Dynamic()
        if e == "~":
            return Jagged()
        raise ValueError(f"Unknown extent specifier: {e!r}")

    shape = Shape(*[_parse_extent(e) for e in extents])
    return Type(shape, stype)


def coord_type(rank: int) -> Type:
    """Type of a coordinate into a rank-r iteration space: (r,) i32."""
    return Type(Shape(Static(rank)), ScalarType.I32)
