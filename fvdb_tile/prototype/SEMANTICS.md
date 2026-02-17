# Prototype Semantic Contract

This document locks the algebraic laws and semantic contracts that the prototype
tests enforce.  Implementations may change freely as long as these contracts
hold.

## Core Principles

- APL/J/K inspiration (especially K) applies at the level of small, orthogonal
  value transformations.
- This DSL does **not** inherit leading-axis/rank-operator semantics directly;
  it uses explicit nested layouts that separate:
  - iteration space, and
  - element type.
- Operations are value-semantic, immutable, and referentially transparent.
- Correctness is prioritized over ergonomics; syntax sugar is deferred.

## Operation Laws

- **Sort**
  - deterministic stable ascending order over leading axis;
  - preserves element multiplicity exactly;
  - does not mutate inputs.
- **Unique**
  - idempotent: `Unique(Unique(x)) == Unique(x)`;
  - output is a subset of input elements with first-occurrence retention after
    sorting in current prototype path;
  - does not mutate inputs.

## Leading Shape Theory

- A type is a recursive nesting terminated by a scalar: `S_1 / S_2 / ... / scalar`.
- The leading shape `S_1` is what operations iterate over.
- The inner type (everything after the first `/`) is what operations receive as elements.
- The nesting structure is a property of the value, not of the operation.
- `cut` deepens nesting (splits the leading shape).
- `reshape` reorganizes within a nesting level.
- Adverbs (Over, EachRight, EachLeft, etc.) always operate on the leading shape.
- To operate at a deeper level, compose adverbs: `Each(Each(f))`.

## Adverb Laws

- **EachRight(f)(x, y: S / E)** = `S / f(x, E)` -- x whole, iterate y.
- **EachLeft(f)(x: S / E, y)** = `S / f(E, y)` -- iterate x, y whole.
- **Composition**:
  - `EachRight(EachLeft(f))(x: S_x / A, y: S_y / B)` = `S_y / (S_x / f(A, B))`
  - `EachLeft(EachRight(f))(x: S_x / A, y: S_y / B)` = `S_x / (S_y / f(A, B))`
- **Over(f)(x: S / E)** = `E` -- consumes the full leading shape.
- **Jaggedness**: Each, EachRight, EachLeft promote Dynamic to Jagged in
  the result's inner type (each element independently determines its size).

## Function Values

- Verbs (Add, Sub, Mul, etc.) are first-class function values (`FnValue`).
- Adverbs take a function and return a new function: `EachLeft(Add)` -> FnValue.
- `Apply(fn, args...)` applies a function value to data.
- `EachLeft(Add, x, y)` is syntactic sugar for `Apply(EachLeft(Add), x, y)`.
- Functions are polymorphic; output types are determined at application time.

## Pipeline Execution Model

- Segment boundaries are explicit planning artifacts:
  - `cutile` -- tile-parallel work fusible into a single @ct.kernel launch.
  - `collective` -- operations requiring cross-thread coordination (Sort,
    Unique, Where, Over), dispatched to torch GPU ops with a synchronization
    barrier between kernel launches.
- Both segment kinds target GPU execution.
- A single pipeline interface executes all segments.
- Segment execution must preserve value semantics at boundaries.
