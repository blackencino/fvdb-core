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

## Pipeline Execution Model

- Segment boundaries are explicit planning artifacts:
  - `cutile` -- tile-parallel work fusible into a single @ct.kernel launch.
  - `collective` -- operations requiring cross-thread coordination (Sort,
    Unique, Where, Over), dispatched to torch GPU ops with a synchronization
    barrier between kernel launches.
- Both segment kinds target GPU execution.
- A single pipeline interface executes all segments.
- Segment execution must preserve value semantics at boundaries.
