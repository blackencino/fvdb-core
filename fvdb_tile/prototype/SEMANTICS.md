# Prototype Semantic Contract

This document locks the semantic contract for the prototype DSL and the
`conv_grid` pipeline work.

## Core Principles

- APL/J/K inspiration (especially K) applies at the level of small, orthogonal
  value transformations.
- This DSL does **not** inherit leading-axis/rank-operator semantics directly;
  it uses explicit nested layouts that separate:
  - iteration space, and
  - element type.
- Operations are value-semantic, immutable, and referentially transparent.
- Correctness is prioritized over ergonomics; syntax sugar is deferred.

## Operation Laws (Prototype Expectations)

- **Sort**
  - deterministic stable ascending order over leading axis;
  - preserves element multiplicity exactly;
  - does not mutate inputs.
- **Unique**
  - idempotent: `Unique(Unique(x)) == Unique(x)`;
  - output is a subset of input elements with first-occurrence retention after
    sorting in current prototype path;
  - does not mutate inputs.
- **Pipeline execution**
  - segment boundaries are explicit planning artifacts (`cutile`, `collective`);
  - a single pipeline interface executes all segments;
  - segment execution must preserve value semantics at boundaries.

## `conv_grid` Contract (Prototype)

Given active input voxel coordinates `x`, kernel offsets `k`, and stride `s`,
an output coordinate `y` is active iff:

`x = y * s + k` component-wise for some active `x` and offset `k`.

Operationally:

1. Expand candidate outputs from active coords and kernel offsets.
2. Deduplicate candidate outputs (`Sort` then `Unique` in the prototype path).
3. Materialize output CIG as an ephemeral lowering artifact.

The semantic output is the deduplicated coordinate set. CIG is execution-time
materialization only; it is not the source of semantic truth.
