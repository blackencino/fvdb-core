# Compact Index Grid -- Design Notes

## Design Goals

A **type system** with **compact notation** (`n`, `*`, `~`) that:

1. Fully describes iteration spaces and element types of compound structures.
2. Supports **adverbial application** of operators: functions declare what
   iteration shape they consume; the type system determines how to decompose
   the argument's iteration space to feed them. Replaces APL/J's rank operator
   with explicit pre-shaped iteration spaces.
3. Enables **automatic code generation**: types carry enough information to
   lower to concrete loops, gathers, and dispatches in a target language.

**Key insight**: nested layouts serve the role of **Halide's schedule** --
they describe how to traverse the data, while operations describe what to
compute. APL/J/K conflate data shape with iteration structure; this system
**decouples** them via the iteration-space / element-type separation.

**North star**: compile compact adverbial expressions into working cuTile /
CUDA code via algebraic optimisation and idiom detection.

---

## NanoVDB OnIndexGrid Recap

Maps `(batchIdx, i, j, k)` to a linear index into an external feature tensor.

Four levels, fixed 3-4-5 configuration:

| Level | Name  | Grid         | Entries | Voxel span |
|-------|-------|--------------|---------|------------|
| 3     | Root  | Variable     | N       | Unbounded  |
| 2     | Upper | 32x32x32     | 32,768  | 4096^3     |
| 1     | Lower | 16x16x16     | 4,096   | 128^3      |
| 0     | Leaf  | 8x8x8 bitmap | 512     | 8^3        |

Internal entries: child pointer or empty. Leaf entries: linear index (via
`mOffset + popcount`) or empty. Nodes at each level stored in contiguous arrays.

fVDB uses none of the tile values, statistics, or world transforms -- only the
sparse `(i,j,k) -> index` map.

---

## Houdstooth Compact Index Grid (Prior Art)

An earlier implementation in the Houdstooth framework. Three levels (no Root),
same 5-4-3 node dimensions as NanoVDB.

**Named_scag**: struct-of-arrays with compile-time named channels. All channels
share the same length (rank-1) but each can have a different scalar type. Names
are type-tag tuples. Not a tensor (heterogeneous types), but iterable with a
single known length.

**Level_shape** (generic across all three levels):
- `blocks`: flat array of child indices (int32, -1 = empty). All nodes at one
  level concatenated. No bitmasks -- -1 is the emptiness sentinel.
- `blocks_meta`: per-node metadata -- parent identity, children end-offsets
  (jagged boundaries), voxel bounds.
- `elements`: per-element inverse mapping -- parent index, local offset within
  parent node.

End-offsets in `blocks_meta` are the jagged structure, equivalent to our
offsets array stored per-node rather than standalone.

**Shape/data split**: cleanly separates `Compact_index_grid_shape` (structural
index data) from `Compact_index_grid_data` (per-voxel values, min/max). We
want to preserve this separation.

---

## Vocabulary

A **scalar type** (`stype`): `f32`, `f16`, `i32`, `i64`, etc.

A **scalar**: a single value of some stype. Rank 0.

A **tensor**: a hyperrectangular, densely-indexed container of a single stype.
Has a **rank** (number of axes) and a **shape** (tuple of axis lengths).
A rank-0 tensor is a scalar.

An **iterable**: anything with a known length. Its elements may be scalars,
tensors, or other iterables. A tensor is a special case of iterable.

**Rank**: the number of axes. Usually statically known (in domain-specific
code, always). Rank-agnostic operators (e.g. elementwise add) exist but are
the exception.

**Extent**: the length of a single axis. Three kinds (see Axis Extent Kinds).

**Indexing rule**: an object of rank `r` requires an index of rank `r` to
produce an element. No partial indexing, no peeling off leading dimensions.
A rank-3 tensor requires a rank-3 index and yields a scalar.

**Coordinate convention**: an index into a rank-`r` iteration space is
represented as `(r,) i32` -- a static-length-r vector of integers. For rank 1,
a scalar `i32` is the degenerate case.

**Elementwise rule**: binary operations (e.g. add) require operands with
identical iteration shapes. No broadcasting -- reshape beforehand if needed.

**Logical vs. physical**: every object has a **logical type** (iteration shape
+ element type -- what the algebra sees) and a **physical storage** (the actual
tensors backing it -- what the GPU sees). For a raw tensor these coincide. For
compound structures they diverge. The algebra operates at the logical level;
lowering to physical storage is a separate concern. The boundary between them
is more like a compilation pass than two separate worlds.

---

## Axis Extent Kinds

Three structurally distinct kinds, with different physical representations:

| Kind | Notation | Meaning | Representation |
|------|----------|---------|----------------|
| Static | `n` (integer) | Compile-time constant | Nothing |
| Dynamic | `*` | Uniform, unknown until runtime | One integer |
| Jagged | `~` | Per-parent-element, varies | Offsets array |

Refinement lattice: static < dynamic < jagged in generality. Any static axis
could be described as dynamic; any uniform dynamic as trivially jagged. But the
representations differ, so the type system keeps them distinct.

**Shape notation**: `(*, ~, 3)` = outer dynamic, middle jagged, inner static 3.

**Compatibility** (for flip, elementwise ops, etc.):
- `n + n` = `n` (must match; mismatch is type error)
- `n + *` = `n` (static wins)
- `* + *` = `*`
- `~ + ~` = `~` (offsets must agree at runtime)
- `n + ~` or `* + ~` = **type error** (uniform vs. non-uniform mismatch)

---

## Layout Wrappers

Every layout wrapper is a **type modifier**, not an operation. It reinterprets
the logical type (iteration space + element type) of an underlying object
without touching physical storage. Layouts compose freely.

**Invariant: type transformations do no computational work.** A layout wrapper
never allocates, copies, or gathers. If a transformation requires data
movement (e.g. stacking a tuple into a contiguous tensor), it is an
**operation**, not a layout. A compiler pass may insert such operations as a
policy decision, but the type system itself only recognises eligibility.

### Cut

Consumes the leading axis of an underlying iterable and splits it into an
outer axis (new iteration shape) and an inner axis (pushed into element type).

Given iteration shape `(D, ...)` and element type `E`:

| Mode | Spec | Result iteration | Result element |
|------|------|-----------------|----------------|
| By count | `cut(n, x)` | `(n,)` | `(D/n, ...) over E` |
| By size | `cut(-s, x)` | `(D/s,)` | `(s, ...) over E` |
| By offsets | `cut(offs, x)` | `(*,)` | `(~, ...) over E` |

By count: outer static, inner static (requires divisibility). By size: outer
dynamic, inner static. By offsets: outer dynamic, inner jagged. Cuts compose.

### Indexed

Associates an **indexer** with a **target**:
- Result iteration space = indexer's iteration space.
- Result element type = target's element type.
- Constraint: indexer's element type must be a valid coordinate into the
  target's iteration space (matching rank per the coordinate convention).

**As a layout**: pure type-level association, no work. **As an operation**
(Gather): materialises the result by performing an actual gather.

### Tuple

Wraps an ordered group of sub-layouts into a single iterable. No constraint
on children's shapes.
- Iteration space: rank 1, length = number of children.
- Element type: heterogeneous (K's generic list).

### Struct

Like tuple, but with **named fields**. Each child is associated with a
**symbol** -- a static label in the type system. Definition-ordered. Access
by name (projection) is primary; positional access also works.
K parallel: struct = dict.

### Flip

Applied to a tuple or struct whose children have **compatible** iteration
spaces. Transposes "collection of arrays" into "array of collections."
- Result iteration space: resolved (tightest) shared shape.
- Result element type: tuple/struct of children's element types.

K parallel: flip(dict) = table.

### Jagged

Binds an **offsets** array to an underlying layout, splitting its leading axis
into variable-length segments. The outer axis (segment count) is `*`; the
inner axis (segment length, pushed into the element type) is `~`.

Given underlying with iteration shape `(D, ...)` and element type `E`:
- Result iteration space: `(*,)`.
- Result element type: `(~, ...) over E`.

This is the cut-by-offsets case with proper extent-kind notation. The `~`
lives in the element type, not the iteration space of the jagged itself.

### Other (Deferred)

- Reshape/view/partition variants -- non-jagged redistribution of axes.
- Repeat -- produce arbitrarily shaped iteration spaces.
- Infinite generators (iota, etc.) -- possible but finite-only for now.

---

## Operations and Adverbs

Unlike layouts, **operations** do computational work: they allocate, compute,
and produce new physical storage. **Adverbs** modify how an operation is
applied across an iteration space.

### Map

Applies a scalar function across the full iteration space.
- Input: `S over E`, function `f: E -> E'`.
- Output: `S over E'`.
- Preserves iteration shape, transforms element type.

### Each

Adverb: applies a function to each element of the outer iteration space.
- Input: `(D, ...) over E`, function `f: E -> R`.
- Output: `(D, ...) over R`.
- If `f` returns an iterable, the result nests (element type gains depth).

### Where

Produces the coordinates of truthy elements.
- Input: `S over bool` where `S` has rank `r`.
- Output: `(*,) over (r,) i32`.
- Dynamic output length (data-dependent). Each element is a coordinate vector
  matching the input's iteration rank.

### Gather

Materialises an Indexed layout.
- Input: indexer `I over (r,) i32`, target `T over E` where `T` has rank `r`.
- Output: `I over E`.
- Checks: indexer element length = target iteration rank.

### Summary

| Name | Kind | Produces new data? |
|------|------|--------------------|
| Cut, Indexed, Tuple, Struct, Flip, Jagged | Layout | No |
| Map, Each | Operation/Adverb | Yes |
| Where | Operation | Yes (data-dependent length) |
| Gather | Operation | Yes (materialises Indexed) |

---

## Worked Example: Single-Leaf Neighbors

Starting from the simplest case to ground the primitives.

### Setup

```
leaf_raw:     (512,) i32       -- one leaf node, flat. >= 0 = active, -1 = empty
features_raw: (N, C) f32       -- external feature array

leaf     = Reshape(leaf_raw, (8,8,8))   -- (8,8,8) i32          [layout]
features = Cut(-C, features_raw)        -- (*,) over (C,) f32   [layout]
```

### Active voxels

```
mask       = Map(leaf, \x -> x >= 0)        -- (8,8,8) bool
active_ijk = Where(mask)                    -- (*,) over (3,) i32
active_idx = Gather(leaf, active_ijk)       -- (*,) over i32
active_feat = Gather(features, active_idx)  -- (*,) over (C,) f32
```

`Where` returns rank-3 coordinates as `(3,) i32` because the mask's iteration
space is rank 3. `Gather(leaf, active_ijk)` checks that element length 3
matches leaf's iteration rank 3. `Gather(features, active_idx)` checks that
scalar i32 matches features' iteration rank 1.

### Neighbor coordinates

```
offsets = Reshape(offsets_raw, (6, 3))      -- (6, 3) i32   [the 6 face-neighbor deltas]
offsets = Cut(-3, offsets)                  -- (6,) over (3,) i32   [layout]

neighbor_ijk = Each(active_ijk, \a ->
    Map(offsets, \o -> Add(a, o))
)
-- (*,) over (6,) over (3,) i32
-- For each active voxel: 6 candidate neighbor coordinates.
```

`Each` applies a function per active voxel. The inner `Map` adds the voxel's
coordinate to each of the 6 offsets. The result nests: the element type gains
a `(6,)` layer.

### Filtering to active neighbors (jagged emerges)

```
active_neighbors = Each(neighbor_ijk, \coords6 ->
    let valid = Map(coords6, \c -> Gather(mask, c))  -- (6,) bool
    in  Gather(coords6, Where(valid))                -- (~,) over (3,) i32
)
-- (*,) over (~,) over (3,) i32
```

The inner `Where` produces a variable number of coordinates (0 to 6 per
voxel). This is where `~` appears: the result type is
`(*,) over (~,) over (3,) i32` -- a dynamic number of active voxels, each
with a **jagged** number of active neighbor coordinates.

### Sentinels and bounds

The sentinel value (-1) is handled at the value level: the predicate `x >= 0`
is an ordinary scalar comparison, not a type-level concept. Similarly,
out-of-bounds neighbor coordinates (negative or >= 8) would need value-level
bounds checking in the predicate. The type system does not model sentinels or
bounds -- it models shapes and extent kinds.

---

## Prototype v0: What It Demonstrated

Code in `docs/wip/prototype/`. Types, layouts, ops, two test files.

### The three phases of an expression

Every test has three distinct phases. The lambdas in the Python code blur
these together, but conceptually they are separate:

**1. Setup (pure Python).** Create numpy arrays, wrap them as `Value` objects
with explicit types. This is boilerplate -- it corresponds to "data already
exists on the GPU" in a real system. Nothing interesting happens here.

```
leaf = Value.from_numpy(np.random.randint(-1, 10, (8,8,8)), I32)
offsets = Value(cut_by_size(3, flat_type), offsets_raw)
```

**2. Expression (the algebra).** A composition of operations and layouts that
describes *what* to compute, with types propagating automatically. This is the
part that matters -- it is what the type system sees, what algebraic
optimisation would transform, and what code generation would lower to GPU
kernels. In the prototype it is written inline as Python calls, but it should
be read as an AST, not as imperative code:

```
mask          = Map(leaf, x >= 0)               -- (8,8,8) bool
active_ijk    = Where(mask)                     -- (*) over (3) i32
neighbor_ijk  = Each(active_ijk,                -- (*) over (6) over (3) i32
                  Map(offsets, Add(a, _)))
filtered      = Each(neighbor_ijk,              -- (*) over (~) over (3) i32
                  Gather(_, Where(valid(_))))
```

The lambdas are an artefact of hosting this in Python -- they are the
anonymous functions passed to Each and Map. In the real system these would be
named operations or inline expressions in the compact syntax. The important
thing is the *type signature at each step*, not the Python mechanics.

**3. Extraction (back to Python).** Pull results out of `Value` objects,
compare against brute-force numpy reference. This is test scaffolding, not
part of the algebra.

### What the prototype actually proved

1. **Type propagation works end-to-end.** Starting from `(8,8,8) i32`, the
   types flow correctly through Map, Where, Each, and Gather without manual
   annotation. Each operation's output type is fully determined by its input
   types and the operation's rules.

2. **Jagged (`~`) emerges automatically.** No code explicitly creates a jagged
   type. The `Each` implementation detects that inner results have varying
   lengths (at the data level, not just the type level) and promotes the inner
   extent from `*` to `~`. The final type `(*) over (~) over (3) i32` falls
   out of the machinery.

3. **The extent-kind distinction matters in practice.** During implementation,
   `Each` had to check *actual numpy shapes*, not just type-level shapes, to
   decide between regular stacking and jagged list-of-Values. Two `Dynamic()`
   extents compare as equal at the type level even when their concrete sizes
   differ. This validates the static < dynamic < jagged refinement lattice:
   the type level carries structural intent, not just "unknown size."

4. **Sentinels stay out of the type system.** The -1 empty sentinel is handled
   entirely by value-level predicates (`x >= 0`, bounds checks). The type
   system models shapes and extent kinds, not value semantics. This keeps the
   algebra clean.

5. **Layouts and operations are cleanly separated.** `cut_by_size`, `reshape`,
   `indexed`, `flip` are all pure type functions (no data). `Map`, `Where`,
   `Gather`, `Each` produce new `Value` objects with new data. The line
   between "free reinterpretation" and "actual work" is maintained.

### What the prototype does NOT yet demonstrate

- Multiple leaf nodes (double nesting via Cut).
- Cross-leaf anything (the hierarchical index chain).
- The CIG type itself (struct of three levels).
- Any optimisation or code generation.
- Separation of the expression AST from its execution (currently interleaved).

---

## Working Theory

**Claim**: nested layouts + rank-matched adverbs + elementwise operations can
fully specify the operations currently defined on NanoVDB / fVDB, in a form
that supports:
- Algebraic transformation and optimisation of expressions.
- Idiom detection (e.g. recognising a scatter-gather pattern).
- Performance-oriented code generation (targeting cuTile / CUDA).

### Exploration Roadmap

1. **Single leaf** -- done above. Establishes Map, Each, Where, Gather, and
   shows jagged emergence from filtering.
2. **Multiple leaf nodes**: `Cut(-512, all_leaves)` gives `(*,) over (8,8,8) i32`.
   Batch Each over leaves. First use of double nesting.
3. **Cross-leaf neighbors**: neighbors at leaf boundaries require the lower
   level of the tree. First use of the hierarchical index chain.
4. **Full CIG type**: express as a struct of three levels, each a cut+reshaped
   tensor. Define the lookup as a chain of Indexed layouts.
5. **Target expression**: "produce the jagged set of neighbor indices for each
   active voxel of one CIG against another." The end-to-end test of the
   framework.
