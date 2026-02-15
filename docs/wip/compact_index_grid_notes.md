# Compact Index Grid -- Design Notes

## Thesis

**The problem.** ML has succeeded in large part because tensors are fungible:
a multidimensional array of a single scalar type is trivially portable across
frameworks, languages, and hardware. Adapting a problem to a tensor
representation is a kind of least-common-denominator type erasure -- it
discards structural information in exchange for universal interoperability.

But tensors on their own are too simple. A mesh stored as `(N, 3) f32` has
lost its topology. A sparse grid stored as flat arrays has lost its hierarchy.
A scene graph stored as a collection of typed objects (Alembic, OpenUSD) has
structure, but that structure is "objectified" -- expressed through
inheritance and framework-specific APIs rather than through composable data
descriptions. Simulation frameworks decompose giant fields into shards of
objects grouped by scene graphs. In every case, the assembly of tensors into
richer structures -- and the operations over those structures -- remains
non-portable, tied to particular implementation code and specific languages.

fVDB wraps NanoVDB as an opaque inner type and provides tensor-framework-style
operations over sparse voxel grids. But it is only as portable as its compiled
C++ components. The opacity of the inner type impedes deconstruction,
analysis, fusion, and optimisation. The same is true of any system that hides
structure behind a compiled API.

**The claim.** A small set of **nested layouts** (Cut, Indexed, Jagged,
Struct, Flip) applied over raw tensors, combined with a small set of
**adverbs** (Map, Each, Where, Gather), form a **portable semantic layer**
that has the same degree of fungibility as tensors but captures the structural
richness that tensors erase.

The physical backing is always tensors. The layouts are metadata -- they
describe how to traverse, partition, and associate the underlying tensor data
without moving it. This means:

- Any system that can store tensors can store the layouts (they're just
  additional small tensors: offsets arrays, index arrays).
- The layouts are already implicit in most tensor-based systems. A mesh
  stored as `(N,3) f32` vertices + `(F,3) i32` face indices has implicitly
  applied `Cut(-3, vertices) + Indexed(faces, vertices)`. The framework
  makes this explicit and portable.
- Every organizational structure needed to represent a geometric scene --
  triangle meshes, variable-topology polygons, sparse grids, point clouds,
  face/vertex/varying attributes (RiSpec), hierarchical LODs -- can be
  expressed as nested layouts over tensors.

**The goal.** A rigorously defined, complete, composable type protocol that
is easy to spread. If adopted, it replaces opaque wrappers (fVDB's GridBatch,
USD's schema types, Alembic's typed object graph) with transparent,
analyzable, optimizable descriptions that lower to GPU tensor operations.

**Key technical insight.** Nested layouts serve the role of **Halide's
schedule** -- they describe how to traverse the data, while operations
describe what to compute. APL/J/K conflate data shape with iteration
structure; this system **decouples** them via the iteration-space /
element-type separation. This decoupling is what makes it possible to write
domain algorithms as tiny compositions of adverbs and layouts:

The entire SPH density calculation can be written as `(R/P/:)\:':` in K9
syntax -- a tacit composition of a reduction, a product, and structured
iteration adverbs over structured tensors. This is what the framework makes
possible: domain algorithms as generic compositions, not framework-specific
imperative code.

**North star.** Compile compact adverbial expressions into working cuTile /
CUDA code via algebraic optimisation and idiom detection.

**Open questions.** (1) Can the protocol surface stay small enough for
incremental adoption? The saving grace: layouts are metadata over tensors, so
fallback to "just send the tensors" is always available. (2) Does the
abstract description compile to competitive GPU code? Deferred to future
work, but the idiom-detection path is promising.

---

## Background: NanoVDB and fVDB

### NanoVDB OnIndexGrid

Maps `(batchIdx, i, j, k)` to a linear index into an external feature tensor.
Four levels, fixed 3-4-5 configuration:

| Level | Name  | Grid         | Entries | Voxel span |
|-------|-------|--------------|---------|------------|
| 3     | Root  | Variable     | N       | Unbounded  |
| 2     | Upper | 32x32x32     | 32,768  | 4096^3     |
| 1     | Lower | 16x16x16     | 4,096   | 128^3      |
| 0     | Leaf  | 8x8x8 bitmap | 512     | 8^3        |

Internal entries: child pointer or empty. Leaf entries: linear index (via
`mOffset + popcount`) or empty. Nodes at each level stored in contiguous
arrays. fVDB uses none of the tile values, statistics, or world transforms --
only the sparse `(i,j,k) -> index` map.

### Houdstooth Compact Index Grid (Prior Art)

An earlier implementation. Three levels (no Root), same 5-4-3 dimensions.
**Named_scag**: struct-of-arrays with compile-time named channels (not a
tensor -- heterogeneous types, but iterable with a single known length).
**Level_shape**: generic across levels -- flat child-index array (-1 =
empty), per-node end-offsets (the jagged structure), inverse mapping.
Cleanly separates shape (structural index data) from data (per-voxel values).

---

## Vocabulary

A **scalar type** (`stype`): `f32`, `f16`, `i32`, `i64`, etc.

A **scalar**: a single value of some stype. Rank 0.

A **tensor**: a hyperrectangular, densely-indexed container of a single stype.
Has a **rank** (number of axes) and a **shape** (tuple of axis lengths).

An **iterable**: anything with a known length whose elements may be scalars,
tensors, or other iterables. A tensor is a special case. Our structure is a
hierarchy of iterables with tensors only at the leaves.

**Rank**: number of axes. Usually statically known. Rank-agnostic operators
exist but are the exception.

**Extent**: length of a single axis. Three kinds:

| Kind | Notation | Meaning | Representation |
|------|----------|---------|----------------|
| Static | `n` (integer) | Compile-time constant | Nothing |
| Dynamic | `*` | Uniform, unknown until runtime | One integer |
| Jagged | `~` | Per-parent-element, varies | Offsets array |

Refinement lattice: static < dynamic < jagged. The representations differ,
so the type system keeps them distinct.

**Compatibility** (for flip, elementwise ops, etc.):
- `n + n` = `n` (must match; mismatch is type error)
- `n + *` = `n` (static wins)
- `* + *` = `*`
- `~ + ~` = `~` (offsets must agree at runtime)
- `n + ~` or `* + ~` = **type error** (uniform vs. non-uniform mismatch)

**Indexing rule**: an object of rank `r` requires an index of rank `r` to
produce an element. No partial indexing. A rank-3 tensor requires a rank-3
index and yields a scalar.

**Coordinate convention**: an index into a rank-`r` iteration space is
`(r,) i32`. For rank 1, a scalar `i32` is the degenerate case.

**Elementwise rule**: binary ops require identical iteration shapes. No
broadcasting -- reshape beforehand.

**Logical vs. physical**: every object has a **logical type** (iteration shape
+ element type) and a **physical storage** (actual tensors). For raw tensors
these coincide. For compound structures they diverge. The algebra operates at
the logical level; lowering to physical storage is a compilation pass.

**Shape notation**: `(*, ~, 3)` = outer dynamic, middle jagged, inner
static 3.

### Shorthand vs. C++ type_traits

```
Shorthand                    C++ (type_traits style)
---------                    ----------------------
i32                          Scalar<int32_t>
(8, 8, 8) over i32          Iter<Shape<S<8>, S<8>, S<8>>, Scalar<int32_t>>
(*) over (3) over i32       Iter<Shape<Dyn>, Iter<Shape<S<3>>, Scalar<int32_t>>>
(*) over (~) over (3) i32   Iter<Shape<Dyn>, Iter<Shape<Jag>, Iter<Shape<S<3>>, Scalar<int32_t>>>>
```

```cpp
template<int N> struct S {};       // Static<N>
struct Dyn {};                     // Dynamic
struct Jag {};                     // Jagged
template<typename... Es> struct Shape {};
template<typename ShapeT, typename ElemT> struct Iter {};
template<typename T> struct Scalar { using stype = T; };

// Extent resolution via trait specialisation
template<typename A, typename B> struct Resolve;
template<int N>         struct Resolve<S<N>, S<N>> { using type = S<N>; };
template<int N>         struct Resolve<S<N>, Dyn>  { using type = S<N>; };
template<int N>         struct Resolve<Dyn, S<N>>  { using type = S<N>; };
template<>              struct Resolve<Dyn, Dyn>   { using type = Dyn; };
template<>              struct Resolve<Jag, Jag>   { using type = Jag; };
// All other combinations: static_assert failure (type error)
```

---

## Layout Wrappers

Every layout wrapper is a **type modifier**, not an operation. It reinterprets
the logical type without touching physical storage. Layouts compose freely.

**Invariant: type transformations do no computational work.** A layout never
allocates, copies, or gathers. If a transformation requires data movement, it
is an **operation**. A compiler pass may insert operations as a policy
decision, but the type system itself only recognises eligibility.

### Cut

Splits the leading axis into outer (new iteration) and inner (element type).

| Mode | Spec | Result iteration | Result element |
|------|------|-----------------|----------------|
| By count | `cut(n, x)` | `(n,)` | `(D/n, ...) over E` |
| By size | `cut(-s, x)` | `(D/s,)` | `(s, ...) over E` |
| By offsets | `cut(offs, x)` | `(*,)` | `(~, ...) over E` |

### Indexed

Associates an **indexer** with a **target**:
- Result iteration space = indexer's iteration space.
- Result element type = target's element type.
- Constraint: indexer element type matches target iteration rank.

As a layout: pure type-level, no work. As an operation (Gather): materialises.

### Tuple

Ordered group of sub-layouts, no shape constraints.
- Iteration space: rank 1, length = number of children.
- Element type: heterogeneous (K's generic list).

### Struct

Like tuple with **named fields** (static labels in the type system).
Definition-ordered. K parallel: struct = dict.

### Flip

Applied to tuple/struct with compatible iteration spaces. Transposes
"collection of arrays" into "array of collections." K parallel:
flip(dict) = table.

### Jagged

Cut-by-offsets with proper extent notation. Outer axis `*`, inner axis `~`.

---

## Operations and Adverbs

Operations do computational work. Adverbs modify how an operation is applied
across an iteration space.

**Key rule: adverbs operate over the full iteration space.** No `axis`
parameter. The layout IS the axis specification. To reduce along a specific
axis, Cut first to isolate it as the iteration space.

### Adverbs (K-style)

**Parallel adverbs** (order-independent, any rank):

| Adverb | Syntax | Input | Output |
|--------|--------|-------|--------|
| Over | `Over(f, xs)` | `S over E` | `E` (full reduction) |
| Each | `Each(xs, x => body)` | `S over E` | `S over R` |
| EachRight | `EachRight(f, x, ys)` | fixed x, iterate ys | `S_ys over R` |
| EachLeft | `EachLeft(f, xs, y)` | iterate xs, fixed y | `S_xs over R` |

**Sequential adverbs** (require rank 1, inherently ordered):

| Adverb | Syntax | Input | Output |
|--------|--------|-------|--------|
| Scan | `Scan(f, xs)` | `(D,) over E` | `(D,) over E` (running accumulation) |
| Prior | `Prior(f, xs)` | `(D,) over E` | `(D-1,) over R` (adjacent pairs) |

**Each promotes Dynamic to Jagged**: the body runs independently per element,
so any `*` in the result becomes `~`. Static extents are preserved.

**Over reduces the full iteration space** to one element. For commutative +
associative operations (Add, Mul, Min, Max), this is well-defined regardless
of rank and maps to GPU parallel tree reduction. For non-commutative ops,
require rank 1.

**Mean is not a primitive** -- it is composed: `Div(Over(Add, xs), Count(xs))`.
Example: face centroids = `Each(faces, f => Div(Over(Add, verts_of_f), Const(3)))`.

### Core Operations

- **Map(xs, x => body)**: scalar function per element. Preserves iteration.
- **Where(xs)**: coords of truthy elements. `S over bool` => `(*,) over (r,) i32`.
- **Gather(target, indexer)**: materialise an Indexed layout.

### Scalar Primitives

`Add`, `Sub`, `Mul`, `Div`, `GE`, `And`, `Not`, `InBounds`, `Count`.

### Domain Primitives

- **Decompose(coord, bit_widths)**: bitfield coordinate split.
- **Morton3d(coord)**: 3D morton encoding.

### Summary

| Name | Kind | New data? |
|------|------|-----------|
| Cut, Indexed, Tuple, Struct, Flip, Jagged | Layout | No |
| Map, Each, Over, Scan, EachRight, EachLeft, Prior | Adverb/Operation | Yes |
| Where | Operation | Yes (data-dependent) |
| Gather | Operation | Yes (materialise) |
| Add, Sub, Mul, Div, GE, And, Not, InBounds, Count | Scalar primitive | Yes |
| Decompose, Morton3d | Domain primitive | Yes |

---

## Worked Example: Single-Leaf Neighbors

```
leaf     = Reshape(leaf_raw, (8,8,8))          -- (8,8,8) i32          [layout]
features = Cut(-C, features_raw)               -- (*,) over (C,) f32   [layout]
mask     = Map(leaf, x >= 0)                   -- (8,8,8) bool
active   = Where(mask)                         -- (*,) over (3,) i32
idx      = Gather(leaf, active)                -- (*,) over i32
feat     = Gather(features, idx)               -- (*,) over (C,) f32
offsets  = Cut(-3, offsets_raw)                -- (6,) over (3,) i32   [layout]
nbrs     = Each(active, a => Map(offsets, o => Add(a, o)))
                                               -- (*,) over (6,) over (3,) i32
filtered = Each(nbrs, cs =>
             Gather(cs, Where(Map(cs, c =>
               And(InBounds(c, 0, 8), Gather(mask, c))))))
                                               -- (*,) over (~,) over (3,) i32
```

The `~` emerges from Each: the inner Where produces a variable number of
coordinates per voxel. Sentinels (-1) and bounds stay at the value level.

---

## Worked Example: Triangle Mesh

An OBJ-style indexed triangle mesh as layouts over two raw tensors:

```
positions_raw: (V * 3,) f32       faces_raw: (F * 3,) i32
vertices = Cut(-3, positions_raw)           -- (V,) over (3,) f32   [layout]
face_idx = Cut(-3, faces_raw)               -- (F,) over (3,) i32   [layout]
tris = Each(face_idx, f => Map(f, i => Gather(vertices, i)))
                                            -- (F,) over (3,) over (3,) f32
```

No custom mesh class. The type `(F,) over (3,) over (3,) f32` fully
describes "F triangles of 3 vertices of 3D positions."

**Type system finding:** `Indexed(face_idx, vertices)` is correctly rejected.
face_idx's element `(3,) i32` is three separate vertex indices, not one
rank-3 coordinate. The type system distinguishes "one multi-dimensional
lookup" from "multiple scalar lookups." The correct expression is
`Each(face_idx, f => Map(f, i => Gather(vertices, i)))` -- for each face,
look up each vertex index separately.

### Materialisation is a scheduling decision

The same algorithm (e.g. face centroids) can be executed with different
materialisation strategies:

**Early materialisation:** Gather all triangle vertices upfront into a dense
`(F, 3, 3) f32` tensor, then compute centroids. Costs memory (repeated
vertices at shared edges), gains data locality.

**Late materialisation:** For each face, gather vertex positions inline
through the index buffer, then compute centroid. No intermediate `(F,3,3)`
tensor exists. Costs indirection, saves memory.

Both produce identical results. The algorithm is the same; the schedule
(when to resolve Indexed to Gather) differs.

This is the Halide analogy made concrete, and it connects directly to a
key fVDB question: should convolution operate on fully-materialised dense
leaf blocks (perfect data coalescing, aligned to thread blocks) or gather
through the sparse index on the fly (saves memory when sparsity is high)?
The answer depends on the sparsity ratio and is problem-specific. The
framework should let you express BOTH paths and choose -- or let an
optimiser choose based on measured characteristics.

---

## Prototype Results

Code in `docs/wip/prototype/`. Run all: `python docs/wip/prototype/run_all_tests.py`.

### v0: Fundamentals

Map, Where, Gather, Each over a single leaf. Type propagation works
end-to-end. Jagged emerges automatically. Layouts and operations cleanly
separated. Sentinels stay out of the type system.

### v1: Composition

Multiple leaves via Cut + Each (double-nested jagged). Indexed layout
predicts Gather type. Struct + Flip composes multi-feature voxels into
`(*) over Struct({pos: (3) f32, color: (3) f32, dens: f32})`.

### v2: Hierarchical Chain

Two-level grid with Decompose + chained Gather. Morton-linearized nodes
produce identical results (swappable indexing strategy). Hierarchical lookup
composes with Each and Where.

### v3: Micro DSL

Programs as text strings, parsed into typed AST, type-checked without data,
executed against numpy. ~15 keywords. `v => body` binding syntax replaces
lambdas. All v0-v2 scenarios expressible. Each correctly promotes `*` to `~`
at type-check time.

### Mesh Exemplar

Triangle mesh as layouts over two raw tensors. Type system correctly rejects
`Indexed(face_idx, vertices)` (mesh indexing is multiple scalar lookups, not
one multi-dimensional lookup). Early vs. late materialisation of face
centroids produce identical results, demonstrating that materialisation
schedule is independent of algorithm. DSL expresses the mesh lookup as a
string program.

### What remains unproven

- Cross-leaf neighbor queries (compose neighbor pattern with hierarchical chain).
- AST optimisation or transformation.
- Code generation from AST to GPU kernels.
- Materialisation scheduling as an automated decision (currently manual).
- Adoption/virality of the protocol.

---

## Working Theory

**Claim**: nested layouts + rank-matched adverbs can fully specify structured
geometric and scientific computations -- sparse grids, meshes, point clouds,
scene graphs -- in a form that is:

- **Portable**: backed by tensors, described by metadata. Any system that can
  store tensors can store and exchange the full structure.
- **Analyzable**: the typed AST can be inspected, transformed, and optimised
  before execution.
- **Compilable**: types carry enough information to lower to concrete GPU
  code via idiom detection and algebraic transformation.

fVDB's sparse voxel operations are the proving ground, not the ceiling.

### Exploration Roadmap

1. **Single leaf** -- done (v0). Map, Each, Where, Gather, jagged emergence.
2. **Multiple leaf nodes** -- done (v1). Cut + Each, double nesting.
3. **Indexed / Struct / Flip** -- done (v1). Layout prediction, FlipStruct.
4. **Hierarchical chain** -- done (v2). Decompose, chained Gather, morton.
5. **Micro DSL** -- done (v3). String -> parse -> type-check -> execute.
6. **Mesh exemplar** -- done. Early/late materialisation, scheduling insight.
7. **Full CIG type**: struct of three levels, lookup as composed function.
8. **Cross-leaf neighbors**: compose neighbor pattern with hierarchical chain.
9. **Target expression**: jagged neighbor indices across two CIGs.
10. **Materialisation scheduling**: automated early/late decisions.
11. **AST optimisation**: idiom detection, fusion, algebraic simplification.
12. **Code generation**: lower DSL to cuTile / CUDA.
13. **Protocol definition**: formal specification for interchange.
