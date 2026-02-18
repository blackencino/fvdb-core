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

**The claim.** A small number of orthogonal concepts -- **nested layouts**,
**leading-shape types**, **verbs**, and **adverbs** -- applied over raw
tensors, form a **portable semantic layer** that has the same degree of
fungibility as tensors but captures the structural richness that tensors
erase.  The minimal conceptual surface:

1. **Nested layouts** (`cut`, `indexed`, `jagged`, `masked`, `fuse`,
   `reshape`, ...) are metadata over tensors.  They describe how to
   traverse, partition, and associate the underlying data without moving it.
   Zero-cost type reinterpretation.

2. **Leading-shape types** (`S_1 / S_2 / ... / scalar`) define what
   operations see as "the iteration space" vs "one element."  The nesting
   structure is a property of the value, not the operation.  Layouts move
   the nesting boundary; the type system tracks the consequences.  `() / i32`
   is the scalar atom -- rank 0, no iterable shape.

3. **Verbs** (`Add`, `Sub`, `Mul`, ...) are scalar function values.
   They operate on `() / scalar` atoms and produce `() / scalar` results.

4. **Adverbs** (`Over`, `Each`, `EachLeft`, `EachRight`, `EachBoth`,
   `Scan`, `Prior`) are function transformers.  They take a verb and return
   a **new function** with a different scheduling pattern.  Adverbs compose:
   `EachLeft(EachRight(EachBoth(Add)))` is a function that, when applied to
   typed inputs, specifies an outer-product-like iteration with componentwise
   addition -- and compiles directly to a GPU kernel with the correct index
   arithmetic.

5. **Structural operations** (`Map`, `Where`, `Gather`) connect layouts
   to computation: Map applies a function over the leading shape, Where
   extracts coordinates of interest, Gather materialises an indexed layout.

The physical backing is always tensors. The layouts are metadata -- they
describe how to traverse, partition, and associate the underlying tensor data
without moving it. This means:

- Any system that can store tensors can store the layouts (they're just
  additional small tensors: offsets arrays, index arrays).
- The layouts are already implicit in most tensor-based systems. A mesh
  stored as `(N,3) f32` vertices + `(F,3) i32` face indices has implicitly
  applied `cut(-3, vertices) + indexed(faces, vertices)`. The framework
  makes this explicit and portable.
- Every organizational structure needed to represent a geometric scene --
  triangle meshes, variable-topology polygons, sparse grids, point clouds,
  face/vertex/varying attributes (RiSpec), hierarchical LODs -- can be
  expressed as nested layouts over tensors.

**The goal.** A rigorously defined, complete, composable type protocol that
is easy to spread. If adopted, it replaces opaque wrappers (fVDB's GridBatch,
USD's schema types, Alembic's typed object graph) with transparent,
analyzable, optimizable descriptions that lower to GPU tensor operations.

**Key technical insight.** The separation into verbs, adverbs, and layouts
serves the role of **Halide's algorithm/schedule split**.  Verbs say *what*
to compute (scalar operations).  Adverbs say *how* to iterate (scheduling
patterns -- outer products, reductions, scans, zips).  Layouts say *what
constitutes one element* (nesting boundaries).  All three compose freely,
and the composition carries enough information for a compiler to emit
concrete GPU code.

APL/J/K conflate data shape with iteration structure; this system
**decouples** them via leading-shape theory: a type is a recursive nesting
`S_1 / S_2 / ... / scalar` where each `/` is a nesting boundary.  Operations
always work on the leading shape `S_1`; the rest is "one element."  Layouts
(`cut`, `fuse`, `reshape`) move the `/` boundary without moving data.
Adverbs compose over verbs to build scheduling specifications that become
concrete only when applied to typed inputs.  This decoupling is what makes
it possible to write domain algorithms as tiny compositions:

The entire SPH density calculation can be written as `(R/P/:)\:':` in K9
syntax -- a tacit composition of a reduction, a product, and structured
iteration over structured tensors. This is what the framework makes possible:
domain algorithms as generic compositions, not framework-specific imperative
code.

**North star.** Compile compact adverbial expressions into working cuTile /
CUDA code via algebraic optimisation and idiom detection.

**Open questions.** (1) Can the protocol surface stay small enough for
incremental adoption? The saving grace: layouts are metadata over tensors, so
fallback to "just send the tensors" is always available. (2) Does the
abstract description compile to competitive GPU code? **Answered (v8):**
the compressed CIG with `masked` layout matches or beats NanoVDB (the
production C++/CUDA sparse grid implementation) on both memory (0.22-0.73x
at typical sparsities) and query performance (1.27-1.34x faster), from a
4-line DSL expression compiled to a cuTile kernel.

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

**Notation.** Types are written as `S_1 / S_2 / ... / scalar`, where `/`
separates nesting levels (see Leading Shape Theory below). The `Type.__repr__`
in Python prints `over` instead of `/`: `(5, 6, 7) over (3, 4) over i32`
means the same as `(5, 6, 7) / (3, 4) / i32`. Both notations appear in this
document; they are interchangeable.

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

**Indexing rule**: a value whose leading shape has rank `r` requires an
index of rank `r` to produce one element (the inner type). No partial
indexing. A value with leading shape `(8, 8, 8)` requires a `(3,) i32`
coordinate and yields its inner type.

**Coordinate convention**: an index into a rank-`r` leading shape is
`(r,) i32`. For rank 1, a scalar `i32` is the degenerate case.

**Elementwise rule**: binary ops require identical leading shapes. No
implicit broadcasting -- use `EachBoth(f)` for explicit zip-iteration over
matching leading shapes, or `EachRight`/`EachLeft` for asymmetric iteration.

**Logical vs. physical**: every object has a **logical type** (recursive
nesting of leading shape / inner type -- see Leading Shape Theory) and a
**physical storage** (actual tensors). For raw tensors these coincide. For
compound structures they diverge. The algebra operates at the logical level;
lowering to physical storage is a compilation pass.

**Shape notation**: `(*, ~, 3)` = outer dynamic, middle jagged, inner
static 3. This describes extents within a single leading shape. The `/`
separator denotes nesting levels: `(5,) / (*, ~, 3) / i32`.

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

## Leading Shape Theory

K's core abstraction is **leading-axis theory**: every value is a list,
operations apply along the leading (outermost) axis, and nesting gives you
depth. To go deeper, you compose adverbs (`f'` = each, `f''` = each-each).

Our system generalises this to **leading-shape theory**: every value has a
**leading shape** (potentially multi-rank), operations apply over the full
leading shape, and nesting gives you depth. The leading shape can be `(8,8,8)`
or `(*)` or `(5,6,7)` -- not just a single-axis list. This is the
generalisation that makes "nested layouts" more than "nested lists."

### Types as recursive nesting

A type is a recursive nesting terminated by a scalar:

```
S_1 / S_2 / ... / S_n / scalar
```

Each `/` is a **nesting boundary**. `S_1` is the **leading shape**.
Everything after the first `/` is the **inner type**. The inner type itself
has the same structure -- its own leading shape, its own inner type. This
recurses until a scalar leaf.

The `Type` dataclass already represents this:

```python
Type(Shape(5,6,7), Type(Shape(3,4), ScalarType.I32))
# is:  (5, 6, 7) / (3, 4) / i32
```

### Why this matters (it is not cosmetic)

**The nesting structure is a property of the value, not of the operation.**
When you `cut` a `(30,) / i32` into `(5,) / (6,) / i32`, you haven't
changed the data -- you've changed what operations see as "one element."
The layout IS the schedule, and the schedule lives in the data's type.

This is the Halide analogy made precise. The same data under different
nesting structures produces different iteration patterns. The algorithm
(what to compute) is separate from the schedule (how to iterate).

### The leading shape rule

Every operation has one rule: **it operates on the leading shape.** The
inner type is what the operation receives as "elements."

| Operation | Input type | Output type | Rule |
|-----------|-----------|-------------|------|
| Over(f) | `S / E` | `E` | Consume leading shape (fold) |
| Each(f) | `S / E` | `S / f(E)` | Preserve leading shape |
| Where | `S / bool` | `(*,) / (rank(S),) i32` | Consume leading shape |
| Gather | `S_t / E`, `S_i / coord` | `S_i / E` | Indexer's shape replaces target's |

No `axis` parameter. To operate on a specific axis, `cut` first to make
it the leading shape. To go deeper, compose adverbs.

### Functions as values, adverbs as function transformers

A **verb** (Add, Sub, Mul, ...) is a function value. An **adverb** (Over,
EachRight, EachLeft, ...) takes a function and returns a **new function**
with different iteration behaviour. Application is always separate:

```
EachLeft(f)          -- adverb applied to f, produces a new function g
g(x, y)              -- g applied to data
EachLeft(f)(x, y)    -- equivalent: two steps in one expression
```

This separation is essential for composition:

```
EachRight(EachLeft(f))   -- a function, not an application
Apply(EachRight(EachLeft(f)), x, y)   -- applying that function
```

### Adverb type rules for dyadic functions

Given a dyadic verb `f`:

- **EachRight(f)(x, y)**: x is passed whole, y is iterated over its
  leading shape. Result type: `S_y / f(T_x, E_y)` where `T_x` is x's
  full type and `E_y` is y's inner type.

- **EachLeft(f)(x, y)**: x is iterated over its leading shape, y is
  passed whole. Result type: `S_x / f(E_x, T_y)` where `E_x` is x's
  inner type and `T_y` is y's full type.

### Nesting composes

```
EachRight(EachLeft(f))(x: S_x / A, y: S_y / B)
  = S_y / EachLeft(f)(S_x / A, B)     -- EachRight iterates y
  = S_y / (S_x / f(A, B))             -- EachLeft iterates x

EachLeft(EachRight(f))(x: S_x / A, y: S_y / B)
  = S_x / EachRight(f)(A, S_y / B)    -- EachLeft iterates x
  = S_x / (S_y / f(A, B))             -- EachRight iterates y
```

Concrete example with `x: (3, 4) / A` and `y: (5, 6, 7) / B`:

- `EachRight(EachLeft(f))(x, y)` produces `(5, 6, 7) / (3, 4) / f(A, B)`
- `EachLeft(EachRight(f))(x, y)` produces `(3, 4) / (5, 6, 7) / f(A, B)`

The adverb nesting order determines the result's nesting order. This
replaces broadcasting: instead of implicit shape-matching rules, the
programmer explicitly controls iteration with adverb composition.

### Jaggedness propagation

When an adverb applies a function independently per element (Each,
EachRight, EachLeft), any Dynamic extent in the result's leading shape
becomes Jagged -- each element could produce a different size. Static
extents are preserved (guaranteed uniform). This is the same rule already
used by Each, extended to all iteration adverbs.

### Relationship to K

| | K | Our system |
|---|---|---|
| **Structure** | Leading axis (rank 1) / rest | Leading shape (any rank) / rest |
| **Depth control** | Implicit nesting of lists | Explicit `cut`, `reshape` |
| **Adverbs** | `f'` = each, `f/:` = each-right | `Each(f)`, `EachRight(f)` |
| **Composition** | `f/:\:` = each-right(each-left) | `EachRight(EachLeft(f))` |
| **Broadcasting** | Atomic extension | None -- use adverb composition |

---

## Notation Convention

**lowercase = free.  PascalCase = work.**

Layouts (type reinterpretations that move no data) use **lowercase** names.
Operations (functions that allocate, compute, or move data) use **PascalCase**
names. When reading a program, if you see lowercase it costs nothing; if you
see PascalCase it does work.

| Category | Convention | Examples |
|----------|-----------|----------|
| Layouts (free) | lowercase | `cut`, `reshape`, `field`, `indexed`, `flip`, `jagged` |
| Operations (work) | PascalCase | `Map`, `Each`, `Over`, `Where`, `Gather`, `Add`, `Decompose` |
| Connectors | PascalCase | `Input`, `Const` |

K-derived adverbs (`Over`, `Scan`, `EachRight`, `EachLeft`, `EachBoth`,
`Prior`) are **function transformers**: they take a verb (function value) and
return a **new function** with different iteration behaviour. Application is
a separate step: `Over(Add, xs)` is syntactic sugar for
`Apply(Over(Add), xs)`. This two-step separation enables composition --
`EachRight(EachLeft(Add))` is a function, not an application. The compiler
recognises reduction, scan, and iteration patterns by matching AST node types,
not by reading case conventions.

---

## Layout Wrappers

Every layout wrapper is a **type modifier**, not an operation. It reinterprets
the logical type without touching physical storage. Layouts compose freely.
**In the DSL, layouts use lowercase names** (`cut`, `reshape`, `field`,
`masked`).

**Invariant: type transformations do no computational work.** A layout never
allocates, copies, or gathers. If a transformation requires data movement, it
is an **operation** (PascalCase). A compiler pass may insert operations as a
policy decision, but the type system itself only recognises eligibility.

### Cut

Deepens nesting by splitting the outermost extent of the leading shape.
Introduces a new `/` boundary: the outer portion becomes the new leading
shape, the inner portion joins the inner type.

| Mode | Spec | New leading shape | New inner type |
|------|------|-------------------|----------------|
| By count | `cut(n, x)` | `(n,)` | `(D/n, ...) / E` |
| By size | `cut(-s, x)` | `(D/s,)` | `(s, ...) / E` |
| By offsets | `cut(offs, x)` | `(*,)` | `(~, ...) / E` |

### Indexed

Associates an **indexer** with a **target**:
- Result leading shape = indexer's leading shape.
- Result inner type = target's inner type.
- Constraint: indexer's inner type must be a coordinate matching the target's
  leading shape rank.

As a layout (`indexed`): pure type-level, no work. As an operation (`Gather`): materialises.

### Tuple

Ordered group of sub-layouts, no shape constraints.
- Leading shape: rank 1, length = number of children.
- Inner type: heterogeneous (K's generic list).

### Struct

Like tuple with **named fields** (static labels in the type system).
Definition-ordered. K parallel: struct = dict.

### Flip

Applied to tuple/struct with compatible leading shapes. Transposes
"collection of arrays" into "array of collections." K parallel:
flip(dict) = table.

### Jagged

Cut-by-offsets with proper extent notation. Outer axis `*`, inner axis `~`.

### Masked

Fixed-shape space with **sparse occupancy**. The counterpart of `jagged`
(variable-length segments): where `jagged` handles "how many elements per
group?", `masked` handles "which positions in a fixed block have data?"

- Leading shape: from the mask shape (e.g. `(8,8,8)` for a leaf block).
- Inner type: from the flat data array.
- Access: bitmask check + **popcount** for dense index. `masked(mask, offset)`
  wraps a packed bitmask and a base offset into a flat data array. Gather
  through a masked layout computes `offset + popcount(mask, position)` if the
  bit is set, else returns sentinel.
- Physical storage: packed bitmask tensor + base offset scalar/tensor + flat
  data tensor. For an (8,8,8) leaf: 512 bits = 8 x u64 words = 64 bytes.
- **This is the layout that makes NanoVDB-competitive CIG memory possible.**
  NanoVDB's `LeafData<ValueIndex>` uses the same structure (64-byte mask +
  `mValueOff` + `countOn(i)` popcount). The `masked` layout exposes the same
  mechanism as a first-class, transparent, composable type-system concept.

### Fuse

Merge the two outermost nesting levels into one. The inverse of `cut`.

`fuse(S_1 / S_2 / E)` = `(S_1 ++ S_2) / E` -- shape concatenation.

No data movement. Constraint: the inner leading shape must not contain
Jagged extents (jagged means "varies per parent" and cannot be fused into
a uniform shape). Dynamic extents are fine.

### Flatten

Recursively fuse ALL nesting levels into a single leading shape.

`flatten(S_1 / S_2 / ... / S_n / scalar)` = `(S_1 ++ ... ++ S_n) / scalar`.

Same jagged constraint as fuse at each level.

### Permute

Reorder axes within the leading shape. Does not cross nesting boundaries.

`permute((a, b, c) / E, [2, 0, 1])` = `(c, a, b) / E`.

To permute across nesting levels: `fuse` first, then `permute`, then `cut`
to re-establish the boundary.

---

## Operations (PascalCase)

Operations do computational work. All use PascalCase.

**Key rule: operations operate on the leading shape.** No `axis` parameter.
The layout IS the axis specification. To reduce along a specific axis,
`cut` first to move it into the leading shape. See "Leading Shape Theory"
above.

### Adverbs (function transformers)

Adverbs take a verb (function value) and return a new function. Application
is a separate step. `Over(Add, xs)` is syntactic sugar for
`Apply(Over(Add), xs)` -- the adverb produces a function, then Apply
consumes data.

**Parallel** (order-independent, any leading shape rank):

| Adverb | Produces | Type rule |
|--------|----------|-----------|
| `Over(f)` | monadic | `S / E` -> `E` (consume leading shape) |
| `Each(f)` | monadic | `S / E` -> `S / f(E)` (preserve leading shape) |
| `EachRight(f)` | dyadic | `(T, S / E)` -> `S / f(T, E)` (x whole, iterate y) |
| `EachLeft(f)` | dyadic | `(S / E, T)` -> `S / f(E, T)` (iterate x, y whole) |

**Sequential** (require rank-1 leading shape):

| Adverb | Produces | Type rule |
|--------|----------|-----------|
| `Scan(f)` | monadic | `(D,) / E` -> `(D,) / E` (running accumulation) |
| `Prior(f)` | monadic | `(D,) / E` -> `(D-1,) / E` (adjacent pairs) |

**Jaggedness**: Each, EachRight, EachLeft promote Dynamic to Jagged in the
result's inner type (each element could independently produce a different
size). Static extents are preserved.

**Over reduces the full leading shape** to one element. For commutative +
associative operations (Add, Mul, Min, Max), this is well-defined regardless
of rank and maps to GPU parallel tree reduction. For non-commutative ops,
require rank 1.

**Mean is not a primitive** -- it is composed: `Div(Over(Add, xs), Count(xs))`.
Example: face centroids = `Each(faces, f => Div(Over(Add, verts_of_f), Const(3)))`.

### Core Operations

- **Map(xs, x => body)**: scalar function per element. Preserves leading shape.
- **Where(xs)**: coords of truthy elements. `S / bool` => `(*,) / (rank(S),) i32`.
- **Gather(target, indexer)**: materialise an Indexed layout.

### Scalar Primitives

`Add`, `Sub`, `Mul`, `Div`, `GE`, `And`, `Not`, `InBounds`, `Count`.

### Domain Primitives

- **Decompose(coord, bit_widths)**: bitfield coordinate split.
- **Morton3d(coord)**: 3D morton encoding.

### Summary

| Name | Convention | Kind | New data? |
|------|-----------|------|-----------|
| `cut`, `indexed`, `tuple`, `struct`, `flip`, `jagged`, `reshape`, `field`, `masked` | lowercase | Layout (free) | No |
| `fuse`, `flatten`, `permute` | lowercase | Layout (free) | No |
| `Over`, `Scan`, `EachRight`, `EachLeft`, `EachBoth`, `Prior` | PascalCase | Adverb (function -> function) | Deferred |
| `Each`, `Map` | PascalCase | Iteration (with lambda body) | Yes |
| `Apply` | PascalCase | Function application | Yes |
| `Where` | PascalCase | Operation | Yes (data-dependent) |
| `Gather` | PascalCase | Operation | Yes (materialise) |
| `Add`, `Sub`, `Mul`, `Div`, `GE`, `And`, `Not`, `InBounds`, `Count` | PascalCase | Verb (function value) | Yes |
| `Decompose`, `Morton3d` | PascalCase | Domain function | Yes |
| `Input`, `Const` | PascalCase | Connector | Introduces data |

---

## Worked Example: Single-Leaf Neighbors

```
leaf     = reshape(leaf_raw, (8,8,8))          -- (8,8,8) i32
features = cut(-C, features_raw)               -- (*,) over (C,) f32
mask     = Map(leaf, x >= 0)                   -- (8,8,8) bool
active   = Where(mask)                         -- (*,) over (3,) i32
idx      = Gather(leaf, active)                -- (*,) over i32
feat     = Gather(features, idx)               -- (*,) over (C,) f32
offsets  = cut(-3, offsets_raw)                -- (6,) over (3,) i32
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
vertices = cut(-3, positions_raw)           -- (V,) over (3,) f32
face_idx = cut(-3, faces_raw)               -- (F,) over (3,) i32
tris = Each(face_idx, f => Map(f, i => Gather(vertices, i)))
                                            -- (F,) over (3,) over (3,) f32
```

No custom mesh class. The type `(F,) over (3,) over (3,) f32` fully
describes "F triangles of 3 vertices of 3D positions."

**Type system finding:** `indexed(face_idx, vertices)` is correctly rejected.
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

Code in `fvdb_tile/prototype/`. Environment:
`source ~/.venvs/fvdb_cutile/bin/activate` (Python 3.12, cuda-tile 1.1.0,
torch 2.10, numpy 2.2; RTX PRO 6000 Blackwell, compute 12.0).
Run all: `python fvdb_tile/tests/run_all_tests.py`.

### v0: Fundamentals

Map, Where, Gather, Each over a single leaf. Type propagation works
end-to-end. Jagged emerges automatically. Layouts and operations cleanly
separated. Sentinels stay out of the type system.

### v1: Composition

Multiple leaves via `cut` + `Each` (double-nested jagged). `indexed` layout
predicts `Gather` type. `struct` + `flip` composes multi-feature voxels into
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
`indexed(face_idx, vertices)` (mesh indexing is multiple scalar lookups, not
one multi-dimensional lookup). Early vs. late materialisation of face
centroids produce identical results, demonstrating that materialisation
schedule is independent of algorithm. DSL expresses the mesh lookup as a
string program.

### v4: cuTile Codegen

First GPU code generation from the DSL. Three components:

1. **Smoke test** (`test_cutile_smoke.py`): `ct.gather`/`ct.scatter` with
   computed indices and 2D multi-dimensional indexing confirmed working on
   Blackwell hardware (RTX PRO 6000, compute 12.0).

2. **Hand-written target kernel** (`test_cutile_gather.py`): the neighbor
   gather predicate -- for each of 453 active voxels in a random (8,8,8)
   leaf, check 6 face-neighbors via `ct.gather` with 3D computed indices and
   `check_bounds=True, padding_value=-1`. Results match numpy exactly (2094
   active neighbors). Corner cases verified: boundary voxels, lone active
   voxel, fully active leaf.

3. **DSL-to-cuTile emitter** (`dsl_to_cutile.py`, `test_cutile_codegen.py`):
   tree-walk emitter that parses a DSL string and emits cuTile Python source.
   The neighbor predicate DSL expression

   ```
   Map(offsets, o => GE(Gather(mask, Add(coord, o)), Const(0)))
   ```

   emits to:

   ```python
   add_2 = coord_tile + offsets_arr
   gath_3 = ct.gather(mask_arr, add_2, check_bounds=True, padding_value=-1)
   ge_4 = gath_3 >= 0
   ```

   Numpy evaluator and cuTile kernel produce identical results for all 453
   voxels.

**Key finding:** `ct.gather` (not `ct.load`) is the right cuTile primitive
for our `Gather` operation. `ct.load` requires power-of-two tile dimensions;
`ct.gather` accepts arbitrary computed indices with broadcasting and built-in
bounds checking. The `check_bounds=True, padding_value=-1` pattern fuses our
`InBounds` + sentinel into a single call.

**Correspondence table** (verified):

| DSL | cuTile |
|-----|--------|
| `cut(input, size)` | `ct.load(array, index, shape)` (power-of-two tiles) |
| `Gather(target, indexer)` | `ct.gather(array, indices, check_bounds=True)` |
| `InBounds` + sentinel | `ct.gather(..., padding_value=-1)` (fused) |
| `Add`, `GE`, `And` | tile arithmetic (`+`, `>=`, `&`) |
| `Each(xs, ...)` | grid-level parallelism (`ct.bid(0)`) |
| `Over(Add, xs)` | `ct.sum(tile, axis)` |
| `Decompose(coord, [3, 4])` | `(coord >> shift) & mask` per level per axis |
| `field(struct, "level_N")` | select per-axis tile variables |
| `Gather(Gather(A, i), j)` | `ct.gather(A, (i, j...))` fused 4D gather |

### v5: End-to-End Codegen, Benchmark, Cross-Leaf

Three milestones closing the remaining gaps:

1. **End-to-end codegen** (`test_cutile_e2e.py`): the emitter now produces
   a complete, compilable `@ct.kernel` with `ct.bid(0)` grid parallelism,
   per-axis coordinate decomposition, power-of-two tile padding, and
   `ct.scatter` output. The DSL string is parsed, emitted to a `.py` file
   (cuTile JIT requires `inspect.getsource()`), imported, launched, and
   verified against both the numpy DSL evaluator and the hand-written kernel.
   **The loop is closed: DSL string -> GPU execution -> correct results.**

2. **First performance number** (`bench_cutile.py`): the hand-written cuTile
   neighbor predicate kernel (which is the emitter's target) benchmarked
   against numpy and PyTorch. On 453 active voxels:

   | Method | Median (us) | vs numpy | vs PyTorch GPU |
   |--------|-------------|----------|----------------|
   | numpy (loop) | 13,708 | 1.0x | -- |
   | PyTorch CPU (vectorized) | 133 | 103x | -- |
   | PyTorch GPU (vectorized) | 284 | 48x | 1.0x |
   | cuTile (hand-written) | 64 | 214x | 4.4x |

   cuTile's advantage comes from fused gather+predicate+scatter in a single
   kernel launch with no intermediate tensors. PyTorch GPU is slower than
   PyTorch CPU at this scale due to kernel launch overhead.

3. **Cross-leaf neighbors** (`test_cross_leaf.py`): the critical DSL
   composability test. For each active voxel, check 6 face-neighbors via
   Decompose + chained Gather through a two-level hierarchy, where neighbors
   may be in different leaf nodes. The DSL expression:

   ```
   nbr = Add(Input("coord"), Input("offset"))
   parts = Decompose(nbr, Const([3, 4]))
   leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
   leaf_node = Gather(Input("leaf_arr"), leaf_idx)
   voxel_val = Gather(leaf_node, field(parts, "level_0"))
   is_active = GE(voxel_val, Const(0))
   ```

   50 voxels x 6 offsets = 300 hierarchical lookups, 48 crossing leaf
   boundaries, all correct. **The DSL composes cleanly for cross-boundary
   traversal.**

### v6: Cross-Leaf GPU Codegen + Scale Benchmark

Three new capabilities closing the compilation gap for hierarchical traversal:

1. **Emitter: Decompose + field + chained Gather** (`dsl_to_cutile.py`):
   three new emission rules extend `_emit_decomposed`:

   - `DecomposeNode`: per-level per-axis bit extractions (`>> shift & mask`).
   - `FieldNode`: project named field from the decomposed struct.
   - `GatherNode` chain flattening (idiom detection):
     `Gather(Gather(leaf_arr, leaf_idx), level_0)` fuses into a single
     4D `ct.gather(leaf_arr, (leaf_idx, l0_x, l0_y, l0_z))`. This is the
     first idiom the emitter recognises -- the two-step "look up leaf block,
     then index into it" becomes one fused gather.

   **Updated correspondence table:**

   | DSL | cuTile |
   |-----|--------|
   | `Decompose(coord, [3, 4])` | `(coord >> shift) & mask` per level per axis |
   | `field(struct, "level_N")` | select per-axis variables for level N |
   | `Gather(Gather(A, i), j)` | `ct.gather(A, (i, j_x, j_y, j_z))` (fused) |

2. **Cross-leaf on GPU** (`test_cutile_cross_leaf.py`): the Map-form DSL
   expression for cross-leaf neighbor finding is parsed, emitted as a
   complete `@ct.kernel`, compiled via cuTile JIT, launched on GPU, and
   verified against both the numpy DSL evaluator and a numpy reference.
   817 voxels across 4 leaves, 1734 active neighbors, 617 cross-leaf
   lookups -- all correct. **DSL string -> GPU execution for hierarchical
   sparse traversal.**

3. **Scale benchmark** (`bench_cutile_cross_leaf.py`): cross-leaf predicate
   benchmarked at 4, 64, and 256 leaves against numpy and PyTorch GPU:

   | Leaves | Voxels | Lookups | PyTorch CPU (us) | PyTorch GPU (us) | cuTile (us) | CT/PT |
   |--------|--------|---------|------------------|------------------|-------------|-------|
   | 4 | 1,211 | 7,266 | 430 | 531 | 61 | 8.7x |
   | 64 | 19,340 | 116,040 | 1,658 | 533 | 70 | 7.6x |
   | 256 | 78,972 | 473,832 | 16,194 | 561 | 101 | 5.6x |

   cuTile's advantage comes from: (a) single fused kernel with no
   intermediate tensors, (b) the 4D fused gather avoids materialising
   intermediate leaf blocks, (c) `ct.gather` built-in bounds checking
   eliminates separate validation passes. The GPU scaling is excellent:
   80x more work (4 -> 256 leaves) costs only 2x more time.

**Key finding:** the chained gather pattern `Gather(Gather(A, i), j)` that
the type system tracks as two distinct operations naturally fuses into one
`ct.gather` at emission time. This is the Halide analogy made concrete: the
**algorithm** is hierarchical lookup expressed as nested Gathers; the
**schedule** is a fused 4D gather. The emitter recognises the pattern and
applies the optimisation automatically.

### v7: CIG Format, Builder, and fVDB Head-to-Head

First apples-to-apples comparison with fVDB. Three components:

1. **CIG builder** (`cig.py`): `build_cig(ijk)` constructs a 2-level CIG
   from `(N, 3)` voxel coordinates. Returns a `CIG` dataclass with `lower`
   `(16,16,16)` and `leaf_arr` `(K,8,8,8)` tensors. Pure PyTorch. The CIG
   equivalent of `fvdb.Grid.from_ijk()`.

2. **CIG ijk_to_index** -- three implementations:
   - Numpy reference (loop-based, correctness verification)
   - PyTorch vectorized (broadcast + advanced indexing)
   - cuTile kernel (`cig_cutile.py`): hand-written from the v6 cross-leaf
     pattern, tile-parallel with TILE=256 queries per block

3. **Head-to-head benchmark** (`bench_cig_vs_fvdb.py`): construction,
   memory, and query time at 1K-200K voxels. Same coordinates, verified
   identical active/inactive classification.

   **Query performance** (50,000 queries):

   | Voxels | CIG cuTile (us) | fVDB NanoVDB (us) | Speedup |
   |--------|----------------|-------------------|---------|
   | 1,000 | 69 | 89 | 1.3x |
   | 10,000 | 68 | 87 | 1.3x |
   | 50,000 | 68 | 89 | 1.3x |
   | 200,000 | 70 | 106 | 1.5x |

   **CIG cuTile is 1.3-1.5x faster than NanoVDB** for ijk_to_index. The
   fused 4D gather avoids NanoVDB's multi-level tree traversal overhead.

   **Memory footprint:**

   | Voxels | CIG (KB) | fVDB NanoVDB (KB) | Ratio |
   |--------|----------|-------------------|-------|
   | 1,000 | 1,798 | 382 | 0.2x |
   | 10,000 | 7,498 | 649 | 0.1x |
   | 200,000 | 8,208 | 682 | 0.1x |

   **NanoVDB uses 8-12x less memory** than the naive CIG. This is expected:
   NanoVDB uses bitmask compression (64-byte active mask per leaf + only
   active voxel data via popcount offsets), while CIG stores dense (8,8,8)
   i32 blocks with -1 sentinels. The CIG trades memory for query simplicity
   -- direct array indexing vs NanoVDB's bitmask + popcount.

   **Construction time:**

   | Voxels | CIG (us) | fVDB (us) | CIG/fVDB |
   |--------|----------|-----------|----------|
   | 1,000 | 3,033 | 3,332 | 0.9x |
   | 10,000 | 29,235 | 3,598 | 8.1x |
   | 200,000 | 594,207 | 7,846 | 75.7x |

   CIG construction is pure Python/PyTorch (sort + scatter), while fVDB
   uses an optimised C++ builder. A GPU-native CIG builder would close
   this gap significantly.

**Key findings:**
- **Query speed validates the approach.** The cuTile fused gather beats
  NanoVDB's compiled C++/CUDA kernel. This is the compilation thesis in
  action: a simple data layout + compiled access pattern outperforms a
  complex tree traversal on the same hardware.
- **Memory is the open problem.** The naive dense-leaf format wastes storage
  on inactive entries. The design question: can we add bitmask compression
  to the CIG while keeping the fused gather pattern? NanoVDB's popcount-
  based offset calculation would need to become part of the Gather chain.
- **Construction needs GPU acceleration.** The pure PyTorch builder is
  acceptable for prototyping but not competitive at scale. A GPU sort +
  scatter builder is straightforward.

### v8: The `masked` Layout -- Bitmask-Compressed CIG

The key structural addition: `masked` as a first-class layout in the type
system. This is the sparse-occupancy counterpart of `jagged` (variable-
length segments). A masked layout wraps a fixed-shape leading shape with a
bitmask indicating which positions have data, plus a base offset into a flat
data array. Access computes: bitmask check + popcount for the dense index.

1. **`masked` layout** (`layouts.py`): `MaskedElement` carries the mask
   shape and element type. `masked(mask_expr, offset_expr)` in the DSL
   constructs the layout. Gather through masked emits the bitmask + popcount
   chain. Zero-cost layout -- no data movement, just type interpretation.

2. **Compressed CIG** (`cig.py`): `CompressedCIG` stores:
   - `leaf_masks: (K, 16) i32` -- 16 words of 32 bits = 512-bit mask/leaf
   - `leaf_offsets: (K,) i32` -- base offset per leaf
   - `voxel_data: (N_active,) i32` -- flat voxel indices
   Per-leaf cost: 68 bytes (vs 2,048 dense, vs NanoVDB's 96 bytes).

3. **cuTile kernel with popcount** (`cig_masked_cutile.py`): software
   Hamming weight in i32 (cuTile lacks i64 support). The popcount chain
   unrolls across 16 mask words with conditional accumulation. The bitmask
   check + popcount + offset calculation adds negligible overhead vs the
   dense Gather.

4. **DSL expression** for compressed CIG `ijk_to_index`:

   ```
   parts = Decompose(query, [3, 4])
   leaf_idx = Gather(lower, field(parts, "level_1"))
   leaf = masked(Gather(leaf_masks, leaf_idx), Gather(leaf_offsets, leaf_idx))
   voxel_idx = Gather(leaf, field(parts, "level_0"))
   ```

   Four lines. The `masked` layout makes the bitmask compression visible
   in the type system, not hidden in an implementation detail.

5. **Head-to-head benchmark** at 1K-200K voxels, 50K queries:

   **Memory:**

   | Voxels | Dense CIG | Compressed CIG | NanoVDB | Comp/fVDB |
   |--------|-----------|---------------|---------|-----------|
   | 1,000 | 1,841 KB | 83 KB | 382 KB | 0.22x |
   | 10,000 | 7,498 KB | 318 KB | 649 KB | 0.49x |
   | 50,000 | 8,208 KB | 499 KB | 682 KB | 0.73x |
   | 200,000 | 8,208 KB | 1,085 KB | 682 KB | 1.59x |

   **Query time (us):**

   | Voxels | Dense CT | Compressed CT | fVDB NanoVDB | Comp/fVDB |
   |--------|---------|--------------|-------------|-----------|
   | 1,000 | 72 | 76 | 96 | 1.27x |
   | 10,000 | 82 | 79 | 101 | 1.28x |
   | 50,000 | 74 | 76 | 102 | 1.34x |
   | 200,000 | 78 | 77 | 98 | 1.28x |

   **The compressed CIG is both smaller AND faster than NanoVDB** at typical
   sparsities (1K-50K voxels). At 200K voxels (high density, nearly all
   lower nodes filled), CIG uses 1.6x more memory due to the dense (16,16,16)
   lower array -- addressable by adding masked compression to internal nodes.

   The popcount overhead is negligible: compressed cuTile (76us) vs dense
   cuTile (74us), a 3% difference. The bitmask + popcount chain runs at
   essentially the same speed as a direct array lookup.

**u64 variant:** cuTile supports `ct.uint64` tiles. A u64 kernel using 8 x
u64 words (instead of 16 x i32) halves the gather count and popcount
accumulations. Benchmarked at ~10% faster for 50K voxels, negligible
difference elsewhere. The u64 variant is cleaner (no i32 mask conversion
step) and is the preferred path. Both i32 and u64 kernels are in
`cig_masked_cutile.py`.

**Key finding:** the `masked` layout simultaneously solves the memory
problem AND preserves the query speed advantage. The bitmask is layout
metadata (like offsets in jagged); the popcount is the access computation
(like offset lookup in jagged). Both are transparent in the type system,
analyzable in the AST, and compilable to GPU code.

### v9: Masked Codegen -- DSL-to-cuTile for `masked` Gather

Closes the compilation loop for the 4-line DSL claim. In v8, the cuTile
kernel was hand-written; the DSL expression was proven correct via the
numpy evaluator but not yet compiled. v9 extends the emitter so that the
DSL produces a kernel structurally identical to the hand-written one.

1. **`MaskedNode` emission** (`dsl_to_cutile.py`): the emitter recognises
   `masked(Gather(Input("leaf_masks"), idx), Gather(Input("leaf_offsets"), idx))`
   as a deferred layout construction. It extracts the mask array name and
   leaf index, emits the base offset gather, and returns a masked sentinel
   `EmitVal`. No popcount code is emitted yet -- that is deferred to the
   downstream Gather.

2. **Masked-Gather idiom** (`dsl_to_cutile.py`): when `GatherNode` detects
   a masked target (via the sentinel), it calls `_emit_masked_gather_u64`
   which emits the full u64 popcount chain: flat bit index, word/bit
   decomposition, bitmask active check, Hamming weight for the partial
   word, 8 unrolled word gathers with conditional popcount accumulation,
   and the final `(base + popcount) * is_active + (-1) * (1 - is_active)`.
   This is the third idiom the emitter recognises, after chained Gather
   fusion (v6) and Decompose + field (v6).

3. **Tile-parallel emission** (`dsl_to_cutile.py`): `emit_runnable_kernel`
   generalised with a `tile_input` parameter for flat parallelism
   (`query_idx = bid * TILE + arange(TILE)`), alongside the existing
   batch+Map pattern. The masked CIG uses tile-parallel: each tile element
   independently processes one query coordinate.

4. **End-to-end test** (`test_cutile_masked_e2e.py`): the 4-line DSL string
   is parsed, emitted as a `@ct.kernel`, compiled via cuTile JIT, launched
   on GPU, and verified against both the numpy DSL evaluator and the
   hand-written u64 kernel. The hand-written kernel is preserved as ground
   truth and regression oracle.

5. **u64 consolidation**: `cig_masked_cutile.py` now uses the u64 kernel
   as the primary entry point (`run_compressed_cig_ijk_to_index`). The i32
   kernel is retained as a labeled reference implementation.

**The generated kernel is 105 lines of cuTile** -- produced mechanically
from 4 lines of DSL. The structure matches the hand-written kernel
one-to-one: same gathers, same bit operations, same popcount chain, same
conditional result. The only difference is variable naming.

**This validates the central claim:** a 4-line DSL expression specifying
the algorithm (what to compute) compiles to a ~100-line GPU kernel
specifying the implementation (how to compute it). The algorithm is
portable, analyzable, and type-checked; the kernel is fast, fused, and
hardware-specific. The emitter is the bridge.

### v10: 3-Level CIG -- Structural Parity with NanoVDB Index Tree

Extends the CIG from 2 levels to 3, achieving structural parity with
NanoVDB's leaf/lower/upper hierarchy.  Bit-widths `[3, 4, 5]` give
8^3 leaves, 16^3 lower nodes, 32^3 upper nodes, covering 4096^3 per
axis.  A root level with variable upper nodes and a torch linear-scan
lookup completes the tree.

1. **Prefix-sum masked layout**: the `masked` layout gains a third
   argument (prefix-sum popcounts) replacing the v8/v9 unrolled popcount
   chain.  Query cost is now O(1) regardless of mask size: one gather
   from the prefix array, one gather from the mask, one partial popcount.
   The emitter produces ~20 lines per masked level instead of ~70.
   Works for any node shape (8^3, 16^3, 32^3, or any other).

2. **CompressedCIG3** (`cig.py`): 3-level data structure with per-level
   masks, prefix sums, and offsets.  `build_compressed_cig3(ijk)` builds
   from voxel coordinates.  `_build_masked_level` is generic across node
   sizes.  `root_lookup(root_coords, query)` provides the torch-side
   linear scan for the root level.

3. **Two-step pipeline**: torch resolves the root (linear scan of
   `root_coords`), then a single fused cuTile kernel handles the 3-level
   masked chain.  The torch barrier is between the root lookup and the
   cuTile kernel; the entire upper -> lower -> leaf -> voxel chain is
   fused.

4. **DSL expression** for 3-level `ijk_to_index`:

   ```
   parts = Decompose(query, [3, 4, 5])
   upper = masked(Gather(upper_masks, upper_idx), Gather(upper_prefix, upper_idx), Gather(upper_offsets, upper_idx))
   lower_idx = Gather(upper, field(parts, "level_2"))
   lower = masked(Gather(lower_masks, lower_idx), Gather(lower_prefix, lower_idx), Gather(lower_offsets, lower_idx))
   leaf_idx = Gather(lower, field(parts, "level_1"))
   leaf = masked(Gather(leaf_masks, leaf_idx), Gather(leaf_prefix, leaf_idx), Gather(leaf_offsets, leaf_idx))
   voxel_idx = Gather(leaf, field(parts, "level_0"))
   ```

   Nine lines.  Three masked levels, each using the same prefix-sum
   Gather pattern.  The bit-widths `[3, 4, 5]` are parameters, not
   hard-coded -- changing them (e.g. four layers of 8^3 with
   `[3, 3, 3, 3]`) requires only a different `Decompose` and one more
   masked level in the chain.

5. **Generated kernel**: 104 lines of cuTile, with three masked-gather
   blocks (32^3, 16^3, 8^3).  Verified against numpy reference for both
   single-upper and multi-upper grids.

**Key findings:**

- The prefix-sum approach scales to any node shape without changing the
  emitter.  The old unrolled approach was O(W) gathers per level (W=8
  for leaves); the new approach is O(1) per level via 2 gathers + 1
  popcount.
- The 3-level chain composes cleanly: the DSL is a repetition of the
  same `masked(Gather, Gather, Gather)` + `Gather(masked, field)`
  pattern.  Adding a fourth level is one more copy of the pattern.
- The root lookup lives outside the cuTile kernel (torch step) by
  design.  This avoids introducing a search primitive into the DSL while
  keeping the heavy computation (3 levels of masked Gather) fully fused.
- The `[3, 4, 5]` configuration matches NanoVDB exactly.  But the
  framework does not hard-code it: `[3, 3, 3, 3]` (four 8^3 layers),
  `[4, 4, 4]` (three 16^3 layers), or any other stacking is equally
  valid.  This generality over level configurations is a structural
  advantage over NanoVDB's fixed tree.

### v11: Performance Optimisation -- Fused Root, Absolute Prefix, CSE

Three optimisations that close the performance gap with NanoVDB:

1. **`Find` primitive**: general-purpose linear scan of a small (R, K)
   table for a matching (K,) key.  Returns the row index or -1.  The
   emitter unrolls the scan at emit time (R is known from the input
   type).  For R=1 (single upper node), this is a single 3-axis
   comparison.  Eliminates the ~190us torch root_lookup barrier.

2. **Absolute prefix sums**: the base offset is folded into the prefix
   table at build time: `abs_prefix[node, word] = offset + cum_popc`.
   Query becomes `abs_prefix[word] + partial_popcount` -- two gathers per
   level instead of three.  `masked` returns to 2 args (cleaner DSL).

3. **Emitter CSE**: Hamming weight constants emitted once per kernel, not
   once per masked level.

**DSL expression** for optimised 3-level `ijk_to_index`:

   ```
   parts = Decompose(query, [3, 4, 5])
   upper_idx = Find(root_coords, field(parts, "which_top"))
   upper = masked(Gather(upper_masks, upper_idx), Gather(upper_abs_prefix, upper_idx))
   lower_idx = Gather(upper, field(parts, "level_2"))
   lower = masked(Gather(lower_masks, lower_idx), Gather(lower_abs_prefix, lower_idx))
   leaf_idx = Gather(lower, field(parts, "level_1"))
   leaf = masked(Gather(leaf_masks, leaf_idx), Gather(leaf_abs_prefix, leaf_idx))
   voxel_idx = Gather(leaf, field(parts, "level_0"))
   ```

   Ten lines.  Fully fused -- single cuTile kernel, no torch barrier.

**Benchmark** (50K queries, [0, 4096)^3):

   | Voxels  | CIG3 (us) | fVDB (us) | CIG3/fVDB | CIG3 memory |
   |---------|-----------|-----------|-----------|-------------|
   | 1,000   | 132       | 123       | 0.94x     | 0.03x fVDB  |
   | 10,000  | 132       | 114       | 0.86x     | 0.03x fVDB  |
   | 50,000  | 133       | 107       | 0.80x     | 0.04x fVDB  |
   | 200,000 | 131       | 113       | 0.86x     | 0.05x fVDB  |

   The 3-level CIG is within 6-20% of NanoVDB query speed while using
   **25-30x less memory**.  The v10 baseline was 3.5x slower (350us)
   due to the torch root barrier and extra offset gathers.

---

## Working Theory

**Claim**: nested layouts + leading-shape-theory adverbs can fully specify
structured geometric and scientific computations -- sparse grids, meshes,
point clouds, scene graphs -- in a form that is:

- **Portable**: backed by tensors, described by metadata. Any system that can
  store tensors can store and exchange the full structure.
- **Analyzable**: the typed AST can be inspected, transformed, and optimised
  before execution.
- **Compilable**: types carry enough information to lower to concrete GPU
  code via idiom detection and algebraic transformation. **Demonstrated** in
  v4-v8: DSL string -> AST -> runnable cuTile `@ct.kernel` -> GPU execution
  -> correct results. v8 demonstrates NanoVDB-competitive memory and query
  performance via the `masked` layout with bitmask + popcount, from a 4-line
  DSL expression.

fVDB's sparse voxel operations are the proving ground, not the ceiling.

---

## Multi-Step Compilation

The current compiler emits a single cuTile `@ct.kernel` from a DSL program.
This works for **gather-dominated pipelines** (ijk_to_index, neighbor
predicates, cross-leaf traversal) where every operation is pointwise or
scatter-gather -- each tile element does its own independent lookup chain.

More complex operations require **collective steps** that cannot fit in a
single kernel: sort, prefix sum, unique/dedup, and data-dependent output
allocation. These arise in grid construction (`from_ijk`), topology
transforms (`conv_grid`), and variable-length output (`Where` on GPU).

### Barrier-based pipeline partitioning

The compiler should partition the AST into **segments** separated by
**barriers**:

```
Segment 1 (cuTile):   pointwise/gather ops  ->  @ct.kernel
     |
     barrier:  data-dependent output size / collective operation
     |
Segment 2 (cuTile):   pointwise/gather ops  ->  @ct.kernel
```

**cuTile segments:** All pointwise, scalar, Gather, Decompose, masked, and
Map operations fuse into a single `@ct.kernel`. This is what the emitter
does today.

**Collective steps:** Dispatched to existing optimised primitives:
- `torch.sort` / CUB radix sort
- `torch.cumsum` / CUB prefix sum
- `torch.unique` / CUB unique
- Output allocation (`torch.empty` with computed size)

These do not need DSL code generation -- they are already highly optimised
library calls. The compiler's job is to **detect where they are needed** and
wire intermediate tensors between segments.

### Barrier detection signals

The AST already carries the information to detect barriers:

- **`Where` node**: output size is data-dependent. Requires: count pass
  (cuTile kernel) -> prefix sum (`torch.cumsum`) -> scatter pass (cuTile
  kernel). This is the canonical two-pass GPU pattern.
- **`Over` (full reduction)**: tree reduction needs multiple passes or a
  CUB reduction call. A barrier between the gather and the reduction.
- **Sort / Unique nodes** (not yet in DSL, but needed for grid construction):
  inherently collective. Explicit barriers.
- **`Each` with jagged output**: collecting variable-length results requires
  knowing all sizes first (scan for offsets), then scatter. Two passes.

### Concrete example: `conv_grid`

```
Step 1 (cuTile):  For each active voxel, compute kernel-offset neighbors
Step 2 (collective): Sort + unique the expanded coordinates (dedup)
Step 3 (cuTile):  For each unique coord, build output CIG entries
```

The compiler sees: Map (cuTile) -> unique (barrier) -> build (cuTile). It
emits two kernels and one `torch.unique` call, wired with intermediate
tensors.

### Relationship to successive lowering

This is the successive lowering architecture discussed earlier, made
concrete with a specific partitioning criterion:

**Pass 1 (logical AST):** The program as written. Operations are abstract.

**Pass 2 (step planning):** Walk the AST, identify barrier nodes, partition
into segments. Each segment is tagged as cuTile or collective.

**Pass 3 (emission):** For cuTile segments, emit a `@ct.kernel` (as today).
For collective steps, emit PyTorch/CUB calls. Wire with intermediate
tensors. The output is a **pipeline** -- a callable sequence of kernel
launches and library calls.

---

## Next Steps (for a new agent)

### Orientation

Read this document top-to-bottom for the thesis, vocabulary, layout wrappers,
adverb semantics, and worked examples. Then:

- **Prototype code**: `fvdb_tile/prototype/`
- **Tests**: `fvdb_tile/tests/` -- run via `python fvdb_tile/tests/run_all_tests.py`
- **Benchmarks**: `fvdb_tile/benchmarks/`
- **Design doc**: this file (`fvdb_tile/README.md`)
- **Decision trail**: `fvdb_tile/HISTORY.md` (append-only)
- **Semantic contracts**: `fvdb_tile/prototype/SEMANTICS.md`

File inventory:

| File | DSL status | Role |
|------|------------|------|
| `prototype/types.py` | core | Extent kinds (Static/Dynamic/Jagged), Shape, Type, ScalarType, FnType |
| `prototype/layouts.py` | core | Layout wrappers (lowercase): cut, indexed, tuple, struct, flip, jagged, masked, fuse, flatten, permute |
| `prototype/ops.py` | core | Python-level operations + CPU-only reference hashmap; FnValue + VERBS registry |
| `prototype/dsl_ast.py` | core | AST node classes (~30 nodes) with `infer_type` methods |
| `prototype/dsl_parse.py` | core | Recursive-descent parser: string -> AST |
| `prototype/dsl_eval.py` | core | Tree-walk evaluator with hooks: type-check pass then torch execution |
| `prototype/dsl_pipeline.py` | core | Barrier-aware pipeline planner, GPU collective dispatch, cutile segment compilation |
| `prototype/dsl_to_cutile.py` | core | DSL AST -> cuTile Python source emitter (preference #1) |
| `prototype/dsl_to_cuda.py` | core | DSL AST -> CUDA C++ emitter, last-resort backend (preference #3) |
| `prototype/dsl_lower.py` | core | Dialect lowering pass: rewrite dialect nodes to core AST |
| `prototype/conv_grid.py` | **fully DSL-driven** | conv_grid topology expansion via DSL pipeline (adverb composition + Sort + Unique) |
| `prototype/conv_grid_leafwise.py` | **fully DSL-driven** | Leafwise conv_grid as unified 13-binding DSL program (DilateLeafMasks + HashMapOccupied + adverb composition) |
| `prototype/cig.py` | **out-of-DSL** | CompressedCIG3 builder (imperative torch), root lookup, reference query |
| `prototype/hashmap_cuda.py` | **out-of-DSL** | GPU hash map build/lookup/scatter_reduce + fused conv_grid_dilate kernel (NVRTC JIT) |
| `prototype/cuda_launch.py` | infra | Generic NVRTC compile-and-launch utility |
| `prototype/dialect_hashmap.py` | core | HashMapBuild/Lookup as DSL dialect nodes |
| `prototype/verify_memory.py` | tool | Byte-level memory comparison CIG3 vs fVDB |
| `tests/run_all_tests.py` | | Single entry point for non-GPU tests |
| `tests/test_where.py` | | v0: Map, Where, Gather pipeline |
| `tests/test_neighbors.py` | | v0: neighbor finding, jagged emergence |
| `tests/test_indexed_flip.py` | | v1: multi-leaf cut, indexed, Struct+Flip |
| `tests/test_two_level.py` | | v2: hierarchical chain, Decompose, morton |
| `tests/test_dsl.py` | | v3: DSL validation (parser + evaluator) |
| `tests/test_mesh.py` | | Triangle mesh, centroids via Over+Div |
| `tests/test_sort_unique.py` | | Sort/Unique DSL primitive correctness |
| `tests/test_pipeline.py` | | Pipeline planning, collective dispatch, cutile segments |
| `tests/test_conv_grid.py` | | conv_grid correctness, stride semantics |
| `tests/test_conv_grid_leafwise.py` | | Leafwise conv_grid CPU + GPU correctness |
| `tests/test_hashmap.py` | | Hash map primitives, bitwise ops, dialect lowering |
| `tests/test_adverbs.py` | | Nested adverbs, outer product, function values |
| `tests/test_layouts_advanced.py` | | fuse, flatten, permute, EachBoth, dot product |
| `tests/test_masked.py` | | v8: masked layout DSL tests |
| `tests/test_cig3.py` | | v10: 3-level CIG builder, root lookup, reference |
| `tests/test_cross_leaf.py` | | v5: cross-leaf neighbors via DSL evaluator |
| `tests/test_cutile_smoke.py` | | v4: cuTile toolchain validation |
| `tests/test_cutile_e2e.py` | | v5: end-to-end DSL -> GPU -> verify |
| `tests/test_cutile_cross_leaf.py` | | v6: cross-leaf GPU codegen + verify |
| `tests/test_cutile_cig3_e2e.py` | | v10/v11: 3-level CIG codegen (fused Find) |
| `tests/test_cutile_hashmap.py` | | GPU hash map + pipeline bitwise ops |
| `benchmarks/bench_cig3_vs_fvdb.py` | | 3-level CIG vs fVDB head-to-head |
| `benchmarks/bench_cutile_cross_leaf.py` | | Cross-leaf scale benchmark |
| `benchmarks/bench_conv_grid.py` | | conv_grid DSL-driven benchmark |
| `benchmarks/bench_conv_grid_leafwise.py` | | Leafwise conv_grid benchmark |
| `benchmarks/bench_hashmap.py` | | CPU hash map benchmark |
| `benchmarks/bench_gpu_hashmap.py` | | GPU hash map benchmark |

### Current state

The DSL has ~25 keywords. **Layouts use lowercase** (`cut`, `reshape`,
`field`, `masked`) -- they reinterpret types without moving data.
**Operations use PascalCase** (`Map`, `Each`, `Where`, `Gather`, `Over`,
`Scan`, `Add`, etc.) -- they do computational work. This convention makes the cost model visible
at a glance. Programs are text strings parsed into typed ASTs, type-checked
without data, then executed against numpy.

Key semantic rules (see Leading Shape Theory section for full details):
- Adverbs are function transformers: `Over(Add)` produces a new function;
  `Apply(Over(Add), xs)` applies it. `Over(Add, xs)` is syntactic sugar.
- All adverbs operate over the **full leading shape**. No axis parameter.
  `cut` to set the nesting boundary. The layout IS the schedule.
- Each/EachRight/EachLeft/EachBoth promote Dynamic to Jagged at type-check
  time (body runs independently per element).
- Over reduces the full leading shape (commutative+associative for rank > 1).
  Scan and Prior require rank-1 leading shape.
- `indexed` is a layout (free, lowercase); `Gather` is its materialisation
  (work, PascalCase). When to resolve is a scheduling decision, not an
  algorithmic one.
- `fuse` merges nesting levels (inverse of `cut`); `flatten` merges all.
  `permute` reorders axes within the leading shape. All free (no data
  movement).

### What has been demonstrated

1. Type propagation end-to-end through all operations.
2. Jagged (`~`) emergence from variable-length filtering within Each.
3. Hierarchical index chain: Decompose + chained Gather through two levels.
4. Swappable indexing strategy (3D vs morton, identical results).
5. Struct + Flip: struct-of-arrays to array-of-structs transposition.
6. Triangle mesh as layouts over raw tensors (no custom mesh class).
7. Mean = `Div(Over(Add, xs), Count(xs))` -- function composition.
8. Expression/execution separation via the string-parsed DSL.
9. cuTile GPU code generation: DSL string -> AST -> cuTile Python source,
   with `ct.gather` for computed-index access and built-in bounds checking.
10. Numerical equivalence: numpy DSL evaluator and cuTile kernel produce
    identical results for the neighbor gather predicate (453 voxels, 2094
    active neighbors).
11. End-to-end codegen loop: DSL string -> emitted `@ct.kernel` -> cuTile
    JIT compile -> GPU launch -> correct results (no hand-written kernel).
12. Performance: cuTile kernel 214x faster than numpy, 4.4x faster than
    vectorized PyTorch GPU for the scatter-gather predicate.
13. Cross-leaf neighbor composability: Decompose + chained Gather traverses
    leaf boundaries correctly (300 lookups, 48 cross-leaf, all correct).
14. Cross-leaf GPU codegen: emitter handles Decompose, field, and chained
    Gather with automatic gather fusion. DSL string -> GPU execution for
    hierarchical sparse traversal (817 voxels, 617 cross-leaf, all correct).
15. Idiom detection: `Gather(Gather(A, i), j)` automatically fused into
    a single 4D `ct.gather(A, (i, j_x, j_y, j_z))` at emission time.
16. Scale benchmark: cuTile 5.6-8.7x faster than PyTorch GPU at 4-256
    leaf scale (up to 79K voxels, 474K lookups). GPU scaling excellent:
    65x more work costs only 1.7x more time.
17. CIG concrete tensor format: 2-level sparse grid as two raw tensors
    (lower + leaf_arr). Programmatic builder from voxel coordinates.
18. First fVDB head-to-head: CIG cuTile ijk_to_index is 1.3-1.5x faster
    than NanoVDB's compiled CUDA kernel at 1K-200K voxel scale.
19. Memory characterisation: NanoVDB 8-12x smaller than naive dense-leaf
    CIG. Bitmask compression is the key difference.
20. `masked` layout: first-class sparse-occupancy layout in the type system.
    Bitmask + popcount for dense index computation, zero-cost construction.
21. Compressed CIG with masked leaves: 0.22-0.73x the memory of NanoVDB
    at typical sparsities, while maintaining 1.27-1.34x faster queries.
22. Software popcount via Hamming weight in cuTile (i32): 16-word unrolled
    bitmask chain adds only ~3% overhead vs dense array lookup.
23. u64 popcount variant: `ct.uint64` tiles work correctly. 8 gathers +
    8 popcounts vs 16 + 16. Eliminates i32 mask conversion step. ~10%
    faster at 50K voxels, negligible difference at other scales.
24. Masked Gather codegen: the masked CIG DSL expression compiles to a
    cuTile `@ct.kernel` via the emitter.  The abs-prefix version produces
    a 56-line kernel (2-level) or 101-line kernel (3-level).  Third idiom
    recognised by the emitter (after chained Gather fusion and Decompose
    + field). See `test_cutile_masked_e2e.py`.
25. Tile-parallel emission: `emit_runnable_kernel` generalised to handle
    flat query parallelism (`query_idx = bid * TILE + arange`) alongside
    the existing batch+Map pattern. The masked CIG uses tile-parallel.
26. Absolute prefix-sum masked layout: `masked(mask, abs_prefix)` with
    base offset folded into the prefix at build time.  Query cost O(1)
    per level via 2 gathers + 1 popcount, regardless of mask width.
    Replaces the O(W) unrolled chain and the separate offset gather.
27. 3-level CIG: upper (32^3) + lower (16^3) + leaf (8^3) with
    bit-widths [3, 4, 5]. Fully fused 101-line cuTile kernel with Find
    root lookup + 3 masked levels. Verified against numpy reference for
    single-upper and multi-upper grids.
28. Configurable level stacking: the `[3, 4, 5]` bit-widths are
    parameters, not hard-coded. Any stacking (e.g. `[3, 3, 3, 3]`)
    works with the same masked Gather pattern.
29. `Find(table, key)`: general-purpose linear scan primitive for small
    tables. Emitter unrolls at emit time. Fuses root lookup into the
    cuTile kernel, eliminating the torch barrier.
30. Absolute prefix sums: base offset folded into prefix at build time.
    `masked` back to 2 args. Two gathers per level instead of three.
31. 3-level CIG within 6-20% of NanoVDB query speed at 25-30x less
    memory. Fully fused single-kernel pipeline (Find + 3 masked levels).
32. Barrier-based pipeline: AST partitioned into cutile + collective
    segments, dispatched to GPU kernel compilation and torch ops
    respectively. Hooks mechanism for pluggable backends.
33. conv_grid as fully DSL-driven pipeline: adverb composition
    (EachLeft/EachRight/EachBoth) + HierarchicalKey + Sort + Unique +
    HierarchicalKeyDecode expressed as DSL string, executed through
    barrier-aware pipeline.
34. GPU hash map: build (atomicCAS), lookup (probe), scatter_reduce
    (atomicOr/Add) via NVRTC JIT. Dialect mechanism wraps as DSL nodes.
35. conv_grid_leafwise: O(L*K) bitmask dilation replaces O(N*K) dense
    expansion. 3 kernel launches, 0 Python loops. Hand-fused GPU kernel
    (out-of-DSL, future fusion target).
36. Pipeline GPU data residency: data stays on target device throughout
    pipeline execution. No CPU round-trips between segments.
37. Adverb emission: Over(commutative) dispatches to torch GPU ops as
    a collective. Over inside Map/Each bodies emits inline in cuTile
    (tile-level reduction, not a barrier). EachNode emits through
    cuTile identically to MapNode.
38. AST normalization: parser's `ApplyNode(AdverbApplyNode("Over"), ...)`
    rewritten to canonical `OverNode` before planning, so barrier
    detection and hooks work correctly.
39. CUDA C++ emitter (last-resort backend): `dsl_to_cuda.py` generates
    CUDA C++ from DSL AST, compiled in-memory via NVRTC. Grid-stride
    loops, arithmetic/bitwise ops, array indexing. No files on disk.
40. Three pipeline segment kinds: cutile, collective, cuda. Strict
    preference order: cuTile > torch GPU > CUDA/NVRTC.
41. `DilateLeafMasksNode`: typed 8x8x8 leaf primitive for fused mask
    dilation. GPU dispatches to `conv_grid_dilate_kernel` via hook.
    CPU evaluates via sequential shift + scatter-OR reference.
42. conv_grid_leafwise fully DSL-driven: unified 13-binding DSL
    program with automatic pipeline segmentation. All 14 tests pass
    (11 CPU + 3 GPU).
43. Leafwise conv_grid beats fVDB conv_grid by 1.5-1.7x at 100K+
    voxels with k=5. The O(L*K) bitmask dilation avoids O(N*K) dense
    expansion. DSL-driven pipeline, not hand-wired.
44. `HashMapOccupied(key_arr)`: returns indices of occupied slots,
    hiding the sentinel value. Composes with `Gather` for extracting
    active entries without leaking implementation details.
45. `DilateLeafMasks` simplified to 4 args (storage_size derived from
    key array shape internally).
46. conv_grid_leafwise unified as a single 13-binding DSL program
    (was three imperative phases). Zero imperative torch in the
    pipeline path; all operations expressed as DSL primitives.
47. `ExpandOffsets` removed from parser. Replaced by adverb
    composition: `reshape(fuse(EachLeft(EachRight(EachBoth(Add)),
    coords, offsets)), [-1])`. The DSL composes from existing
    primitives rather than special-casing.
48. Pipeline barrier detection extended: `ApplyNode` wrapping
    `EachLeft`/`EachRight` adverbs with data arguments classified as
    barriers (they create new iteration dimensions).
49. cuTile fallback: cutile segments that fail compilation gracefully
    fall back to the evaluator. cuTile is an optimisation; the
    evaluator's torch ops are device-agnostic and correct on CUDA.
50. Adverb cuTile emission: composed adverb chains
    (`EachLeft(EachRight(EachBoth(Add)))`) compile directly to cuTile
    GPU kernels. The adverb chain is the scheduling specification:
    EachLeft/EachRight determine index decomposition (`qidx // K`,
    `qidx % K`), EachBoth maps to per-axis gather, the verb becomes
    the scalar operation. Layout ops (fuse, reshape) are no-ops in
    the emitter. The planner classifies emittable adverb barriers as
    `kind="cutile"` instead of `"collective"`. Multi-axis output
    scatter and K as a kernel constant. No special-case lowering --
    the adverb composition IS the code generation input.

### Completed milestones (previously "next steps")

- **Cross-leaf neighbors** (v5): done. The DSL composes Decompose + chained
  Gather across leaf boundaries cleanly. See `test_cross_leaf.py`.
- **GPU codegen loop** (v5): done. DSL string -> emitted `@ct.kernel` ->
  cuTile JIT -> GPU launch -> correct results. See `test_cutile_e2e.py`.
- **Cross-leaf GPU codegen** (v6): done. Emitter extended with Decompose,
  field, and chained Gather fusion. DSL string -> GPU execution for
  hierarchical traversal. See `test_cutile_cross_leaf.py`,
  `bench_cutile_cross_leaf.py`.
- **`masked` layout + compressed CIG** (v8): done. Bitmask-compressed
  nodes with popcount access. See `test_masked.py`.
- **Masked codegen** (v9): done. The 4-line masked CIG DSL expression
  compiles to a cuTile kernel. `MaskedNode` emission, masked-Gather
  idiom detection (u64 popcount chain). See `dsl_to_cutile.py`.
- **3-level CIG** (v10): done. Prefix-sum popcounts (O(1) per level).
  3-level CIG with root linear scan + fused cuTile chain. See
  `test_cig3.py`, `test_cutile_cig3_e2e.py`.
- **Fused root + abs prefix + CSE** (v11): done. Find primitive fuses
  root into cuTile. Absolute prefix sums. 3-level CIG within 22-24%
  of NanoVDB query speed at 20-30x less memory. See
  `bench_cig3_vs_fvdb.py`.

**Latest benchmark** (50K queries, [0, 4096)^3, RTX PRO 6000 Blackwell):

| Voxels | CIG3 (us) | fVDB (us) | CIG3/fVDB | CIG3 memory |
|--------|-----------|-----------|-----------|-------------|
| 1,000  | 137       | 113       | 0.82x     | 0.03x fVDB  |
| 10,000 | 132       | 111       | 0.84x     | 0.03x fVDB  |
| 50,000 | 137       | 109       | 0.79x     | 0.04x fVDB  |
| 200,000| 139       | 110       | 0.80x     | 0.05x fVDB  |

### Completed since v11

**Multi-step compilation (barrier-based pipeline).**  The pipeline
planner (`dsl_pipeline.py`) partitions programs into `cutile`
(kernel-fusible) and `collective` (torch GPU barrier) segments.  The
executor dispatches: collective segments to `torch.sort` /
`torch.unique` / `torch.nonzero`; cutile segments to compiled cuTile
`@ct.kernel` launches (when `device="cuda"`).  AST rewriting promotes
inter-segment references to kernel inputs automatically.  The hooks
mechanism on `EvalEnv` makes the dispatch pluggable.

**conv_grid (fully DSL-driven).**  First multi-step pipeline
application.  Algorithm expressed as a DSL program string using adverb
composition for coordinate expansion (EachLeft/EachRight/EachBoth) +
HierarchicalKey + Sort + Unique + HierarchicalKeyDecode.  Parsed,
planned, and executed through the pipeline executor.  Correctness
verified against fVDB reference at multiple scales.

**GPU hash map.**  Hand-written CUDA kernels for hash map build
(atomicCAS), lookup (probe loop), and scatter-reduce (atomicOr/Add).
Exposed to the DSL via dialect nodes (HashMapBuild, HashMapLookup) and
pipeline hooks for GPU dispatch.

**conv_grid_leafwise (now DSL-driven).**  Topology expansion via
leaf-level bitmask dilation.  O(L*K) word-level ops vs conv_grid's
O(N*K) dense expansion.  Originally out-of-DSL with a hand-fused CUDA
kernel.  Now expressed as a DSL pipeline with three phases: torch
collectives for hash map build, `DilateLeafMasksNode` (typed 8x8x8
leaf primitive dispatched to the fused CUDA kernel via hook), and
torch collectives for coordinate extraction.

**Leafwise conv_grid benchmark** (RTX PRO 6000 Blackwell,
`bench_conv_grid_leafwise.py`):

| Voxels | Kernel | Leaves | Output coords | Leafwise GPU (us) | fVDB conv (us) | Speedup |
|--------|--------|--------|---------------|--------------------|----------------|---------|
| 1,000 | 3x3x3 | 986 | 26,983 | 5,248 | 5,456 | 1.04x |
| 10,000 | 3x3x3 | 8,626 | 267,962 | 5,995 | 5,384 | 0.90x |
| 50,000 | 3x3x3 | 25,586 | 1,298,977 | 5,906 | 6,918 | 1.17x |
| 100,000 | 3x3x3 | 31,206 | 2,502,578 | 6,617 | 9,963 | 1.51x |
| 200,000 | 3x3x3 | 32,682 | 4,651,077 | 9,525 | 14,587 | 1.53x |
| 50,000 | 5x5x5 | 25,586 | 5,237,753 | 10,011 | 15,672 | 1.57x |
| 100,000 | 5x5x5 | 31,206 | 8,887,300 | 17,369 | 28,962 | 1.67x |

The leafwise approach matches fVDB at small scale and **beats fVDB by
1.5-1.7x at large scale** (100K+ voxels, k=5).  The O(L*K) bitmask
dilation avoids the O(N*K) dense expansion that dominates fVDB's cost
at high voxel counts.  This is now a DSL-driven pipeline (not
hand-wired), with the fused CUDA kernel as the backend for the
DilateLeafMasks primitive.

**Adverb emission.**  `OverNode` (commutative: Add, Mul, Max, Min, Or)
and `EachNode` are now emittable in both the cuTile and CUDA emitters.
Over dispatches to torch GPU ops (torch.sum, etc.) as a collective.
Over inside Map/Each bodies is cuTile-emittable (tile-level reduction,
not a barrier).  AST normalization rewrites the parser's composed
`ApplyNode(AdverbApplyNode("Over"), ...)` form to canonical `OverNode`.

**CUDA C++ emitter (last-resort backend).**  New `dsl_to_cuda.py`
generates CUDA C++ from DSL AST, compiled in-memory via NVRTC.  No
files on disk.  Handles grid-stride loops, arithmetic/bitwise ops,
array indexing, and atomics.  Strict preference order: cuTile > torch
GPU > CUDA/NVRTC.  Used only when the first two options are
insufficient (e.g. atomicOr in DilateLeafMasks).

**Pipeline segment routing.**  Three segment kinds: `cutile`, `collective`,
`cuda`.  `_run_cuda_segment` compiles and launches CUDA kernels through
the existing NVRTC infrastructure.

**Pipeline cleanup pass.**  Removed CPU-in-GPU-hot-path violations
(`.cpu()` calls in collective hooks and cutile segment results).
Vectorized CPU loops over tensor data in CIG construction and evaluator
unique.  Added `is_cuda` guards on CPU-only reference hashmap.  Added
`# DSL status:` / `# OUT_OF_DSL:` markers to all application files.

### The OUT_OF_DSL problem

Two remaining components bypass the DSL/AST pipeline.  A third
(`conv_grid_leafwise`) has been resolved.

**1. `conv_grid_leafwise.py` -- RESOLVED.**  Now DSL-driven via
`DilateLeafMasksNode`, a typed 8x8x8 leaf primitive.  The DSL pipeline
has three phases: torch collectives (hash map build), DilateLeafMasks
(dispatched to the fused CUDA kernel via hook), torch collectives
(MaskToCoords).  The hand-written CUDA kernel is the backend, not an
escape hatch.  See Phase 4 of the OUT_OF_DSL compliance work.

**2. `cig.py` -- CIG construction.**  `build_compressed_cig3` is
imperative torch code (sort + unique + scatter).  The query path is
fully DSL-driven; construction is not.  Construction is structurally a
sequence of Sort + Unique + scatter operations -- the same pattern the
pipeline already handles for conv_grid -- but with bitmask packing that
needs vectorized scatter-OR.

**3. `hashmap_cuda.py` -- GPU hash map kernels.**  Build, lookup,
scatter_reduce, and conv_grid_dilate are hand-written NVRTC CUDA.  The
dialect mechanism (`dialect_hashmap.py`) wraps build/lookup as DSL
nodes, so pipeline programs can reference them.  But the actual compute
is opaque to the emitter.

### Approaches to resolving OUT_OF_DSL

Several paths exist, with different trade-offs.  These need to be
weighed before committing to an implementation.

**Approach A: AST-level fusion.**  Express the full algorithm in the
DSL.  Add a fusion pass that recognises patterns spanning multiple
pipeline segments and emits a single fused kernel.  The DSL program is
the source of truth; the fused kernel is a compiler optimisation.

- Pro: purest validation of the thesis.  The DSL can express the
  algorithm and the compiler can make it fast.
- Con: requires a non-trivial fusion pass.  The conv_grid_leafwise
  kernel fuses across collective barriers (hash map + scatter-reduce),
  which means the fusion pass must reason about atomics and cross-thread
  coordination.
- Candidate: conv_grid_leafwise, since the DSL program already exists
  in the docstring and the hand-fused kernel serves as ground truth.

**Approach B: Intrinsic dialect expansion.**  Promote the hash map
and scatter-reduce operations to first-class DSL primitives with
dedicated GPU backends.  The emitter generates code that calls into
the existing CUDA kernels as intrinsics.

- Pro: reuses proven kernels.  Minimal emitter changes.
- Con: the DSL vocabulary grows with implementation-specific operations
  (HashMapBuild, ScatterReduce(OR), etc.).  The "minimal coding"
  thesis weakens.
- This is partially done: `dialect_hashmap.py` already wraps
  HashMapBuild/Lookup.

**Approach C: Hybrid with idiom recognition.**  Express the algorithm
in the DSL using generic primitives (Sort, Unique, Gather,
ScatterReduce).  The emitter recognises the specific pattern
(expand + scatter_reduce(OR) + hash_probe) and replaces the segment
sequence with the fused kernel.  Similar to how the emitter already
fuses `Gather(Gather(A, i), j)` into a single 4D gather.

- Pro: the DSL stays generic; fusion is an optimisation, not a language
  feature.  Idiom recognition is already a proven pattern in the emitter
  (three idioms exist: chained Gather, masked Gather, Find).
- Con: the idiom needs to span pipeline segment boundaries (the hash map
  build is a barrier).  This is harder than intra-segment idiom
  recognition.

**Approach D: Construction via pipeline.**  Express CIG construction
as a DSL pipeline program (like conv_grid).  The sort + unique +
scatter pattern maps directly to existing pipeline primitives.  The
bitmask packing step becomes a new `PackMask` or `ScatterBitOr` node.

- Pro: construction and query share the same DSL infrastructure.
- Con: CIG construction is a one-time cost and the current imperative
  torch code works.  Lower priority than the hot-path dilation kernel.

**Recommended sequence:**  Start with Approach C (hybrid idiom
recognition) targeting `conv_grid_leafwise`.  The DSL program already
exists.  The pattern -- expand pairs, scatter-reduce with fused hash
probe -- is specific enough for reliable idiom detection.  If the
fusion pass generalises cleanly, apply it to CIG construction
(Approach D) and then to the hash map kernels themselves.  Approach A
is the long-term goal; Approach C is the pragmatic bridge.

### Kernel fusion: current state and cross-barrier challenge

The emitter already performs **intra-segment fusion** -- merging adjacent
AST nodes within a single pipeline segment into one block of generated
code.  Three idioms are recognised today:

1. **Chained Gather**: `Gather(Gather(A, i), j)` fuses into a single
   4D `ct.gather(A, (i, j_x, j_y, j_z))`.  The two-step "look up leaf
   block, then index into it" becomes one fused gather.
2. **Masked Gather**: `masked(mask, prefix)` + `Gather(masked, field)`
   fuses into bitmask check + popcount + offset in one code block.
3. **Find**: `Find(table, key)` unrolls the linear scan at emit time,
   inlining the entire loop into the kernel body.

These are all **intra-segment**: the fusion happens inside the emitter's
tree-walk over a single segment's AST.  The pipeline planner is not
involved -- it sees one segment, the emitter fuses internally.

The unsolved problem is **cross-barrier fusion**.  The pipeline planner
splits the AST at barrier nodes (Sort, Unique, HashMapBuild,
HashMapLookup, ShiftLeafMask, MaskToCoords).  Each segment becomes a
separate kernel launch or torch op.  The `conv_grid_leafwise` problem
requires fusing across these barriers:

```
Segment 1 (collective): HashMapBuild(unique_keys)       -- build output leaf map
Segment 2 (fused):      for each (leaf, offset) pair:
                           shift mask, decompose boundary,
                           hash-probe target leaf,
                           atomicOr shifted bits into output -- accumulate masks
Segment 3 (cutile):     MaskToCoords(output_masks)       -- extract voxel coords
```

The hand-fused `conv_grid_dilate_kernel` combines segments 1-2 into a
single kernel where each thread does its own hash probe (open addressing)
and `atomicOr` into the output.  This works because OR is commutative
and associative, so concurrent writes produce the same result as
sequential scatter-reduce.

**What cross-barrier fusion requires:**

- A **fusion analysis pass** that examines adjacent segments and
  determines whether the inter-segment data dependency can be replaced
  by atomic operations.
- For `ScatterReduce(op, ...)` where `op` is commutative+associative
  (OR, ADD, MAX), the build-then-scatter pattern can become
  probe-and-atomic in a single kernel.
- The hash map itself must be pre-built (or built with `atomicCAS` in
  the same kernel -- which is what `gpu_hash_map_build` already does).
- The fusion pass recognises the pattern: `HashMapBuild(keys)` followed
  by element-wise work that scatters into the map with a commutative
  reducer, and replaces the two segments with a single fused kernel that
  combines hash probe and atomic scatter.

This is the compiler-mechanics counterpart of Approach C above.
Approach C describes the product strategy (express in DSL, fuse via
idiom recognition); this section describes the specific compiler pass
that makes it work.  The existing intra-segment idiom detection is the
proof that pattern-based fusion works; the challenge is extending it
across segment boundaries where atomics replace sequential dependencies.

### Other recommended steps

**1. Close the query performance gap (16-21% vs NanoVDB).**  The
3-level CIG is within 16-21% of NanoVDB at 20-30x less memory.
Investigate hardware popcount via `CUDA_TILE_LOGS=CUTILEIR`.  If
cuTile does not lower the Hamming weight pattern to `__popcll()`, add
a `Popc` DSL primitive that maps to the hardware intrinsic.

**2. Next fVDB operations.**  `neighbor_indexes`, `dilated_grid`,
`sample_trilinear`, `inject_from`/`inject_to`.  The cross-leaf pattern
(v5/v6) and pipeline architecture already support these.

**3. Batch dimension.**  One kernel launch per grid (simplest), jagged
outer `Each` (DSL-native), or packed contiguous storage.

### cuTile backend notes

**Target backend: cuTile** (NVIDIA's tile-based Python GPU DSL on TileIR).

**Why cuTile:**
1. `ct.gather` with computed indices + `check_bounds` + `padding_value`
   maps directly to our `Gather` + `InBounds` + sentinel pattern.
2. Grid-level parallelism (`ct.bid`) maps to `Each`.
3. Tile-level elementwise ops map to `Map` over static-size collections.
4. `ct.load` with power-of-two tiles maps to `cut` for aligned access.
5. Hardware-aware: leverages TMA, tensor cores on Blackwell.

**Key constraint discovered:** `ct.load` requires power-of-two tile
dimensions; `ct.gather` does not. For our scatter-gather patterns,
`ct.gather` is the primary primitive. `ct.load` is used only for
tile-aligned block access (e.g. loading a full (8,8,8) leaf).

**Environment** (satisfied): RTX PRO 6000 Blackwell (compute 12.0),
driver 591.59, cuda-tile 1.1.0,
`source ~/.venvs/fvdb_cutile/bin/activate`.

### Kernel synthesis: file-on-disk problem and direct IR emission

The prototype synthesises GPU kernels via two paths.  Both generate
source code in a higher-level language that a downstream framework
compiles to an intermediate representation (TileIR or PTX).  This works,
but both paths have engineering trade-offs -- particularly around the
files they leave on disk.

**Two synthesis paths today (strict preference: cuTile > torch > CUDA):**

| Path | Input | Framework | IR | Files on disk | Preference |
|------|-------|-----------|----|---------------|------------|
| cuTile | Python source (`@ct.kernel`) | cuda-tile JIT | TileIR -> CUBIN | Yes (`_generated/*.py`) | #1 (preferred) |
| CUDA/NVRTC | CUDA C++ source string | NVRTC | PTX -> CUBIN | No (in-memory) | #3 (last resort) |

- **cuTile path** (`dsl_to_cutile.py`, preference #1): the emitter
  produces Python source containing `@ct.kernel` decorated functions,
  writes them to `_generated/*.py` files, then imports via `importlib`.
  This is forced by cuTile's reliance on `inspect.getsource()` -- the
  JIT compiler needs to read the function's source text from a real file.
  Currently ~25 generated files across `prototype/_generated/` and
  `tests/_generated/`.  These are breadcrumbs that a library user would
  find surprising.  Handles: Map, Each, Over (inline), Gather (with
  chained/masked fusion), Decompose, Find, HierarchicalKey.
- **CUDA/NVRTC path** (`dsl_to_cuda.py`, preference #3): CUDA C++
  source strings are generated from DSL AST or hard-coded, compiled to
  PTX in-memory via NVRTC, loaded as `CUmodule`, and launched via the
  driver API.  No files on disk.  Used ONLY when cuTile and torch
  cannot handle the operation (atomics, inline hash probing, boundary
  decomposition).  Current use: `DilateLeafMasksNode` dispatched via
  pipeline hook to `conv_grid_dilate_kernel`.

**Direct IR emission -- eliminating the file problem:**

- **TileIR direct emission.**  The `cuda_tile` package exposes an
  MLIR-based `cuda_tile` dialect.  The emitter could target TileIR
  directly (as MLIR text or through the TileIR builder API), then
  compile via `tileiras` to CUBIN without writing Python files.  The
  emitter's core logic (idiom detection, masked Gather, Find, chained
  Gather fusion) stays the same -- only the output representation
  changes from Python cuTile source strings to TileIR nodes.  This
  eliminates the `_generated/` directory entirely for tile-parallel
  kernels.
- **PTX direct emission.**  For CUDA-path kernels, the existing NVRTC
  flow already avoids files.  Emitting PTX directly would eliminate the
  NVRTC dependency, but PTX is a low-level virtual ISA -- the loss of
  readability and maintainability is not worth the marginal gain.  The
  NVRTC path is already clean; TileIR is the path that needs fixing.
- **Hybrid target model.**  Tile-parallel segments (gather/scatter,
  pointwise, masked Gather) emit TileIR.  Atomic/collective segments
  (hash map build, scatter-reduce with atomicCAS/atomicOr) emit CUDA
  C++ via NVRTC.  Each path uses the IR that matches its abstraction
  level.  No generated files in either case.  This also opens the door
  to AOT compilation: TileIR segments can be serialised as `.cutile`
  bytecode, and NVRTC segments can be cached as PTX or CUBIN -- both
  without writing Python source.

The hybrid model aligns with the pipeline architecture: the planner
already knows whether a segment is `cutile` or `collective`.  Routing
each segment kind to its natural IR backend is a straightforward
extension.

### Medium-term items

- **Cross-barrier fusion** (generalise the DilateLeafMasks pattern):
  the current `DilateLeafMasksNode` is a hand-specified fused primitive.
  The next step is automatic fusion: the planner recognises adjacent
  segments where atomics can replace sequential dependencies and emits
  a single fused CUDA kernel.  See "Kernel fusion" section above.
- **Direct IR emission** (eliminate `_generated/` files): retarget the
  cuTile emitter to TileIR and keep the NVRTC path in-memory.  See
  "Kernel synthesis" section above for the hybrid model and trade-offs.
- **Materialisation scheduling**: automatically decide where to insert
  Gather (early vs late) based on data characteristics.  The Halide
  schedule analogy.
- **Let-bindings in Map bodies**: the parser supports only single
  expressions.  Let-bindings would avoid deeply nested expressions.
- **`flip` in DSL**: currently Python-only.  Add as a lowercase keyword.
- **World-to-grid transforms**: design decision is that transforms are
  instancing metadata, not structural properties.  They live outside the
  CIG (deliberate departure from NanoVDB).

### North star

Compile DSL expressions to working cuTile / CUDA code via algebraic
optimisation and idiom detection. The entire SPH density calculation can be
written as `(R/P/:)\:':` in K9 syntax -- a tacit composition of reduction,
product, and structured iteration. This is what the framework makes possible:
domain algorithms as tiny compositions of functions over structured tensors,
compiled to GPU code without framework-specific imperative implementations.

The compressed CIG `ijk_to_index` -- bitmask-compressed sparse grid lookup
with hierarchical traversal, popcount, and root search -- has a similar
ultra-minimal form.  The full 3-level version with fused root lookup:

```
(F[R];m[M,P];m[M,P];m[M,P])/@q\[3;4;5]
```

Decompose query `q` at bit-widths [3, 4, 5], then fold-index (`/@`)
through four stages: `F[R]` (Find in root table `R`), then three masked
levels `m[M,P]` (bitmask `M` + absolute prefix `P`, popcount access).
Changing the tree configuration is changing the fold: `[3,3,3,3]` for
four 8^3 layers, `[4,4,4]` for three 16^3, or any other stacking.

Most programmers would find this impenetrable, and readability matters --
the 10-line DSL form is the one intended for human use. But the point is
not the syntax. The point is that a sparse grid query which previously
required a large C++/CUDA library and thousands of lines of code can be
expressed as a single composed expression over structured tensors, and
that expression compiles to a GPU kernel that is within 6-20% of the
hand-written implementation's speed at 25-30x less memory.
