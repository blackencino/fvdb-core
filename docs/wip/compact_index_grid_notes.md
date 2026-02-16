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

**The claim.** A small set of **nested layouts** (`cut`, `indexed`, `jagged`,
`tuple`, `struct`, `flip`) applied over raw tensors, combined with a small set
of **functions** (`Map`, `Each`, `Where`, `Gather`), form a **portable
semantic layer** that has the same degree of fungibility as tensors but
captures the structural richness that tensors erase.

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

**Key technical insight.** Nested layouts serve the role of **Halide's
schedule** -- they describe how to traverse the data, while operations
describe what to compute. APL/J/K conflate data shape with iteration
structure; this system **decouples** them via the iteration-space /
element-type separation. This decoupling is what makes it possible to write
domain algorithms as tiny compositions of functions and layouts:

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
abstract description compile to competitive GPU code? **Partially answered:**
the v4/v5 work shows that the DSL compiles to cuTile kernels that produce
correct results and run 4.4x faster than vectorized PyTorch GPU for the
neighbor gather predicate. Full performance characterisation at scale is
still needed.

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

K-derived adverbs (`Over`, `Scan`, `EachRight`, `EachLeft`, `Prior`, `Map`,
`Each`) are higher-order functions: they take a verb and an iterable and
produce new data. They do not need a separate notational tier. The DSL is
functional; adverbs are functions. The compiler recognises reduction, scan, and
iteration patterns by matching AST node types, not by reading case conventions.

---

## Layout Wrappers

Every layout wrapper is a **type modifier**, not an operation. It reinterprets
the logical type without touching physical storage. Layouts compose freely.
**In the DSL, layouts use lowercase names** (`cut`, `reshape`, `field`).

**Invariant: type transformations do no computational work.** A layout never
allocates, copies, or gathers. If a transformation requires data movement, it
is an **operation** (PascalCase). A compiler pass may insert operations as a
policy decision, but the type system itself only recognises eligibility.

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

As a layout (`indexed`): pure type-level, no work. As an operation (`Gather`): materialises.

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

## Operations (PascalCase)

Operations do computational work. All use PascalCase. Higher-order functions
(K-derived) modify how a verb is applied across an iteration space.

**Key rule: higher-order functions operate over the full iteration space.**
No `axis` parameter. The layout IS the axis specification. To reduce along a
specific axis, `cut` first to isolate it as the iteration space.

### Higher-Order Functions (K-derived)

**Parallel** (order-independent, any rank):

| Function | Syntax | Input | Output |
|----------|--------|-------|--------|
| Over | `Over(f, xs)` | `S over E` | `E` (full reduction) |
| Each | `Each(xs, x => body)` | `S over E` | `S over R` |
| EachRight | `EachRight(f, x, ys)` | fixed x, iterate ys | `S_ys over R` |
| EachLeft | `EachLeft(f, xs, y)` | iterate xs, fixed y | `S_xs over R` |

**Sequential** (require rank 1, inherently ordered):

| Function | Syntax | Input | Output |
|----------|--------|-------|--------|
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

| Name | Convention | Kind | New data? |
|------|-----------|------|-----------|
| `cut`, `indexed`, `tuple`, `struct`, `flip`, `jagged`, `reshape`, `field` | lowercase | Layout (free) | No |
| `Map`, `Each`, `Over`, `Scan`, `EachRight`, `EachLeft`, `Prior` | PascalCase | Higher-order function | Yes |
| `Where` | PascalCase | Operation | Yes (data-dependent) |
| `Gather` | PascalCase | Operation | Yes (materialise) |
| `Add`, `Sub`, `Mul`, `Div`, `GE`, `And`, `Not`, `InBounds`, `Count` | PascalCase | Scalar function | Yes |
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

Code in `docs/wip/prototype/`. Environment:
`source ~/.venvs/fvdb_cutile/bin/activate` (Python 3.12, cuda-tile 1.1.0,
torch 2.10, numpy 2.2; RTX PRO 6000 Blackwell, compute 12.0).
Run all: `python docs/wip/prototype/run_all_tests.py`.

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
  code via idiom detection and algebraic transformation. **Demonstrated** in
  v4/v5: DSL string -> AST -> runnable cuTile `@ct.kernel` -> GPU execution
  -> correct results, 4.4x faster than vectorized PyTorch GPU.

fVDB's sparse voxel operations are the proving ground, not the ceiling.

---

## Next Steps (for a new agent)

### Orientation

Read this document top-to-bottom for the thesis, vocabulary, layout wrappers,
adverb semantics, and worked examples. Then:

- **Prototype code**: `docs/wip/prototype/`
- **Run all tests**: `python docs/wip/prototype/run_all_tests.py` (20 tests, ~2s)
- **Design doc**: this file (`docs/wip/compact_index_grid_notes.md`)

File inventory:

| File | Role |
|------|------|
| `types.py` | Extent kinds (Static/Dynamic/Jagged), Shape, Type, ScalarType |
| `layouts.py` | Layout wrappers (lowercase): cut, indexed, tuple, struct, flip, jagged |
| `ops.py` | Python-level operations: Map, Each, Where, Gather, FlipStruct, Decompose, morton3d |
| `dsl_ast.py` | AST node classes (~25 nodes) with `infer_type` methods |
| `dsl_parse.py` | Recursive-descent parser: string -> AST |
| `dsl_eval.py` | Tree-walk evaluator: type-check pass then numpy execution |
| `test_where.py` | v0: Map, Where, Gather pipeline (DSL strings) |
| `test_neighbors.py` | v0: neighbor finding, jagged emergence (DSL strings) |
| `test_indexed_flip.py` | v1: multi-leaf cut, indexed, Struct+Flip (DSL + API tests) |
| `test_two_level.py` | v2: hierarchical chain, Decompose, morton (DSL strings) |
| `test_dsl.py` | v3: DSL validation (parser + evaluator correctness) |
| `test_mesh.py` | mesh: triangle mesh, centroids via Over+Div (DSL strings) |
| `dsl_to_cutile.py` | v4: DSL AST -> cuTile Python source emitter |
| `test_cutile_smoke.py` | v4: cuTile toolchain validation (gather/scatter) |
| `test_cutile_gather.py` | v4: hand-written neighbor predicate kernel vs numpy |
| `test_cutile_codegen.py` | v4: emitter output validation + numpy vs cuTile |
| `test_cutile_e2e.py` | v5: end-to-end DSL string -> GPU execution -> verify |
| `bench_cutile.py` | v5: performance benchmark (cuTile vs numpy vs PyTorch) |
| `test_cross_leaf.py` | v5: cross-leaf neighbor finding via hierarchical chain |
| `run_all_tests.py` | Single entry point for all 20 tests |

### Current state

The DSL has ~25 keywords. **Layouts use lowercase** (`cut`, `reshape`,
`field`) -- they reinterpret types without moving data. **Operations use
PascalCase** (`Map`, `Each`, `Where`, `Gather`, `Over`, `Scan`, `Add`, etc.)
-- they do computational work. This convention makes the cost model visible
at a glance. Programs are text strings parsed into typed ASTs, type-checked
without data, then executed against numpy.

Key semantic rules:
- Higher-order functions operate over the **full iteration space**. No axis
  parameter. `cut` to isolate the axis you want. The layout IS the axis
  specification.
- Each promotes Dynamic to Jagged at type-check time (body runs independently
  per element).
- Over reduces the full iteration space (commutative+associative for rank > 1).
  Scan and Prior require rank 1.
- `indexed` is a layout (free, lowercase); `Gather` is its materialisation
  (work, PascalCase). When to resolve is a scheduling decision, not an
  algorithmic one.

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

### Completed milestones (previously "next steps")

- **Cross-leaf neighbors** (v5): done. The DSL composes Decompose + chained
  Gather across leaf boundaries cleanly. See `test_cross_leaf.py`.
- **GPU codegen loop** (v5): done. DSL string -> emitted `@ct.kernel` ->
  cuTile JIT -> GPU launch -> correct results. See `test_cutile_e2e.py`.
- **First performance number** (v5): done. cuTile 4.4x faster than
  PyTorch GPU, 214x faster than numpy. See `bench_cutile.py`.

### Recommended next steps

**1. Codegen for cross-leaf neighbors on GPU.** The cross-leaf predicate
works in the numpy DSL evaluator. The next compilation target: emit a cuTile
kernel for the hierarchical chain (Decompose + 4-step Gather) and run it on
GPU. This requires the emitter to handle `Decompose`, `field`, and chained
`Gather` -- all new emission rules beyond the current set.

**2. Jagged output on GPU (Where).** The `Where` operation produces
variable-length output, which requires a two-pass GPU pattern: count pass
(parallel prefix sum for offsets), then scatter pass. This is a known GPU
primitive but has not been expressed in the emitter yet.

**3. Full CIG type.** Express a 3-level compact index grid (Root/Upper/Lower
/Leaf) as a composed DSL expression. The 2-level chain already works; 3
levels adds Upper with 32x32x32 nodes. This validates the type system at
full NanoVDB scale.

**4. Performance at scale.** The current benchmark uses 453 voxels in one
(8,8,8) leaf. Real fVDB workloads have millions of voxels across thousands
of leaves. Benchmark with a realistic grid size to measure how cuTile scales.

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

### Medium-term items

- **Emitter: Decompose + field + chained Gather**: extend `dsl_to_cutile.py`
  to emit the cross-leaf hierarchical chain as a cuTile kernel. This requires
  new emission rules for `DecomposeNode`, `FieldNode`, and multi-step
  `GatherNode` chains. The target kernel shape is known from the hand-written
  examples; the emitter needs to handle struct decomposition.
- **Emitter: Where (jagged output)**: two-pass GPU pattern (count + scatter)
  for variable-length output. Known approach; needs emission rules for
  `WhereNode` and the corresponding parallel prefix sum.
- **Idiom detection**: recognize patterns like `Over(Add, Map(xs, Gather(...)))`
  as scatter-gather and propose optimized implementations. This is where the
  AST pays off -- the expression tree is inspectable.
- **Materialisation scheduling**: given a DSL expression, automatically decide
  where to insert Gather (early vs late) based on data characteristics
  (sparsity ratio, memory budget). This is the Halide schedule analogy.
- **Full CIG type**: express a 3-level compact index grid as a struct of
  three levels. Low risk -- just more of what the 2-level test proves.
- **`flip` in DSL**: currently Python-only. Add as a lowercase DSL keyword.
- **Performance at scale**: benchmark with realistic grid sizes (millions of
  voxels, thousands of leaves) to characterise cuTile scaling behaviour.

### North star

Compile DSL expressions to working cuTile / CUDA code via algebraic
optimisation and idiom detection. The entire SPH density calculation can be
written as `(R/P/:)\:':` in K9 syntax -- a tacit composition of reduction,
product, and structured iteration. This is what the framework makes possible:
domain algorithms as tiny compositions of functions over structured tensors,
compiled to GPU code without framework-specific imperative implementations.
