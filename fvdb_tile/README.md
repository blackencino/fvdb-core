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
**In the DSL, layouts use lowercase names** (`cut`, `reshape`, `field`,
`masked`).

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

### Masked

Fixed-shape space with **sparse occupancy**. The counterpart of `jagged`
(variable-length segments): where `jagged` handles "how many elements per
group?", `masked` handles "which positions in a fixed block have data?"

- Iteration space: from the mask shape (e.g. `(8,8,8)` for a leaf block).
- Element type: from the flat data array.
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
| `cut`, `indexed`, `tuple`, `struct`, `flip`, `jagged`, `reshape`, `field`, `masked` | lowercase | Layout (free) | No |
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

   | Leaves | Voxels | Lookups | numpy (us) | PyTorch GPU (us) | cuTile (us) | CT/PT |
   |--------|--------|---------|------------|------------------|-------------|-------|
   | 4 | 1,217 | 7,302 | 58,653 | 471 | 57 | 8.3x |
   | 64 | 19,778 | 118,668 | 980,862 | 478 | 58 | 8.2x |
   | 256 | 79,206 | 475,236 | 3,828,976 | 498 | 102 | 4.9x |

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
length segments). A masked layout wraps a fixed-shape iteration space with a
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

**Claim**: nested layouts + rank-matched adverbs can fully specify structured
geometric and scientific computations -- sparse grids, meshes, point clouds,
scene graphs -- in a form that is:

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

| File | Role |
|------|------|
| `prototype/types.py` | Extent kinds (Static/Dynamic/Jagged), Shape, Type, ScalarType |
| `prototype/layouts.py` | Layout wrappers (lowercase): cut, indexed, tuple, struct, flip, jagged, masked |
| `prototype/ops.py` | Python-level operations: Map, Each, Where, Gather, FlipStruct, Decompose, morton3d |
| `prototype/dsl_ast.py` | AST node classes (~25 nodes) with `infer_type` methods |
| `prototype/dsl_parse.py` | Recursive-descent parser: string -> AST |
| `prototype/dsl_eval.py` | Tree-walk evaluator with hooks: type-check pass then numpy execution |
| `prototype/dsl_pipeline.py` | Barrier-aware pipeline planner, GPU collective dispatch, cutile segment compilation |
| `prototype/dsl_to_cutile.py` | DSL AST -> cuTile Python source emitter (`emit_runnable_kernel`) |
| `prototype/cig.py` | 3-level CompressedCIG3 builder, root lookup, numpy reference query |
| `prototype/conv_grid.py` | conv_grid topology expansion via pipeline (expand + Sort + Unique) |
| `tests/test_where.py` | v0: Map, Where, Gather pipeline (DSL strings) |
| `tests/test_neighbors.py` | v0: neighbor finding, jagged emergence |
| `tests/test_indexed_flip.py` | v1: multi-leaf cut, indexed, Struct+Flip |
| `tests/test_two_level.py` | v2: hierarchical chain, Decompose, morton |
| `tests/test_dsl.py` | v3: DSL validation (parser + evaluator correctness) |
| `tests/test_mesh.py` | mesh: triangle mesh, centroids via Over+Div |
| `tests/test_sort_unique.py` | Sort/Unique DSL primitive correctness |
| `tests/test_pipeline.py` | Pipeline planning, collective dispatch, cutile segment compilation |
| `tests/test_conv_grid.py` | conv_grid correctness, stride semantics, immutability |
| `tests/test_masked.py` | v8: masked layout DSL tests |
| `tests/test_cig3.py` | v10: 3-level CIG builder, root lookup, numpy reference |
| `tests/test_cross_leaf.py` | v5: cross-leaf neighbors via DSL evaluator |
| `tests/test_cutile_smoke.py` | v4: cuTile toolchain validation (gather/scatter) |
| `tests/test_cutile_e2e.py` | v5: end-to-end DSL -> GPU execution -> verify |
| `tests/test_cutile_cross_leaf.py` | v6: cross-leaf GPU codegen, compile, launch, verify |
| `tests/test_cutile_cig3_e2e.py` | v10/v11: 3-level CIG DSL -> cuTile codegen (fused Find) |
| `benchmarks/bench_cig3_vs_fvdb.py` | v11: 3-level CIG vs fVDB head-to-head |
| `benchmarks/bench_cutile_cross_leaf.py` | v6: cross-leaf scale benchmark |
| `bench_cig3_vs_fvdb.py` | v11: 3-level CIG vs fVDB head-to-head benchmark |
| `verify_memory.py` | v11: detailed byte-level memory comparison CIG3 vs fVDB |
| `run_all_tests.py` | Single entry point for non-GPU tests |

### Current state

The DSL has ~25 keywords. **Layouts use lowercase** (`cut`, `reshape`,
`field`, `masked`) -- they reinterpret types without moving data.
**Operations use PascalCase** (`Map`, `Each`, `Where`, `Gather`, `Over`,
`Scan`, `Add`, etc.) -- they do computational work. This convention makes the cost model visible
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
14. Cross-leaf GPU codegen: emitter handles Decompose, field, and chained
    Gather with automatic gather fusion. DSL string -> GPU execution for
    hierarchical sparse traversal (817 voxels, 617 cross-leaf, all correct).
15. Idiom detection: `Gather(Gather(A, i), j)` automatically fused into
    a single 4D `ct.gather(A, (i, j_x, j_y, j_z))` at emission time.
16. Scale benchmark: cuTile 4.9-8.3x faster than PyTorch GPU at 4-256
    leaf scale (up to 79K voxels, 475K lookups). GPU scaling excellent:
    80x more work costs only 2x more time.
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
| 1,000  | 134       | 102       | 0.77x     | 0.03x fVDB  |
| 10,000 | 131       | 103       | 0.78x     | 0.03x fVDB  |
| 50,000 | 136       | 107       | 0.78x     | 0.04x fVDB  |
| 200,000| 143       | 108       | 0.76x     | 0.05x fVDB  |

### Completed since v11

**Multi-step compilation (barrier-based pipeline).**  The pipeline
planner (`dsl_pipeline.py`) partitions programs into `cutile`
(kernel-fusible) and `collective` (torch GPU barrier) segments.  The
executor dispatches: collective segments to `torch.sort` /
`torch.unique` / `torch.nonzero`; cutile segments to compiled cuTile
`@ct.kernel` launches (when `device="cuda"`).  AST rewriting promotes
inter-segment references to kernel inputs automatically.  The hooks
mechanism on `EvalEnv` makes the dispatch pluggable.

**conv_grid.**  First multi-step pipeline application.  Computes unique
output coordinates for sparse convolution topology: broadcast expansion
of active coords by kernel offsets, stride filtering, then Sort + Unique
dedup via the pipeline executor.  Correctness verified against numpy
reference at multiple scales.

### Recommended next steps

**1. Next fVDB operations.**  Broaden from `ijk_to_index` and
`conv_grid` to other `GridBatch` operations:

- `neighbor_indexes(ijk, extent)`: given query coords + extent, return
  neighbor voxel indices.  The cross-leaf neighbor pattern from v5/v6 is
  already demonstrated in the DSL; this is a formalisation for the
  3-level CIG.
- `dilated_grid(dilation)`: structurally similar to conv_grid but simpler
  (fixed extent).
- `sample_trilinear(points, voxel_data)`: interpolation at continuous
  coordinates.  Requires world-to-voxel transform, 8-corner Gather for
  trilinear weights, weighted sum.
- `inject_from` / `inject_to`: map data between grids with different
  topologies.  Uses `ijk_to_index` on both grids -- largely a
  composition of existing primitives.

**2. Close the query performance gap (22-24% vs NanoVDB).**  The 3-level
CIG is within 22-24% of NanoVDB at 20-30x less memory.  Sources of the
remaining gap:

- Software Hamming weight (4 bit-manipulation steps per popcount) vs
  NanoVDB's `__popcll()` (single PTX instruction).  Investigate whether
  cuTile's compiler lowers the Hamming weight pattern to `__popcll()` via
  the TileIR dump (`CUDA_TILE_LOGS=CUTILEIR`).  If not, add a `Popc`
  DSL primitive that maps to a hardware intrinsic.
- The `Find` scan for R=1 emits a comparison + conditional select that
  could be strength-reduced to a constant `upper_idx = 0` when R is
  known to be 1 at emit time.

**3. Batch dimension.**  fVDB's `GridBatch` wraps multiple grids.
Can be handled externally (one kernel launch per grid) without framework
changes.  More advanced options: jagged outer `Each`, or packed
contiguous storage with per-grid offsets.

**4. Per-node bounds (half-open intervals).**  Store `[min, max)` per
node.  Derivable from node position and bit-widths, but explicit storage
enables ray-box intersection without recomputing decomposition.

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

- **Idiom detection**: the emitter now recognises three idioms: chained
  Gather fusion (v6), masked Gather with abs-prefix (v8-v11), and Find
  linear scan (v11).  Next target: the full CIG chain (Find + N masked
  levels) as a macro-idiom for whole-program optimisation.
- **Hardware popcount**: investigate whether cuTile's Hamming weight
  bit-manipulation pattern lowers to `__popcll()` (single PTX instruction)
  in the TileIR compilation pipeline.  Use `CUDA_TILE_LOGS=CUTILEIR` to
  inspect the intermediate representation.  If not, add a `Popc` DSL
  primitive that maps directly to the hardware intrinsic.
- **TileIR emission backend**: the `cuda_tile` MLIR dialect is designed as
  a code generation target for DSLs and compilers.  Retargeting the emitter
  from Python cuTile to the MLIR dialect would enable AOT compilation via
  `tileiras`, custom MLIR optimisation passes, and `.cutile` bytecode
  serialisation.  The emitter logic (idiom detection, masked Gather, Find)
  would not change -- only the string templates.  Investigated in this
  session via `CUDA_TILE_LOGS=CUTILEIR`.
- **Batch dimension**: fVDB's `GridBatch` wraps multiple grids.  The CIG
  currently has no batch dimension.  Options: one kernel launch per grid
  (simplest), jagged outer `Each` (DSL-native), or packed contiguous
  storage (like NanoVDB's buffer).  See Recommended next steps item 3.
- **Materialisation scheduling**: given a DSL expression, automatically
  decide where to insert Gather (early vs late) based on data
  characteristics (sparsity ratio, memory budget).  The Halide schedule
  analogy.
- **Let-bindings in Map bodies**: the parser only supports single
  expressions as Map/Each bodies.  Adding let-bindings would avoid deeply
  nested expressions for complex Map bodies.  Straightforward parser
  extension.
- **`flip` in DSL**: currently Python-only.  Add as a lowercase DSL
  keyword.
- **GPU-accelerated CIG builder**: `build_compressed_cig3` is pure
  PyTorch (sort + scatter on CPU).  A GPU-native builder using CUB radix
  sort + scatter would close the construction-time gap with fVDB.
- **No world-to-grid transforms in CIG**: the design decision is that
  `origin`, `voxel_size`, and transform matrices are instancing metadata,
  not structural properties of the grid.  They should live outside the
  CIG, allowing the same grid to be instanced with different transforms
  (as in modern geometry frameworks like USD).  This is a deliberate
  departure from NanoVDB, which embeds the transform in the grid buffer.

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
