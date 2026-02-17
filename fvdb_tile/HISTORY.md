# fvdb_tile History

**This document is append-only.** New entries go at the bottom. It records
the decision trail for the fvdb_tile prototype: what was explored, what
conclusions were drawn, and what code was removed when its purpose was
served. The companion `README.md` is the living design document.

---

## Milestone Progression

### v0 -- Single-leaf fundamentals (Feb 13 2026)

Map, Where, Gather, Each over a single `(8,8,8)` leaf node. Type
propagation works end-to-end. Jagged extent `(~)` emerges automatically
from variable-length filtering within Each. Layouts and operations cleanly
separated. Sentinels stay out of the type system.

### v1 -- Composition (Feb 13 2026)

Multiple leaves via `cut` + `Each`. `indexed` layout predicts `Gather`
type. `struct` + `flip` composes multi-feature voxels.

### v2 -- Hierarchical chain (Feb 13 2026)

Two-level grid with Decompose + chained Gather. Morton-linearized nodes
produce identical results (swappable indexing strategy).

### v3 -- Micro DSL (Feb 13 2026)

Programs as text strings, parsed into typed AST, type-checked, executed
against numpy. All v0-v2 scenarios expressible as DSL strings.

### Mesh exemplar (Feb 14 2026)

Triangle mesh as layouts over two raw tensors. Type system correctly
rejects `indexed(face_idx, vertices)` (mesh indexing is multiple scalar
lookups, not one multi-dimensional lookup). Early vs late materialisation
produce identical results.

### v4 -- cuTile codegen (Feb 15 2026)

First GPU code generation. `ct.gather` (not `ct.load`) identified as
the right cuTile primitive for Gather. Hand-written target kernel for
neighbor predicate. Text-only emitter (`emit_program`) as proof of concept.

### v5 -- End-to-end codegen + cross-leaf (Feb 16 2026)

`emit_runnable_kernel` closes the loop: DSL string -> compilable
`@ct.kernel` -> GPU execution -> correct results. First performance
numbers (cuTile 4.4x faster than PyTorch GPU). Cross-leaf neighbor
finding composes cleanly in the DSL.

### v6 -- Cross-leaf GPU codegen (Feb 16 2026)

Emitter handles Decompose + field + chained Gather. Idiom detection:
`Gather(Gather(A, i), j)` fuses into a single 4D `ct.gather`. Scale
benchmark: 80x more work costs only 2x more time.

### v7 -- CIG format and fVDB comparison (Feb 16 2026)

Dense 2-level CIG (`CIG` class) with `(16,16,16)` lower + `(K,8,8,8)`
leaf blocks. cuTile query 1.3-1.5x faster than NanoVDB. But NanoVDB uses
8-12x less memory due to bitmask compression.

**Conclusion carried forward:** Fused 4D `ct.gather` beats NanoVDB tree
traversal on query speed. Dense storage is the memory bottleneck.

### v8 -- Masked layout (Feb 16 2026)

`masked` as a first-class layout: bitmask + popcount for sparse
occupancy. `CompressedCIG` with `(K,8) i64` masks + absolute prefix
per leaf. Beats NanoVDB on both memory (0.22-0.73x) and query speed
(1.27-1.34x) at typical sparsities, from a 4-line DSL expression.

**Conclusion carried forward:** Masked layout simultaneously solves the
memory problem and preserves the query speed advantage. Popcount is
O(1) per level via prefix sums.

### v9 -- Masked codegen (Feb 16 2026)

The 4-line DSL expression compiles to a cuTile kernel via the extended
emitter. `MaskedNode` emission + masked-Gather idiom detection (u64
popcount chain).

### v10 -- 3-level CIG (Feb 16 2026)

`CompressedCIG3` with root + upper (32^3) + lower (16^3) + leaf (8^3).
Prefix-sum popcounts replace unrolled chain. Configurable bit-widths.
Structural parity with NanoVDB's index tree.

### v11 -- Performance optimisation (Feb 16 2026)

`Find` fuses root lookup into cuTile kernel. Absolute prefix sums
eliminate offset gathers. Hamming weight constants CSE'd. 3-level CIG
within 6-20% of NanoVDB at 25-30x less memory.

### Sort/Unique + pipeline planning (Feb 16 2026)

`Sort` and `Unique` as first-class DSL primitives with correct type
inference (Dynamic leading extent for Unique). Barrier-aware pipeline
planner partitions programs into `cutile` (kernel-fusible) and
`collective` (torch GPU barrier) segments.

---

## Cleanup: Feb 17 2026

Removed superseded stepping-stone code whose conclusions were
incorporated into the current implementation.

### Removed: Dense 2-level CIG (v7)

- `CIG` class, `build_cig()`, `cig_ijk_to_index()`, `cig_ijk_to_index_numpy()` from `cig.py`
- `cig_cutile.py` (hand-written cuTile kernel for dense CIG)
- `test_cig.py` (tests for dense CIG and 2-level CompressedCIG)
- `bench_cig_vs_fvdb.py` (benchmark for dense + 2-level compressed CIG)

The dense CIG proved the query-speed thesis (fused gather > tree
traversal). The `CompressedCIG3` carries this forward with masked
compression at all levels.

### Removed: 2-level CompressedCIG (v8)

- `CompressedCIG` class, `build_compressed_cig()`, `compressed_cig_ijk_to_index()`, `compressed_cig_ijk_to_index_numpy()` from `cig.py`

The 2-level CompressedCIG proved the masked-layout thesis. The 3-level
`CompressedCIG3` carries this forward with full coordinate range and
root lookup.

### Removed: hand-written masked CIG cuTile kernels

- `cig_masked_cutile.py` (entire file: i32 reference kernel + u64 kernel + launchers)
- `test_cutile_masked_e2e.py` (tested 2-level masked CIG codegen)

The 2-level masked CIG kernels were stepping stones. The u64 masked-gather
pattern is now tested in triplicate by `test_cutile_cig3_e2e.py` (the
3-level CIG generates three masked levels from the same DSL pattern).

### Removed: v4 text-only emitter

- `emit_node()`, `emit_program()` from `dsl_to_cutile.py`
- `test_cutile_codegen.py` (tested the text-only emitter)

The text-only emitter was the first codegen proof. `emit_runnable_kernel()`
(v5+) produces complete, compilable kernels and is tested by four
end-to-end test suites.

### Removed: v4 hand-written neighbor kernel

- `test_cutile_gather.py` (hand-written cuTile neighbor predicate)
- `bench_cutile.py` (single-leaf benchmark using that kernel)

The hand-written kernel was the target for the v4 emitter. The emitter
now generates equivalent kernels, tested in `test_cutile_e2e.py`.
Toolchain validation remains in `test_cutile_smoke.py`.

### Removed: dead code

- `inv_morton3d()` from `ops.py` (morton decoder, never used)
- `scalar_type()` from `types.py` (identity function, its own docstring said "use ScalarType.X directly")

### Relocated

- `cig3_ijk_to_index_numpy()` moved from `test_cig3.py` to `cig.py`
  (it is a reference query implementation, not a test fixture).

---

## Current State After Cleanup

### What remains

**Core DSL:** `types.py`, `ops.py`, `layouts.py`, `dsl_ast.py`,
`dsl_eval.py`, `dsl_parse.py`, `dsl_pipeline.py`, `dsl_to_cutile.py`

**CIG implementation:** `CompressedCIG3` in `cig.py` (3-level
bitmask-compressed grid with root lookup and numpy reference query).

**Tests (13):** v0-v3 fundamentals (`test_where`, `test_neighbors`,
`test_indexed_flip`, `test_two_level`, `test_dsl`, `test_mesh`),
Sort/Unique + pipeline (`test_sort_unique`, `test_pipeline`),
masked + CIG3 (`test_masked`, `test_cig3`), GPU codegen
(`test_cutile_smoke`, `test_cutile_e2e`, `test_cutile_cross_leaf`,
`test_cutile_cig3_e2e`), cross-leaf (`test_cross_leaf`).

**Benchmarks (2):** `bench_cig3_vs_fvdb.py` (3-level CIG vs NanoVDB),
`bench_cutile_cross_leaf.py` (cross-leaf scale benchmark).

### Next steps

1. **conv_grid** -- topology expansion using Sort + Unique collectives
   via the pipeline executor (now that collectives dispatch to torch).
2. **Close the query performance gap** -- investigate hardware popcount
   via cuTile TileIR.
3. **Batch dimension** -- extend CIG3 to handle GridBatch.
4. **Cutile segment compilation** -- wire cutile segments to
   `emit_runnable_kernel` for full GPU execution end-to-end.

---

## GPU-Backed Pipeline Collectives: Feb 17 2026

The pipeline executor now dispatches collective segments (Where, Sort,
Unique) to torch ops via the `device` parameter on
`PipelineExecutable.run()`.

### What was added

- `EvalEnv.hooks` in `dsl_eval.py`: optional dict mapping node types to
  callable overrides.  The evaluator checks hooks before its normal
  dispatch, and hooks propagate through child environments automatically.
- `_torch_where`, `_torch_sort`, `_torch_unique` in `dsl_pipeline.py`:
  torch-backed implementations of collective operations.
- `_make_collective_hooks(device)`: builds a hooks dict for a given
  torch device.  Used by `PipelineExecutable.run(device=...)`.
- `PipelineExecutable.run(device=...)`: when `device` is `"cpu"` or
  `"cuda"`, collective operations dispatch to torch.  When `None`
  (default), the pure numpy evaluator handles everything (backwards
  compatible).
- Four new tests in `test_pipeline.py` verifying collective dispatch
  matches the pure evaluator for Where, Sort+Unique, mixed segments,
  and nested barriers.

### Design notes

Data stays as numpy `Value` objects at segment boundaries.  Collective
hooks convert numpy to torch, run the op, and convert back.  This keeps
the existing test infrastructure working.  The hooks mechanism is general:
any node type can be overridden, enabling future backends (cuTile segment
compilation) via the same interface.

---

## CuTile Segment Compilation: Feb 17 2026

The pipeline executor now compiles cutile segments to cuTile GPU kernels
when `device="cuda"`.  Combined with torch-backed collectives, this
gives full GPU execution: cutile segments as compiled `@ct.kernel`
launches, collective segments as `torch.sort` / `torch.unique` calls.

### What was added

- **AST rewriting** in `dsl_pipeline.py`: `_segment_external_refs`
  finds names referenced but not defined within a segment;
  `_rewrite_refs_to_inputs` rewrites `RefNode` to `InputNode` for
  those names so the segment can be emitted as a standalone kernel.
- **`_compile_cutile_segment`**: reconstructs a sub-program from
  segment bindings, determines the tile input and parallelism pattern
  automatically, calls `emit_runnable_kernel`, compiles via the
  file-writing JIT pattern, launches on GPU, and returns the result
  as a numpy Value.
- **`_compile_kernel`**: shared file-based JIT helper with caching
  (cuTile requires `inspect.getsource()`).
- **`SubNode` emission** in `dsl_to_cutile.py`: mirrors AddNode
  with per-axis decomposition support.
- **Modified `run()` loop**: when `device="cuda"`, cutile segments
  dispatch to `_run_cutile_segment`; all other configurations use
  the evaluator (backwards compatible).
- Two GPU-guarded tests in `test_pipeline.py`.

---

## conv_grid: Feb 17 2026

First multi-step pipeline application.  `conv_grid` computes the unique
output coordinates for a sparse convolution topology.

### Semantic contract

Output coordinate `y` is active iff there exists active input `x` and
kernel offset `k` such that `x = y * stride + k` (component-wise).

### Implementation

`conv_grid.py` implements:
1. Broadcast expansion: `active[:, None, :] - offsets[None, :, :]`
2. Stride filter: keep candidates where `cand % stride == 0`, divide
3. Dedup via the pipeline executor: `Sort` + `Unique` dispatched to
   torch ops via `DEDUP_PIPELINE.run(device=...)`

The expansion is a torch broadcast (not DSL-compiled); Sort + Unique
are DSL collectives dispatched to torch.  This demonstrates the
pipeline architecture: different stages use different backends, wired
by the executor.

### Tests

Five tests in `test_conv_grid.py`: correctness vs numpy reference,
stride semantics, input immutability, torch tensor input, larger scale
(500 unique active coords -> expanded and deduped).

---

## Leading Shape Theory + Functions as Values

### Conceptual reframe

Renamed the core abstraction from "iteration space + element type" to
**leading shape theory**: a generalisation of K's leading-axis theory to
multi-rank shapes.  A type is a recursive nesting `S_1 / S_2 / ... / scalar`
where `S_1` is the leading shape.  Operations always operate on the leading
shape.  The nesting structure is a property of the value, not the operation.
`cut` deepens nesting, `reshape` reorganises within a level.

This is not cosmetic.  The nesting boundary determines what operations see
as "one element."  The layout IS the schedule, and the schedule lives in
the data's type.

### Functions as values

Verbs (Add, Sub, Mul, etc.) are first-class `FnValue` objects with both
`apply_fn` (numpy execution) and `type_fn` (type inference without data).
`FnType` added to the type system `ElementType` union.  `VERBS` registry
in `ops.py` holds all built-in verb FnValues.

### Adverbs as function transformers

Adverbs (Over, EachRight, EachLeft, Scan, Prior) take a function and
return a **new** function.  Application is always separate:
`EachLeft(Add)(x, y)` is two steps -- produce a function, then apply it.

Three new AST nodes:
- `VerbRefNode(name)`: reference to a built-in verb as a function value.
- `AdverbApplyNode(adverb, fn)`: apply an adverb to a function, producing
  a new function.
- `ApplyNode(fn, args)`: apply a function value to data arguments.

The parser desugars `Over(Add, xs)` into
`ApplyNode(AdverbApplyNode("Over", VerbRefNode("Add")), [xs])`.

### Nested adverb composition

`EachRight(EachLeft(f))` is a nested `AdverbApplyNode`.  The type rules
compose structurally:

```
EachRight(EachLeft(f))(x: S_x / A, y: S_y / B) = S_y / (S_x / f(A, B))
EachLeft(EachRight(f))(x: S_x / A, y: S_y / B) = S_x / (S_y / f(A, B))
```

This replaces broadcasting: the programmer explicitly controls iteration
with adverb composition.

### Tests

Eight tests in `test_adverbs.py`: Over(Add) via new path, basic
EachRight/EachLeft, outer product via nested adverbs (both nesting
orders), type inference for multi-rank leading shapes, let-bound composed
functions, and backward compatibility with the mesh centroid program.
All 30 existing tests continue to pass.

---

## GPU Hash Map + Dialect Lowering: Feb 17 2026

Hand-written CUDA kernels for hash map build (atomicCAS), lookup (probe
loop), and scatter-reduce (atomicOr/Add).  JIT-compiled via NVRTC and
launched on torch's CUDA stream.  MurmurHash3 64-bit finalizer.

### What was added

- `hashmap_cuda.py`: GPU hash map build/lookup/scatter_reduce kernels
  via NVRTC JIT.  `conv_grid_dilate_kernel`: fused mask shift + boundary
  decomposition + hash probe + atomicOr (see conv_grid_leafwise below).
- `cuda_launch.py`: generic NVRTC compile-and-launch utility (shared by
  all CUDA kernels).
- `dialect_hashmap.py`: HashMapBuild and HashMapLookup as DSL dialect
  nodes.  `dsl_lower.py` lowers dialect nodes to core AST via rewrite
  passes.
- Pipeline hooks for GPU collectives: `hashmap_build_hook` and
  `hashmap_lookup_hook` in `dsl_pipeline.py` dispatch to GPU kernels
  when `device="cuda"`.
- `test_hashmap.py`, `test_cutile_hashmap.py`, `bench_hashmap.py`,
  `bench_gpu_hashmap.py`.

### Design note

The GPU hash map is OUT_OF_DSL infrastructure.  The kernels are
hand-written CUDA, not DSL-generated.  The dialect mechanism wraps them
so pipeline programs can reference hash map operations, but the actual
work bypasses the emitter.  Future: express as DSL programs with idiom
recognition that emits the fused kernels.

---

## conv_grid_leafwise: Feb 17 2026

Topology expansion via leaf-level bitmask dilation.  Instead of the
N*K dense expansion of `conv_grid`, shifts entire 8x8x8 leaf masks
and accumulates via hash-map scatter-reduce with OR.  Work is O(L*K)
word-level ops where L is the number of input leaves (typically
50-100x fewer than active voxels N).

### GPU path (3 kernel launches, 0 Python loops)

1. HashMapBuild: build output leaf hash map from expanded leaf coords.
2. `conv_grid_dilate_kernel`: fused mask shift + boundary decomposition
   + hash probe + atomicOr over all L*K pairs.  Single launch.
3. MaskToCoords: popcount + extract voxel coords from accumulated masks.

### OUT_OF_DSL status

The algorithm corresponds to a DSL program (documented in the
`conv_grid_leafwise.py` docstring), but the GPU path is a hand-fused
kernel that bypasses the DSL/AST pipeline.  This is the prime candidate
for AST-level fusion: express in the DSL, then have the compiler
recognise the pattern and emit the fused kernel.

### What was added

- `conv_grid_leafwise.py`: public API + CPU reference + GPU path.
- `conv_grid_dilate_kernel` in `hashmap_cuda.py`: the fused CUDA kernel.
- `test_conv_grid_leafwise.py`, `bench_conv_grid.py`,
  `bench_conv_grid_leafwise.py`.

---

## Pipeline Cleanup Pass: Feb 17 2026

Systematic audit and consolidation of the prototype, removing CPU
operations from the GPU hot path and annotating out-of-DSL code for
future fusion work.

### GPU pipeline CPU round-trip removed (hypothesis-breaking)

The collective hooks in `_make_collective_hooks` (dsl_pipeline.py)
previously moved every GPU result back to CPU after each collective
operation.  The `_run_cutile_segment` result was also moved to CPU.
This was a workaround for the evaluator assuming CPU tensors.

Fixed: removed all `.cpu()` calls from hooks and cutile segment
results.  Data now stays on the target device throughout pipeline
execution.  Also removed the double-compute pattern in `sort_hook`
(was sorting on CPU then re-sorting on GPU) and `unique_hook` (same).

### conv_grid pipeline device pass-through (hypothesis-weakening)

`conv_grid.py` hardcoded `device=None` in `pipeline.run()`, meaning the
GPU compilation path was never used even when called with `device="cuda"`.

Fixed: the `device` parameter now passes through to `pipeline.run()`.

### CPU loops over tensor data vectorized

- `_build_masked_level` in `cig.py`: Python loop with O(M) implicit
  `.item()` calls replaced with vectorized `scatter_add_` over bit
  values (non-overlapping single bits, so add = OR).
- `_unique_leading_axis` in `dsl_eval.py`: Python loop with O(N) GPU
  syncs for first-occurrence finding replaced with vectorized
  `scatter_reduce_("amin")`.

### CPU-only reference hashmap guarded

`hash_map_build`, `hash_map_lookup`, `hash_map_scatter_reduce` in
`ops.py` now assert `not keys.is_cuda` to prevent accidental GPU use.
Docstrings updated to point to GPU alternatives.

### Dead code removed

- `cig3_ijk_to_index_numpy` alias (zero callers).

### OUT_OF_DSL annotations added

Consistent `# DSL status:` headers and `# OUT_OF_DSL:` markers added
to the four application-level files:

- `conv_grid.py`: fully DSL-driven
- `conv_grid_leafwise.py`: out-of-DSL (GPU path is hand-fused)
- `cig.py`: out-of-DSL (construction is imperative torch)
- `hashmap_cuda.py`: out-of-DSL (hand-written CUDA kernels)

### Test fixes

Two tests that compared GPU pipeline output against CPU references now
move the GPU result to CPU at the test level before comparison,
reflecting the corrected behaviour (results stay on device).
