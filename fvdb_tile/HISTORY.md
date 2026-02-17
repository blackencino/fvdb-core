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
