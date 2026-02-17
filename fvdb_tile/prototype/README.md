# fvdb_tile prototype

A DSL-driven prototype for sparse grid computation with GPU code generation.
The DSL uses explicit nested layouts to separate iteration space from element
type, inspired by APL/J/K (especially K) at the level of small, orthogonal
value transformations.

Run all tests:

```
source ~/.venvs/fvdb_cutile/bin/activate   # for GPU/cuTile tests
python fvdb_tile/prototype/run_all_tests.py
```

## Semantic contract

[SEMANTICS.md](SEMANTICS.md) locks the algebraic laws (Sort stability, Unique
idempotence, multiset preservation) and execution model (pipeline segment
boundaries) that the tests enforce.  Implementations can change freely --
including the evaluator, code generator, and runtime -- as long as the contracts
in that document continue to hold.

## Milestone progression

Tests in `run_all_tests.py` are organized by milestone:

- **v0** -- Single leaf fundamentals: Map, Where, Gather, Each, jagged emergence
- **v1** -- Multiple leaves (Cut), Indexed layout, Struct + Flip
- **v2** -- Two-level hierarchical chain: Decompose, chained Gather, morton
- **v3** -- Micro DSL: string -> parse -> type-check -> execute; Sort/Unique
  primitives; barrier-aware pipeline planning
- **mesh** -- Triangle mesh as layouts over tensors
- **v5** -- Cross-leaf neighbors via DSL evaluator (numpy)

GPU tests (require cuTile venv):

- **v4** -- cuTile codegen: smoke tests, gather, codegen
- **v5** -- cuTile end-to-end
- **v6** -- Cross-leaf neighbors on GPU (cuTile codegen), CIG3 benchmarks
