# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: hash map build/lookup vs sort-based dedup.

Measures:
  - hash_map_build: insert N keys, time per key
  - hash_map_lookup: query N keys, time per query
  - Comparison: hash map round-trip vs torch.sort + torch.unique
  - ScatterReduce: DSL-level scatter-reduce via direct eval

Run:  source ~/.venvs/fvdb_cutile/bin/activate && python fvdb_tile/benchmarks/bench_hashmap.py
"""

from __future__ import annotations

import time

import torch

from fvdb_tile.prototype.ops import (
    hash_map_build,
    hash_map_compute_storage_size,
    hash_map_lookup,
    hash_map_scatter_reduce,
)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_cpu_fn(fn, warmup=1, repeats=5):
    """CPU timing: warmup, then median of *repeats* runs (microseconds)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _gen_unique_keys(n: int, seed: int = 42) -> torch.Tensor:
    """Generate N unique i64 keys (non-negative, not the sentinel)."""
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randint(0, n * 100, size=(n * 2,), generator=gen, dtype=torch.int64)
    keys = torch.unique(keys)[:n]
    return keys


# ---------------------------------------------------------------------------
# Benchmark: build
# ---------------------------------------------------------------------------


def bench_build(n: int) -> dict:
    keys = _gen_unique_keys(n)
    storage_size = hash_map_compute_storage_size(n)

    us = _time_cpu_fn(lambda: hash_map_build(keys))
    return {
        "n": n,
        "storage_size": storage_size,
        "build_us": us,
        "build_us_per_key": us / n,
    }


# ---------------------------------------------------------------------------
# Benchmark: lookup (all hit)
# ---------------------------------------------------------------------------


def bench_lookup(n: int) -> dict:
    keys = _gen_unique_keys(n)
    key_arr = hash_map_build(keys)

    us = _time_cpu_fn(lambda: hash_map_lookup(key_arr, keys))
    return {
        "n": n,
        "lookup_us": us,
        "lookup_us_per_query": us / n,
    }


# ---------------------------------------------------------------------------
# Benchmark: sort-based dedup comparison
# ---------------------------------------------------------------------------


def bench_sort_dedup(n: int) -> dict:
    keys = _gen_unique_keys(n)

    def sort_dedup():
        sorted_keys, _ = torch.sort(keys)
        return torch.unique(sorted_keys)

    sort_us = _time_cpu_fn(sort_dedup)

    def hashmap_dedup():
        key_arr = hash_map_build(keys)
        return key_arr[key_arr != -1]

    hashmap_us = _time_cpu_fn(hashmap_dedup)

    return {
        "n": n,
        "sort_dedup_us": sort_us,
        "hashmap_dedup_us": hashmap_us,
        "speedup": sort_us / hashmap_us if hashmap_us > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Benchmark: scatter_reduce
# ---------------------------------------------------------------------------


def bench_scatter_reduce(n: int) -> dict:
    gen = torch.Generator().manual_seed(42)
    n_unique = max(1, n // 5)
    keys = torch.randint(0, n_unique, size=(n,), generator=gen, dtype=torch.int64)
    values = torch.randint(0, 256, size=(n,), generator=gen, dtype=torch.int64)

    def run_scatter():
        key_arr = hash_map_build(keys)
        return hash_map_scatter_reduce(key_arr, keys, values, reduce_fn="or")

    us = _time_cpu_fn(run_scatter)
    return {
        "n": n,
        "n_unique": n_unique,
        "scatter_reduce_us": us,
        "scatter_reduce_us_per_elem": us / n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    scales = [100, 1000, 10000]

    print("=" * 70)
    print("Hash Map Build Benchmark")
    print("=" * 70)
    print(f"{'N':>8} {'storage':>10} {'build_us':>12} {'us/key':>10}")
    print("-" * 44)
    for n in scales:
        r = bench_build(n)
        print(f"{r['n']:>8} {r['storage_size']:>10} {r['build_us']:>12.1f} {r['build_us_per_key']:>10.3f}")

    print()
    print("=" * 70)
    print("Hash Map Lookup Benchmark (all hit)")
    print("=" * 70)
    print(f"{'N':>8} {'lookup_us':>12} {'us/query':>10}")
    print("-" * 34)
    for n in scales:
        r = bench_lookup(n)
        print(f"{r['n']:>8} {r['lookup_us']:>12.1f} {r['lookup_us_per_query']:>10.3f}")

    print()
    print("=" * 70)
    print("Dedup Comparison: sort+unique vs hash map")
    print("=" * 70)
    print(f"{'N':>8} {'sort_us':>12} {'hashmap_us':>12} {'speedup':>10}")
    print("-" * 46)
    for n in scales:
        r = bench_sort_dedup(n)
        print(f"{r['n']:>8} {r['sort_dedup_us']:>12.1f} {r['hashmap_dedup_us']:>12.1f} {r['speedup']:>10.2f}x")

    print()
    print("=" * 70)
    print("ScatterReduce Benchmark (OR, ~5 dups per key)")
    print("=" * 70)
    print(f"{'N':>8} {'unique':>8} {'scatter_us':>12} {'us/elem':>10}")
    print("-" * 42)
    for n in scales:
        r = bench_scatter_reduce(n)
        print(f"{r['n']:>8} {r['n_unique']:>8} {r['scatter_reduce_us']:>12.1f} {r['scatter_reduce_us_per_elem']:>10.3f}")

    print()
    print("Legend:")
    print("  N           -- number of input keys/elements")
    print("  storage     -- hash map storage size (power-of-two with 4x load factor)")
    print("  build_us    -- median build time (microseconds)")
    print("  us/key      -- build time per key")
    print("  lookup_us   -- median lookup time (all keys hit)")
    print("  us/query    -- lookup time per query")
    print("  sort_us     -- sort + unique dedup time")
    print("  hashmap_us  -- hash map build + extract dedup time")
    print("  speedup     -- sort_us / hashmap_us")
    print("  scatter_us  -- scatter-reduce time (build + reduce)")
    print("  us/elem     -- scatter-reduce time per input element")
    print()
    print("NOTE: These are CPU-only torch reference timings.")
    print("GPU benchmarks will follow when cuTile emission is added.")


if __name__ == "__main__":
    main()
