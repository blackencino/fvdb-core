# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: GPU hash map build/lookup/scatter-reduce.

Compares GPU CUDA kernel performance against CPU reference and
torch.sort + torch.unique for deduplication tasks.

Requires CUDA.

Run:  source ~/.venvs/fvdb_cutile/bin/activate && python fvdb_tile/benchmarks/bench_gpu_hashmap.py
"""

from __future__ import annotations

import time

import torch

HAS_CUDA = torch.cuda.is_available()

if not HAS_CUDA:
    print("ERROR: No CUDA device available. This benchmark requires GPU.")
    raise SystemExit(1)

from fvdb_tile.prototype.hashmap_cuda import (
    gpu_hash_map_build,
    gpu_hash_map_lookup,
    gpu_hash_map_scatter_reduce,
)
from fvdb_tile.prototype.ops import (
    hash_map_build,
    hash_map_compute_storage_size,
    hash_map_lookup,
)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_gpu_fn(fn, warmup=3, repeats=20):
    """GPU-synced timing: warmup, then median of *repeats* runs (microseconds)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


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


def _gen_unique_keys(n: int, seed: int = 42, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randint(0, n * 100, size=(n * 2,), generator=gen, dtype=torch.int64)
    keys = torch.unique(keys)[:n]
    return keys.to(device)


# ---------------------------------------------------------------------------
# Benchmark: GPU build
# ---------------------------------------------------------------------------


def bench_gpu_build(n: int) -> dict:
    keys_gpu = _gen_unique_keys(n, device="cuda")
    keys_cpu = keys_gpu.cpu()

    gpu_us = _time_gpu_fn(lambda: gpu_hash_map_build(keys_gpu))
    cpu_us = _time_cpu_fn(lambda: hash_map_build(keys_cpu))

    return {
        "n": n,
        "gpu_build_us": gpu_us,
        "cpu_build_us": cpu_us,
        "speedup": cpu_us / gpu_us if gpu_us > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Benchmark: GPU lookup
# ---------------------------------------------------------------------------


def bench_gpu_lookup(n: int) -> dict:
    keys_gpu = _gen_unique_keys(n, device="cuda")
    keys_cpu = keys_gpu.cpu()

    key_arr_gpu, _ = gpu_hash_map_build(keys_gpu)
    key_arr_cpu = hash_map_build(keys_cpu)
    torch.cuda.synchronize()

    gpu_us = _time_gpu_fn(lambda: gpu_hash_map_lookup(key_arr_gpu, keys_gpu))
    cpu_us = _time_cpu_fn(lambda: hash_map_lookup(key_arr_cpu, keys_cpu))

    return {
        "n": n,
        "gpu_lookup_us": gpu_us,
        "cpu_lookup_us": cpu_us,
        "speedup": cpu_us / gpu_us if gpu_us > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Benchmark: GPU vs sort+unique dedup
# ---------------------------------------------------------------------------


def bench_gpu_vs_sort_dedup(n: int) -> dict:
    keys_gpu = _gen_unique_keys(n, device="cuda")

    def sort_dedup():
        sorted_keys, _ = torch.sort(keys_gpu)
        return torch.unique(sorted_keys)

    def hashmap_dedup():
        key_arr, _ = gpu_hash_map_build(keys_gpu)
        return key_arr[key_arr != -1]

    sort_us = _time_gpu_fn(sort_dedup)
    hashmap_us = _time_gpu_fn(hashmap_dedup)

    return {
        "n": n,
        "sort_dedup_us": sort_us,
        "hashmap_dedup_us": hashmap_us,
        "speedup": sort_us / hashmap_us if hashmap_us > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Benchmark: GPU scatter-reduce
# ---------------------------------------------------------------------------


def bench_gpu_scatter_reduce(n: int) -> dict:
    gen = torch.Generator().manual_seed(42)
    n_unique = max(1, n // 5)
    keys = torch.randint(0, n_unique, size=(n,), generator=gen, dtype=torch.int64).cuda()
    values = torch.randint(0, 256, size=(n,), generator=gen, dtype=torch.int64).cuda()

    key_arr, _ = gpu_hash_map_build(keys)
    torch.cuda.synchronize()

    def run_scatter():
        return gpu_hash_map_scatter_reduce(key_arr, keys, values, reduce_fn="or")

    us = _time_gpu_fn(run_scatter)
    return {
        "n": n,
        "n_unique": n_unique,
        "scatter_reduce_us": us,
        "us_per_elem": us / n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print()

    scales = [1000, 10000, 100000]

    # --- Build ---
    print("=" * 70)
    print("GPU Hash Map Build")
    print("=" * 70)
    print(f"{'N':>8} {'GPU us':>12} {'CPU us':>12} {'Speedup':>10}")
    print("-" * 46)
    for n in scales:
        r = bench_gpu_build(n)
        print(f"{r['n']:>8} {r['gpu_build_us']:>12.1f} {r['cpu_build_us']:>12.1f} {r['speedup']:>10.1f}x")

    # --- Lookup ---
    print()
    print("=" * 70)
    print("GPU Hash Map Lookup (all hit)")
    print("=" * 70)
    print(f"{'N':>8} {'GPU us':>12} {'CPU us':>12} {'Speedup':>10}")
    print("-" * 46)
    for n in scales:
        r = bench_gpu_lookup(n)
        print(f"{r['n']:>8} {r['gpu_lookup_us']:>12.1f} {r['cpu_lookup_us']:>12.1f} {r['speedup']:>10.1f}x")

    # --- GPU dedup comparison ---
    print()
    print("=" * 70)
    print("GPU Dedup: sort+unique vs hash map")
    print("=" * 70)
    print(f"{'N':>8} {'sort us':>12} {'hashmap us':>12} {'HM speedup':>12}")
    print("-" * 48)
    for n in scales:
        r = bench_gpu_vs_sort_dedup(n)
        print(f"{r['n']:>8} {r['sort_dedup_us']:>12.1f} {r['hashmap_dedup_us']:>12.1f} {r['speedup']:>12.2f}x")

    # --- Scatter-reduce ---
    print()
    print("=" * 70)
    print("GPU ScatterReduce (OR, ~5 dups per key)")
    print("=" * 70)
    print(f"{'N':>8} {'unique':>8} {'scatter us':>12} {'us/elem':>10}")
    print("-" * 42)
    for n in scales:
        r = bench_gpu_scatter_reduce(n)
        print(f"{r['n']:>8} {r['n_unique']:>8} {r['scatter_reduce_us']:>12.1f} {r['us_per_elem']:>10.3f}")

    print()
    print("Legend:")
    print("  GPU us     -- median GPU-synced time (microseconds)")
    print("  CPU us     -- median CPU time (microseconds)")
    print("  Speedup    -- CPU time / GPU time")
    print("  HM speedup -- sort+unique time / hashmap time (both GPU)")


if __name__ == "__main__":
    main()
