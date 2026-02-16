# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Head-to-head comparison: CIG vs fVDB Grid.

Compares construction time, memory footprint, and ijk_to_index query
performance at multiple grid sizes, using the same input coordinates.

Three query implementations:
  - fVDB Grid.ijk_to_index  (NanoVDB C++/CUDA kernel)
  - CIG PyTorch vectorized   (advanced indexing)
  - CIG cuTile kernel        (hand-written from the cross-leaf pattern)
"""

import time

import numpy as np
import torch

from docs.wip.prototype.cig import CIG, build_cig, cig_ijk_to_index
from docs.wip.prototype.cig_cutile import run_cig_ijk_to_index

try:
    import fvdb

    HAS_FVDB = True
except ImportError:
    HAS_FVDB = False
    print("WARNING: fvdb not available -- fVDB measurements will be skipped.\n")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_grid_coords(n_voxels: int, seed: int = 42) -> torch.Tensor:
    """Generate n_voxels unique random coordinates in [0, 128)^3."""
    rng = np.random.RandomState(seed)
    # Generate extra to ensure enough unique coords after dedup
    coords_set = set()
    while len(coords_set) < n_voxels:
        batch = rng.randint(0, 128, (n_voxels * 2, 3))
        for row in batch:
            coords_set.add(tuple(row))
            if len(coords_set) >= n_voxels:
                break
    coords = np.array(sorted(coords_set)[:n_voxels], dtype=np.int32)
    return torch.from_numpy(coords)


def make_query_coords(grid_coords: torch.Tensor, n_queries: int, seed: int = 99) -> torch.Tensor:
    """Generate query coords: ~50% from the grid (hits) + ~50% random (mostly misses)."""
    rng = np.random.RandomState(seed)
    N = grid_coords.shape[0]
    n_hits = n_queries // 2
    n_random = n_queries - n_hits

    # Sample from grid coords (guaranteed hits)
    hit_indices = rng.choice(N, n_hits, replace=True)
    hits = grid_coords[hit_indices]

    # Random coords (mostly misses for sparse grids)
    randoms = torch.from_numpy(rng.randint(0, 128, (n_random, 3)).astype(np.int32))

    return torch.cat([hits, randoms], dim=0)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def _time_fn(fn, warmup=3, repeats=20):
    """Time a function. Returns median in microseconds."""
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
    """Time a CPU function. Returns median in microseconds."""
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
# Correctness verification
# ---------------------------------------------------------------------------


def verify_agreement(fvdb_result, cig_result, label=""):
    """Verify fVDB and CIG agree on active/inactive classification."""
    fvdb_active = fvdb_result >= 0
    cig_active = cig_result >= 0

    agree = (fvdb_active == cig_active).all()
    n_fvdb_active = fvdb_active.sum().item()
    n_cig_active = cig_active.sum().item()

    if not agree:
        n_disagree = (fvdb_active != cig_active).sum().item()
        print(f"  WARNING {label}: {n_disagree} disagreements! fVDB={n_fvdb_active}, CIG={n_cig_active}")
    return agree, n_fvdb_active


# ---------------------------------------------------------------------------
# Benchmark for one grid size
# ---------------------------------------------------------------------------


def bench_one(n_voxels: int, n_queries: int = 50000):
    print(f"\n--- {n_voxels:,} voxels, {n_queries:,} queries ---")

    grid_coords = make_grid_coords(n_voxels)
    query_coords = make_query_coords(grid_coords, n_queries)
    grid_coords_cuda = grid_coords.cuda()
    query_coords_cuda = query_coords.cuda()

    results = {"n_voxels": n_voxels, "n_queries": n_queries}

    # ===== Construction =====

    # CIG construction (on CPU, then move to GPU)
    t_cig_build = _time_cpu_fn(lambda: build_cig(grid_coords), warmup=2, repeats=5)
    cig = build_cig(grid_coords)
    cig_cuda = cig.cuda()
    results["cig_build_us"] = t_cig_build
    results["cig_n_leaves"] = cig.n_leaves
    results["cig_n_active"] = cig.n_active
    results["cig_bytes"] = cig.num_bytes

    if HAS_FVDB:
        t_fvdb_build = _time_fn(lambda: fvdb.Grid.from_ijk(grid_coords_cuda))
        grid = fvdb.Grid.from_ijk(grid_coords_cuda)
        results["fvdb_build_us"] = t_fvdb_build
        results["fvdb_n_voxels"] = grid.num_voxels
        results["fvdb_n_leaves"] = grid.num_leaf_nodes
        results["fvdb_bytes"] = grid.num_bytes
    else:
        results["fvdb_build_us"] = None
        results["fvdb_bytes"] = None

    # Print structure comparison
    print(f"  CIG:  {cig.n_active:>7,} voxels, {cig.n_leaves:>5,} leaves, {cig.num_bytes:>10,} bytes ({cig.num_bytes / 1024:.1f} KB)")
    if HAS_FVDB:
        print(
            f"  fVDB: {grid.num_voxels:>7,} voxels, {grid.num_leaf_nodes:>5,} leaves, "
            f"{grid.num_bytes:>10,} bytes ({grid.num_bytes / 1024:.1f} KB)"
        )
        ratio = grid.num_bytes / max(cig.num_bytes, 1)
        print(f"  Memory ratio (fVDB / CIG): {ratio:.2f}x")

    # ===== Query correctness =====

    cig_pt_result = cig_ijk_to_index(cig_cuda, query_coords_cuda)
    cig_ct_result = run_cig_ijk_to_index(query_coords_cuda, cig_cuda.lower, cig_cuda.leaf_arr)

    # CIG PyTorch vs cuTile should be identical
    np.testing.assert_array_equal(cig_pt_result.cpu().numpy(), cig_ct_result.cpu().numpy())

    if HAS_FVDB:
        fvdb_result = grid.ijk_to_index(query_coords_cuda)
        agree, n_hits = verify_agreement(fvdb_result.cpu(), cig_pt_result.cpu(), "fVDB vs CIG")
        if agree:
            print(f"  Query agreement: OK ({n_hits} hits out of {n_queries})")
        results["n_hits"] = n_hits
    else:
        n_hits = (cig_pt_result >= 0).sum().item()
        results["n_hits"] = n_hits
        print(f"  CIG queries: {n_hits} hits out of {n_queries}")

    # ===== Query timing =====

    # CIG PyTorch vectorized
    t_cig_pt = _time_fn(lambda: cig_ijk_to_index(cig_cuda, query_coords_cuda))
    results["cig_pt_us"] = t_cig_pt

    # CIG cuTile kernel
    t_cig_ct = _time_fn(lambda: run_cig_ijk_to_index(query_coords_cuda, cig_cuda.lower, cig_cuda.leaf_arr))
    results["cig_ct_us"] = t_cig_ct

    if HAS_FVDB:
        t_fvdb = _time_fn(lambda: grid.ijk_to_index(query_coords_cuda))
        results["fvdb_us"] = t_fvdb

    # Print timing
    print(f"\n  {'Method':<25} {'Build (us)':>12} {'Query (us)':>12}")
    print(f"  {'-' * 52}")
    if HAS_FVDB:
        print(f"  {'fVDB (NanoVDB)':<25} {results['fvdb_build_us']:>12.1f} {results['fvdb_us']:>12.1f}")
    print(f"  {'CIG PyTorch':<25} {t_cig_build:>12.1f} {t_cig_pt:>12.1f}")
    print(f"  {'CIG cuTile':<25} {'--':>12} {t_cig_ct:>12.1f}")

    if HAS_FVDB:
        print(f"\n  Query speedup (CIG cuTile vs fVDB): {results['fvdb_us'] / t_cig_ct:.2f}x")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 65)
    print("CIG vs fVDB: ijk_to_index comparison")
    print("=" * 65)

    sizes = [1_000, 10_000, 50_000, 200_000]
    all_results = []

    for n in sizes:
        r = bench_one(n, n_queries=50_000)
        all_results.append(r)

    # Summary table
    print("\n\n" + "=" * 65)
    print("Summary")
    print("=" * 65)

    header_mem = f"{'Voxels':>8} {'CIG (KB)':>10} "
    header_query = f"{'CIG-PT (us)':>12} {'CIG-CT (us)':>12} "
    if HAS_FVDB:
        header_mem += f"{'fVDB (KB)':>10} {'Ratio':>7}"
        header_query += f"{'fVDB (us)':>10} {'CT/fVDB':>8}"
    print(f"\n  Memory:")
    print(f"  {header_mem}")
    print(f"  {'-' * len(header_mem)}")
    for r in all_results:
        line = f"  {r['n_voxels']:>8,} {r['cig_bytes'] / 1024:>10.1f} "
        if HAS_FVDB and r["fvdb_bytes"] is not None:
            ratio = r["fvdb_bytes"] / max(r["cig_bytes"], 1)
            line += f"{r['fvdb_bytes'] / 1024:>10.1f} {ratio:>6.1f}x"
        print(line)

    print(f"\n  Query time ({all_results[0]['n_queries']:,} queries):")
    print(f"  {header_query}")
    print(f"  {'-' * len(header_query)}")
    for r in all_results:
        line = f"  {r['cig_pt_us']:>12.1f} {r['cig_ct_us']:>12.1f} "
        if HAS_FVDB and r.get("fvdb_us") is not None:
            speedup = r["fvdb_us"] / r["cig_ct_us"]
            line += f"{r['fvdb_us']:>10.1f} {speedup:>7.2f}x"
        print(line)

    if HAS_FVDB:
        print(f"\n  Build time:")
        print(f"  {'Voxels':>8} {'CIG (us)':>12} {'fVDB (us)':>12} {'Ratio':>8}")
        print(f"  {'-' * 45}")
        for r in all_results:
            ratio = r["fvdb_build_us"] / max(r["cig_build_us"], 1)
            print(f"  {r['n_voxels']:>8,} {r['cig_build_us']:>12.1f} {r['fvdb_build_us']:>12.1f} {ratio:>7.2f}x")


if __name__ == "__main__":
    main()
