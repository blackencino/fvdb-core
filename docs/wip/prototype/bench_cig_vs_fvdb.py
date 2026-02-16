# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Head-to-head comparison: CIG vs fVDB Grid.

Compares construction time, memory footprint, and ijk_to_index query
performance at multiple grid sizes, using the same input coordinates.

Includes both dense CIG (v7) and compressed CIG with masked layout (v8).
"""

import time

import numpy as np
import torch

from docs.wip.prototype.cig import CIG, build_cig, cig_ijk_to_index
from docs.wip.prototype.cig import CompressedCIG, build_compressed_cig, compressed_cig_ijk_to_index
from docs.wip.prototype.cig_cutile import run_cig_ijk_to_index
from docs.wip.prototype.cig_masked_cutile import run_compressed_cig_ijk_to_index

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

    # Compressed CIG
    ccig = build_compressed_cig(grid_coords)
    ccig_cuda = ccig.cuda()
    results["ccig_bytes"] = ccig.num_bytes

    # Print structure comparison
    print(f"  Dense CIG:      {cig.n_active:>7,} voxels, {cig.n_leaves:>5,} leaves, {cig.num_bytes:>10,} bytes ({cig.num_bytes / 1024:.1f} KB)")
    print(f"  Compressed CIG: {ccig.n_active:>7,} voxels, {ccig.n_leaves:>5,} leaves, {ccig.num_bytes:>10,} bytes ({ccig.num_bytes / 1024:.1f} KB)")
    if HAS_FVDB:
        print(
            f"  fVDB (NanoVDB): {grid.num_voxels:>7,} voxels, {grid.num_leaf_nodes:>5,} leaves, "
            f"{grid.num_bytes:>10,} bytes ({grid.num_bytes / 1024:.1f} KB)"
        )
        ratio = ccig.num_bytes / max(grid.num_bytes, 1)
        print(f"  Compressed CIG / fVDB memory: {ratio:.2f}x")

    # ===== Query correctness =====

    cig_pt_result = cig_ijk_to_index(cig_cuda, query_coords_cuda)

    # Compressed CIG correctness (PyTorch)
    ccig_pt_result = compressed_cig_ijk_to_index(ccig, query_coords_cuda.cpu())
    np.testing.assert_array_equal(
        (cig_pt_result.cpu() >= 0).numpy(), (ccig_pt_result >= 0).numpy()
    )

    # Compressed CIG cuTile correctness (u64 path)
    ccig_ct_result = run_compressed_cig_ijk_to_index(
        query_coords_cuda, ccig_cuda.lower, ccig_cuda.leaf_masks, ccig_cuda.leaf_offsets.int()
    )
    np.testing.assert_array_equal(ccig_pt_result.numpy(), ccig_ct_result.cpu().numpy())

    if HAS_FVDB:
        fvdb_result = grid.ijk_to_index(query_coords_cuda)
        agree, n_hits = verify_agreement(fvdb_result.cpu(), ccig_ct_result.cpu(), "fVDB vs Compressed CIG")
        if agree:
            print(f"  Query agreement: OK ({n_hits} hits out of {n_queries})")
        results["n_hits"] = n_hits
    else:
        n_hits = (ccig_ct_result >= 0).sum().item()
        results["n_hits"] = n_hits
        print(f"  CIG queries: {n_hits} hits out of {n_queries}")

    # ===== Query timing =====

    # Dense CIG cuTile kernel
    t_dense_ct = _time_fn(lambda: run_cig_ijk_to_index(query_coords_cuda, cig_cuda.lower, cig_cuda.leaf_arr))
    results["dense_ct_us"] = t_dense_ct

    # Compressed CIG cuTile kernel (u64 path)
    t_comp_ct = _time_fn(
        lambda: run_compressed_cig_ijk_to_index(
            query_coords_cuda, ccig_cuda.lower, ccig_cuda.leaf_masks, ccig_cuda.leaf_offsets.int()
        )
    )
    results["comp_ct_us"] = t_comp_ct

    if HAS_FVDB:
        t_fvdb = _time_fn(lambda: grid.ijk_to_index(query_coords_cuda))
        results["fvdb_us"] = t_fvdb

    # Print timing
    print(f"\n  {'Method':<30} {'Memory (KB)':>12} {'Query (us)':>12}")
    print(f"  {'-' * 58}")
    if HAS_FVDB:
        print(f"  {'fVDB (NanoVDB)':<30} {results['fvdb_bytes'] / 1024:>12.1f} {results['fvdb_us']:>12.1f}")
    print(f"  {'Dense CIG cuTile':<30} {cig.num_bytes / 1024:>12.1f} {t_dense_ct:>12.1f}")
    print(f"  {'Compressed CIG cuTile':<30} {ccig.num_bytes / 1024:>12.1f} {t_comp_ct:>12.1f}")

    if HAS_FVDB:
        print(f"\n  Compressed CIG vs fVDB query: {results['fvdb_us'] / t_comp_ct:.2f}x")

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
    print("\n\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\n  Memory:")
    hdr = f"  {'Voxels':>8} {'Dense CIG':>12} {'Comp. CIG':>12}"
    if HAS_FVDB:
        hdr += f" {'NanoVDB':>12} {'Comp/fVDB':>10}"
    print(hdr)
    print(f"  {'-' * len(hdr)}")
    for r in all_results:
        line = f"  {r['n_voxels']:>8,} {r['cig_bytes']:>12,} {r['ccig_bytes']:>12,}"
        if HAS_FVDB and r.get("fvdb_bytes"):
            ratio = r["ccig_bytes"] / r["fvdb_bytes"]
            line += f" {r['fvdb_bytes']:>12,} {ratio:>9.2f}x"
        print(line)

    print(f"\n  Query time (us, {all_results[0]['n_queries']:,} queries):")
    hdr2 = f"  {'Voxels':>8} {'Dense CT':>10} {'Comp CT':>10}"
    if HAS_FVDB:
        hdr2 += f" {'fVDB':>10} {'Comp/fVDB':>10}"
    print(hdr2)
    print(f"  {'-' * len(hdr2)}")
    for r in all_results:
        line = f"  {r['n_voxels']:>8,} {r['dense_ct_us']:>10.1f} {r['comp_ct_us']:>10.1f}"
        if HAS_FVDB and r.get("fvdb_us"):
            speedup = r["fvdb_us"] / r["comp_ct_us"]
            line += f" {r['fvdb_us']:>10.1f} {speedup:>9.2f}x"
        print(line)


if __name__ == "__main__":
    main()
