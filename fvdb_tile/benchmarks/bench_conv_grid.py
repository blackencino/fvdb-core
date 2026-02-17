# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: fvdb_tile conv_grid vs fVDB GridBatch.conv_grid().

Compares the adverb-based DSL pipeline (EachLeft(EachRight(EachBoth(Add)))
+ fuse + reshape + Morton3dSigned + Sort + Unique + MortonDecode3d) against
fVDB's compiled C++/CUDA implementation at multiple scales, kernel sizes,
and strides.

The fvdb_tile path currently runs through the Python evaluator (not cuTile
compiled).  This benchmark establishes the baseline for future GPU codegen.

Run:  source ~/.venvs/fvdb_cutile/bin/activate && python fvdb_tile/benchmarks/bench_conv_grid.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from fvdb_tile.prototype.conv_grid import conv_grid
from fvdb_tile.prototype.ops import morton3d_signed

try:
    import fvdb
    from fvdb import GridBatch, JaggedTensor

    HAS_FVDB = torch.cuda.is_available()
except (ImportError, RuntimeError):
    HAS_FVDB = False
    print("WARNING: fvdb or CUDA not available -- fVDB measurements will be skipped.\n")


# ---------------------------------------------------------------------------
# Timing helpers (match bench_cig3_vs_fvdb.py patterns)
# ---------------------------------------------------------------------------


def _time_fn(fn, warmup=3, repeats=20):
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


def _generate_sparse_coords(n_voxels: int, extent: int = 256, seed: int = 42) -> np.ndarray:
    """Generate *n_voxels* unique random coordinates in [0, extent)^3."""
    rng = np.random.RandomState(seed)
    coords_set: set[tuple[int, int, int]] = set()
    while len(coords_set) < n_voxels:
        batch = rng.randint(0, extent, (n_voxels * 2, 3))
        for row in batch:
            coords_set.add((int(row[0]), int(row[1]), int(row[2])))
            if len(coords_set) >= n_voxels:
                break
    arr = np.array(sorted(coords_set)[:n_voxels], dtype=np.int32)
    return arr


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _assert_same_coord_set(actual_np: np.ndarray, expected_np: np.ndarray, label: str = "") -> bool:
    """Check that two (N, 3) coordinate arrays are the same SET."""
    if actual_np.shape[0] != expected_np.shape[0]:
        print(f"  MISMATCH {label}: count {actual_np.shape[0]} vs {expected_np.shape[0]}")
        return False
    a_codes = morton3d_signed(actual_np)
    e_codes = morton3d_signed(expected_np)
    a_sorted = actual_np[np.argsort(a_codes, kind="stable")]
    e_sorted = expected_np[np.argsort(e_codes, kind="stable")]
    if not np.array_equal(a_sorted, e_sorted):
        n_diff = np.sum(np.any(a_sorted != e_sorted, axis=1))
        print(f"  MISMATCH {label}: {n_diff} differing rows")
        return False
    return True


# ---------------------------------------------------------------------------
# Benchmark runner for one configuration
# ---------------------------------------------------------------------------


def bench_one(
    n_voxels: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> dict:
    """Benchmark a single (voxels, kernel, stride) configuration."""
    active_np = _generate_sparse_coords(n_voxels)
    kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]

    results: dict = {
        "n_voxels": n_voxels,
        "kernel_size": kernel_size,
        "stride": stride,
        "kernel_volume": kernel_volume,
    }

    # --- fvdb_tile path ---
    # Run once to get result + verify
    tile_result = conv_grid(active_np, kernel_size=kernel_size, stride=stride, device="cpu")
    results["dst_voxels"] = tile_result.shape[0]

    # Time (CPU evaluator path)
    t_tile = _time_cpu_fn(
        lambda: conv_grid(active_np, kernel_size=kernel_size, stride=stride, device="cpu"),
        warmup=1,
        repeats=5,
    )
    results["tile_us"] = t_tile

    # --- fVDB path ---
    if HAS_FVDB:
        dev = torch.device("cuda")
        coords_t = torch.tensor(active_np, dtype=torch.int32, device=dev)
        ijks = JaggedTensor(coords_t)
        grid = GridBatch.from_ijk(ijks, device=dev)

        # Run once to get result
        dst_grid = grid.conv_grid(kernel_size=kernel_size, stride=stride)
        fvdb_coords = dst_grid.ijk.jdata.cpu().numpy().astype(np.int32)

        # Verify agreement
        ok = _assert_same_coord_set(tile_result, fvdb_coords, f"n={n_voxels} k={kernel_size} s={stride}")
        results["agreement"] = ok

        # Time (end-to-end: from_ijk + conv_grid)
        def fvdb_fn():
            g = GridBatch.from_ijk(JaggedTensor(coords_t), device=dev)
            _ = g.conv_grid(kernel_size=kernel_size, stride=stride)

        t_fvdb = _time_fn(fvdb_fn, warmup=3, repeats=20)
        results["fvdb_us"] = t_fvdb

        # Also time just the conv_grid call (grid already built)
        def fvdb_conv_only():
            _ = grid.conv_grid(kernel_size=kernel_size, stride=stride)

        t_fvdb_conv = _time_fn(fvdb_conv_only, warmup=3, repeats=20)
        results["fvdb_conv_us"] = t_fvdb_conv

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 90)
    print("conv_grid Benchmark: fvdb_tile (DSL pipeline) vs fVDB (C++/CUDA)")
    print("=" * 90)

    configs = []
    sizes = [1_000, 10_000, 50_000, 200_000]
    kernels = [(3, 3, 3), (3, 5, 7)]
    strides = [(1, 1, 1), (2, 2, 2)]

    for n in sizes:
        for ks in kernels:
            for st in strides:
                configs.append((n, ks, st))

    all_results = []
    for n, ks, st in configs:
        print(f"\n--- {n:>7,} voxels, kernel={ks}, stride={st} ---")
        r = bench_one(n, ks, st)

        status = ""
        if HAS_FVDB:
            status = "OK" if r.get("agreement", False) else "MISMATCH"
            print(f"  Agreement: {status}")
        print(f"  dst_voxels: {r['dst_voxels']:,}")
        print(f"  tile (evaluator): {r['tile_us']:>12,.0f} us")
        if HAS_FVDB:
            print(f"  fVDB (build+conv): {r['fvdb_us']:>10,.0f} us")
            print(f"  fVDB (conv only):  {r['fvdb_conv_us']:>10,.0f} us")

        all_results.append(r)

    # --- Summary table ---
    print("\n\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    hdr = (
        f"  {'Voxels':>8}  {'Kernel':>9}  {'Stride':>9}"
        f"  {'dst_vox':>9}"
        f"  {'tile (us)':>12}"
    )
    if HAS_FVDB:
        hdr += f"  {'fVDB e2e':>12}  {'fVDB conv':>12}  {'tile/conv':>10}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for r in all_results:
        ks_str = f"({r['kernel_size'][0]},{r['kernel_size'][1]},{r['kernel_size'][2]})"
        st_str = f"({r['stride'][0]},{r['stride'][1]},{r['stride'][2]})"
        line = (
            f"  {r['n_voxels']:>8,}"
            f"  {ks_str:>9}"
            f"  {st_str:>9}"
            f"  {r['dst_voxels']:>9,}"
            f"  {r['tile_us']:>12,.0f}"
        )
        if HAS_FVDB:
            ratio = r["tile_us"] / r["fvdb_conv_us"] if r.get("fvdb_conv_us") else 0
            line += (
                f"  {r['fvdb_us']:>12,.0f}"
                f"  {r['fvdb_conv_us']:>12,.0f}"
                f"  {ratio:>9.1f}x"
            )
        print(line)

    if HAS_FVDB:
        print(f"\n  tile/conv = ratio of fvdb_tile evaluator time to fVDB conv_grid-only time.")
        print(f"  fVDB e2e = fVDB from_ijk + conv_grid (includes grid construction).")
        print(f"  tile includes expansion + morton encode + sort + unique + decode (Python evaluator).")
        all_ok = all(r.get("agreement", False) for r in all_results)
        print(f"\n  All results agree: {'YES' if all_ok else 'NO -- CHECK MISMATCHES'}")


if __name__ == "__main__":
    main()
