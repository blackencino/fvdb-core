# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Head-to-head comparison: 3-level CIG vs fVDB Grid.

Compares construction time, memory footprint, and ijk_to_index query
performance at multiple grid sizes in [0, 4096)^3, using the same input
coordinates.

The 3-level CIG uses bit-widths [3, 4, 5] (8^3 leaf, 16^3 lower, 32^3
upper) with bitmask-compressed nodes and prefix-sum popcounts at every
level.  The query is a two-step pipeline: torch root_lookup + fused
cuTile 3-level masked chain.
"""

import importlib
import math
import os
import time

import numpy as np
import torch

import cuda.tile as ct

from docs.wip.prototype.cig import build_compressed_cig3, root_lookup
from docs.wip.prototype.dsl_to_cutile import emit_runnable_kernel
from docs.wip.prototype.test_cig3 import cig3_ijk_to_index_numpy
from docs.wip.prototype.types import Dynamic, ScalarType, Shape, Static, Type

try:
    import fvdb

    HAS_FVDB = True
except ImportError:
    HAS_FVDB = False
    print("WARNING: fvdb not available -- fVDB measurements will be skipped.\n")

TILE = 256

# ---------------------------------------------------------------------------
# Kernel compilation (done once)
# ---------------------------------------------------------------------------

CIG3_KERNEL_PROGRAM = """
parts = Decompose(Input("query"), Const([3, 4, 5]))
upper = masked(Gather(Input("upper_masks"), Input("upper_idx")), Gather(Input("upper_prefix"), Input("upper_idx")), Gather(Input("upper_offsets"), Input("upper_idx")))
lower_idx = Gather(upper, field(parts, "level_2"))
lower = masked(Gather(Input("lower_masks"), lower_idx), Gather(Input("lower_prefix"), lower_idx), Gather(Input("lower_offsets"), lower_idx))
leaf_idx = Gather(lower, field(parts, "level_1"))
leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_prefix"), leaf_idx), Gather(Input("leaf_offsets"), leaf_idx))
voxel_idx = Gather(leaf, field(parts, "level_0"))
voxel_idx
"""

CIG3_INPUT_TYPES = {
    "query": Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.I32)),
    "upper_idx": Type(Shape(Dynamic()), ScalarType.I32),
    "upper_masks": Type(Shape(Dynamic()), Type(Shape(Static(512)), ScalarType.I64)),
    "upper_prefix": Type(Shape(Dynamic()), Type(Shape(Static(512)), ScalarType.I32)),
    "upper_offsets": Type(Shape(Dynamic()), ScalarType.I32),
    "lower_masks": Type(Shape(Dynamic()), Type(Shape(Static(64)), ScalarType.I64)),
    "lower_prefix": Type(Shape(Dynamic()), Type(Shape(Static(64)), ScalarType.I32)),
    "lower_offsets": Type(Shape(Dynamic()), ScalarType.I32),
    "leaf_masks": Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I64)),
    "leaf_prefix": Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I32)),
    "leaf_offsets": Type(Shape(Dynamic()), ScalarType.I32),
}

_GEN_DIR = os.path.join(os.path.dirname(__file__), "_generated")
_KERNEL_FN = None


def _get_kernel():
    """Emit and compile the 3-level CIG kernel (cached)."""
    global _KERNEL_FN
    if _KERNEL_FN is not None:
        return _KERNEL_FN

    code, _, _ = emit_runnable_kernel(
        CIG3_KERNEL_PROGRAM,
        CIG3_INPUT_TYPES,
        kernel_name="bench_cig3_kernel",
        tile_input="query",
        tile_input_rank=3,
        tile_size=TILE,
        tile_scalar_inputs=["upper_idx"],
    )

    os.makedirs(_GEN_DIR, exist_ok=True)
    init_path = os.path.join(_GEN_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")
    filepath = os.path.join(_GEN_DIR, "_gen_bench_cig3_kernel.py")
    with open(filepath, "w") as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location("_gen_bench_cig3_kernel", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _KERNEL_FN = getattr(mod, "bench_cig3_kernel")
    return _KERNEL_FN


# ---------------------------------------------------------------------------
# CIG3 query: torch root + cuTile 3-level chain
# ---------------------------------------------------------------------------


def run_cig3_ijk_to_index(cig_cuda, queries_cuda):
    """Full 3-level CIG ijk_to_index: root lookup + cuTile kernel."""
    kernel_fn = _get_kernel()
    N = queries_cuda.shape[0]
    n_blocks = math.ceil(N / TILE)

    upper_idx = root_lookup(cig_cuda.root_coords, queries_cuda)

    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device="cuda")
    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        kernel_fn,
        (
            queries_cuda, upper_idx,
            cig_cuda.upper_masks, cig_cuda.upper_prefix.int(), cig_cuda.upper_offsets.int(),
            cig_cuda.lower_masks, cig_cuda.lower_prefix.int(), cig_cuda.lower_offsets.int(),
            cig_cuda.leaf_masks, cig_cuda.leaf_prefix.int(), cig_cuda.leaf_offsets.int(),
            result_t, TILE,
        ),
    )
    return result_t[:N]


def run_cig3_kernel_only(cig_cuda, queries_cuda, upper_idx):
    """cuTile kernel only (root already resolved)."""
    kernel_fn = _get_kernel()
    N = queries_cuda.shape[0]
    n_blocks = math.ceil(N / TILE)

    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device="cuda")
    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        kernel_fn,
        (
            queries_cuda, upper_idx,
            cig_cuda.upper_masks, cig_cuda.upper_prefix.int(), cig_cuda.upper_offsets.int(),
            cig_cuda.lower_masks, cig_cuda.lower_prefix.int(), cig_cuda.lower_offsets.int(),
            cig_cuda.leaf_masks, cig_cuda.leaf_prefix.int(), cig_cuda.leaf_offsets.int(),
            result_t, TILE,
        ),
    )
    return result_t[:N]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_grid_coords(n_voxels: int, seed: int = 42) -> torch.Tensor:
    """Generate n_voxels unique random coordinates in [0, 4096)^3."""
    rng = np.random.RandomState(seed)
    coords_set = set()
    while len(coords_set) < n_voxels:
        batch = rng.randint(0, 4096, (n_voxels * 2, 3))
        for row in batch:
            coords_set.add(tuple(row))
            if len(coords_set) >= n_voxels:
                break
    coords = np.array(sorted(coords_set)[:n_voxels], dtype=np.int32)
    return torch.from_numpy(coords)


def make_query_coords(grid_coords: torch.Tensor, n_queries: int, seed: int = 99) -> torch.Tensor:
    """~50% grid hits + ~50% random in [0, 4096)^3."""
    rng = np.random.RandomState(seed)
    N = grid_coords.shape[0]
    n_hits = n_queries // 2
    n_random = n_queries - n_hits
    hits = grid_coords[rng.choice(N, n_hits, replace=True)]
    randoms = torch.from_numpy(rng.randint(0, 4096, (n_random, 3)).astype(np.int32))
    return torch.cat([hits, randoms], dim=0)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def _time_fn(fn, warmup=3, repeats=20):
    """Time a GPU function. Returns median in microseconds."""
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
    if not agree:
        n_disagree = (fvdb_active != cig_active).sum().item()
        print(f"  WARNING {label}: {n_disagree} disagreements! fVDB={n_fvdb_active}, CIG={cig_active.sum().item()}")
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

    # 3-level CIG
    t_cig3_build = _time_cpu_fn(lambda: build_compressed_cig3(grid_coords), warmup=1, repeats=3)
    cig3 = build_compressed_cig3(grid_coords)
    cig3_cuda = cig3.cuda()
    results["cig3_build_us"] = t_cig3_build
    results["cig3_bytes"] = cig3.num_bytes

    if HAS_FVDB:
        t_fvdb_build = _time_fn(lambda: fvdb.Grid.from_ijk(grid_coords_cuda))
        grid = fvdb.Grid.from_ijk(grid_coords_cuda)
        results["fvdb_build_us"] = t_fvdb_build
        results["fvdb_bytes"] = grid.num_bytes
    else:
        results["fvdb_build_us"] = None
        results["fvdb_bytes"] = None

    # Print structure
    print(
        f"  3-level CIG:    {cig3.n_active:>7,} voxels, "
        f"{cig3.n_leaves:>5,} leaves, {cig3.n_lower:>5,} lower, {cig3.n_upper:>3,} upper, "
        f"{cig3.num_bytes:>10,} bytes ({cig3.num_bytes / 1024:.1f} KB)"
    )
    if HAS_FVDB:
        print(
            f"  fVDB (NanoVDB): {grid.num_voxels:>7,} voxels, {grid.num_leaf_nodes:>5,} leaves, "
            f"{grid.num_bytes:>10,} bytes ({grid.num_bytes / 1024:.1f} KB)"
        )
        ratio = cig3.num_bytes / max(grid.num_bytes, 1)
        print(f"  CIG3 / fVDB memory: {ratio:.2f}x")

    # ===== Query correctness =====

    cig3_result = run_cig3_ijk_to_index(cig3_cuda, query_coords_cuda)

    if HAS_FVDB:
        fvdb_result = grid.ijk_to_index(query_coords_cuda)
        agree, n_hits = verify_agreement(fvdb_result.cpu(), cig3_result.cpu(), "fVDB vs CIG3")
        if agree:
            print(f"  Query agreement: OK ({n_hits} hits out of {n_queries})")
        results["n_hits"] = n_hits
    else:
        n_hits = (cig3_result >= 0).sum().item()
        results["n_hits"] = n_hits
        print(f"  CIG3 queries: {n_hits} hits out of {n_queries}")

    # ===== Query timing =====

    # Pre-compute upper_idx for separated timing
    upper_idx = root_lookup(cig3_cuda.root_coords, query_coords_cuda)

    # Full pipeline (root + kernel)
    t_cig3_total = _time_fn(lambda: run_cig3_ijk_to_index(cig3_cuda, query_coords_cuda))
    results["cig3_total_us"] = t_cig3_total

    # Root lookup only
    t_root = _time_fn(lambda: root_lookup(cig3_cuda.root_coords, query_coords_cuda))
    results["cig3_root_us"] = t_root

    # Kernel only (root pre-resolved)
    t_kernel = _time_fn(lambda: run_cig3_kernel_only(cig3_cuda, query_coords_cuda, upper_idx))
    results["cig3_kernel_us"] = t_kernel

    if HAS_FVDB:
        t_fvdb = _time_fn(lambda: grid.ijk_to_index(query_coords_cuda))
        results["fvdb_us"] = t_fvdb

    # Print timing
    print(f"\n  {'Method':<35} {'Memory (KB)':>12} {'Query (us)':>12}")
    print(f"  {'-' * 63}")
    if HAS_FVDB:
        print(f"  {'fVDB (NanoVDB)':<35} {results['fvdb_bytes'] / 1024:>12.1f} {results['fvdb_us']:>12.1f}")
    print(f"  {'CIG3 cuTile (kernel only)':<35} {'':>12} {t_kernel:>12.1f}")
    print(f"  {'CIG3 root lookup':<35} {'':>12} {t_root:>12.1f}")
    print(f"  {'CIG3 total (root + kernel)':<35} {cig3.num_bytes / 1024:>12.1f} {t_cig3_total:>12.1f}")

    if HAS_FVDB:
        print(f"\n  CIG3 total vs fVDB: {results['fvdb_us'] / t_cig3_total:.2f}x")
        print(f"  CIG3 kernel-only vs fVDB: {results['fvdb_us'] / t_kernel:.2f}x")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("3-Level CIG vs fVDB: ijk_to_index comparison")
    print("=" * 70)

    # Warm up the kernel JIT
    print("\nCompiling cuTile kernel...")
    _get_kernel()
    print("  Done.\n")

    sizes = [1_000, 10_000, 50_000, 200_000]
    all_results = []

    for n in sizes:
        r = bench_one(n, n_queries=50_000)
        all_results.append(r)

    # Summary tables
    print("\n\n" + "=" * 75)
    print("Summary")
    print("=" * 75)

    print(f"\n  Memory:")
    hdr = f"  {'Voxels':>8} {'CIG3 (KB)':>12} {'Leaves':>8} {'Lower':>8} {'Upper':>6}"
    if HAS_FVDB:
        hdr += f" {'fVDB (KB)':>12} {'CIG3/fVDB':>10}"
    print(hdr)
    print(f"  {'-' * len(hdr)}")
    for r in all_results:
        cig3_kb = r["cig3_bytes"] / 1024
        line = f"  {r['n_voxels']:>8,} {cig3_kb:>12.1f}"
        # Get structure info from the cig3 object (rebuild quickly)
        cig = build_compressed_cig3(make_grid_coords(r["n_voxels"]))
        line += f" {cig.n_leaves:>8,} {cig.n_lower:>8,} {cig.n_upper:>6,}"
        if HAS_FVDB and r.get("fvdb_bytes"):
            fvdb_kb = r["fvdb_bytes"] / 1024
            ratio = r["cig3_bytes"] / r["fvdb_bytes"]
            line += f" {fvdb_kb:>12.1f} {ratio:>9.2f}x"
        print(line)

    print(f"\n  Query time (us, {all_results[0]['n_queries']:,} queries):")
    hdr2 = f"  {'Voxels':>8} {'Root':>8} {'Kernel':>8} {'Total':>8}"
    if HAS_FVDB:
        hdr2 += f" {'fVDB':>8} {'Tot/fVDB':>10} {'Kern/fVDB':>10}"
    print(hdr2)
    print(f"  {'-' * len(hdr2)}")
    for r in all_results:
        line = f"  {r['n_voxels']:>8,} {r['cig3_root_us']:>8.1f} {r['cig3_kernel_us']:>8.1f} {r['cig3_total_us']:>8.1f}"
        if HAS_FVDB and r.get("fvdb_us"):
            total_ratio = r["fvdb_us"] / r["cig3_total_us"]
            kernel_ratio = r["fvdb_us"] / r["cig3_kernel_us"]
            line += f" {r['fvdb_us']:>8.1f} {total_ratio:>9.2f}x {kernel_ratio:>9.2f}x"
        print(line)


if __name__ == "__main__":
    main()
