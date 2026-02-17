# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Head-to-head comparison: 3-level CIG vs fVDB Grid.

Fully fused: the root lookup (Find) runs inside the cuTile kernel alongside
the 3-level masked chain. No torch barrier. Single kernel launch.
"""

import importlib
import math
import os
import time

import numpy as np
import torch

import cuda.tile as ct

from docs.wip.prototype.cig import build_compressed_cig3
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

CIG3_KERNEL_PROGRAM = """
parts = Decompose(Input("query"), Const([3, 4, 5]))
upper_idx = Find(Input("root_coords"), field(parts, "which_top"))
upper = masked(Gather(Input("upper_masks"), upper_idx), Gather(Input("upper_abs_prefix"), upper_idx))
lower_idx = Gather(upper, field(parts, "level_2"))
lower = masked(Gather(Input("lower_masks"), lower_idx), Gather(Input("lower_abs_prefix"), lower_idx))
leaf_idx = Gather(lower, field(parts, "level_1"))
leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_abs_prefix"), leaf_idx))
voxel_idx = Gather(leaf, field(parts, "level_0"))
voxel_idx
"""

_GEN_DIR = os.path.join(os.path.dirname(__file__), "_generated")
_KERNEL_CACHE = {}


def _get_kernel(n_upper: int):
    """Emit and compile a kernel for the given number of upper nodes (cached)."""
    if n_upper in _KERNEL_CACHE:
        return _KERNEL_CACHE[n_upper]

    input_types = {
        "query": Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.I32)),
        "root_coords": Type(Shape(Static(n_upper)), Type(Shape(Static(3)), ScalarType.I32)),
        "upper_masks": Type(Shape(Dynamic()), Type(Shape(Static(512)), ScalarType.I64)),
        "upper_abs_prefix": Type(Shape(Dynamic()), Type(Shape(Static(512)), ScalarType.I32)),
        "lower_masks": Type(Shape(Dynamic()), Type(Shape(Static(64)), ScalarType.I64)),
        "lower_abs_prefix": Type(Shape(Dynamic()), Type(Shape(Static(64)), ScalarType.I32)),
        "leaf_masks": Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I64)),
        "leaf_abs_prefix": Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I32)),
    }

    kname = f"bench_cig3_R{n_upper}"
    code, _, _ = emit_runnable_kernel(
        CIG3_KERNEL_PROGRAM,
        input_types,
        kernel_name=kname,
        tile_input="query",
        tile_input_rank=3,
        tile_size=TILE,
    )

    os.makedirs(_GEN_DIR, exist_ok=True)
    init_path = os.path.join(_GEN_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")
    filepath = os.path.join(_GEN_DIR, f"_gen_{kname}.py")
    with open(filepath, "w") as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location(f"_gen_{kname}", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, kname)
    _KERNEL_CACHE[n_upper] = fn
    return fn


def run_cig3_fused(cig_cuda, queries_cuda):
    """Fully fused 3-level CIG ijk_to_index (Find + masked chain, single launch)."""
    kernel_fn = _get_kernel(cig_cuda.root_coords.shape[0])
    N = queries_cuda.shape[0]
    n_blocks = math.ceil(N / TILE)
    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device="cuda")
    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        kernel_fn,
        (
            queries_cuda, cig_cuda.root_coords,
            cig_cuda.upper_masks, cig_cuda.upper_abs_prefix.int(),
            cig_cuda.lower_masks, cig_cuda.lower_abs_prefix.int(),
            cig_cuda.leaf_masks, cig_cuda.leaf_abs_prefix.int(),
            result_t, TILE,
        ),
    )
    return result_t[:N]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_grid_coords(n_voxels: int, seed: int = 42) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    coords_set = set()
    while len(coords_set) < n_voxels:
        batch = rng.randint(0, 4096, (n_voxels * 2, 3))
        for row in batch:
            coords_set.add(tuple(row))
            if len(coords_set) >= n_voxels:
                break
    return torch.from_numpy(np.array(sorted(coords_set)[:n_voxels], dtype=np.int32))


def make_query_coords(grid_coords: torch.Tensor, n_queries: int, seed: int = 99) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    N = grid_coords.shape[0]
    n_hits = n_queries // 2
    hits = grid_coords[rng.choice(N, n_hits, replace=True)]
    randoms = torch.from_numpy(rng.randint(0, 4096, (n_queries - n_hits, 3)).astype(np.int32))
    return torch.cat([hits, randoms], dim=0)


def _time_fn(fn, warmup=3, repeats=20):
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


def verify_agreement(fvdb_result, cig_result, label=""):
    fvdb_active = fvdb_result >= 0
    cig_active = cig_result >= 0
    agree = (fvdb_active == cig_active).all()
    n_fvdb_active = fvdb_active.sum().item()
    if not agree:
        n_disagree = (fvdb_active != cig_active).sum().item()
        print(f"  WARNING {label}: {n_disagree} disagreements!")
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

    # Build CIG3
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

    print(
        f"  CIG3: {cig3.n_active:>7,} voxels, "
        f"{cig3.n_leaves:>5,} leaves, {cig3.n_lower:>5,} lower, {cig3.n_upper:>3,} upper, "
        f"{cig3.num_bytes:>10,} bytes ({cig3.num_bytes / 1024:.1f} KB)"
    )
    if HAS_FVDB:
        print(
            f"  fVDB: {grid.num_voxels:>7,} voxels, {grid.num_leaf_nodes:>5,} leaves, "
            f"{grid.num_bytes:>10,} bytes ({grid.num_bytes / 1024:.1f} KB)"
        )

    # Warm up kernel for this R
    _get_kernel(cig3.n_upper)

    # Correctness
    cig3_result = run_cig3_fused(cig3_cuda, query_coords_cuda)

    if HAS_FVDB:
        fvdb_result = grid.ijk_to_index(query_coords_cuda)
        agree, n_hits = verify_agreement(fvdb_result.cpu(), cig3_result.cpu(), "fVDB vs CIG3")
        if agree:
            print(f"  Agreement: OK ({n_hits} hits)")
        results["n_hits"] = n_hits
    else:
        n_hits = (cig3_result >= 0).sum().item()
        results["n_hits"] = n_hits

    # Timing
    t_cig3 = _time_fn(lambda: run_cig3_fused(cig3_cuda, query_coords_cuda))
    results["cig3_us"] = t_cig3

    if HAS_FVDB:
        t_fvdb = _time_fn(lambda: grid.ijk_to_index(query_coords_cuda))
        results["fvdb_us"] = t_fvdb

    print(f"\n  {'Method':<25} {'Memory (KB)':>12} {'Query (us)':>12}")
    print(f"  {'-' * 53}")
    if HAS_FVDB:
        print(f"  {'fVDB (NanoVDB)':<25} {results['fvdb_bytes'] / 1024:>12.1f} {results['fvdb_us']:>12.1f}")
    print(f"  {'CIG3 fused cuTile':<25} {cig3.num_bytes / 1024:>12.1f} {t_cig3:>12.1f}")

    if HAS_FVDB:
        print(f"\n  CIG3 vs fVDB: {results['fvdb_us'] / t_cig3:.2f}x query, {cig3.num_bytes / results['fvdb_bytes']:.2f}x memory")

    return results


def main():
    print("=" * 65)
    print("3-Level CIG (fused) vs fVDB: ijk_to_index")
    print("=" * 65)

    sizes = [1_000, 10_000, 50_000, 200_000]
    all_results = []

    for n in sizes:
        r = bench_one(n, n_queries=50_000)
        all_results.append(r)

    print("\n\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\n  Memory:")
    hdr = f"  {'Voxels':>8} {'CIG3 (KB)':>12}"
    if HAS_FVDB:
        hdr += f" {'fVDB (KB)':>12} {'CIG3/fVDB':>10}"
    print(hdr)
    print(f"  {'-' * len(hdr)}")
    for r in all_results:
        line = f"  {r['n_voxels']:>8,} {r['cig3_bytes'] / 1024:>12.1f}"
        if HAS_FVDB and r.get("fvdb_bytes"):
            line += f" {r['fvdb_bytes'] / 1024:>12.1f} {r['cig3_bytes'] / r['fvdb_bytes']:>9.2f}x"
        print(line)

    print(f"\n  Query time (us, {all_results[0]['n_queries']:,} queries):")
    hdr2 = f"  {'Voxels':>8} {'CIG3':>10}"
    if HAS_FVDB:
        hdr2 += f" {'fVDB':>10} {'CIG3/fVDB':>10}"
    print(hdr2)
    print(f"  {'-' * len(hdr2)}")
    for r in all_results:
        line = f"  {r['n_voxels']:>8,} {r['cig3_us']:>10.1f}"
        if HAS_FVDB and r.get("fvdb_us"):
            line += f" {r['fvdb_us']:>10.1f} {r['fvdb_us'] / r['cig3_us']:>9.2f}x"
        print(line)


if __name__ == "__main__":
    main()
