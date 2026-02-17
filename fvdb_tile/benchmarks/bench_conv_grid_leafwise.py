# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: leafwise conv_grid (bitmask dilation) vs dense expansion vs fVDB.

Compares three implementations of sparse convolution topology expansion:
  1. Dense expansion: N*K candidates, sort, unique (CPU torch)
  2. Leafwise GPU: fused CUDA kernel, bitmask dilation via hash map
  3. fVDB: C++/CUDA GridBatch.conv_grid() (if available)

Run:  fvdb_tile/run_gpu.sh fvdb_tile/benchmarks/bench_conv_grid_leafwise.py
"""

from __future__ import annotations

import time

import torch

from fvdb_tile.prototype.cig import build_compressed_cig3
from fvdb_tile.prototype.conv_grid import conv_grid
from fvdb_tile.prototype.conv_grid_leafwise import conv_grid_leafwise
from fvdb_tile.prototype.ops import hierarchical_key

try:
    import fvdb
    from fvdb import GridBatch, JaggedTensor

    HAS_FVDB = torch.cuda.is_available()
except (ImportError, RuntimeError):
    HAS_FVDB = False

HAS_CUDA = torch.cuda.is_available()

if not HAS_CUDA:
    print("ERROR: This benchmark requires a CUDA device.")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_gpu(fn, warmup=3, repeats=20):
    """GPU-synced timing: warmup, then median of repeats (microseconds)."""
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


def _time_cpu(fn, warmup=1, repeats=5):
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


def _gen_coords(n: int, extent: int = 256, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    coords = set()
    while len(coords) < n:
        batch = torch.randint(0, extent, (n * 2, 3), generator=gen)
        for row in batch:
            coords.add((int(row[0]), int(row[1]), int(row[2])))
            if len(coords) >= n:
                break
    return torch.tensor(sorted(coords)[:n], dtype=torch.int32)


def _assert_same(a: torch.Tensor, b: torch.Tensor, label: str = "") -> bool:
    if a.shape[0] != b.shape[0]:
        print(f"  MISMATCH {label}: {a.shape[0]} vs {b.shape[0]}")
        return False
    ka = hierarchical_key(a, [3, 4, 5])
    kb = hierarchical_key(b, [3, 4, 5])
    return torch.equal(a[torch.argsort(ka, stable=True)], b[torch.argsort(kb, stable=True)])


# ---------------------------------------------------------------------------
# Benchmark one configuration
# ---------------------------------------------------------------------------


def bench_one(n_voxels: int, kernel_size: tuple[int, int, int]) -> dict:
    active = _gen_coords(n_voxels)
    cig = build_compressed_cig3(active)

    r: dict = {
        "n_voxels": active.shape[0],
        "n_leaves": cig.n_leaves,
        "avg_occ": active.shape[0] / max(cig.n_leaves, 1),
        "kernel_size": kernel_size[0],
    }

    # --- Dense expansion (CPU, torch sort/unique) ---
    ref = conv_grid(active, kernel_size=kernel_size, stride=1, device="cpu")
    r["dst_voxels"] = ref.shape[0]
    r["dense_cpu_us"] = _time_cpu(
        lambda: conv_grid(active, kernel_size=kernel_size, stride=1, device="cpu")
    )

    # --- Leafwise GPU (fused CUDA kernel) ---
    leaf_result = conv_grid_leafwise(cig, kernel_size=kernel_size, stride=1, device="cuda")
    leaf_ok = _assert_same(ref, leaf_result.cpu(), "leaf_gpu")
    r["leaf_ok"] = leaf_ok
    r["leaf_gpu_us"] = _time_gpu(
        lambda: conv_grid_leafwise(cig, kernel_size=kernel_size, stride=1, device="cuda")
    )

    # --- fVDB (C++/CUDA) ---
    if HAS_FVDB:
        dev = torch.device("cuda")
        coords_cuda = active.to(device=dev, dtype=torch.int32)
        ijks = JaggedTensor(coords_cuda)
        grid = GridBatch.from_ijk(ijks, device=dev)

        fvdb_dst = grid.conv_grid(kernel_size=kernel_size, stride=(1, 1, 1))
        fvdb_coords = fvdb_dst.ijk.jdata.cpu().to(torch.int32)
        fvdb_ok = _assert_same(ref, fvdb_coords, "fvdb")
        r["fvdb_ok"] = fvdb_ok

        # End-to-end: from_ijk + conv_grid
        def fvdb_e2e():
            g = GridBatch.from_ijk(JaggedTensor(coords_cuda), device=dev)
            _ = g.conv_grid(kernel_size=kernel_size, stride=(1, 1, 1))

        r["fvdb_e2e_us"] = _time_gpu(fvdb_e2e)

        # Conv only (grid already built)
        r["fvdb_conv_us"] = _time_gpu(
            lambda: grid.conv_grid(kernel_size=kernel_size, stride=(1, 1, 1))
        )

    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    if HAS_FVDB:
        print(f"fVDB: available")
    else:
        print(f"fVDB: not available (fVDB columns will be blank)")
    print()

    print("=" * 95)
    print("Conv Grid: Dense (CPU) vs Leafwise GPU vs fVDB")
    print("=" * 95)

    configs = [
        (1_000, (3, 3, 3)),
        (10_000, (3, 3, 3)),
        (50_000, (3, 3, 3)),
        (100_000, (3, 3, 3)),
        (200_000, (3, 3, 3)),
        (10_000, (5, 5, 5)),
        (50_000, (5, 5, 5)),
        (100_000, (5, 5, 5)),
    ]

    header = (f"{'n_vox':>8} {'k':>5} {'leaves':>7} {'dst':>10}"
              f" {'leaf_gpu':>11}")
    if HAS_FVDB:
        header += f" {'fvdb_conv':>11} {'speedup':>9}"
    print(header)
    print("-" * len(header))

    for n, ks in configs:
        r = bench_one(n, ks)
        status = "ok" if r.get("leaf_ok", False) else "FAIL"
        k_str = f"{ks[0]}x{ks[1]}x{ks[2]}"
        line = (f"{r['n_voxels']:>8,} {k_str:>5} {r['n_leaves']:>7,}"
                f" {r['dst_voxels']:>10,}"
                f" {r['leaf_gpu_us']:>10,.0f}u")

        if HAS_FVDB:
            fvdb_conv = r.get('fvdb_conv_us', 0)
            speedup = fvdb_conv / r['leaf_gpu_us'] if r['leaf_gpu_us'] > 0 else 0
            line += f" {fvdb_conv:>10,.0f}u {speedup:>8.2f}x"

        print(f"{line}  [{status}]")

    print()
    print("Legend:")
    print("  leaf_gpu   -- leafwise bitmask dilation, GPU (fused CUDA kernel)")
    if HAS_FVDB:
        print("  fvdb_conv  -- fVDB conv_grid only (grid already built)")
        print("  speedup    -- fvdb_conv / leaf_gpu (> 1.0 = leaf wins)")
    print()
    print("NOTE: leaf_gpu uses a hand-fused CUDA kernel, not yet DSL-expressed.")
    print("Future: DSL program + idiom-recognition lowering -> fused kernel.")


if __name__ == "__main__":
    main()
