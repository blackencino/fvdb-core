# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
conv_grid prototype benchmark with stage breakdown.

This benchmark is correctness-first and CPU-friendly:
  - expand candidates
  - dedup via DSL pipeline (Sort + Unique)
  - CIG3 materialization

It compares against a simple numpy/torch baseline implementation of the same
semantic contract.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from fvdb_tile.prototype.cig import build_compressed_cig3, build_compressed_cig3_from_unique
from fvdb_tile.prototype.conv_grid import CONV_GRID_COLLECTIVE_PIPELINE, conv_grid_ephemeral_cig
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import ScalarType, Shape, Static, Type


def _as_vec3(value):
    if isinstance(value, int):
        return (value, value, value)
    return tuple(value)


def _make_active_coords(n_active: int, coord_limit: int = 256, seed: int = 123) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    coords = rng.randint(0, coord_limit, size=(n_active, 3)).astype(np.int32)
    # Dedup for stable workload definition.
    coords = np.unique(coords, axis=0)
    return torch.from_numpy(coords)


def _expand_candidates(active: torch.Tensor, kernel_size, stride):
    ks = _as_vec3(kernel_size)
    st = _as_vec3(stride)
    xs = torch.arange(ks[0], dtype=torch.int32)
    ys = torch.arange(ks[1], dtype=torch.int32)
    zs = torch.arange(ks[2], dtype=torch.int32)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    offsets = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)

    cand = active[:, None, :] - offsets[None, :, :]
    if st != (1, 1, 1):
        stride_vec = torch.tensor(st, dtype=torch.int32)
        valid = ((cand % stride_vec) == 0).all(dim=2)
        cand = cand[valid]
        if cand.numel() == 0:
            return cand.reshape(0, 3)
        return cand.reshape(-1, 3) // stride_vec
    return cand.reshape(-1, 3)


def _dedup_pipeline(coords: torch.Tensor) -> torch.Tensor:
    coords_np = coords.numpy().astype(np.int32, copy=False)
    coords_val = Value(
        Type(Shape(Static(coords_np.shape[0])), Type(Shape(Static(3)), ScalarType.I32)),
        coords_np.copy(),
    )
    out = CONV_GRID_COLLECTIVE_PIPELINE.run({"coords": coords_val}).output
    return torch.from_numpy(out.data.astype(np.int32, copy=False))


def _dedup_baseline(coords: torch.Tensor) -> torch.Tensor:
    rows = coords.numpy()
    if rows.shape[0] == 0:
        return coords
    order = np.lexsort(tuple(rows[:, i] for i in range(rows.shape[1] - 1, -1, -1)))
    rows = rows[order]
    _, first_idx = np.unique(rows, axis=0, return_index=True)
    first_idx = np.sort(first_idx)
    return torch.from_numpy(rows[first_idx].astype(np.int32, copy=False))


def _time_us(fn, warmup=1, repeats=5):
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


def bench_one(n_active: int, kernel_size=3, stride=1):
    active = _make_active_coords(n_active=n_active)

    t_expand = _time_us(lambda: _expand_candidates(active, kernel_size, stride), warmup=1, repeats=8)
    cand = _expand_candidates(active, kernel_size, stride)

    t_dedup_pipeline = _time_us(lambda: _dedup_pipeline(cand), warmup=1, repeats=8)
    unique_pipeline = _dedup_pipeline(cand)

    t_dedup_baseline = _time_us(lambda: _dedup_baseline(cand), warmup=1, repeats=8)
    unique_baseline = _dedup_baseline(cand)

    t_build_generic = _time_us(lambda: build_compressed_cig3(unique_pipeline), warmup=1, repeats=4)
    t_build_from_unique = _time_us(lambda: build_compressed_cig3_from_unique(unique_pipeline), warmup=1, repeats=4)
    cig3 = build_compressed_cig3_from_unique(unique_pipeline)

    t_total_pipeline = _time_us(lambda: conv_grid_ephemeral_cig(active, kernel_size=kernel_size, stride=stride), warmup=1, repeats=4)

    np.testing.assert_array_equal(unique_pipeline.numpy(), unique_baseline.numpy())

    print(f"\n--- n_active={active.shape[0]:,}, kernel={_as_vec3(kernel_size)}, stride={_as_vec3(stride)} ---")
    print(f"expand_us:          {t_expand:10.1f}")
    print(f"dedup_pipeline_us:  {t_dedup_pipeline:10.1f}")
    print(f"dedup_baseline_us:  {t_dedup_baseline:10.1f}")
    print(f"build_cig3_us:      {t_build_generic:10.1f}  (generic)")
    print(f"build_cig3_uq_us:   {t_build_from_unique:10.1f}  (from_unique)")
    print(f"total_pipeline_us:  {t_total_pipeline:10.1f}")
    print(
        f"output: coords={unique_pipeline.shape[0]:,}, "
        f"cig3 bytes={cig3.num_bytes:,}, upper/lower/leaf={cig3.n_upper}/{cig3.n_lower}/{cig3.n_leaves}"
    )


def main():
    print("=== conv_grid prototype benchmark (correctness-first) ===")
    for n in [2000, 8000, 20000]:
        bench_one(n_active=n, kernel_size=3, stride=1)
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
