# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: cuTile neighbor predicate kernel vs numpy vs PyTorch.

Measures wall-clock time for the neighbor gather predicate on a random
(8,8,8) leaf with ~450 active voxels. Provides the project's first
quantitative performance claim.
"""

import time

import numpy as np
import torch

import cuda.tile as ct

from fvdb_tile.prototype.test_cutile_gather import (
    FACE_OFFSETS,
    TILE_OFFSETS,
    neighbor_predicate_kernel,
)

ConstInt = ct.Constant[int]


# ---------------------------------------------------------------------------
# Numpy reference
# ---------------------------------------------------------------------------

def numpy_neighbor_predicate(mask_np, active_coords_np, offsets_np):
    """Brute-force numpy: per-voxel, per-offset loop."""
    N = active_coords_np.shape[0]
    N_OFF = offsets_np.shape[0]
    result = np.zeros((N, N_OFF), dtype=bool)
    for i in range(N):
        coord = active_coords_np[i]
        for j in range(N_OFF):
            nbr = coord + offsets_np[j]
            if np.all(nbr >= 0) and np.all(nbr < 8):
                if mask_np[nbr[0], nbr[1], nbr[2]] >= 0:
                    result[i, j] = True
    return result


# ---------------------------------------------------------------------------
# PyTorch (CPU) -- vectorized
# ---------------------------------------------------------------------------

def pytorch_cpu_neighbor_predicate(mask_np, active_coords_np, offsets_np):
    """Vectorized PyTorch on CPU."""
    mask_t = torch.from_numpy(mask_np)
    coords_t = torch.from_numpy(active_coords_np)
    offsets_t = torch.from_numpy(offsets_np)

    # (N, 1, 3) + (1, 6, 3) -> (N, 6, 3)
    nbr = coords_t.unsqueeze(1) + offsets_t.unsqueeze(0)

    in_bounds = (nbr >= 0).all(dim=2) & (nbr < 8).all(dim=2)

    ni = nbr[:, :, 0].clamp(0, 7)
    nj = nbr[:, :, 1].clamp(0, 7)
    nk = nbr[:, :, 2].clamp(0, 7)
    vals = mask_t[ni, nj, nk]

    result = in_bounds & (vals >= 0)
    return result.numpy()


# ---------------------------------------------------------------------------
# PyTorch (GPU) -- vectorized
# ---------------------------------------------------------------------------

def pytorch_gpu_neighbor_predicate(mask_t, coords_t, offsets_t):
    """Vectorized PyTorch on GPU."""
    nbr = coords_t.unsqueeze(1) + offsets_t.unsqueeze(0)

    in_bounds = (nbr >= 0).all(dim=2) & (nbr < 8).all(dim=2)

    ni = nbr[:, :, 0].clamp(0, 7)
    nj = nbr[:, :, 1].clamp(0, 7)
    nk = nbr[:, :, 2].clamp(0, 7)
    vals = mask_t[ni, nj, nk]

    return in_bounds & (vals >= 0)


# ---------------------------------------------------------------------------
# cuTile kernel launcher
# ---------------------------------------------------------------------------

def cutile_neighbor_predicate(mask_t, coords_t, offsets_t, N):
    """Launch the hand-written cuTile kernel."""
    result_t = torch.zeros(N, TILE_OFFSETS, dtype=torch.int32, device="cuda")
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        neighbor_predicate_kernel,
        (mask_t, coords_t, offsets_t, result_t, TILE_OFFSETS),
    )
    return result_t[:, :6]


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def _time_fn(fn, warmup=3, repeats=50):
    """Time a function with warmup and repeats. Returns median time in us."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]


def bench():
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    mask_bool = leaf_data >= 0
    active_coords = np.argwhere(mask_bool).astype(np.int32)
    N = active_coords.shape[0]

    # Verify all implementations agree
    ref = numpy_neighbor_predicate(leaf_data, active_coords, FACE_OFFSETS)
    pt_cpu = pytorch_cpu_neighbor_predicate(leaf_data, active_coords, FACE_OFFSETS)
    np.testing.assert_array_equal(ref, pt_cpu)

    mask_t = torch.from_numpy(leaf_data.copy()).cuda()
    coords_t = torch.from_numpy(active_coords.copy()).cuda()
    offsets_t = torch.from_numpy(FACE_OFFSETS.copy()).cuda()

    pt_gpu = pytorch_gpu_neighbor_predicate(mask_t, coords_t, offsets_t).cpu().numpy()
    np.testing.assert_array_equal(ref, pt_gpu)

    ct_result = cutile_neighbor_predicate(mask_t, coords_t, offsets_t, N)
    ct_result_np = ct_result.cpu().numpy().astype(bool)
    np.testing.assert_array_equal(ref, ct_result_np)

    total_nbrs = int(ref.sum())
    print(f"Data: {N} active voxels, {total_nbrs} active neighbors")
    print(f"All implementations agree.\n")

    # Benchmark
    t_numpy = _time_fn(lambda: numpy_neighbor_predicate(leaf_data, active_coords, FACE_OFFSETS))
    t_pt_cpu = _time_fn(lambda: pytorch_cpu_neighbor_predicate(leaf_data, active_coords, FACE_OFFSETS))
    t_pt_gpu = _time_fn(lambda: pytorch_gpu_neighbor_predicate(mask_t, coords_t, offsets_t))
    t_cutile = _time_fn(lambda: cutile_neighbor_predicate(mask_t, coords_t, offsets_t, N))

    print(f"{'Method':<25} {'Median (us)':>12} {'vs numpy':>10} {'vs GPU PyTorch':>15}")
    print("-" * 65)
    print(f"{'numpy (loop)':<25} {t_numpy:>12.1f} {'1.0x':>10} {'':<15}")
    print(f"{'PyTorch CPU (vectorized)':<25} {t_pt_cpu:>12.1f} {t_numpy/t_pt_cpu:>9.1f}x {'':<15}")
    print(f"{'PyTorch GPU (vectorized)':<25} {t_pt_gpu:>12.1f} {t_numpy/t_pt_gpu:>9.1f}x {'1.0x':>15}")
    print(f"{'cuTile (hand-written)':<25} {t_cutile:>12.1f} {t_numpy/t_cutile:>9.1f}x {t_pt_gpu/t_cutile:>14.1f}x")


if __name__ == "__main__":
    print("=== Neighbor predicate benchmark ===\n")
    bench()
