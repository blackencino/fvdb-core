# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for conv_grid topology expansion.
"""

import numpy as np
import torch

from fvdb_tile.prototype.conv_grid import conv_grid


def _reference_conv_grid(active: np.ndarray, kernel_size, stride):
    """Numpy reference: brute-force expand + dedup."""
    ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
    st = stride if isinstance(stride, tuple) else (stride, stride, stride)

    xs = np.arange(ks[0], dtype=np.int32)
    ys = np.arange(ks[1], dtype=np.int32)
    zs = np.arange(ks[2], dtype=np.int32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    offsets = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    cand = active[:, None, :] - offsets[None, :, :]
    if st != (1, 1, 1):
        stride_arr = np.array(st, dtype=np.int32)
        divisible = np.all(cand % stride_arr == 0, axis=2)
        cand = cand[divisible]
        if cand.size == 0:
            return np.empty((0, 3), dtype=np.int32)
        cand = cand.reshape(-1, 3) // stride_arr
    else:
        cand = cand.reshape(-1, 3)

    rows = cand.astype(np.int32)
    order = np.lexsort(tuple(rows[:, i] for i in range(rows.shape[1] - 1, -1, -1)))
    rows = rows[order]
    _, first_idx = np.unique(rows, axis=0, return_index=True)
    first_idx = np.sort(first_idx)
    return rows[first_idx]


def test_conv_grid_matches_reference():
    """conv_grid produces the same unique coords as the numpy reference."""
    active = np.array(
        [[4, 4, 4], [4, 4, 5], [6, 5, 4], [8, 8, 8]],
        dtype=np.int32,
    )
    ref = _reference_conv_grid(active, kernel_size=3, stride=1)
    result = conv_grid(active, kernel_size=3, stride=1, device="cpu")

    np.testing.assert_array_equal(result, ref)
    assert result.shape[1] == 3
    assert result.dtype == np.int32


def test_conv_grid_stride():
    """Strided conv_grid produces correct output."""
    active = np.array([[6, 6, 6], [7, 6, 6], [8, 8, 8]], dtype=np.int32)
    ref = _reference_conv_grid(active, kernel_size=3, stride=2)
    result = conv_grid(active, kernel_size=3, stride=2, device="cpu")

    np.testing.assert_array_equal(result, ref)


def test_conv_grid_preserves_input():
    """Input array is not mutated (value semantics)."""
    active = np.array([[6, 6, 6], [7, 6, 6], [8, 8, 8]], dtype=np.int32)
    before = active.copy()
    _ = conv_grid(active, kernel_size=3, stride=1, device="cpu")

    np.testing.assert_array_equal(active, before)


def test_conv_grid_accepts_torch_tensor():
    """conv_grid accepts torch.Tensor inputs."""
    active = torch.tensor([[4, 4, 4], [8, 8, 8]], dtype=torch.int32)
    result = conv_grid(active, kernel_size=3, stride=1, device="cpu")

    ref = _reference_conv_grid(active.numpy(), kernel_size=3, stride=1)
    np.testing.assert_array_equal(result, ref)


def test_conv_grid_larger_scale():
    """Runs conv_grid at a larger scale to exercise dedup pipeline."""
    rng = np.random.RandomState(42)
    active = rng.randint(0, 64, size=(500, 3)).astype(np.int32)
    active = np.unique(active, axis=0)

    ref = _reference_conv_grid(active, kernel_size=3, stride=1)
    result = conv_grid(active, kernel_size=3, stride=1, device="cpu")

    np.testing.assert_array_equal(result, ref)
    assert result.shape[0] > active.shape[0]
