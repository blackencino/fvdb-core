# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Correctness-first tests for conv_grid prototype with ephemeral CIG output.
"""

import numpy as np
import torch

from fvdb_tile.prototype.conv_grid import (
    CONV_GRID_COLLECTIVE_PIPELINE,
    conv_grid_ephemeral_cig,
)
from fvdb_tile.prototype.test_cig3 import cig3_ijk_to_index_numpy


def _reference_conv_grid(active: torch.Tensor, kernel_size, stride):
    ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
    st = stride if isinstance(stride, tuple) else (stride, stride, stride)

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
        cand = cand.reshape(-1, 3) // stride_vec
    else:
        cand = cand.reshape(-1, 3)

    rows = cand.numpy()
    order = np.lexsort(tuple(rows[:, i] for i in range(rows.shape[1] - 1, -1, -1)))
    rows = rows[order]
    _, first_idx = np.unique(rows, axis=0, return_index=True)
    first_idx = np.sort(first_idx)
    return torch.from_numpy(rows[first_idx]).to(torch.int32)


def test_conv_grid_ephemeral_matches_reference():
    active = torch.tensor(
        [
            [4, 4, 4],
            [4, 4, 5],
            [6, 5, 4],
            [8, 8, 8],
        ],
        dtype=torch.int32,
    )
    ref = _reference_conv_grid(active, kernel_size=(3, 3, 3), stride=(1, 1, 1))
    result = conv_grid_ephemeral_cig(active, kernel_size=(3, 3, 3), stride=(1, 1, 1))

    np.testing.assert_array_equal(result.active_coords.cpu().numpy(), ref.cpu().numpy())

    # Ephemeral materialization check: all produced coords are active in output CIG.
    lookup = cig3_ijk_to_index_numpy(result.cig3, result.active_coords.cpu().numpy())
    assert np.all(lookup >= 0)


def test_conv_grid_stride_semantics():
    active = torch.tensor([[6, 6, 6], [7, 6, 6], [8, 8, 8]], dtype=torch.int32)
    ref = _reference_conv_grid(active, kernel_size=3, stride=2)
    result = conv_grid_ephemeral_cig(active, kernel_size=3, stride=2)
    np.testing.assert_array_equal(result.active_coords.cpu().numpy(), ref.cpu().numpy())


def test_conv_grid_uses_collective_pipeline_and_preserves_input():
    active = torch.tensor([[6, 6, 6], [7, 6, 6], [8, 8, 8]], dtype=torch.int32)
    before = active.clone()

    result = conv_grid_ephemeral_cig(active, kernel_size=3, stride=1)
    kinds = [seg.kind for seg in CONV_GRID_COLLECTIVE_PIPELINE.plan.segments]

    assert "collective" in kinds
    assert result.active_coords.dtype == torch.int32
    assert torch.equal(active, before)
