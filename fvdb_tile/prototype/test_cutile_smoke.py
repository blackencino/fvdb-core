# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
cuTile smoke test: vector add via ct.gather / ct.scatter.

Validates the toolchain and confirms computed-index access works on
the current Blackwell hardware. Pattern adapted from TileGym softmax.
"""

import numpy as np
import torch

import cuda.tile as ct

ConstInt = ct.Constant[int]


# ---------------------------------------------------------------------------
# Kernel: vector add using gather/scatter (computed-index, not tile-aligned)
# ---------------------------------------------------------------------------

@ct.kernel
def vector_add_gather(a, b, result, N: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    offsets = ct.arange(TILE, dtype=ct.int32) + bid * TILE
    a_tile = ct.gather(a, offsets, check_bounds=True, padding_value=0.0)
    b_tile = ct.gather(b, offsets, check_bounds=True, padding_value=0.0)
    ct.scatter(result, offsets, a_tile + b_tile, check_bounds=True)


# ---------------------------------------------------------------------------
# Kernel: 2D gather with computed row+col indices
# ---------------------------------------------------------------------------

@ct.kernel
def gather_2d_smoke(data, result, TILE: ConstInt, COLS: ConstInt):
    """Gather from a 2D array using computed (row, col) indices."""
    bid = ct.bid(0)
    col_offsets = ct.arange(TILE, dtype=ct.int32)
    row = ct.gather(data, (bid, col_offsets), check_bounds=True, padding_value=0.0)
    ct.scatter(result, (bid, col_offsets), row * 2.0, check_bounds=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_vector_add():
    """1D vector add via gather/scatter with computed offsets."""
    N = 1024
    TILE = 128
    a_np = np.random.randn(N).astype(np.float32)
    b_np = np.random.randn(N).astype(np.float32)
    expected = a_np + b_np

    a = torch.from_numpy(a_np).cuda()
    b = torch.from_numpy(b_np).cuda()
    result = torch.zeros(N, dtype=torch.float32, device="cuda")

    grid = (N // TILE,)
    ct.launch(torch.cuda.current_stream(), grid, vector_add_gather, (a, b, result, N, TILE))

    result_np = result.cpu().numpy()
    np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-5)
    print(f"  vector_add: {N} elements, TILE={TILE} -- PASSED")


def test_gather_2d():
    """2D gather with computed (row, col) indices -- validates multi-dim indexing."""
    ROWS, COLS = 32, 64
    TILE = 64
    data_np = np.random.randn(ROWS, COLS).astype(np.float32)
    expected = data_np * 2.0

    data = torch.from_numpy(data_np).cuda()
    result = torch.zeros(ROWS, COLS, dtype=torch.float32, device="cuda")

    grid = (ROWS,)
    ct.launch(torch.cuda.current_stream(), grid, gather_2d_smoke, (data, result, TILE, COLS))

    result_np = result.cpu().numpy()
    np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-5)
    print(f"  gather_2d: ({ROWS},{COLS}), TILE={TILE} -- PASSED")


if __name__ == "__main__":
    print("=== cuTile smoke tests ===")
    test_vector_add()
    test_gather_2d()
    print("\nAll cuTile smoke tests passed.")
