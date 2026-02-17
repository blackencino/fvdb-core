# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for conv_grid topology expansion.

Tests are divided into two groups:
  1. Self-consistency tests using a numpy reference (always runnable).
  2. fVDB-reference tests that compare against GridBatch.conv_grid()
     (require the fvdb package -- skipped if not importable).
"""

import numpy as np
import pytest
import torch

from fvdb_tile.prototype.conv_grid import conv_grid, _kernel_offsets_centered
from fvdb_tile.prototype.ops import morton3d_signed


# ---------------------------------------------------------------------------
# Numpy reference (uses centered offsets + addition, matching fVDB/new code)
# ---------------------------------------------------------------------------


def _reference_conv_grid(active: np.ndarray, kernel_size, stride):
    """Numpy reference: brute-force expand + dedup with centered offsets."""
    ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
    st = stride if isinstance(stride, tuple) else (stride, stride, stride)

    offsets = _kernel_offsets_centered(ks)

    cand = active[:, None, :] + offsets[None, :, :]
    if st != (1, 1, 1):
        stride_arr = np.array(st, dtype=np.int32)
        cand_flat = cand.reshape(-1, 3)
        divisible = np.all(cand_flat % stride_arr == 0, axis=1)
        cand_flat = cand_flat[divisible]
        if cand_flat.size == 0:
            return np.empty((0, 3), dtype=np.int32)
        cand_flat = cand_flat // stride_arr
    else:
        cand_flat = cand.reshape(-1, 3)

    rows = cand_flat.astype(np.int32)
    # Dedup via set
    unique_rows = np.unique(rows, axis=0)
    # Sort by morton code for deterministic order
    codes = morton3d_signed(unique_rows)
    order = np.argsort(codes, kind="stable")
    return unique_rows[order]


def _assert_same_coord_set(actual: np.ndarray, expected: np.ndarray, msg: str = ""):
    """Assert two (N, 3) coordinate arrays are the same SET (order-independent)."""
    assert actual.shape[1] == 3, f"actual shape {actual.shape}"
    assert expected.shape[1] == 3, f"expected shape {expected.shape}"
    assert len(actual) == len(expected), (
        f"Count mismatch: got {len(actual)}, expected {len(expected)}. {msg}"
    )
    # Sort both by morton code for comparison
    a_codes = morton3d_signed(actual)
    e_codes = morton3d_signed(expected)
    a_order = np.argsort(a_codes, kind="stable")
    e_order = np.argsort(e_codes, kind="stable")
    np.testing.assert_array_equal(actual[a_order], expected[e_order], err_msg=msg)


# =========================================================================
# 1. Self-consistency tests (numpy reference)
# =========================================================================


def test_conv_grid_matches_reference():
    """conv_grid produces the same unique coords as the numpy reference."""
    active = np.array(
        [[4, 4, 4], [4, 4, 5], [6, 5, 4], [8, 8, 8]],
        dtype=np.int32,
    )
    ref = _reference_conv_grid(active, kernel_size=3, stride=1)
    result = conv_grid(active, kernel_size=3, stride=1, device="cpu")

    _assert_same_coord_set(result, ref)
    assert result.dtype == np.int32


def test_conv_grid_stride():
    """Strided conv_grid produces correct output."""
    active = np.array([[6, 6, 6], [7, 6, 6], [8, 8, 8]], dtype=np.int32)
    ref = _reference_conv_grid(active, kernel_size=3, stride=2)
    result = conv_grid(active, kernel_size=3, stride=2, device="cpu")

    _assert_same_coord_set(result, ref)


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
    _assert_same_coord_set(result, ref)


def test_conv_grid_larger_scale():
    """Runs conv_grid at a larger scale to exercise dedup pipeline."""
    rng = np.random.RandomState(42)
    active = rng.randint(0, 64, size=(500, 3)).astype(np.int32)
    active = np.unique(active, axis=0)

    ref = _reference_conv_grid(active, kernel_size=3, stride=1)
    result = conv_grid(active, kernel_size=3, stride=1, device="cpu")

    _assert_same_coord_set(result, ref)
    assert result.shape[0] > active.shape[0]


def test_conv_grid_negative_coords():
    """conv_grid handles coordinates near origin that produce negative output."""
    active = np.array(
        [[0, 0, 0], [0, 1, 1], [1, 0, 2]],
        dtype=np.int32,
    )
    ref = _reference_conv_grid(active, kernel_size=3, stride=1)
    result = conv_grid(active, kernel_size=3, stride=1, device="cpu")

    _assert_same_coord_set(result, ref)
    assert np.any(result < 0), "Expected some negative output coordinates"


def test_conv_grid_nonuniform_stride():
    """Non-uniform stride (1,2,3) produces correct output."""
    active = np.array(
        [[3, 4, 6], [4, 4, 6], [3, 6, 9], [6, 6, 9]],
        dtype=np.int32,
    )
    ref = _reference_conv_grid(active, kernel_size=3, stride=(1, 2, 3))
    result = conv_grid(active, kernel_size=3, stride=(1, 2, 3), device="cpu")

    _assert_same_coord_set(result, ref)


def test_conv_grid_asymmetric_kernel():
    """Asymmetric kernel (3, 5, 7) works correctly."""
    active = np.array(
        [[2, 3, 4], [5, 5, 5]],
        dtype=np.int32,
    )
    ref = _reference_conv_grid(active, kernel_size=(3, 5, 7), stride=1)
    result = conv_grid(active, kernel_size=(3, 5, 7), stride=1, device="cpu")

    _assert_same_coord_set(result, ref)


def test_conv_grid_single_voxel():
    """Single input voxel with kernel (3,5,7) produces kernel_volume outputs."""
    active = np.array([[2, 3, 4]], dtype=np.int32)
    result = conv_grid(active, kernel_size=(3, 5, 7), stride=1, device="cpu")
    assert result.shape[0] == 3 * 5 * 7


# =========================================================================
# 2. fVDB-reference tests (skipped if fvdb is not available)
# =========================================================================

try:
    import fvdb
    from fvdb import GridBatch, JaggedTensor

    _HAS_FVDB = torch.cuda.is_available()
except (ImportError, RuntimeError):
    _HAS_FVDB = False

requires_fvdb = pytest.mark.skipif(not _HAS_FVDB, reason="fvdb not installed or CUDA not available")


def _fvdb_conv_grid(coords_np: np.ndarray, kernel_size, stride, device="cuda"):
    """Run fVDB's GridBatch.conv_grid and return (M, 3) i32 numpy coords."""
    dev = torch.device(device)
    coords_t = torch.tensor(coords_np, dtype=torch.int32, device=dev)
    ijks = JaggedTensor(coords_t)
    grid = GridBatch.from_ijk(ijks, device=dev)
    ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
    st = stride if isinstance(stride, tuple) else (stride, stride, stride)
    dst_grid = grid.conv_grid(kernel_size=ks, stride=st)
    return dst_grid.ijk.jdata.cpu().numpy().astype(np.int32)


@requires_fvdb
def test_conv_grid_vs_fvdb_stride1():
    """conv_grid matches fVDB for stride=1 with a small cluster."""
    active = np.array(
        [[0, 0, 0], [0, 1, 1], [1, 0, 2], [3, 4, 5], [4, 4, 5], [7, 8, 9]],
        dtype=np.int32,
    )
    fvdb_coords = _fvdb_conv_grid(active, kernel_size=3, stride=1)
    tile_coords = conv_grid(active, kernel_size=3, stride=1, device="cpu")

    _assert_same_coord_set(tile_coords, fvdb_coords, msg="stride=1, k=3")


@requires_fvdb
def test_conv_grid_vs_fvdb_asymmetric_kernel():
    """conv_grid matches fVDB for asymmetric kernel (3, 5, 7)."""
    active = np.array(
        [[2, 3, 4], [4, 4, 5], [3, 5, 6], [7, 8, 9]],
        dtype=np.int32,
    )
    fvdb_coords = _fvdb_conv_grid(active, kernel_size=(3, 5, 7), stride=1)
    tile_coords = conv_grid(active, kernel_size=(3, 5, 7), stride=1, device="cpu")

    _assert_same_coord_set(tile_coords, fvdb_coords, msg="stride=1, k=(3,5,7)")


@requires_fvdb
def test_conv_grid_vs_fvdb_stride_uniform():
    """conv_grid matches fVDB for uniform stride (2,2,2)."""
    ks = (3, 5, 7)
    kernel_half = tuple(k // 2 for k in ks)
    base = tuple(k + 1 for k in kernel_half)
    active = np.array(
        [
            [base[0] + 0, base[1] + 0, base[2] + 0],
            [base[0] + 2, base[1] + 0, base[2] + 1],
            [base[0] + 1, base[1] + 2, base[2] + 0],
            [base[0] + 3, base[1] + 3, base[2] + 3],
            [base[0] + 5, base[1] + 4, base[2] + 5],
        ],
        dtype=np.int32,
    )
    fvdb_coords = _fvdb_conv_grid(active, kernel_size=ks, stride=(2, 2, 2))
    tile_coords = conv_grid(active, kernel_size=ks, stride=(2, 2, 2), device="cpu")

    _assert_same_coord_set(tile_coords, fvdb_coords, msg="stride=(2,2,2)")


@requires_fvdb
def test_conv_grid_vs_fvdb_stride_nonuniform():
    """conv_grid matches fVDB for non-uniform stride (1,2,3)."""
    ks = (3, 5, 7)
    kernel_half = tuple(k // 2 for k in ks)
    base = tuple(k + 1 for k in kernel_half)
    active = np.array(
        [
            [base[0] + 0, base[1] + 0, base[2] + 0],
            [base[0] + 2, base[1] + 0, base[2] + 1],
            [base[0] + 1, base[1] + 2, base[2] + 0],
            [base[0] + 3, base[1] + 3, base[2] + 3],
            [base[0] + 5, base[1] + 4, base[2] + 5],
        ],
        dtype=np.int32,
    )
    fvdb_coords = _fvdb_conv_grid(active, kernel_size=ks, stride=(1, 2, 3))
    tile_coords = conv_grid(active, kernel_size=ks, stride=(1, 2, 3), device="cpu")

    _assert_same_coord_set(tile_coords, fvdb_coords, msg="stride=(1,2,3)")


@requires_fvdb
def test_conv_grid_vs_fvdb_larger_scale():
    """conv_grid matches fVDB at larger scale (500 random voxels, k=3)."""
    rng = np.random.RandomState(42)
    active = rng.randint(0, 64, size=(500, 3)).astype(np.int32)
    active = np.unique(active, axis=0)

    fvdb_coords = _fvdb_conv_grid(active, kernel_size=3, stride=1)
    tile_coords = conv_grid(active, kernel_size=3, stride=1, device="cpu")

    _assert_same_coord_set(tile_coords, fvdb_coords, msg="larger scale, k=3")


# =========================================================================

if __name__ == "__main__":
    print("=== Self-consistency tests ===")
    test_conv_grid_matches_reference()
    print("  test_conv_grid_matches_reference: PASS")
    test_conv_grid_stride()
    print("  test_conv_grid_stride: PASS")
    test_conv_grid_preserves_input()
    print("  test_conv_grid_preserves_input: PASS")
    test_conv_grid_accepts_torch_tensor()
    print("  test_conv_grid_accepts_torch_tensor: PASS")
    test_conv_grid_larger_scale()
    print("  test_conv_grid_larger_scale: PASS")
    test_conv_grid_negative_coords()
    print("  test_conv_grid_negative_coords: PASS")
    test_conv_grid_nonuniform_stride()
    print("  test_conv_grid_nonuniform_stride: PASS")
    test_conv_grid_asymmetric_kernel()
    print("  test_conv_grid_asymmetric_kernel: PASS")
    test_conv_grid_single_voxel()
    print("  test_conv_grid_single_voxel: PASS")

    if _HAS_FVDB:
        print("\n=== fVDB-reference tests ===")
        test_conv_grid_vs_fvdb_stride1()
        print("  test_conv_grid_vs_fvdb_stride1: PASS")
        test_conv_grid_vs_fvdb_asymmetric_kernel()
        print("  test_conv_grid_vs_fvdb_asymmetric_kernel: PASS")
        test_conv_grid_vs_fvdb_stride_uniform()
        print("  test_conv_grid_vs_fvdb_stride_uniform: PASS")
        test_conv_grid_vs_fvdb_stride_nonuniform()
        print("  test_conv_grid_vs_fvdb_stride_nonuniform: PASS")
        test_conv_grid_vs_fvdb_larger_scale()
        print("  test_conv_grid_vs_fvdb_larger_scale: PASS")
    else:
        print("\n(fvdb not available -- skipping fVDB-reference tests)")

    print("\nAll conv_grid tests passed.")
