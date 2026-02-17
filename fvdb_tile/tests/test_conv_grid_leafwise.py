# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for leafwise conv_grid (bitmask dilation).

Every test compares conv_grid_leafwise against the dense-expansion
conv_grid reference.  If they produce the same coordinate set, the
bitmask dilation is correct.
"""

import torch

from fvdb_tile.prototype.cig import build_compressed_cig3
from fvdb_tile.prototype.conv_grid import conv_grid
from fvdb_tile.prototype.conv_grid_leafwise import conv_grid_leafwise
from fvdb_tile.prototype.ops import hierarchical_key


_BIT_WIDTHS = [3, 4, 5]


def _assert_same_coord_set(actual: torch.Tensor, expected: torch.Tensor, msg: str = ""):
    """Assert two (N, 3) coordinate arrays are the same set."""
    assert actual.shape[1] == 3 and expected.shape[1] == 3
    assert len(actual) == len(expected), (
        f"Count mismatch: got {len(actual)}, expected {len(expected)}. {msg}"
    )
    a_codes = hierarchical_key(actual, _BIT_WIDTHS)
    e_codes = hierarchical_key(expected, _BIT_WIDTHS)
    a_order = torch.argsort(a_codes, stable=True)
    e_order = torch.argsort(e_codes, stable=True)
    torch.testing.assert_close(actual[a_order], expected[e_order], atol=0, rtol=0, msg=msg)


def _run_both(active, kernel_size, msg=""):
    """Run both conv_grid variants and assert same result."""
    ref = conv_grid(active, kernel_size=kernel_size, stride=1, device="cpu")
    cig = build_compressed_cig3(active)
    result = conv_grid_leafwise(cig, kernel_size=kernel_size, stride=1)
    _assert_same_coord_set(result, ref, msg=msg)
    return result


# =========================================================================
# Tests
# =========================================================================


def test_small_cluster():
    """Small cluster with voxels within a single leaf."""
    active = torch.tensor(
        [[4, 4, 4], [4, 4, 5], [6, 5, 4], [5, 5, 5]],
        dtype=torch.int32,
    )
    _run_both(active, kernel_size=3, msg="small_cluster k=3")


def test_cross_leaf_boundary():
    """Voxels straddling a leaf boundary (x=7 and x=8)."""
    active = torch.tensor([[7, 7, 7], [8, 8, 8]], dtype=torch.int32)
    _run_both(active, kernel_size=3, msg="cross_leaf k=3")


def test_negative_output_coords():
    """Voxels near origin producing negative output coordinates."""
    active = torch.tensor(
        [[0, 0, 0], [0, 1, 1], [1, 0, 2]],
        dtype=torch.int32,
    )
    result = _run_both(active, kernel_size=3, msg="negative_coords k=3")
    assert torch.any(result < 0), "Expected some negative output coordinates"


def test_single_voxel():
    """Single voxel produces exactly kernel_volume outputs."""
    active = torch.tensor([[2, 3, 4]], dtype=torch.int32)
    result = _run_both(active, kernel_size=3, msg="single_voxel k=3")
    assert result.shape[0] == 27


def test_single_voxel_at_leaf_corner():
    """Single voxel at leaf corner (0,0,0) -- all 27 offsets cross."""
    active = torch.tensor([[0, 0, 0]], dtype=torch.int32)
    result = _run_both(active, kernel_size=3, msg="corner_voxel k=3")
    assert result.shape[0] == 27


def test_single_voxel_at_leaf_edge():
    """Single voxel at (7,7,7) -- offsets cross in all directions."""
    active = torch.tensor([[7, 7, 7]], dtype=torch.int32)
    _run_both(active, kernel_size=3, msg="edge_voxel k=3")


def test_asymmetric_kernel():
    """Asymmetric kernel (3, 5, 7) produces correct result."""
    active = torch.tensor(
        [[2, 3, 4], [5, 5, 5]],
        dtype=torch.int32,
    )
    _run_both(active, kernel_size=(3, 5, 7), msg="asymmetric k=(3,5,7)")


def test_larger_scale():
    """500 random voxels with 3x3x3 kernel."""
    gen = torch.Generator().manual_seed(42)
    active = torch.randint(0, 64, size=(500, 3), generator=gen, dtype=torch.int32)
    active = torch.unique(active, dim=0)
    _run_both(active, kernel_size=3, msg="large_scale k=3")


def test_multi_leaf_sparse():
    """Sparse voxels across many leaves."""
    gen = torch.Generator().manual_seed(99)
    active = torch.randint(0, 128, size=(200, 3), generator=gen, dtype=torch.int32)
    active = torch.unique(active, dim=0)
    _run_both(active, kernel_size=3, msg="multi_leaf_sparse k=3")


def test_kernel_5():
    """5x5x5 kernel with cross-leaf voxels."""
    active = torch.tensor(
        [[4, 4, 4], [7, 7, 7], [8, 8, 8], [12, 12, 12]],
        dtype=torch.int32,
    )
    _run_both(active, kernel_size=5, msg="k=5")


def test_empty_input():
    """Empty CIG produces empty output."""
    active = torch.empty((0, 3), dtype=torch.int32)
    ref = conv_grid(active, kernel_size=3, stride=1, device="cpu")
    assert ref.shape[0] == 0


# =========================================================================
# GPU tests (skip if no CUDA)
# =========================================================================


def test_gpu_leafwise_matches_cpu():
    """GPU leafwise conv_grid matches CPU reference."""
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA device")
        return
    active = torch.tensor(
        [[4, 4, 4], [4, 4, 5], [6, 5, 4], [8, 8, 8]],
        dtype=torch.int32,
    )
    ref = conv_grid(active, kernel_size=3, stride=1, device="cpu")
    cig = build_compressed_cig3(active)
    result = conv_grid_leafwise(cig, kernel_size=3, stride=1, device="cuda")
    _assert_same_coord_set(result.cpu(), ref, msg="gpu_small k=3")


def test_gpu_leafwise_cross_leaf():
    """GPU leafwise handles cross-leaf boundaries."""
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA device")
        return
    active = torch.tensor([[7, 7, 7], [8, 8, 8]], dtype=torch.int32)
    ref = conv_grid(active, kernel_size=3, stride=1, device="cpu")
    cig = build_compressed_cig3(active)
    result = conv_grid_leafwise(cig, kernel_size=3, stride=1, device="cuda")
    _assert_same_coord_set(result.cpu(), ref, msg="gpu_cross_leaf k=3")


def test_gpu_leafwise_larger_scale():
    """GPU leafwise at 500 voxels matches CPU reference."""
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA device")
        return
    gen = torch.Generator().manual_seed(42)
    active = torch.randint(0, 64, size=(500, 3), generator=gen, dtype=torch.int32)
    active = torch.unique(active, dim=0)
    ref = conv_grid(active, kernel_size=3, stride=1, device="cpu")
    cig = build_compressed_cig3(active)
    result = conv_grid_leafwise(cig, kernel_size=3, stride=1, device="cuda")
    _assert_same_coord_set(result.cpu(), ref, msg="gpu_large k=3")


# =========================================================================

if __name__ == "__main__":
    cpu_tests = [
        ("small_cluster", test_small_cluster),
        ("cross_leaf_boundary", test_cross_leaf_boundary),
        ("negative_output_coords", test_negative_output_coords),
        ("single_voxel", test_single_voxel),
        ("single_voxel_at_leaf_corner", test_single_voxel_at_leaf_corner),
        ("single_voxel_at_leaf_edge", test_single_voxel_at_leaf_edge),
        ("asymmetric_kernel", test_asymmetric_kernel),
        ("larger_scale", test_larger_scale),
        ("multi_leaf_sparse", test_multi_leaf_sparse),
        ("kernel_5", test_kernel_5),
        ("empty_input", test_empty_input),
    ]
    gpu_tests = [
        ("gpu_leafwise_matches_cpu", test_gpu_leafwise_matches_cpu),
        ("gpu_leafwise_cross_leaf", test_gpu_leafwise_cross_leaf),
        ("gpu_leafwise_larger_scale", test_gpu_leafwise_larger_scale),
    ]

    print("=== Leafwise conv_grid tests (CPU) ===")
    for name, fn in cpu_tests:
        fn()
        print(f"  {name}: PASS")

    print("\n=== Leafwise conv_grid tests (GPU) ===")
    for name, fn in gpu_tests:
        fn()
        print(f"  {name}: PASS")

    print("\nAll leafwise conv_grid tests passed.")
