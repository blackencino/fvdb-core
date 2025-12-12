# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Utilities for testing 3D convolution operations.

This module provides:
1. Baseline references for 3D convolution using PyTorch dense operations
2. Helper functions for comparing sparse and dense convolution results
3. Utilities for coordinate manipulation and comparison

fVDB uses the following order for tensors in convolution:

[BATCH, SPATIAL_AXIS_0, SPATIAL_AXIS_1, SPATIAL_AXIS_2, FEATURES]

SPATIAL_AXIS_0 is the major axis (slowest-changing spatial coord in contiguous tensor layout)
SPATIAL_AXIS_2 is the minor axis (fastest-changing spatial coord in contiguous tensor layout)

In fVDB voxel coordinates, x is the major axis, z is the minor axis.

It is important that when spatial axes are referred to, we avoid calling them
"width", "height", or "depth", and we ignore the application of those terms in the torch
documentation. Because the spatial axes don't always have the same physical meaning, for example
for Z-up interpretations of x, y, z, the concept of the "height" of the volume would be ambiguous.

When we interact with torch's convolution, we swap the order of the channels and the spatial
axes, but we otherwise keep the spatial axes in the same order as fVDB:

[BATCH, FEATURES, SPATIAL_AXIS_0, SPATIAL_AXIS_1, SPATIAL_AXIS_2]

That way, spatial function arguments like kernel_size, stride, bias don't need to be reversed.
"""

import math
from contextlib import contextmanager
from typing import Sequence

import torch
from fvdb.types import NumericMaxRank1, ValueConstraint, to_Vec3i

from fvdb import GridBatch, JaggedTensor

# =============================================================================
# TF32 Control
# =============================================================================


@contextmanager
def disable_tf32():
    """
    Context manager to temporarily disable TF32 for consistent precision.

    TF32 (TensorFloat-32) can cause numerical differences between CPU and CUDA.
    Use this when comparing results across devices or when exact precision matters.

    Example:
        with disable_tf32():
            output = torch.nn.functional.conv3d(input, weight, padding="same")
    """
    old_setting = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = old_setting


# =============================================================================
# Coordinate Utilities
# =============================================================================


def sort_coords_by_ijk(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort coordinates by a unique encoding and return sorted coords and permutation.

    This is useful for comparing coordinate sets that may be in different orders
    (e.g., due to different tiling strategies in sparse representations).

    Args:
        coords: Tensor of shape (N, 3) containing ijk coordinates.
                Supports negative coordinates.

    Returns:
        Tuple of (sorted_coords, permutation_indices) where:
        - sorted_coords: The input coordinates sorted by (i, j, k) order
        - permutation_indices: Indices such that coords[permutation_indices] == sorted_coords
    """
    # Use large multipliers to ensure unique encoding even with negative coords
    encoding = (
        coords[:, 0].to(torch.int64) * 1000000000
        + coords[:, 1].to(torch.int64) * 1000000
        + coords[:, 2].to(torch.int64)
    )
    perm = torch.argsort(encoding)
    return coords[perm], perm


def assert_coords_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    msg: str = "",
) -> None:
    """
    Assert that two coordinate tensors contain the same coordinates (order-independent).

    Args:
        actual: Tensor of shape (N, 3) with actual coordinates
        expected: Tensor of shape (M, 3) with expected coordinates
        msg: Optional message to include in assertion errors

    Raises:
        AssertionError: If the coordinate sets don't match
    """
    assert len(actual) == len(
        expected
    ), f"Coordinate count mismatch: got {len(actual)}, expected {len(expected)}. {msg}"

    actual_sorted, _ = sort_coords_by_ijk(actual)
    expected_sorted, _ = sort_coords_by_ijk(expected)

    assert torch.equal(actual_sorted, expected_sorted), f"Coordinates do not match. {msg}"


def normalize_stride(stride: int | Sequence[int]) -> tuple[int, int, int]:
    """Normalize stride to a 3-tuple."""
    if isinstance(stride, int):
        return (stride, stride, stride)
    return tuple(stride)  # type: ignore


# =============================================================================
# Topology Ground Truth
# =============================================================================


def compute_conv_grid_topology_ground_truth(
    input_coords: torch.Tensor,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute the expected output coordinates for conv_grid using dense convolution.

    This function computes which output coordinates would be non-zero after
    convolving sparse input with an all-ones kernel. This is useful for validating
    the topology (coordinate set) produced by sparse convolution operations.

    For stride=1:
        Uses dense convolution with 'same' padding and finds non-zero locations.

    For stride>1:
        Computes output coordinates analytically based on receptive field overlap.
        The output at coordinate o receives contributions from input coordinates
        in the range [o*stride - kernel_half, o*stride + kernel_half].

    Args:
        input_coords: Tensor of shape (N, 3) with input ijk coordinates
        kernel_size: Tuple of kernel dimensions (k0, k1, k2)
        stride: Tuple of stride values (s0, s1, s2)
        device: Target device
        dtype: Data type for computation

    Returns:
        Tensor of shape (M, 3) with expected output ijk coordinates
    """
    kernel_half = tuple(k // 2 for k in kernel_size)

    # Compute coordinate ranges
    input_min = input_coords.min(dim=0).values.tolist()
    input_max = input_coords.max(dim=0).values.tolist()

    if stride == (1, 1, 1):
        # Stride 1: output extends by half-kernel beyond input
        output_min = tuple(input_min[i] - kernel_half[i] for i in range(3))
        output_max = tuple(input_max[i] + kernel_half[i] for i in range(3))
        coord_offset = output_min
        dense_shape = tuple(output_max[i] - output_min[i] + 1 for i in range(3))

        # Create dense input
        dense_input = torch.zeros((1, 1) + dense_shape, device=device, dtype=dtype)
        for coord in input_coords:
            idx = tuple(coord[i].item() - coord_offset[i] for i in range(3))
            dense_input[0, 0, idx[0], idx[1], idx[2]] = 1

        # All-ones kernel
        kernel = torch.ones((1, 1) + kernel_size, device=device, dtype=dtype)

        # Dense convolution with 'same' padding
        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=kernel, padding="same")

        # Find non-zero coordinates and convert back to grid coordinates
        nonzero_indices = torch.nonzero(dense_output[0, 0] != 0)
        offset_tensor = torch.tensor(coord_offset, device=device, dtype=torch.int32)
        expected_coords = nonzero_indices.to(torch.int32) + offset_tensor

    else:
        # For stride > 1, compute analytically based on receptive field overlap
        # For an input at coord c, it contributes to outputs at:
        #   o where o*stride - kernel_half <= c <= o*stride + kernel_half
        #   i.e., (c - kernel_half) / stride <= o <= (c + kernel_half) / stride
        # Using ceiling/floor for the bounds:
        #   o_min = ceil((c - kernel_half) / stride)
        #   o_max = floor((c + kernel_half) / stride)

        output_coords_set: set[tuple[int, int, int]] = set()

        for coord in input_coords:
            c = coord.tolist()

            # For each input coordinate, find all output coordinates it contributes to
            o_ranges = []
            for dim in range(3):
                c_val = c[dim]
                kh = kernel_half[dim]
                s = stride[dim]
                o_min = math.ceil((c_val - kh) / s)
                o_max = math.floor((c_val + kh) / s)
                o_ranges.append(range(o_min, o_max + 1))

            # Add all combinations
            for o0 in o_ranges[0]:
                for o1 in o_ranges[1]:
                    for o2 in o_ranges[2]:
                        output_coords_set.add((o0, o1, o2))

        expected_coords = torch.tensor(list(output_coords_set), device=device, dtype=torch.int32)

    return expected_coords


# =============================================================================
# Dense Convolution Ground Truth
# =============================================================================


def conv_ground_truth_stride_1(
    grid_batch: GridBatch,
    activation: JaggedTensor,
    weights: torch.Tensor,
    *,
    dense_dims: NumericMaxRank1 | None = None,
    ijk_min: NumericMaxRank1 | None = None,
    allow_tf32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ground truth 3D convolution (with stride 1) over a GridBatch using PyTorch.

    This function first densifies the sparse input activation to a dense tensor in
    channel-major ("C-major") order as required by PyTorch's `conv3d`. The dense region
    is determined by the optional `dense_dims`/`ijk_min` arguments or, if not provided,
    by the total bounding box of the grid batch.

    The function then performs a 3D convolution using `torch.nn.functional.conv3d`
    with "same" padding (which is supported only for stride 1 in PyTorch). The resulting
    dense tensor is mapped back into a sparse JaggedTensor, matching the original sparse layout.

    Args:
        grid_batch (GridBatch): The input spatial grid batch over which to convolve.
        activation (JaggedTensor): Voxel features or activations over the grid (sparse).
            Shape: (batch_size, total_voxels, channels)
        weights (torch.Tensor): Convolution kernel weights in
            PyTorch conv3d format. Shape:
            (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
        dense_dims (NumericMaxRank1 | None, optional): The spatial dimensions
            of the dense tensor region to extract. If None, uses the bounding box of `grid_batch`.
        ijk_min (NumericMaxRank1 | None, optional): The minimum IJK coordinate
            (origin) for the dense region. If None, uses the bbox origin of `grid_batch`.
        allow_tf32 (bool, optional): If True, enables TF32 on supported hardware for
            faster computation. Default is False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - dense_activation (torch.Tensor): The densified input features in C-major order.
                Shape: (batch_size, in_channels, dim0, dim1, dim2)
            - convolved (torch.Tensor): The dense convolved features (same shape as dense_activation).
    """
    bbox = grid_batch.total_bbox
    if ijk_min is None:
        ijk_min = torch.tensor(bbox[0], device="cpu")
    else:
        ijk_min = to_Vec3i(ijk_min)

    if dense_dims is None:
        dense_dims = 1 + (torch.tensor(bbox[1], device="cpu") - ijk_min)
    else:
        dense_dims = to_Vec3i(dense_dims, value_constraint=ValueConstraint.POSITIVE)

    dense_activation = grid_batch.inject_to_dense_cmajor(
        sparse_data=activation, min_coord=ijk_min, grid_size=dense_dims
    )

    if allow_tf32:
        convolved = torch.nn.functional.conv3d(input=dense_activation, weight=weights, padding="same")
    else:
        with disable_tf32():
            convolved = torch.nn.functional.conv3d(input=dense_activation, weight=weights, padding="same")

    if dense_activation.shape != convolved.shape:
        raise ValueError(
            f"Dense activation shape {dense_activation.shape} does not match convolved shape {convolved.shape}"
        )

    return dense_activation, convolved
