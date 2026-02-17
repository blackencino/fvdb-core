# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
conv_grid: topology expansion for sparse convolution.

Given active input voxel coordinates, kernel size, and stride, compute the
set of unique output coordinates.  The semantic equation:

    output y is active iff there exists active x and kernel offset k
    such that (x + k) is divisible by stride and y = (x + k) / stride.

**Algorithm (expand + dedup):**

The algorithm decomposes into three composable primitives::

    expanded = expand_offsets(active, offsets)     # N*K candidates
    filtered = stride_filter(expanded, stride)     # M candidates
    result   = dedup_coords(filtered, bit_widths)  # unique output

In DSL / leading-shape terms::

    expanded  = fuse(EachLeft(EachRight(EachBoth(Add)), coords, offsets))
    filtered  = stride_filter(expanded, stride)
    codes     = HierarchicalKey(filtered, bit_widths)
    sorted    = Sort(codes)
    unique    = Unique(sorted)
    result    = HierarchicalKeyDecode(unique, bit_widths)

This is the same algorithm used by fVDB's ``BuildGridForConv.cu``:
for each active voxel, generate all kernel-offset neighbors, filter by
stride divisibility, divide, then deduplicate via grid construction.

The algorithm works identically for all strides.  For stride=1 the
filter is a no-op; the expand + dedup dominates.  Complexity is
O(N*K * log(N*K)) where N is the number of active voxels and K is
the kernel volume.

Voxel ordering uses ``hierarchical_key`` with configurable ``bit_widths``
for CIG-compatible tree-traversal order.
"""

from __future__ import annotations

import torch

from .ops import dedup_coords, expand_offsets, stride_filter


# ---------------------------------------------------------------------------
# Default CIG bit-widths (leaf=8^3, lower=16^3, upper=32^3)
# ---------------------------------------------------------------------------

DEFAULT_BIT_WIDTHS: list[int] = [3, 4, 5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_vec3(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"Expected scalar or length-3, got {value}")
    return (int(value[0]), int(value[1]), int(value[2]))


def _kernel_offsets_centered(kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """Generate (K, 3) i32 kernel offset tensor matching fVDB convention.

    For odd kernel sizes: centered offsets [-(k-1)/2, (k-1)/2].
    For even kernel sizes: offsets [0, k-1].
    """
    ranges = []
    for k in kernel_size:
        if k <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if k % 2 == 0:
            ranges.append(torch.arange(0, k, dtype=torch.int32))
        else:
            half = (k - 1) // 2
            ranges.append(torch.arange(-half, half + 1, dtype=torch.int32))
    gx, gy, gz = torch.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def conv_grid(
    active_coords: torch.Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    device: str = "cpu",
    bit_widths: list[int] | None = None,
) -> torch.Tensor:
    """Compute unique output coordinates for a sparse convolution topology.

    Uses expand + stride_filter + dedup_coords, matching the fVDB
    ``BuildGridForConv.cu`` algorithm.

    Args:
        active_coords: (N, 3) i32 active input voxel coordinates.
        kernel_size: scalar or (kx, ky, kz) kernel dimensions.
        stride: scalar or (sx, sy, sz) stride.
        device: torch device for computation ("cpu" or "cuda").
        bit_widths: CIG level bit-widths, leaf-first.  Default [3, 4, 5].

    Returns:
        (M, 3) i32 torch tensor of unique output coordinates, sorted by
        hierarchical key (CIG-compatible order).
    """
    if bit_widths is None:
        bit_widths = DEFAULT_BIT_WIDTHS

    dev = torch.device(device)

    if isinstance(active_coords, torch.Tensor):
        active_t = active_coords.to(device=dev, dtype=torch.int32)
    else:
        raise TypeError(f"Expected Tensor, got {type(active_coords)}")

    if active_t.ndim != 2 or active_t.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) coords, got {active_t.shape}")
    if active_t.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=dev)

    ks = _as_vec3(kernel_size)
    st = _as_vec3(stride)
    if min(st) <= 0:
        raise ValueError(f"stride must be positive, got {st}")

    offsets = _kernel_offsets_centered(ks).to(dev)

    # Three-step pipeline: expand -> stride_filter -> dedup
    cand = expand_offsets(active_t, offsets)
    cand = stride_filter(cand, st)
    if cand.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=dev)

    return dedup_coords(cand, bit_widths)
