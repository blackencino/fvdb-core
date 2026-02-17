# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
conv_grid: topology expansion for sparse convolution.

Given active input voxel coordinates, kernel size, and stride, compute the
set of unique output coordinates.  The semantic equation:

    output y is active iff there exists active x and kernel offset k
    such that x = y * stride + k  (component-wise).

**Adverb decomposition (reference, stride=1):**

The algorithm is expressed as a composition of leading-shape adverbs::

    expanded  = EachLeft(EachRight(EachBoth(Add)), coords, offsets)
    flat      = fuse(expanded)
    reshaped  = reshape(flat, [-1])
    codes     = HierarchicalKey(reshaped, [3, 4, 5])
    sorted    = Sort(codes)
    unique    = Unique(sorted)
    result    = HierarchicalKeyDecode(unique, [3, 4, 5])

**Leaf-level algorithm (production path):**

Phase 1 -- Coarse (leaf expansion):
  1. Extract source leaf coords: unique(active_coords >> leaf_bits)
  2. Compute leaf-level kernel offsets (typically {-1, 0, 1}^3 = 27)
  3. Expand via broadcasting + dedup -> candidate destination leaves
  This is tiny: L_src ~ N/256, so L_src * 27 << N * K.

Phase 2 -- Fine fill (dense lookup on device):
  Build a 3D bool tensor on the target device from active coords.
  For each candidate dest leaf, generate all 512 voxel positions.
  For each kernel offset k (loop of ~27 iterations):
    Compute source positions = dest_voxels - k
    Batch-check via dense tensor indexing (single GPU gather)
    OR into active accumulator
  Collect active dest voxels.

The entire computation runs on torch tensors, using the ``device``
parameter.

Voxel ordering uses ``hierarchical_key`` with configurable ``bit_widths``
for CIG-compatible tree-traversal order.
"""

from __future__ import annotations

import math

import torch

from .ops import hierarchical_key, hierarchical_key_decode


# ---------------------------------------------------------------------------
# Default CIG bit-widths (leaf=8^3, lower=16^3, upper=32^3)
# ---------------------------------------------------------------------------

DEFAULT_BIT_WIDTHS: list[int] = [3, 4, 5]


# ---------------------------------------------------------------------------
# Pre-computed 8x8x8 local grid (shared across calls, moved to device lazily)
# ---------------------------------------------------------------------------

_LOCAL_GRID_CACHE: dict[torch.device, torch.Tensor] = {}


def _local_grid(device: torch.device) -> torch.Tensor:
    """(512, 3) i32 tensor of all positions in an 8^3 block, on *device*."""
    if device not in _LOCAL_GRID_CACHE:
        xs = torch.arange(8, dtype=torch.int32)
        grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing="ij"), dim=-1)
        _LOCAL_GRID_CACHE[device] = grid.reshape(-1, 3).to(device)
    return _LOCAL_GRID_CACHE[device]


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


def _leaf_kernel_offsets(kernel_half: tuple[int, int, int], leaf_size: int = 8) -> torch.Tensor:
    """Compute leaf-level kernel offsets for coarse expansion."""
    ranges = []
    for kh in kernel_half:
        lo = -math.ceil(kh / leaf_size)
        hi = (leaf_size - 1 + kh) // leaf_size
        ranges.append(torch.arange(lo, hi + 1, dtype=torch.int32))
    gx, gy, gz = torch.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)


def _dedup_coords(coords: torch.Tensor, bit_widths: list[int]) -> torch.Tensor:
    """Deduplicate (N, 3) i32 coords via hierarchical key sort + unique.

    Returns deduplicated coords sorted by hierarchical key (CIG-compatible).
    """
    if coords.shape[0] == 0:
        return coords
    keys = hierarchical_key(coords.to(torch.int32), bit_widths)
    keys_t = keys.to(torch.int64).to(coords.device)
    sorted_keys, sort_idx = torch.sort(keys_t, stable=True)
    sorted_coords = coords[sort_idx]

    keep = torch.ones(sorted_keys.shape[0], dtype=torch.bool, device=coords.device)
    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return sorted_coords[keep]


# ---------------------------------------------------------------------------
# Dense source lookup
# ---------------------------------------------------------------------------


def _build_dense_source(
    active: torch.Tensor,
    kernel_half: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense 3D bool tensor marking active source voxels.

    Returns (dense_tensor, offset) where offset is the min coordinate
    (subtracted to make all indices non-negative).  The tensor is padded
    by kernel_half on each side so that out-of-range lookups return False.
    """
    kh_t = torch.tensor(kernel_half, dtype=torch.int32, device=active.device)
    coord_min = active.min(dim=0).values - kh_t
    coord_max = active.max(dim=0).values + kh_t
    shape = (coord_max - coord_min + 1).tolist()

    dense = torch.zeros(shape, dtype=torch.bool, device=active.device)
    shifted = active - coord_min.unsqueeze(0)
    dense[shifted[:, 0].long(), shifted[:, 1].long(), shifted[:, 2].long()] = True

    return dense, coord_min


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

    Uses a two-phase leaf-level algorithm with all computation on torch
    tensors.

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
    kernel_half = tuple(k // 2 if k % 2 == 1 else k - 1 for k in ks)
    leaf_bits = bit_widths[0]

    if st == (1, 1, 1):
        result_t = _conv_grid_leaf_stride1(active_t, offsets, kernel_half, leaf_bits, bit_widths)
    else:
        result_t = _conv_grid_leaf_strided(active_t, offsets, kernel_half, st, leaf_bits, bit_widths)

    return result_t


# ---------------------------------------------------------------------------
# Stride=1 leaf-level implementation (all torch)
# ---------------------------------------------------------------------------


def _conv_grid_leaf_stride1(
    active: torch.Tensor,
    offsets: torch.Tensor,
    kernel_half: tuple[int, int, int],
    leaf_bits: int,
    bit_widths: list[int],
) -> torch.Tensor:
    """Leaf-level conv_grid for stride=1, all on torch tensors."""
    dev = active.device
    leaf_size = 1 << leaf_bits

    source_leaf_coords = torch.unique(active >> leaf_bits, dim=0)
    leaf_offsets = _leaf_kernel_offsets(kernel_half, leaf_size).to(dev)

    expanded_leaves = source_leaf_coords.unsqueeze(1) + leaf_offsets.unsqueeze(0)
    expanded_leaves = expanded_leaves.reshape(-1, 3)
    dest_leaves = _dedup_coords(expanded_leaves, bit_widths)

    dense_src, offset = _build_dense_source(active, kernel_half)
    local = _local_grid(dev)

    all_dest = (dest_leaves.unsqueeze(1) * leaf_size + local.unsqueeze(0)).reshape(-1, 3)

    any_active = torch.zeros(all_dest.shape[0], dtype=torch.bool, device=dev)

    for k_idx in range(offsets.shape[0]):
        if any_active.all():
            break
        src = all_dest - offsets[k_idx].unsqueeze(0) - offset.unsqueeze(0)
        in_bounds = (
            (src[:, 0] >= 0) & (src[:, 0] < dense_src.shape[0])
            & (src[:, 1] >= 0) & (src[:, 1] < dense_src.shape[1])
            & (src[:, 2] >= 0) & (src[:, 2] < dense_src.shape[2])
        )
        valid_src = src.clamp(min=0)
        valid_src[:, 0].clamp_(max=dense_src.shape[0] - 1)
        valid_src[:, 1].clamp_(max=dense_src.shape[1] - 1)
        valid_src[:, 2].clamp_(max=dense_src.shape[2] - 1)
        hits = dense_src[valid_src[:, 0].long(), valid_src[:, 1].long(), valid_src[:, 2].long()]
        any_active |= hits & in_bounds

    result = all_dest[any_active]
    if result.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=dev)

    return _dedup_coords(result, bit_widths)


# ---------------------------------------------------------------------------
# Strided leaf-level implementation
# ---------------------------------------------------------------------------


def _conv_grid_leaf_strided(
    active: torch.Tensor,
    offsets: torch.Tensor,
    kernel_half: tuple[int, int, int],
    stride: tuple[int, int, int],
    leaf_bits: int,
    bit_widths: list[int],
) -> torch.Tensor:
    """Leaf-level conv_grid for stride > 1, all on torch tensors."""
    dev = active.device
    stride_t = torch.tensor(stride, dtype=torch.int32, device=dev)

    cand = active.unsqueeze(1) + offsets.unsqueeze(0)
    cand = cand.reshape(-1, 3)

    divisible = (cand % stride_t.unsqueeze(0) == 0).all(dim=1)
    cand = cand[divisible]
    if cand.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=dev)
    cand = cand // stride_t.unsqueeze(0)

    return _dedup_coords(cand, bit_widths)
