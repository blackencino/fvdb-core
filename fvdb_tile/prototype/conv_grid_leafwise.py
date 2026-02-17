# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
conv_grid_leafwise: topology expansion via leaf-level bitmask dilation.

Given a CIG (with leaf masks and leaf coords) and kernel size, compute
the set of unique output voxel coordinates **without** the N*K dense
expansion of the original conv_grid.

Instead of expanding every active voxel by every kernel offset (N*K
candidates), this algorithm shifts entire 8x8x8 leaf masks and
accumulates them into output leaf slots via hash-map scatter-reduce
with OR.  The work is O(L * K_voxel * 8) word-level ops, where L is
the number of input leaves -- typically 50-100x fewer than N.

**GPU path** (3 kernel launches, 0 Python loops over data):

  Launch 1: HashMapBuild -- build output leaf index from
      ExpandOffsets(leaf_coords, leaf_level_offsets) + unique.
  Launch 2: conv_grid_dilate_kernel -- fused mask shift + boundary
      decomposition + hash probe + atomicOr, over all L*K pairs.
  Launch 3: MaskToCoords -- popcount + extract voxel coords.

**CPU path** (reference): Python-level loop over kernel offsets,
  calling shift_leaf_masks + hash_map_scatter_reduce per offset.
  Correct but slow -- exists for correctness verification.

**DSL expressiveness gap (TODO)**: The algorithm corresponds to this
DSL program, but is not yet executed through the DSL/AST pipeline::

    pairs     = ExpandOffsets(leaf_indices, offset_indices)
    shifted   = ShiftLeafMask(leaf_masks, offsets, pairs)
    target_k  = HierarchicalKey(target_coords, leaf_bw)
    map       = HashMapBuild(Unique(target_k))
                ScatterReduce(target_k, shifted, Or)
    result    = MaskToCoords(output_masks, output_leaf_coords)

The GPU path is a hand-fused kernel that implements the above.  Future
work: express this in the DSL with idiom-recognition lowering that emits
the fused kernel, so the source of truth is the DSL program and the
fused kernel is a compiler optimization.
"""

from __future__ import annotations

import torch

from .cig import CompressedCIG3
from .conv_grid import _as_vec3, _kernel_offsets_centered
from .ops import (
    HASH_MAP_EMPTY_KEY,
    hash_map_build,
    hash_map_scatter_reduce,
    mask_to_coords,
    shift_leaf_masks,
    hierarchical_key,
    hierarchical_key_decode,
)

# Leaf-level bit widths: [4, 5] covers lower(16^3) + upper(32^3).
# The leaf level (3 bits) is consumed by the leaf mask itself.
_LEAF_BIT_WIDTHS = [4, 5]
_VOXEL_BIT_WIDTHS = [3, 4, 5]


def _leaf_level_offsets(kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """Compute the set of leaf-level offset deltas for a voxel kernel.

    For a voxel kernel of size k, a single leaf can contribute to
    target leaves at delta -1, 0, or +1 per axis (for k <= 8).
    Returns (K_leaf, 3) i32.
    """
    ranges = []
    for k in kernel_size:
        half = (k - 1) // 2
        lo = -(half + 7) // 8
        hi = (half + 7) // 8
        ranges.append(torch.arange(lo, hi + 1, dtype=torch.int32))
    gx, gy, gz = torch.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)


def conv_grid_leafwise(
    cig: CompressedCIG3,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    device: str | None = None,
) -> torch.Tensor:
    """Compute unique output coordinates for sparse convolution via bitmask dilation.

    Uses the CIG's leaf masks directly -- no N*K expansion.

    Args:
        cig: CompressedCIG3 with leaf_masks and leaf_coords.
        kernel_size: scalar or (kx, ky, kz) kernel dimensions.
        stride: scalar or (sx, sy, sz) stride.  Only stride=1 supported initially.
        device: "cpu", "cuda", or None (use CIG's device).

    Returns:
        (M, 3) i32 torch tensor of unique output coordinates, sorted by
        hierarchical key (CIG-compatible order).
    """
    ks = _as_vec3(kernel_size)
    st = _as_vec3(stride)

    if st != (1, 1, 1):
        raise NotImplementedError("Leafwise conv_grid currently supports stride=1 only")

    target_device = torch.device(device) if device is not None else cig.leaf_masks.device
    use_gpu = target_device.type == "cuda"

    if cig.leaf_masks.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=target_device)

    offsets = _kernel_offsets_centered(ks)  # (K_voxel, 3) i32

    if use_gpu:
        return _conv_grid_leafwise_gpu(cig, ks, offsets, target_device)
    else:
        return _conv_grid_leafwise_cpu(cig, ks, offsets)


# ---------------------------------------------------------------------------
# GPU path: 3 launches, 0 Python loops over data
# ---------------------------------------------------------------------------


def _conv_grid_leafwise_gpu(
    cig: CompressedCIG3,
    ks: tuple[int, int, int],
    offsets: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    from .hashmap_cuda import gpu_hash_map_build, gpu_conv_grid_dilate

    leaf_masks = cig.leaf_masks.to(device)
    leaf_coords = cig.leaf_coords.to(device)
    offsets_gpu = offsets.to(device=device, dtype=torch.int32)

    # Launch 1: Build output leaf hash map.
    # Expand leaf coords by leaf-level deltas to get all possible target leaves,
    # then unique + build hash map.  This is vectorized torch + one GPU kernel.
    leaf_deltas = _leaf_level_offsets(ks).to(device)
    expanded_lc = (leaf_coords.unsqueeze(1) + leaf_deltas.unsqueeze(0)).reshape(-1, 3)
    expanded_keys = hierarchical_key(expanded_lc, _LEAF_BIT_WIDTHS)
    unique_keys = torch.unique(expanded_keys)
    key_arr, _ = gpu_hash_map_build(unique_keys)
    storage_size = key_arr.shape[0]

    # Launch 2: Fused dilation kernel.
    output_masks = gpu_conv_grid_dilate(
        leaf_masks, leaf_coords, offsets_gpu, key_arr, storage_size,
    )

    # Launch 3: Extract voxel coordinates from accumulated masks.
    active_leaf_mask = key_arr != HASH_MAP_EMPTY_KEY
    active_masks = output_masks[active_leaf_mask]
    active_keys = key_arr[active_leaf_mask]
    output_leaf_coords = hierarchical_key_decode(active_keys, _LEAF_BIT_WIDTHS)
    result = mask_to_coords(active_masks, output_leaf_coords)

    if result.shape[0] > 0:
        voxel_keys = hierarchical_key(result, _VOXEL_BIT_WIDTHS)
        order = torch.argsort(voxel_keys, stable=True)
        result = result[order]

    return result


# ---------------------------------------------------------------------------
# CPU path: reference implementation (Python loop, correct but slow)
# ---------------------------------------------------------------------------


def _conv_grid_leafwise_cpu(
    cig: CompressedCIG3,
    ks: tuple[int, int, int],
    offsets: torch.Tensor,
) -> torch.Tensor:
    leaf_masks = cig.leaf_masks.cpu()
    leaf_coords = cig.leaf_coords.cpu()

    all_keys = []
    all_masks = []

    for i in range(offsets.shape[0]):
        ox, oy, oz = int(offsets[i, 0]), int(offsets[i, 1]), int(offsets[i, 2])
        pairs = shift_leaf_masks(leaf_masks, ox, oy, oz)
        for shifted_mask, (dx, dy, dz) in pairs:
            delta = torch.tensor([[dx, dy, dz]], dtype=torch.int32)
            target_lc = leaf_coords + delta
            keys = hierarchical_key(target_lc, _LEAF_BIT_WIDTHS)
            all_keys.append(keys)
            all_masks.append(shifted_mask)

    cat_keys = torch.cat(all_keys, dim=0)
    cat_masks = torch.cat(all_masks, dim=0)
    unique_keys = torch.unique(cat_keys)

    key_arr = hash_map_build(unique_keys)
    storage_size = key_arr.shape[0]
    output_masks = torch.zeros(storage_size, 8, dtype=torch.int64)
    for w in range(8):
        output_masks[:, w] = hash_map_scatter_reduce(
            key_arr, cat_keys, cat_masks[:, w], reduce_fn="or"
        )

    active_leaf_mask = key_arr != HASH_MAP_EMPTY_KEY
    active_masks = output_masks[active_leaf_mask]
    active_keys = key_arr[active_leaf_mask]
    output_leaf_coords = hierarchical_key_decode(active_keys, _LEAF_BIT_WIDTHS)
    result = mask_to_coords(active_masks, output_leaf_coords)

    if result.shape[0] > 0:
        voxel_keys = hierarchical_key(result, _VOXEL_BIT_WIDTHS)
        order = torch.argsort(voxel_keys, stable=True)
        result = result[order]

    return result
