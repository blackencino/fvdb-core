# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Compact Index Grid (CIG) -- concrete tensor format and operations.

A CIG is a 2-level sparse index grid stored as two raw tensors:

  lower:    (16, 16, 16)   i32  -- maps lower-level 3D coords to leaf index (-1 = empty)
  leaf_arr: (K, 8, 8, 8)   i32  -- K leaf blocks, each maps local 3D coords to
                                    a linear voxel index (-1 = inactive)

Bit-width decomposition [3, 4]:
  level_0 (leaf-local):  coord & 7          (bottom 3 bits)
  level_1 (lower-node):  (coord >> 3) & 15  (next 4 bits)

This file provides:
  build_cig()           -- construct a CIG from (N, 3) voxel coordinates
  cig_ijk_to_index()    -- vectorized PyTorch query: (N, 3) -> (N,) indices
  cig_num_bytes()       -- total memory footprint in bytes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CIG:
    """A 2-level Compact Index Grid."""

    lower: torch.Tensor  # (16, 16, 16) i32
    leaf_arr: torch.Tensor  # (K, 8, 8, 8) i32
    n_active: int  # total number of active voxels
    n_leaves: int  # number of allocated leaf blocks (K)

    @property
    def num_bytes(self) -> int:
        """Total memory footprint of the CIG tensors in bytes."""
        return self.lower.nelement() * self.lower.element_size() + self.leaf_arr.nelement() * self.leaf_arr.element_size()

    @property
    def device(self) -> torch.device:
        return self.lower.device

    def cuda(self) -> "CIG":
        if self.lower.is_cuda:
            return self
        return CIG(
            lower=self.lower.cuda(),
            leaf_arr=self.leaf_arr.cuda(),
            n_active=self.n_active,
            n_leaves=self.n_leaves,
        )


def build_cig(ijk: torch.Tensor) -> CIG:
    """Build a 2-level CIG from voxel coordinates.

    Args:
        ijk: (N, 3) i32 tensor of voxel coordinates. Duplicates are
             deduplicated (only one voxel per coordinate).

    Returns:
        A CIG with lower and leaf_arr tensors, plus metadata.
    """
    assert ijk.ndim == 2 and ijk.shape[1] == 3, f"Expected (N, 3), got {ijk.shape}"
    device = ijk.device

    # Decompose coordinates into lower-level and leaf-local parts
    lower_coords = (ijk >> 3) & 15  # (N, 3)
    leaf_local = ijk & 7  # (N, 3)

    # Deduplicate: unique (lower, local) pairs
    full_coords = torch.cat([lower_coords, leaf_local], dim=1)  # (N, 6)
    unique_full, _ = torch.unique(full_coords, dim=0, return_inverse=True)
    lower_coords_u = unique_full[:, :3]
    leaf_local_u = unique_full[:, 3:]
    N_unique = unique_full.shape[0]

    # Find unique lower nodes and assign leaf indices
    unique_lower, lower_inverse = torch.unique(lower_coords_u, dim=0, return_inverse=True)
    K = unique_lower.shape[0]

    # Build the lower array: (16, 16, 16) i32, -1 = empty
    lower = torch.full((16, 16, 16), -1, dtype=torch.int32, device=device)
    leaf_indices = torch.arange(K, dtype=torch.int32, device=device)
    lower[unique_lower[:, 0], unique_lower[:, 1], unique_lower[:, 2]] = leaf_indices

    # Build leaf array: (K, 8, 8, 8) i32, -1 = inactive
    leaf_arr = torch.full((K, 8, 8, 8), -1, dtype=torch.int32, device=device)

    # Assign sequential voxel indices within each leaf.
    # Sort by leaf index for stable within-leaf ordering.
    leaf_of_voxel = lower_inverse  # (N_unique,) -- which leaf each voxel belongs to
    sort_key = leaf_of_voxel * 512 + leaf_local_u[:, 0] * 64 + leaf_local_u[:, 1] * 8 + leaf_local_u[:, 2]
    sort_order = torch.argsort(sort_key)

    sorted_leaf = leaf_of_voxel[sort_order]
    sorted_local = leaf_local_u[sort_order]

    # Compute per-leaf offsets for sequential index assignment
    # Count voxels per leaf
    counts = torch.zeros(K, dtype=torch.int64, device=device)
    counts.scatter_add_(0, sorted_leaf.long(), torch.ones(N_unique, dtype=torch.int64, device=device))
    offsets = torch.zeros(K, dtype=torch.int64, device=device)
    offsets[1:] = counts[:-1].cumsum(0)

    # Assign linear voxel indices: offset[leaf] + position_within_leaf
    position_within_leaf = torch.arange(N_unique, dtype=torch.int64, device=device) - offsets[sorted_leaf.long()]
    voxel_indices = offsets[sorted_leaf.long()] + position_within_leaf
    leaf_arr[sorted_leaf.long(), sorted_local[:, 0].long(), sorted_local[:, 1].long(), sorted_local[:, 2].long()] = voxel_indices.int()

    return CIG(
        lower=lower,
        leaf_arr=leaf_arr,
        n_active=int(N_unique),
        n_leaves=int(K),
    )


def cig_ijk_to_index(cig: CIG, query: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch ijk_to_index: map (N, 3) coordinates to linear indices.

    Returns (N,) i32 tensor. -1 for coordinates not in the grid.
    """
    assert query.ndim == 2 and query.shape[1] == 3
    device = query.device

    # Decompose
    l1 = (query >> 3) & 15  # (N, 3) lower-level coords
    l0 = query & 7  # (N, 3) leaf-local coords

    lower = cig.lower.to(device)
    leaf_arr = cig.leaf_arr.to(device)
    K = leaf_arr.shape[0]

    # Bounds check on lower coords
    l1_valid = (l1 >= 0).all(dim=1) & (l1 < 16).all(dim=1)  # (N,)
    l1_clamped = l1.clamp(0, 15)

    # Gather leaf index from lower
    leaf_idx = lower[l1_clamped[:, 0], l1_clamped[:, 1], l1_clamped[:, 2]]  # (N,)
    valid_leaf = l1_valid & (leaf_idx >= 0) & (leaf_idx < K)
    safe_leaf_idx = leaf_idx.clamp(min=0, max=max(K - 1, 0)).long()

    # Gather voxel index from leaf_arr
    l0_clamped = l0.clamp(0, 7)
    voxel_idx = leaf_arr[safe_leaf_idx, l0_clamped[:, 0].long(), l0_clamped[:, 1].long(), l0_clamped[:, 2].long()]

    # Mask invalid lookups
    result = torch.where(valid_leaf & (voxel_idx >= 0), voxel_idx, torch.tensor(-1, dtype=torch.int32, device=device))
    return result


# ---------------------------------------------------------------------------
# Compressed CIG: bitmask + popcount (the masked layout in tensor form)
# ---------------------------------------------------------------------------


@dataclass
class CompressedCIG:
    """A 2-level CIG with bitmask-compressed leaves.

    Physical storage:
      lower:        (16, 16, 16) i32  -- maps lower-level 3D coords to leaf index
      leaf_masks:   (K, 8)       i64  -- 512-bit bitmask per leaf (8 x 64-bit words)
      leaf_offsets: (K,)         i64  -- base offset per leaf into voxel_data
      voxel_data:   (N_active,)  i32  -- flat contiguous voxel indices

    Per-leaf cost: 72 bytes (vs 2,048 bytes dense, vs NanoVDB's 96 bytes).
    """

    lower: torch.Tensor
    leaf_masks: torch.Tensor
    leaf_offsets: torch.Tensor
    voxel_data: torch.Tensor
    n_active: int
    n_leaves: int

    @property
    def num_bytes(self) -> int:
        total = 0
        for t in [self.lower, self.leaf_masks, self.leaf_offsets, self.voxel_data]:
            total += t.nelement() * t.element_size()
        return total

    @property
    def device(self) -> torch.device:
        return self.lower.device

    def cuda(self) -> "CompressedCIG":
        if self.lower.is_cuda:
            return self
        return CompressedCIG(
            lower=self.lower.cuda(),
            leaf_masks=self.leaf_masks.cuda(),
            leaf_offsets=self.leaf_offsets.cuda(),
            voxel_data=self.voxel_data.cuda(),
            n_active=self.n_active,
            n_leaves=self.n_leaves,
        )


def build_compressed_cig(ijk: torch.Tensor) -> CompressedCIG:
    """Build a bitmask-compressed 2-level CIG from voxel coordinates.

    Same input as build_cig, but produces compressed leaf storage:
    bitmask + offset per leaf instead of dense (8,8,8) i32 blocks.
    """
    assert ijk.ndim == 2 and ijk.shape[1] == 3, f"Expected (N, 3), got {ijk.shape}"
    device = ijk.device

    lower_coords = (ijk >> 3) & 15
    leaf_local = ijk & 7

    # Deduplicate
    full_coords = torch.cat([lower_coords, leaf_local], dim=1)
    unique_full, _ = torch.unique(full_coords, dim=0, return_inverse=True)
    lower_coords_u = unique_full[:, :3]
    leaf_local_u = unique_full[:, 3:]
    N_unique = unique_full.shape[0]

    # Unique lower nodes -> leaf indices
    unique_lower, lower_inverse = torch.unique(lower_coords_u, dim=0, return_inverse=True)
    K = unique_lower.shape[0]

    # Build lower array
    lower = torch.full((16, 16, 16), -1, dtype=torch.int32, device=device)
    lower[unique_lower[:, 0], unique_lower[:, 1], unique_lower[:, 2]] = torch.arange(K, dtype=torch.int32, device=device)

    # Sort by (leaf_idx, flat_local) for deterministic ordering
    leaf_of_voxel = lower_inverse
    flat_local = leaf_local_u[:, 0] * 64 + leaf_local_u[:, 1] * 8 + leaf_local_u[:, 2]
    sort_key = leaf_of_voxel * 512 + flat_local
    sort_order = torch.argsort(sort_key)
    sorted_leaf = leaf_of_voxel[sort_order]
    sorted_flat_local = flat_local[sort_order]

    # Build bitmasks: for each leaf, set bits at active local positions
    leaf_masks = torch.zeros(K, 8, dtype=torch.int64, device=device)
    for i in range(N_unique):
        li = int(sorted_leaf[i])
        fl = int(sorted_flat_local[i])
        word_idx = fl >> 6
        bit_pos = fl & 63
        leaf_masks[li, word_idx] |= 1 << bit_pos

    # Compute per-leaf voxel counts and offsets
    counts = torch.zeros(K, dtype=torch.int64, device=device)
    counts.scatter_add_(0, sorted_leaf.long(), torch.ones(N_unique, dtype=torch.int64, device=device))
    leaf_offsets = torch.zeros(K, dtype=torch.int64, device=device)
    if K > 1:
        leaf_offsets[1:] = counts[:-1].cumsum(0)

    # Voxel data: sequential indices [0, N_unique)
    voxel_data = torch.arange(N_unique, dtype=torch.int32, device=device)

    return CompressedCIG(
        lower=lower,
        leaf_masks=leaf_masks,
        leaf_offsets=leaf_offsets,
        voxel_data=voxel_data,
        n_active=int(N_unique),
        n_leaves=int(K),
    )


def _popcount_i64(x: torch.Tensor) -> torch.Tensor:
    """Popcount for i64 tensors using the parallel bit-count algorithm."""
    u = x.long()  # ensure i64
    # Hamming weight via the standard parallel bit-count
    m1 = 0x5555555555555555
    m2 = 0x3333333333333333
    m4 = 0x0F0F0F0F0F0F0F0F
    h01 = 0x0101010101010101
    u = u - ((u >> 1) & m1)
    u = (u & m2) + ((u >> 2) & m2)
    u = (u + (u >> 4)) & m4
    return ((u * h01) >> 56).int()


def compressed_cig_ijk_to_index(cig: CompressedCIG, query: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch ijk_to_index for compressed CIG.

    Returns (N,) i32 tensor. -1 for coordinates not in the grid.
    """
    assert query.ndim == 2 and query.shape[1] == 3
    device = query.device
    N = query.shape[0]
    K = cig.n_leaves

    l1 = (query >> 3) & 15
    l0 = query & 7

    lower = cig.lower.to(device)
    leaf_masks = cig.leaf_masks.to(device)
    leaf_offsets = cig.leaf_offsets.to(device)

    # Lower lookup
    l1_clamped = l1.clamp(0, 15)
    leaf_idx = lower[l1_clamped[:, 0], l1_clamped[:, 1], l1_clamped[:, 2]]
    valid_leaf = (l1 >= 0).all(dim=1) & (l1 < 16).all(dim=1) & (leaf_idx >= 0) & (leaf_idx < K)
    safe_leaf = leaf_idx.clamp(0, max(K - 1, 0)).long()

    # Compute flat bit index within leaf
    flat_idx = l0[:, 0].long() * 64 + l0[:, 1].long() * 8 + l0[:, 2].long()
    word_idx = flat_idx >> 6  # (N,) which of the 8 u64 words
    bit_pos = flat_idx & 63  # (N,) position within word

    # Gather the relevant mask word
    mask_word = leaf_masks[safe_leaf, word_idx.clamp(0, 7)]  # (N,) i64

    # Check if bit is set
    is_active = ((mask_word >> bit_pos) & 1).bool()

    # Popcount: count set bits before flat_idx in the leaf's mask
    # For each query: sum popcount of all words before word_idx, plus partial word
    all_words = leaf_masks[safe_leaf]  # (N, 8) i64

    # Cumulative popcount per word
    word_popcounts = torch.zeros(N, 8, dtype=torch.int64, device=device)
    for w in range(8):
        word_popcounts[:, w] = _popcount_i64(all_words[:, w])

    # Sum of full words before word_idx
    cumsum = word_popcounts.cumsum(dim=1)  # (N, 8)
    # full_words_count = cumsum[word_idx - 1] if word_idx > 0 else 0
    # Use gather to select
    shifted_cumsum = torch.zeros(N, 9, dtype=torch.int64, device=device)
    shifted_cumsum[:, 1:] = cumsum
    full_count = shifted_cumsum.gather(1, word_idx.unsqueeze(1)).squeeze(1)

    # Partial word: count bits below bit_pos
    partial_mask = mask_word & ((torch.ones(N, dtype=torch.int64, device=device) << bit_pos) - 1)
    partial_count = _popcount_i64(partial_mask)

    total_popcount = full_count + partial_count

    # Compute result
    base = leaf_offsets[safe_leaf]
    voxel_idx = (base + total_popcount).int()

    result = torch.where(valid_leaf & is_active, voxel_idx, torch.tensor(-1, dtype=torch.int32, device=device))
    return result


def compressed_cig_ijk_to_index_numpy(cig: CompressedCIG, query_np: np.ndarray) -> np.ndarray:
    """Numpy reference ijk_to_index for compressed CIG."""
    lower_np = cig.lower.cpu().numpy()
    masks_np = cig.leaf_masks.cpu().numpy()
    offsets_np = cig.leaf_offsets.cpu().numpy()
    K = cig.n_leaves
    N = query_np.shape[0]
    result = np.full(N, -1, dtype=np.int32)

    for i in range(N):
        coord = query_np[i]
        l1 = (coord >> 3) & 15
        l0 = coord & 7
        if np.any(l1 < 0) or np.any(l1 >= 16):
            continue
        leaf_idx = lower_np[l1[0], l1[1], l1[2]]
        if leaf_idx < 0 or leaf_idx >= K:
            continue

        flat_idx = int(l0[0]) * 64 + int(l0[1]) * 8 + int(l0[2])
        word_idx = flat_idx >> 6
        bit_pos = flat_idx & 63
        word = int(masks_np[leaf_idx, word_idx])
        if not ((word >> bit_pos) & 1):
            continue

        # Popcount
        total = 0
        for w in range(word_idx):
            total += bin(int(masks_np[leaf_idx, w]) & 0xFFFFFFFFFFFFFFFF).count("1")
        partial = word & ((1 << bit_pos) - 1)
        total += bin(partial & 0xFFFFFFFFFFFFFFFF).count("1")

        result[i] = int(offsets_np[leaf_idx]) + total

    return result


def cig_ijk_to_index_numpy(cig: CIG, query_np: np.ndarray) -> np.ndarray:
    """Numpy reference ijk_to_index: loop-based, for correctness verification."""
    lower_np = cig.lower.cpu().numpy()
    leaf_arr_np = cig.leaf_arr.cpu().numpy()
    K = leaf_arr_np.shape[0]
    N = query_np.shape[0]
    result = np.full(N, -1, dtype=np.int32)

    for i in range(N):
        coord = query_np[i]
        l1 = (coord >> 3) & 15
        l0 = coord & 7

        if np.any(l1 < 0) or np.any(l1 >= 16):
            continue
        leaf_idx = lower_np[l1[0], l1[1], l1[2]]
        if leaf_idx < 0 or leaf_idx >= K:
            continue
        voxel_idx = leaf_arr_np[leaf_idx, l0[0], l0[1], l0[2]]
        result[i] = voxel_idx

    return result
