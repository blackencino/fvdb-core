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
      leaf_prefix:  (K, 8)       i32  -- cumulative popcount before each word
      leaf_offsets: (K,)         i64  -- base offset per leaf into voxel_data
      voxel_data:   (N_active,)  i32  -- flat contiguous voxel indices

    Per-leaf cost: 100 bytes (64 mask + 32 prefix + 4 offset).
    """

    lower: torch.Tensor
    leaf_masks: torch.Tensor
    leaf_prefix: torch.Tensor
    leaf_offsets: torch.Tensor
    voxel_data: torch.Tensor
    n_active: int
    n_leaves: int

    @property
    def num_bytes(self) -> int:
        total = 0
        for t in [self.lower, self.leaf_masks, self.leaf_prefix, self.leaf_offsets, self.voxel_data]:
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
            leaf_prefix=self.leaf_prefix.cuda(),
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

    # Build prefix-sum popcounts per leaf
    leaf_prefix = _build_prefix_sums(leaf_masks)

    # Voxel data: sequential indices [0, N_unique)
    voxel_data = torch.arange(N_unique, dtype=torch.int32, device=device)

    return CompressedCIG(
        lower=lower,
        leaf_masks=leaf_masks,
        leaf_prefix=leaf_prefix,
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


def _build_prefix_sums(masks: torch.Tensor) -> torch.Tensor:
    """Build prefix-sum popcount array from mask words.

    Input:  (K, W) i64  -- W packed u64 words per node
    Output: (K, W) i32  -- prefix[k, i] = sum(popcount(masks[k, 0..i-1]))

    prefix[k, 0] = 0 always. prefix[k, i] = cumulative popcount of
    words before word i for node k.
    """
    per_word_popc = _popcount_i64(masks)  # (K, W) i32
    cum = per_word_popc.cumsum(dim=1)  # (K, W) i32
    # Shift right: prefix[0] = 0, prefix[i] = cum[i-1]
    prefix = torch.zeros_like(cum)
    if cum.shape[1] > 1:
        prefix[:, 1:] = cum[:, :-1]
    return prefix


def _build_masked_level(node_of_child, child_local_flat, n_nodes, n_words, device):
    """Build bitmasks, prefix sums, and offsets for one masked level.

    Generic across node sizes (8 words for 8^3, 64 for 16^3, 512 for 32^3).

    Args:
        node_of_child: (M,) i64 -- which node each child belongs to
        child_local_flat: (M,) i64 -- flat local position within the node
        n_nodes: int -- number of nodes at this level
        n_words: int -- number of u64 words per node mask
        device: torch device

    Returns:
        masks: (N, W) i64, prefix: (N, W) i32, offsets: (N,) i64
    """
    M = node_of_child.shape[0]

    # Sort by (node, flat_local)
    total_positions = n_words * 64
    sort_key = node_of_child * total_positions + child_local_flat
    sort_order = torch.argsort(sort_key)
    sorted_node = node_of_child[sort_order]
    sorted_flat = child_local_flat[sort_order]

    # Build bitmasks
    masks = torch.zeros(n_nodes, n_words, dtype=torch.int64, device=device)
    for i in range(M):
        ni = int(sorted_node[i])
        fl = int(sorted_flat[i])
        word_idx = fl >> 6
        bit_pos = fl & 63
        masks[ni, word_idx] |= 1 << bit_pos

    # Prefix sums
    prefix = _build_prefix_sums(masks)

    # Offsets: cumulative child counts
    counts = torch.zeros(n_nodes, dtype=torch.int64, device=device)
    counts.scatter_add_(0, sorted_node.long(), torch.ones(M, dtype=torch.int64, device=device))
    offsets = torch.zeros(n_nodes, dtype=torch.int64, device=device)
    if n_nodes > 1:
        offsets[1:] = counts[:-1].cumsum(0)

    return masks, prefix, offsets


# ---------------------------------------------------------------------------
# 3-level Compressed CIG: upper (32^3) + lower (16^3) + leaf (8^3)
# ---------------------------------------------------------------------------


@dataclass
class CompressedCIG3:
    """A 3-level CIG with bitmask-compressed nodes at every level.

    Bit-widths [3, 4, 5]:  leaf=8^3, lower=16^3, upper=32^3.
    Total coordinate range: 2^(3+4+5) = 4096 per axis.

    Root level: variable number of upper nodes, identified by their
    which_top coordinates. Root lookup is external (torch linear scan).
    """

    # Root
    root_coords: torch.Tensor  # (R, 3) i32 -- which_top coords of each upper node

    # Upper nodes (32^3 = 32768 positions per node, 512 u64 words)
    upper_masks: torch.Tensor  # (U, 512) i64
    upper_prefix: torch.Tensor  # (U, 512) i32
    upper_offsets: torch.Tensor  # (U,) i64

    # Lower nodes (16^3 = 4096 positions per node, 64 u64 words)
    lower_masks: torch.Tensor  # (L, 64) i64
    lower_prefix: torch.Tensor  # (L, 64) i32
    lower_offsets: torch.Tensor  # (L,) i64

    # Leaf nodes (8^3 = 512 positions per node, 8 u64 words)
    leaf_masks: torch.Tensor  # (K, 8) i64
    leaf_prefix: torch.Tensor  # (K, 8) i32
    leaf_offsets: torch.Tensor  # (K,) i64

    n_active: int
    n_leaves: int
    n_lower: int
    n_upper: int

    @property
    def num_bytes(self) -> int:
        total = 0
        for t in [
            self.root_coords,
            self.upper_masks, self.upper_prefix, self.upper_offsets,
            self.lower_masks, self.lower_prefix, self.lower_offsets,
            self.leaf_masks, self.leaf_prefix, self.leaf_offsets,
        ]:
            total += t.nelement() * t.element_size()
        return total

    @property
    def device(self) -> torch.device:
        return self.root_coords.device

    def cuda(self) -> "CompressedCIG3":
        if self.root_coords.is_cuda:
            return self
        return CompressedCIG3(
            root_coords=self.root_coords.cuda(),
            upper_masks=self.upper_masks.cuda(),
            upper_prefix=self.upper_prefix.cuda(),
            upper_offsets=self.upper_offsets.cuda(),
            lower_masks=self.lower_masks.cuda(),
            lower_prefix=self.lower_prefix.cuda(),
            lower_offsets=self.lower_offsets.cuda(),
            leaf_masks=self.leaf_masks.cuda(),
            leaf_prefix=self.leaf_prefix.cuda(),
            leaf_offsets=self.leaf_offsets.cuda(),
            n_active=self.n_active,
            n_leaves=self.n_leaves,
            n_lower=self.n_lower,
            n_upper=self.n_upper,
        )


def build_compressed_cig3(ijk: torch.Tensor) -> CompressedCIG3:
    """Build a 3-level bitmask-compressed CIG from voxel coordinates.

    Bit-widths [3, 4, 5]:
      level_0 (leaf-local):  coord & 7              (3 bits)
      level_1 (lower-local): (coord >> 3) & 15      (4 bits)
      level_2 (upper-local): (coord >> 7) & 31      (5 bits)
      which_top:             coord >> 12             (root key)

    Children at every level are contiguous: prefix sums slice perfectly.
    """
    assert ijk.ndim == 2 and ijk.shape[1] == 3, f"Expected (N, 3), got {ijk.shape}"
    device = ijk.device

    # Decompose coordinates at bit-widths [3, 4, 5]
    leaf_local = ijk & 7  # (N, 3) -- 3 bits
    lower_local = (ijk >> 3) & 15  # (N, 3) -- 4 bits
    upper_local = (ijk >> 7) & 31  # (N, 3) -- 5 bits
    which_top = ijk >> 12  # (N, 3) -- remaining bits

    # Deduplicate voxels
    all_parts = torch.cat([which_top, upper_local, lower_local, leaf_local], dim=1)  # (N, 12)
    unique_parts, _ = torch.unique(all_parts, dim=0, return_inverse=True)
    wt = unique_parts[:, :3]
    ul = unique_parts[:, 3:6]
    ll = unique_parts[:, 6:9]
    fl = unique_parts[:, 9:]
    N_voxels = unique_parts.shape[0]

    # --- Upper nodes: unique which_top values ---
    unique_wt, wt_inverse = torch.unique(wt, dim=0, return_inverse=True)
    U = unique_wt.shape[0]
    root_coords = unique_wt.int()

    # --- Lower nodes: unique (which_top, upper_local) pairs ---
    wt_ul = torch.cat([wt, ul], dim=1)  # (N_voxels, 6)
    unique_wt_ul, wt_ul_inverse = torch.unique(wt_ul, dim=0, return_inverse=True)
    L = unique_wt_ul.shape[0]

    # Which upper node does each lower node belong to?
    lower_wt = unique_wt_ul[:, :3]
    lower_ul = unique_wt_ul[:, 3:]
    _, lower_to_upper = torch.unique(lower_wt, dim=0, return_inverse=True)

    # Flat local position of each lower node within its upper node (32^3)
    lower_flat_in_upper = lower_ul[:, 0] * 1024 + lower_ul[:, 1] * 32 + lower_ul[:, 2]

    upper_masks, upper_prefix, upper_offsets = _build_masked_level(
        lower_to_upper, lower_flat_in_upper, U, 512, device
    )

    # --- Leaf nodes: unique (which_top, upper_local, lower_local) triples ---
    wt_ul_ll = torch.cat([wt, ul, ll], dim=1)  # (N_voxels, 9)
    unique_wt_ul_ll, wt_ul_ll_inverse = torch.unique(wt_ul_ll, dim=0, return_inverse=True)
    K = unique_wt_ul_ll.shape[0]

    # Which lower node does each leaf belong to?
    leaf_wt_ul = unique_wt_ul_ll[:, :6]
    leaf_ll = unique_wt_ul_ll[:, 6:]
    _, leaf_to_lower = torch.unique(leaf_wt_ul, dim=0, return_inverse=True)

    # Flat local position of each leaf within its lower node (16^3)
    leaf_flat_in_lower = leaf_ll[:, 0] * 256 + leaf_ll[:, 1] * 16 + leaf_ll[:, 2]

    lower_masks, lower_prefix, lower_offsets = _build_masked_level(
        leaf_to_lower, leaf_flat_in_lower, L, 64, device
    )

    # --- Voxels: which leaf does each voxel belong to? ---
    voxel_wt_ul_ll = torch.cat([wt, ul, ll], dim=1)  # (N_voxels, 9)
    _, voxel_to_leaf = torch.unique(voxel_wt_ul_ll, dim=0, return_inverse=True)

    # Flat local position of each voxel within its leaf (8^3)
    voxel_flat_in_leaf = fl[:, 0] * 64 + fl[:, 1] * 8 + fl[:, 2]

    leaf_masks, leaf_prefix, leaf_offsets = _build_masked_level(
        voxel_to_leaf, voxel_flat_in_leaf, K, 8, device
    )

    return CompressedCIG3(
        root_coords=root_coords,
        upper_masks=upper_masks,
        upper_prefix=upper_prefix,
        upper_offsets=upper_offsets,
        lower_masks=lower_masks,
        lower_prefix=lower_prefix,
        lower_offsets=lower_offsets,
        leaf_masks=leaf_masks,
        leaf_prefix=leaf_prefix,
        leaf_offsets=leaf_offsets,
        n_active=int(N_voxels),
        n_leaves=int(K),
        n_lower=int(L),
        n_upper=int(U),
    )


def root_lookup(root_coords: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Root-level linear scan: find which upper node each query belongs to.

    Args:
        root_coords: (R, 3) i32 -- which_top coordinates of each upper node
        query: (N, 3) i32 -- full query coordinates

    Returns:
        (N,) i32 tensor of upper node indices, -1 for no match.
    """
    which_top = query >> 12  # (N, 3) -- extract root-level bits
    R = root_coords.shape[0]
    # Vectorized: (N, 1, 3) == (1, R, 3) -> (N, R) all-match -> argmax
    match = (which_top.unsqueeze(1) == root_coords.unsqueeze(0)).all(dim=2)  # (N, R) bool
    any_match = match.any(dim=1)  # (N,) bool
    upper_idx = match.int().argmax(dim=1).int()  # (N,) i32
    upper_idx[~any_match] = -1
    return upper_idx


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
