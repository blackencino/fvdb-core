# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Compact Index Grid (CIG) -- 3-level bitmask-compressed sparse index grid.

A CompressedCIG3 maps (i, j, k) voxel coordinates to linear indices via
three masked levels: upper (32^3), lower (16^3), leaf (8^3), with a root
lookup over variable upper nodes.

Bit-width decomposition [3, 4, 5]:
  level_0 (leaf-local):  coord & 7              (3 bits)
  level_1 (lower-local): (coord >> 3) & 15      (4 bits)
  level_2 (upper-local): (coord >> 7) & 31      (5 bits)
  which_top:             coord >> 12             (root key)

This file provides:
  build_compressed_cig3()       -- construct from (N, 3) voxel coordinates
  root_lookup()                 -- torch root-level linear scan
  cig3_ijk_to_index_numpy()     -- numpy reference query (loop-based)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal helpers: popcount and masked-level builder
# ---------------------------------------------------------------------------


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
    """Build bitmasks and absolute prefix sums for one masked level.

    Generic across node sizes (8 words for 8^3, 64 for 16^3, 512 for 32^3).

    The absolute prefix folds the base offset into the cumulative popcount:
    abs_prefix[node, word] = node_offset + cum_popc_before_word.
    Query is just abs_prefix[word] + partial_popcount. Two arrays, two gathers.

    Args:
        node_of_child: (M,) i64 -- which node each child belongs to
        child_local_flat: (M,) i64 -- flat local position within the node
        n_nodes: int -- number of nodes at this level
        n_words: int -- number of u64 words per node mask
        device: torch device

    Returns:
        masks: (N, W) i64, abs_prefix: (N, W) i32
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

    # Relative prefix sums (within each node)
    rel_prefix = _build_prefix_sums(masks)

    # Offsets: cumulative child counts per node
    counts = torch.zeros(n_nodes, dtype=torch.int64, device=device)
    counts.scatter_add_(0, sorted_node.long(), torch.ones(M, dtype=torch.int64, device=device))
    offsets = torch.zeros(n_nodes, dtype=torch.int64, device=device)
    if n_nodes > 1:
        offsets[1:] = counts[:-1].cumsum(0)

    # Absolute prefix: fold offset into prefix
    # abs_prefix[node, word] = offsets[node] + rel_prefix[node, word]
    abs_prefix = rel_prefix + offsets.unsqueeze(1).int()

    return masks, abs_prefix


# ---------------------------------------------------------------------------
# 3-level Compressed CIG: upper (32^3) + lower (16^3) + leaf (8^3)
# ---------------------------------------------------------------------------


@dataclass
class CompressedCIG3:
    """A 3-level CIG with bitmask-compressed nodes at every level.

    Bit-widths [3, 4, 5]:  leaf=8^3, lower=16^3, upper=32^3.
    Total coordinate range: 2^(3+4+5) = 4096 per axis.

    Each level stores masks + absolute prefix (offset folded in).
    Two arrays and two gathers per level.

    Root level: variable number of upper nodes, identified by their
    which_top coordinates.
    """

    # Root
    root_coords: torch.Tensor  # (R, 3) i32 -- which_top coords of each upper node

    # Upper nodes (32^3 = 32768 positions per node, 512 u64 words)
    upper_masks: torch.Tensor  # (U, 512) i64
    upper_abs_prefix: torch.Tensor  # (U, 512) i32

    # Lower nodes (16^3 = 4096 positions per node, 64 u64 words)
    lower_masks: torch.Tensor  # (L, 64) i64
    lower_abs_prefix: torch.Tensor  # (L, 64) i32

    # Leaf nodes (8^3 = 512 positions per node, 8 u64 words)
    leaf_masks: torch.Tensor  # (K, 8) i64
    leaf_abs_prefix: torch.Tensor  # (K, 8) i32

    n_active: int
    n_leaves: int
    n_lower: int
    n_upper: int

    @property
    def num_bytes(self) -> int:
        total = 0
        for t in [
            self.root_coords,
            self.upper_masks, self.upper_abs_prefix,
            self.lower_masks, self.lower_abs_prefix,
            self.leaf_masks, self.leaf_abs_prefix,
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
            upper_abs_prefix=self.upper_abs_prefix.cuda(),
            lower_masks=self.lower_masks.cuda(),
            lower_abs_prefix=self.lower_abs_prefix.cuda(),
            leaf_masks=self.leaf_masks.cuda(),
            leaf_abs_prefix=self.leaf_abs_prefix.cuda(),
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

    upper_masks, upper_abs_prefix = _build_masked_level(
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

    lower_masks, lower_abs_prefix = _build_masked_level(
        leaf_to_lower, leaf_flat_in_lower, L, 64, device
    )

    # --- Voxels: which leaf does each voxel belong to? ---
    voxel_wt_ul_ll = torch.cat([wt, ul, ll], dim=1)  # (N_voxels, 9)
    _, voxel_to_leaf = torch.unique(voxel_wt_ul_ll, dim=0, return_inverse=True)

    # Flat local position of each voxel within its leaf (8^3)
    voxel_flat_in_leaf = fl[:, 0] * 64 + fl[:, 1] * 8 + fl[:, 2]

    leaf_masks, leaf_abs_prefix = _build_masked_level(
        voxel_to_leaf, voxel_flat_in_leaf, K, 8, device
    )

    return CompressedCIG3(
        root_coords=root_coords,
        upper_masks=upper_masks,
        upper_abs_prefix=upper_abs_prefix,
        lower_masks=lower_masks,
        lower_abs_prefix=lower_abs_prefix,
        leaf_masks=leaf_masks,
        leaf_abs_prefix=leaf_abs_prefix,
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


# ---------------------------------------------------------------------------
# Numpy reference: 3-level ijk_to_index
# ---------------------------------------------------------------------------


def _masked_lookup_numpy(mask_words, abs_prefix, flat_idx):
    """Single masked lookup: check bitmask + absolute prefix popcount."""
    word_idx = flat_idx >> 6
    bit_pos = flat_idx & 63
    if word_idx < 0 or word_idx >= len(mask_words):
        return -1
    word = int(mask_words[word_idx])
    if not ((word >> bit_pos) & 1):
        return -1
    cum = int(abs_prefix[word_idx])
    partial = bin(word & ((1 << bit_pos) - 1) & 0xFFFFFFFFFFFFFFFF).count("1")
    return cum + partial


def cig3_ijk_to_index_numpy(cig: CompressedCIG3, query_np: np.ndarray) -> np.ndarray:
    """Numpy reference ijk_to_index for 3-level CIG."""
    root_np = cig.root_coords.cpu().numpy()
    u_masks = cig.upper_masks.cpu().numpy()
    u_abs_prefix = cig.upper_abs_prefix.cpu().numpy()
    l_masks = cig.lower_masks.cpu().numpy()
    l_abs_prefix = cig.lower_abs_prefix.cpu().numpy()
    k_masks = cig.leaf_masks.cpu().numpy()
    k_abs_prefix = cig.leaf_abs_prefix.cpu().numpy()

    N = query_np.shape[0]
    result = np.full(N, -1, dtype=np.int32)

    for i in range(N):
        coord = query_np[i]
        wt = coord >> 12
        ul = (coord >> 7) & 31
        ll = (coord >> 3) & 15
        fl = coord & 7

        # Root: linear scan for matching which_top
        upper_idx = -1
        for r in range(root_np.shape[0]):
            if np.array_equal(root_np[r], wt):
                upper_idx = r
                break
        if upper_idx < 0:
            continue

        # Upper -> lower
        flat_ul = int(ul[0]) * 1024 + int(ul[1]) * 32 + int(ul[2])
        lower_idx = _masked_lookup_numpy(u_masks[upper_idx], u_abs_prefix[upper_idx], flat_ul)
        if lower_idx < 0:
            continue

        # Lower -> leaf
        flat_ll = int(ll[0]) * 256 + int(ll[1]) * 16 + int(ll[2])
        leaf_idx = _masked_lookup_numpy(l_masks[lower_idx], l_abs_prefix[lower_idx], flat_ll)
        if leaf_idx < 0:
            continue

        # Leaf -> voxel
        flat_fl = int(fl[0]) * 64 + int(fl[1]) * 8 + int(fl[2])
        voxel_idx = _masked_lookup_numpy(k_masks[leaf_idx], k_abs_prefix[leaf_idx], flat_fl)
        result[i] = voxel_idx

    return result
