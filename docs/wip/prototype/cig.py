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
