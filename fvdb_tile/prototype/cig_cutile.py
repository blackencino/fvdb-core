# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
cuTile kernel for CIG ijk_to_index.

Hand-written from the cross-leaf pattern (v6), simplified: no inner Map,
just tile-parallel queries per block. Each block handles TILE queries via
ct.arange, performing the Decompose + fused chained Gather lookup.

Pattern:
  query -> Decompose([3, 4]) -> Gather(lower, level_1) -> leaf_idx
                               -> Gather(leaf_arr, (leaf_idx, level_0)) -> voxel_idx
"""

import math

import torch

import cuda.tile as ct

ConstInt = ct.Constant[int]

TILE = 256  # queries per block (power of 2)


@ct.kernel
def cig_ijk_to_index_kernel(
    query_arr,
    lower_arr,
    leaf_arr_arr,
    result_arr,
    TILE: ConstInt,
):
    """For each query coordinate, look up the voxel index via two-level CIG.

    query_arr:    (N, 3)       i32 -- query coordinates
    lower_arr:    (16, 16, 16) i32 -- lower-level grid (-1 = empty)
    leaf_arr_arr: (K, 8, 8, 8) i32 -- leaf blocks (-1 = inactive)
    result_arr:   (N,)         i32 -- output indices
    TILE:         256          -- queries per block
    """
    bid = ct.bid(0)
    idx = ct.arange(TILE, dtype=ct.int32)
    query_idx = bid * TILE + idx

    # Gather query coordinate components
    qx = ct.gather(query_arr, (query_idx, 0), check_bounds=True, padding_value=0)
    qy = ct.gather(query_arr, (query_idx, 1), check_bounds=True, padding_value=0)
    qz = ct.gather(query_arr, (query_idx, 2), check_bounds=True, padding_value=0)

    # Decompose: bit_widths=[3, 4]
    l0_x = qx & 7
    l0_y = qy & 7
    l0_z = qz & 7
    l1_x = (qx >> 3) & 15
    l1_y = (qy >> 3) & 15
    l1_z = (qz >> 3) & 15

    # Hierarchical gather: lower -> leaf_idx, then fused (leaf_idx, l0) -> voxel_idx
    leaf_idx = ct.gather(lower_arr, (l1_x, l1_y, l1_z), check_bounds=True, padding_value=-1)
    voxel_idx = ct.gather(leaf_arr_arr, (leaf_idx, l0_x, l0_y, l0_z), check_bounds=True, padding_value=-1)

    # Scatter output
    ct.scatter(result_arr, query_idx, voxel_idx, check_bounds=True)


def run_cig_ijk_to_index(
    query_t: torch.Tensor,
    lower_t: torch.Tensor,
    leaf_arr_t: torch.Tensor,
) -> torch.Tensor:
    """Launch the cuTile CIG ijk_to_index kernel.

    Args:
        query_t:    (N, 3) i32 CUDA tensor
        lower_t:    (16, 16, 16) i32 CUDA tensor
        leaf_arr_t: (K, 8, 8, 8) i32 CUDA tensor

    Returns:
        (N,) i32 tensor of voxel indices (-1 for inactive)
    """
    N = query_t.shape[0]
    n_blocks = math.ceil(N / TILE)

    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device="cuda")

    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        cig_ijk_to_index_kernel,
        (query_t, lower_t, leaf_arr_t, result_t, TILE),
    )

    return result_t[:N]
