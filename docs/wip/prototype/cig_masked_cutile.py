# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
cuTile kernels for compressed CIG ijk_to_index with bitmask + popcount.

Primary path: u64 (8 x uint64 words = 512 bits per leaf).
Reference path: i32 (16 x int32 words) -- kept for comparison.

The Gather-through-masked chain:
  1. Decompose query into level_1 (lower) and level_0 (leaf-local)
  2. Gather leaf_idx from lower using level_1
  3. Compute flat bit index from level_0
  4. Gather the mask word, check if bit is set
  5. Popcount: count set bits before the target position
  6. Result = leaf_offset + popcount if active, else -1
"""

import math

import numpy as np
import torch

import cuda.tile as ct

ConstInt = ct.Constant[int]

TILE = 256


@ct.kernel
def cig_masked_kernel(
    query_arr,
    lower_arr,
    leaf_masks_arr,
    leaf_offsets_arr,
    result_arr,
    TILE: ConstInt,
):
    """Compressed CIG ijk_to_index via bitmask + popcount.

    query_arr:      (N, 3)       i32 -- query coordinates
    lower_arr:      (16, 16, 16) i32 -- lower-level grid
    leaf_masks_arr: (K, 16)      i32 -- 16 x 32-bit bitmask per leaf
    leaf_offsets_arr: (K,)       i32 -- base offset per leaf
    result_arr:     (N,)         i32 -- output
    """
    bid = ct.bid(0)
    idx = ct.arange(TILE, dtype=ct.int32)
    query_idx = bid * TILE + idx

    # Load query components
    qx = ct.gather(query_arr, (query_idx, 0), check_bounds=True, padding_value=0)
    qy = ct.gather(query_arr, (query_idx, 1), check_bounds=True, padding_value=0)
    qz = ct.gather(query_arr, (query_idx, 2), check_bounds=True, padding_value=0)

    # Decompose
    l0_x = qx & 7
    l0_y = qy & 7
    l0_z = qz & 7
    l1_x = (qx >> 3) & 15
    l1_y = (qy >> 3) & 15
    l1_z = (qz >> 3) & 15

    # Gather leaf index from lower
    leaf_idx = ct.gather(lower_arr, (l1_x, l1_y, l1_z), check_bounds=True, padding_value=-1)

    # Flat bit index within the 512-bit leaf mask
    bit_idx = l0_x * 64 + l0_y * 8 + l0_z

    # Which i32 word (0-15) and bit position (0-31)
    word_idx = (bit_idx >> 5) & 15
    bit_pos = bit_idx & 31

    # Gather the target mask word
    target_word = ct.gather(leaf_masks_arr, (leaf_idx, word_idx), check_bounds=True, padding_value=0)

    # Bitmask check: is this voxel active?
    is_active = (target_word >> bit_pos) & 1

    # Popcount: count set bits before bit_idx across all 16 words.
    # Sum popcount of words 0..word_idx-1, plus partial popcount of word_idx.

    # Partial word: mask out bits >= bit_pos
    partial_mask = target_word & ((1 << bit_pos) - 1)

    # Hamming weight of partial word
    m1 = 0x55555555
    m2 = 0x33333333
    m4 = 0x0F0F0F0F
    u = partial_mask - ((partial_mask >> 1) & m1)
    u = (u & m2) + ((u >> 2) & m2)
    u = (u + (u >> 4)) & m4
    partial_popc = (u * 0x01010101) >> 24

    # Sum popcount of all full words before word_idx.
    # Unrolled: gather each of 16 words, compute popcount, accumulate
    # for words where their index < word_idx.
    full_popc = 0
    w0 = ct.gather(leaf_masks_arr, (leaf_idx, 0), check_bounds=True, padding_value=0)
    w1 = ct.gather(leaf_masks_arr, (leaf_idx, 1), check_bounds=True, padding_value=0)
    w2 = ct.gather(leaf_masks_arr, (leaf_idx, 2), check_bounds=True, padding_value=0)
    w3 = ct.gather(leaf_masks_arr, (leaf_idx, 3), check_bounds=True, padding_value=0)
    w4 = ct.gather(leaf_masks_arr, (leaf_idx, 4), check_bounds=True, padding_value=0)
    w5 = ct.gather(leaf_masks_arr, (leaf_idx, 5), check_bounds=True, padding_value=0)
    w6 = ct.gather(leaf_masks_arr, (leaf_idx, 6), check_bounds=True, padding_value=0)
    w7 = ct.gather(leaf_masks_arr, (leaf_idx, 7), check_bounds=True, padding_value=0)
    w8 = ct.gather(leaf_masks_arr, (leaf_idx, 8), check_bounds=True, padding_value=0)
    w9 = ct.gather(leaf_masks_arr, (leaf_idx, 9), check_bounds=True, padding_value=0)
    w10 = ct.gather(leaf_masks_arr, (leaf_idx, 10), check_bounds=True, padding_value=0)
    w11 = ct.gather(leaf_masks_arr, (leaf_idx, 11), check_bounds=True, padding_value=0)
    w12 = ct.gather(leaf_masks_arr, (leaf_idx, 12), check_bounds=True, padding_value=0)
    w13 = ct.gather(leaf_masks_arr, (leaf_idx, 13), check_bounds=True, padding_value=0)
    w14 = ct.gather(leaf_masks_arr, (leaf_idx, 14), check_bounds=True, padding_value=0)
    w15 = ct.gather(leaf_masks_arr, (leaf_idx, 15), check_bounds=True, padding_value=0)

    # Popcount helper inline (Hamming weight for each word, conditional on index)
    def _popc(w):
        v = w - ((w >> 1) & m1)
        v = (v & m2) + ((v >> 2) & m2)
        v = (v + (v >> 4)) & m4
        return (v * 0x01010101) >> 24

    # Accumulate: add popcount only if word index < word_idx
    # Use masking: if word_idx > i, add popcount(w_i), else add 0
    # (word_idx > i) produces 1 or 0 which we multiply
    p0 = _popc(w0) * (word_idx > 0)
    p1 = _popc(w1) * (word_idx > 1)
    p2 = _popc(w2) * (word_idx > 2)
    p3 = _popc(w3) * (word_idx > 3)
    p4 = _popc(w4) * (word_idx > 4)
    p5 = _popc(w5) * (word_idx > 5)
    p6 = _popc(w6) * (word_idx > 6)
    p7 = _popc(w7) * (word_idx > 7)
    p8 = _popc(w8) * (word_idx > 8)
    p9 = _popc(w9) * (word_idx > 9)
    p10 = _popc(w10) * (word_idx > 10)
    p11 = _popc(w11) * (word_idx > 11)
    p12 = _popc(w12) * (word_idx > 12)
    p13 = _popc(w13) * (word_idx > 13)
    p14 = _popc(w14) * (word_idx > 14)
    p15 = _popc(w15) * (word_idx > 15)
    full_popc = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11 + p12 + p13 + p14 + p15

    total_popc = full_popc + partial_popc

    # Base offset
    base = ct.gather(leaf_offsets_arr, leaf_idx, check_bounds=True, padding_value=0)

    # Result: active -> base + popcount, inactive -> -1
    voxel_idx = (base + total_popc) * is_active + (-1) * (1 - is_active)

    ct.scatter(result_arr, query_idx, voxel_idx, check_bounds=True)


# ---------------------------------------------------------------------------
# u64 variant: 8 x uint64 words instead of 16 x i32
# ---------------------------------------------------------------------------


@ct.kernel
def cig_masked_u64_kernel(
    query_arr,
    lower_arr,
    leaf_masks_arr,
    leaf_offsets_arr,
    result_arr,
    TILE: ConstInt,
):
    """Compressed CIG ijk_to_index via u64 bitmask + popcount.

    query_arr:      (N, 3)       i32 -- query coordinates
    lower_arr:      (16, 16, 16) i32 -- lower-level grid
    leaf_masks_arr: (K, 8)       i64 -- 8 x 64-bit bitmask per leaf (treated as u64)
    leaf_offsets_arr: (K,)       i32 -- base offset per leaf
    result_arr:     (N,)         i32 -- output
    """
    bid = ct.bid(0)
    idx = ct.arange(TILE, dtype=ct.int32)
    query_idx = bid * TILE + idx

    # Load query components
    qx = ct.gather(query_arr, (query_idx, 0), check_bounds=True, padding_value=0)
    qy = ct.gather(query_arr, (query_idx, 1), check_bounds=True, padding_value=0)
    qz = ct.gather(query_arr, (query_idx, 2), check_bounds=True, padding_value=0)

    # Decompose
    l0_x = qx & 7
    l0_y = qy & 7
    l0_z = qz & 7
    l1_x = (qx >> 3) & 15
    l1_y = (qy >> 3) & 15
    l1_z = (qz >> 3) & 15

    # Gather leaf index from lower
    leaf_idx = ct.gather(lower_arr, (l1_x, l1_y, l1_z), check_bounds=True, padding_value=-1)

    # Flat bit index within the 512-bit leaf mask
    bit_idx = l0_x * 64 + l0_y * 8 + l0_z

    # Which u64 word (0-7) and bit position (0-63)
    word_idx = (bit_idx >> 6) & 7
    bit_pos = ct.astype(bit_idx & 63, ct.uint64)

    # Gather the target mask word (i64 tensor -> cast to u64 for clean bit ops)
    target_word = ct.astype(
        ct.gather(leaf_masks_arr, (leaf_idx, word_idx), check_bounds=True, padding_value=0),
        ct.uint64,
    )

    # Bitmask check: is this voxel active?
    is_active_u = (target_word >> bit_pos) & ct.uint64(1)
    is_active = ct.astype(is_active_u, ct.int32)

    # Partial word: mask out bits >= bit_pos
    one_u = ct.uint64(1)
    partial_mask = target_word & ((one_u << bit_pos) - one_u)

    # u64 Hamming weight constants
    m1 = ct.uint64(0x5555555555555555)
    m2 = ct.uint64(0x3333333333333333)
    m4 = ct.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = ct.uint64(0x0101010101010101)

    def _popc64(w):
        v = w - ((w >> ct.uint64(1)) & m1)
        v = (v & m2) + ((v >> ct.uint64(2)) & m2)
        v = (v + (v >> ct.uint64(4))) & m4
        return ct.astype((v * h01) >> ct.uint64(56), ct.int32)

    partial_popc = _popc64(partial_mask)

    # Gather and popcount all 8 words, accumulate for words before word_idx
    w0 = ct.astype(ct.gather(leaf_masks_arr, (leaf_idx, 0), check_bounds=True, padding_value=0), ct.uint64)
    w1 = ct.astype(ct.gather(leaf_masks_arr, (leaf_idx, 1), check_bounds=True, padding_value=0), ct.uint64)
    w2 = ct.astype(ct.gather(leaf_masks_arr, (leaf_idx, 2), check_bounds=True, padding_value=0), ct.uint64)
    w3 = ct.astype(ct.gather(leaf_masks_arr, (leaf_idx, 3), check_bounds=True, padding_value=0), ct.uint64)
    w4 = ct.astype(ct.gather(leaf_masks_arr, (leaf_idx, 4), check_bounds=True, padding_value=0), ct.uint64)
    w5 = ct.astype(ct.gather(leaf_masks_arr, (leaf_idx, 5), check_bounds=True, padding_value=0), ct.uint64)
    w6 = ct.astype(ct.gather(leaf_masks_arr, (leaf_idx, 6), check_bounds=True, padding_value=0), ct.uint64)
    w7 = ct.astype(ct.gather(leaf_masks_arr, (leaf_idx, 7), check_bounds=True, padding_value=0), ct.uint64)

    full_popc = (
        _popc64(w0) * (word_idx > 0)
        + _popc64(w1) * (word_idx > 1)
        + _popc64(w2) * (word_idx > 2)
        + _popc64(w3) * (word_idx > 3)
        + _popc64(w4) * (word_idx > 4)
        + _popc64(w5) * (word_idx > 5)
        + _popc64(w6) * (word_idx > 6)
        + _popc64(w7) * (word_idx > 7)
    )

    total_popc = full_popc + partial_popc

    # Base offset
    base = ct.gather(leaf_offsets_arr, leaf_idx, check_bounds=True, padding_value=0)

    # Result: active -> base + popcount, inactive -> -1
    voxel_idx = (base + total_popc) * is_active + (-1) * (1 - is_active)

    ct.scatter(result_arr, query_idx, voxel_idx, check_bounds=True)


# ---------------------------------------------------------------------------
# Primary entry point (u64)
# ---------------------------------------------------------------------------


def run_compressed_cig_ijk_to_index(query_t, lower_t, leaf_masks_i64_t, leaf_offsets_t):
    """Launch the compressed CIG cuTile kernel (u64 path).

    Args:
        query_t:          (N, 3) i32 CUDA
        lower_t:          (16, 16, 16) i32 CUDA
        leaf_masks_i64_t: (K, 8) i64 CUDA -- native 8 x u64 bitmasks
        leaf_offsets_t:   (K,) i32 CUDA -- base offsets

    Returns:
        (N,) i32 tensor of voxel indices (-1 for inactive)
    """
    N = query_t.shape[0]
    n_blocks = math.ceil(N / TILE)

    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device="cuda")

    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        cig_masked_u64_kernel,
        (query_t, lower_t, leaf_masks_i64_t, leaf_offsets_t, result_t, TILE),
    )

    return result_t[:N]


# ---------------------------------------------------------------------------
# i32 reference variant (kept for comparison, not the primary path)
# ---------------------------------------------------------------------------


def build_i32_masks(cig_compressed) -> torch.Tensor:
    """Convert CompressedCIG i64 masks to i32 masks for the i32 reference kernel.

    Input:  (K, 8) i64  -- 8 words of 64 bits
    Output: (K, 16) i32 -- 16 words of 32 bits
    """
    masks_i64 = cig_compressed.leaf_masks
    K = masks_i64.shape[0]
    masks_np = masks_i64.cpu().numpy()

    result_np = np.zeros((K, 16), dtype=np.int32)
    for k in range(K):
        for w in range(8):
            val = int(masks_np[k, w])
            lo = val & 0xFFFFFFFF
            hi = (val >> 32) & 0xFFFFFFFF
            result_np[k, w * 2] = np.array(lo, dtype=np.uint32).view(np.int32)
            result_np[k, w * 2 + 1] = np.array(hi, dtype=np.uint32).view(np.int32)

    return torch.from_numpy(result_np).to(masks_i64.device)


def run_compressed_cig_ijk_to_index_i32(query_t, lower_t, leaf_masks_i32_t, leaf_offsets_t):
    """Launch the i32 reference compressed CIG cuTile kernel.

    Args:
        query_t:          (N, 3) i32 CUDA
        lower_t:          (16, 16, 16) i32 CUDA
        leaf_masks_i32_t: (K, 16) i32 CUDA -- i32 packed bitmasks
        leaf_offsets_t:   (K,) i32 CUDA -- base offsets (i32)

    Returns:
        (N,) i32 tensor of voxel indices (-1 for inactive)
    """
    N = query_t.shape[0]
    n_blocks = math.ceil(N / TILE)

    result_t = torch.full((n_blocks * TILE,), -1, dtype=torch.int32, device="cuda")

    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks,),
        cig_masked_kernel,
        (query_t, lower_t, leaf_masks_i32_t, leaf_offsets_t, result_t, TILE),
    )

    return result_t[:N]
