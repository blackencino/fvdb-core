import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: parts = Decompose(Input("query"), Const([3, 4]))
# leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
# leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_prefix"), leaf_idx), Gather(Input("leaf_offsets"), leaf_idx))
# voxel_idx = Gather(leaf, field(parts, "level_0"))
# voxel_idx
# Tile input: query (rank=3), TILE=256

@ct.kernel
def gen_masked_cig_np(query_arr, lower_arr, leaf_masks_arr, leaf_prefix_arr, leaf_offsets_arr, result_arr, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    idx_1 = ct.arange(256, dtype=ct.int32)
    qidx_2 = bid * 256 + idx_1
    
    qi_3 = ct.gather(query_arr, (qidx_2, 0), check_bounds=True, padding_value=0)
    qi_4 = ct.gather(query_arr, (qidx_2, 1), check_bounds=True, padding_value=0)
    qi_5 = ct.gather(query_arr, (qidx_2, 2), check_bounds=True, padding_value=0)
    
    
    # Decompose: bit_widths=[3, 4]
    d0_6 = qi_3 & 7
    d0_7 = qi_4 & 7
    d0_8 = qi_5 & 7
    d1_9 = (qi_3 >> 3) & 15
    d1_10 = (qi_4 >> 3) & 15
    d1_11 = (qi_5 >> 3) & 15
    dt_12 = qi_3 >> 7
    dt_13 = qi_4 >> 7
    dt_14 = qi_5 >> 7
    gath_15 = ct.gather(lower_arr, (d1_9, d1_10, d1_11), check_bounds=True, padding_value=-1)
    mbase_16 = ct.gather(leaf_offsets_arr, gath_15, check_bounds=True, padding_value=0)
    # --- Masked gather (prefix-sum, 8^3 node) ---
    bit_idx_17 = d0_6 * 64 + d0_7 * 8 + d0_8
    word_idx_18 = (bit_idx_17 >> 6) & 7
    bit_pos_19 = ct.astype(bit_idx_17 & 63, ct.uint64)
    tgt_word_20 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, word_idx_18), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_21 = (tgt_word_20 >> bit_pos_19) & ct.uint64(1)
    is_active_22 = ct.astype(is_active_u_21, ct.int32)
    cum_popc_23 = ct.gather(leaf_prefix_arr, (gath_15, word_idx_18), check_bounds=True, padding_value=0)
    pmask_24 = tgt_word_20 & ((ct.uint64(1) << bit_pos_19) - ct.uint64(1))
    m1_u64 = ct.uint64(0x5555555555555555)
    m2_u64 = ct.uint64(0x3333333333333333)
    m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)
    h01_u64 = ct.uint64(0x0101010101010101)
    pc_25 = pmask_24 - ((pmask_24 >> ct.uint64(1)) & m1_u64)
    pc_26 = (pc_25 & m2_u64) + ((pc_25 >> ct.uint64(2)) & m2_u64)
    pc_27 = (pc_26 + (pc_26 >> ct.uint64(4))) & m4_u64
    pc_28 = ct.astype((pc_27 * h01_u64) >> ct.uint64(56), ct.int32)
    total_popc_29 = cum_popc_23 + pc_28
    masked_idx_30 = (mbase_16 + total_popc_29) * is_active_22 + (-1) * (1 - is_active_22)
    
    out_31 = ct.astype(masked_idx_30, ct.int32)
    ct.scatter(result_arr, qidx_2, out_31, check_bounds=True)
