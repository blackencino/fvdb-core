import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: parts = Decompose(Input("query"), Const([3, 4]))
# leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
# leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_abs_prefix"), leaf_idx))
# voxel_idx = Gather(leaf, field(parts, "level_0"))
# voxel_idx
# Tile input: query (rank=3), TILE=256

@ct.kernel
def gen_masked_cig_hw(query_arr, lower_arr, leaf_masks_arr, leaf_abs_prefix_arr, result_arr, TILE: ct.Constant[int]):
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
    # --- Masked gather (abs-prefix, 8^3 node) ---
    bit_idx_16 = d0_6 * 64 + d0_7 * 8 + d0_8
    word_idx_17 = (bit_idx_16 >> 6) & 7
    bit_pos_18 = ct.astype(bit_idx_16 & 63, ct.uint64)
    tgt_word_19 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, word_idx_17), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_20 = (tgt_word_19 >> bit_pos_18) & ct.uint64(1)
    is_active_21 = ct.astype(is_active_u_20, ct.int32)
    abs_popc_22 = ct.gather(leaf_abs_prefix_arr, (gath_15, word_idx_17), check_bounds=True, padding_value=0)
    pmask_23 = tgt_word_19 & ((ct.uint64(1) << bit_pos_18) - ct.uint64(1))
    m1_u64 = ct.uint64(0x5555555555555555)
    m2_u64 = ct.uint64(0x3333333333333333)
    m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)
    h01_u64 = ct.uint64(0x0101010101010101)
    pc_24 = pmask_23 - ((pmask_23 >> ct.uint64(1)) & m1_u64)
    pc_25 = (pc_24 & m2_u64) + ((pc_24 >> ct.uint64(2)) & m2_u64)
    pc_26 = (pc_25 + (pc_25 >> ct.uint64(4))) & m4_u64
    pc_27 = ct.astype((pc_26 * h01_u64) >> ct.uint64(56), ct.int32)
    masked_idx_28 = (abs_popc_22 + pc_27) * is_active_21 + (-1) * (1 - is_active_21)
    
    out_29 = ct.astype(masked_idx_28, ct.int32)
    ct.scatter(result_arr, qidx_2, out_29, check_bounds=True)
