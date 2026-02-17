import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: parts = Decompose(Input("query"), Const([3, 4, 5]))
# upper = masked(Gather(Input("upper_masks"), Input("upper_idx")), Gather(Input("upper_prefix"), Input("upper_idx")), Gather(Input("upper_offsets"), Input("upper_idx")))
# lower_idx = Gather(upper, field(parts, "level_2"))
# lower = masked(Gather(Input("lower_masks"), lower_idx), Gather(Input("lower_prefix"), lower_idx), Gather(Input("lower_offsets"), lower_idx))
# leaf_idx = Gather(lower, field(parts, "level_1"))
# leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_prefix"), leaf_idx), Gather(Input("leaf_offsets"), leaf_idx))
# voxel_idx = Gather(leaf, field(parts, "level_0"))
# voxel_idx
# Tile input: query (rank=3), TILE=256

@ct.kernel
def bench_cig3_kernel(query_arr, upper_idx_arr, upper_masks_arr, upper_prefix_arr, upper_offsets_arr, lower_masks_arr, lower_prefix_arr, lower_offsets_arr, leaf_masks_arr, leaf_prefix_arr, leaf_offsets_arr, result_arr, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    idx_1 = ct.arange(256, dtype=ct.int32)
    qidx_2 = bid * 256 + idx_1
    
    qi_3 = ct.gather(query_arr, (qidx_2, 0), check_bounds=True, padding_value=0)
    qi_4 = ct.gather(query_arr, (qidx_2, 1), check_bounds=True, padding_value=0)
    qi_5 = ct.gather(query_arr, (qidx_2, 2), check_bounds=True, padding_value=0)
    
    ti_6 = ct.gather(upper_idx_arr, qidx_2, check_bounds=True, padding_value=-1)
    
    # Decompose: bit_widths=[3, 4, 5]
    d0_7 = qi_3 & 7
    d0_8 = qi_4 & 7
    d0_9 = qi_5 & 7
    d1_10 = (qi_3 >> 3) & 15
    d1_11 = (qi_4 >> 3) & 15
    d1_12 = (qi_5 >> 3) & 15
    d2_13 = (qi_3 >> 7) & 31
    d2_14 = (qi_4 >> 7) & 31
    d2_15 = (qi_5 >> 7) & 31
    dt_16 = qi_3 >> 12
    dt_17 = qi_4 >> 12
    dt_18 = qi_5 >> 12
    mbase_19 = ct.gather(upper_offsets_arr, ti_6, check_bounds=True, padding_value=0)
    # --- Masked gather (prefix-sum, 32^3 node) ---
    bit_idx_20 = d2_13 * 1024 + d2_14 * 32 + d2_15
    word_idx_21 = (bit_idx_20 >> 6) & 511
    bit_pos_22 = ct.astype(bit_idx_20 & 63, ct.uint64)
    tgt_word_23 = ct.astype(ct.gather(upper_masks_arr, (ti_6, word_idx_21), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_24 = (tgt_word_23 >> bit_pos_22) & ct.uint64(1)
    is_active_25 = ct.astype(is_active_u_24, ct.int32)
    cum_popc_26 = ct.gather(upper_prefix_arr, (ti_6, word_idx_21), check_bounds=True, padding_value=0)
    pmask_27 = tgt_word_23 & ((ct.uint64(1) << bit_pos_22) - ct.uint64(1))
    m1_u64 = ct.uint64(0x5555555555555555)
    m2_u64 = ct.uint64(0x3333333333333333)
    m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)
    h01_u64 = ct.uint64(0x0101010101010101)
    pc_28 = pmask_27 - ((pmask_27 >> ct.uint64(1)) & m1_u64)
    pc_29 = (pc_28 & m2_u64) + ((pc_28 >> ct.uint64(2)) & m2_u64)
    pc_30 = (pc_29 + (pc_29 >> ct.uint64(4))) & m4_u64
    pc_31 = ct.astype((pc_30 * h01_u64) >> ct.uint64(56), ct.int32)
    total_popc_32 = cum_popc_26 + pc_31
    masked_idx_33 = (mbase_19 + total_popc_32) * is_active_25 + (-1) * (1 - is_active_25)
    mbase_34 = ct.gather(lower_offsets_arr, masked_idx_33, check_bounds=True, padding_value=0)
    # --- Masked gather (prefix-sum, 16^3 node) ---
    bit_idx_35 = d1_10 * 256 + d1_11 * 16 + d1_12
    word_idx_36 = (bit_idx_35 >> 6) & 63
    bit_pos_37 = ct.astype(bit_idx_35 & 63, ct.uint64)
    tgt_word_38 = ct.astype(ct.gather(lower_masks_arr, (masked_idx_33, word_idx_36), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_39 = (tgt_word_38 >> bit_pos_37) & ct.uint64(1)
    is_active_40 = ct.astype(is_active_u_39, ct.int32)
    cum_popc_41 = ct.gather(lower_prefix_arr, (masked_idx_33, word_idx_36), check_bounds=True, padding_value=0)
    pmask_42 = tgt_word_38 & ((ct.uint64(1) << bit_pos_37) - ct.uint64(1))
    m1_u64 = ct.uint64(0x5555555555555555)
    m2_u64 = ct.uint64(0x3333333333333333)
    m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)
    h01_u64 = ct.uint64(0x0101010101010101)
    pc_43 = pmask_42 - ((pmask_42 >> ct.uint64(1)) & m1_u64)
    pc_44 = (pc_43 & m2_u64) + ((pc_43 >> ct.uint64(2)) & m2_u64)
    pc_45 = (pc_44 + (pc_44 >> ct.uint64(4))) & m4_u64
    pc_46 = ct.astype((pc_45 * h01_u64) >> ct.uint64(56), ct.int32)
    total_popc_47 = cum_popc_41 + pc_46
    masked_idx_48 = (mbase_34 + total_popc_47) * is_active_40 + (-1) * (1 - is_active_40)
    mbase_49 = ct.gather(leaf_offsets_arr, masked_idx_48, check_bounds=True, padding_value=0)
    # --- Masked gather (prefix-sum, 8^3 node) ---
    bit_idx_50 = d0_7 * 64 + d0_8 * 8 + d0_9
    word_idx_51 = (bit_idx_50 >> 6) & 7
    bit_pos_52 = ct.astype(bit_idx_50 & 63, ct.uint64)
    tgt_word_53 = ct.astype(ct.gather(leaf_masks_arr, (masked_idx_48, word_idx_51), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_54 = (tgt_word_53 >> bit_pos_52) & ct.uint64(1)
    is_active_55 = ct.astype(is_active_u_54, ct.int32)
    cum_popc_56 = ct.gather(leaf_prefix_arr, (masked_idx_48, word_idx_51), check_bounds=True, padding_value=0)
    pmask_57 = tgt_word_53 & ((ct.uint64(1) << bit_pos_52) - ct.uint64(1))
    m1_u64 = ct.uint64(0x5555555555555555)
    m2_u64 = ct.uint64(0x3333333333333333)
    m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)
    h01_u64 = ct.uint64(0x0101010101010101)
    pc_58 = pmask_57 - ((pmask_57 >> ct.uint64(1)) & m1_u64)
    pc_59 = (pc_58 & m2_u64) + ((pc_58 >> ct.uint64(2)) & m2_u64)
    pc_60 = (pc_59 + (pc_59 >> ct.uint64(4))) & m4_u64
    pc_61 = ct.astype((pc_60 * h01_u64) >> ct.uint64(56), ct.int32)
    total_popc_62 = cum_popc_56 + pc_61
    masked_idx_63 = (mbase_49 + total_popc_62) * is_active_55 + (-1) * (1 - is_active_55)
    
    out_64 = ct.astype(masked_idx_63, ct.int32)
    ct.scatter(result_arr, qidx_2, out_64, check_bounds=True)
