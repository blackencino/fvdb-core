import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: parts = Decompose(Input("query"), Const([3, 4, 5]))
# upper_idx = Find(Input("root_coords"), field(parts, "which_top"))
# upper = masked(Gather(Input("upper_masks"), upper_idx), Gather(Input("upper_abs_prefix"), upper_idx))
# lower_idx = Gather(upper, field(parts, "level_2"))
# lower = masked(Gather(Input("lower_masks"), lower_idx), Gather(Input("lower_abs_prefix"), lower_idx))
# leaf_idx = Gather(lower, field(parts, "level_1"))
# leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_abs_prefix"), leaf_idx))
# voxel_idx = Gather(leaf, field(parts, "level_0"))
# voxel_idx
# Tile input: query (rank=3), TILE=256

@ct.kernel
def gen_cig3_fused1(query_arr, root_coords_arr, upper_masks_arr, upper_abs_prefix_arr, lower_masks_arr, lower_abs_prefix_arr, leaf_masks_arr, leaf_abs_prefix_arr, result_arr, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    idx_1 = ct.arange(256, dtype=ct.int32)
    qidx_2 = bid * 256 + idx_1
    
    qi_3 = ct.gather(query_arr, (qidx_2, 0), check_bounds=True, padding_value=0)
    qi_4 = ct.gather(query_arr, (qidx_2, 1), check_bounds=True, padding_value=0)
    qi_5 = ct.gather(query_arr, (qidx_2, 2), check_bounds=True, padding_value=0)
    
    
    # Decompose: bit_widths=[3, 4, 5]
    d0_6 = qi_3 & 7
    d0_7 = qi_4 & 7
    d0_8 = qi_5 & 7
    d1_9 = (qi_3 >> 3) & 15
    d1_10 = (qi_4 >> 3) & 15
    d1_11 = (qi_5 >> 3) & 15
    d2_12 = (qi_3 >> 7) & 31
    d2_13 = (qi_4 >> 7) & 31
    d2_14 = (qi_5 >> 7) & 31
    dt_15 = qi_3 >> 12
    dt_16 = qi_4 >> 12
    dt_17 = qi_5 >> 12
    # --- Find: linear scan of 1 entries ---
    find_idx_18 = -1
    rc_19 = ct.gather(root_coords_arr, (0, 0), check_bounds=True, padding_value=-9999)
    fm_20 = ct.astype(dt_15 == rc_19, ct.int32)
    rc_21 = ct.gather(root_coords_arr, (0, 1), check_bounds=True, padding_value=-9999)
    fm_22 = ct.astype(dt_16 == rc_21, ct.int32)
    rc_23 = ct.gather(root_coords_arr, (0, 2), check_bounds=True, padding_value=-9999)
    fm_24 = ct.astype(dt_17 == rc_23, ct.int32)
    fm_25 = fm_20 & fm_22
    fm_26 = fm_25 & fm_24
    find_idx_27 = find_idx_18 * (1 - fm_26) + 0 * fm_26
    # --- Masked gather (abs-prefix, 32^3 node) ---
    bit_idx_28 = d2_12 * 1024 + d2_13 * 32 + d2_14
    word_idx_29 = (bit_idx_28 >> 6) & 511
    bit_pos_30 = ct.astype(bit_idx_28 & 63, ct.uint64)
    tgt_word_31 = ct.astype(ct.gather(upper_masks_arr, (find_idx_27, word_idx_29), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_32 = (tgt_word_31 >> bit_pos_30) & ct.uint64(1)
    is_active_33 = ct.astype(is_active_u_32, ct.int32)
    abs_popc_34 = ct.gather(upper_abs_prefix_arr, (find_idx_27, word_idx_29), check_bounds=True, padding_value=0)
    pmask_35 = tgt_word_31 & ((ct.uint64(1) << bit_pos_30) - ct.uint64(1))
    m1_u64 = ct.uint64(0x5555555555555555)
    m2_u64 = ct.uint64(0x3333333333333333)
    m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)
    h01_u64 = ct.uint64(0x0101010101010101)
    pc_36 = pmask_35 - ((pmask_35 >> ct.uint64(1)) & m1_u64)
    pc_37 = (pc_36 & m2_u64) + ((pc_36 >> ct.uint64(2)) & m2_u64)
    pc_38 = (pc_37 + (pc_37 >> ct.uint64(4))) & m4_u64
    pc_39 = ct.astype((pc_38 * h01_u64) >> ct.uint64(56), ct.int32)
    masked_idx_40 = (abs_popc_34 + pc_39) * is_active_33 + (-1) * (1 - is_active_33)
    # --- Masked gather (abs-prefix, 16^3 node) ---
    bit_idx_41 = d1_9 * 256 + d1_10 * 16 + d1_11
    word_idx_42 = (bit_idx_41 >> 6) & 63
    bit_pos_43 = ct.astype(bit_idx_41 & 63, ct.uint64)
    tgt_word_44 = ct.astype(ct.gather(lower_masks_arr, (masked_idx_40, word_idx_42), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_45 = (tgt_word_44 >> bit_pos_43) & ct.uint64(1)
    is_active_46 = ct.astype(is_active_u_45, ct.int32)
    abs_popc_47 = ct.gather(lower_abs_prefix_arr, (masked_idx_40, word_idx_42), check_bounds=True, padding_value=0)
    pmask_48 = tgt_word_44 & ((ct.uint64(1) << bit_pos_43) - ct.uint64(1))
    pc_49 = pmask_48 - ((pmask_48 >> ct.uint64(1)) & m1_u64)
    pc_50 = (pc_49 & m2_u64) + ((pc_49 >> ct.uint64(2)) & m2_u64)
    pc_51 = (pc_50 + (pc_50 >> ct.uint64(4))) & m4_u64
    pc_52 = ct.astype((pc_51 * h01_u64) >> ct.uint64(56), ct.int32)
    masked_idx_53 = (abs_popc_47 + pc_52) * is_active_46 + (-1) * (1 - is_active_46)
    # --- Masked gather (abs-prefix, 8^3 node) ---
    bit_idx_54 = d0_6 * 64 + d0_7 * 8 + d0_8
    word_idx_55 = (bit_idx_54 >> 6) & 7
    bit_pos_56 = ct.astype(bit_idx_54 & 63, ct.uint64)
    tgt_word_57 = ct.astype(ct.gather(leaf_masks_arr, (masked_idx_53, word_idx_55), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_58 = (tgt_word_57 >> bit_pos_56) & ct.uint64(1)
    is_active_59 = ct.astype(is_active_u_58, ct.int32)
    abs_popc_60 = ct.gather(leaf_abs_prefix_arr, (masked_idx_53, word_idx_55), check_bounds=True, padding_value=0)
    pmask_61 = tgt_word_57 & ((ct.uint64(1) << bit_pos_56) - ct.uint64(1))
    pc_62 = pmask_61 - ((pmask_61 >> ct.uint64(1)) & m1_u64)
    pc_63 = (pc_62 & m2_u64) + ((pc_62 >> ct.uint64(2)) & m2_u64)
    pc_64 = (pc_63 + (pc_63 >> ct.uint64(4))) & m4_u64
    pc_65 = ct.astype((pc_64 * h01_u64) >> ct.uint64(56), ct.int32)
    masked_idx_66 = (abs_popc_60 + pc_65) * is_active_59 + (-1) * (1 - is_active_59)
    
    out_67 = ct.astype(masked_idx_66, ct.int32)
    ct.scatter(result_arr, qidx_2, out_67, check_bounds=True)
