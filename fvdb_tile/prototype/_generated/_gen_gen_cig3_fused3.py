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
def gen_cig3_fused3(query_arr, root_coords_arr, upper_masks_arr, upper_abs_prefix_arr, lower_masks_arr, lower_abs_prefix_arr, leaf_masks_arr, leaf_abs_prefix_arr, result_arr, TILE: ct.Constant[int]):
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
    # --- Find: linear scan of 3 entries ---
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
    rc_28 = ct.gather(root_coords_arr, (1, 0), check_bounds=True, padding_value=-9999)
    fm_29 = ct.astype(dt_15 == rc_28, ct.int32)
    rc_30 = ct.gather(root_coords_arr, (1, 1), check_bounds=True, padding_value=-9999)
    fm_31 = ct.astype(dt_16 == rc_30, ct.int32)
    rc_32 = ct.gather(root_coords_arr, (1, 2), check_bounds=True, padding_value=-9999)
    fm_33 = ct.astype(dt_17 == rc_32, ct.int32)
    fm_34 = fm_29 & fm_31
    fm_35 = fm_34 & fm_33
    find_idx_36 = find_idx_27 * (1 - fm_35) + 1 * fm_35
    rc_37 = ct.gather(root_coords_arr, (2, 0), check_bounds=True, padding_value=-9999)
    fm_38 = ct.astype(dt_15 == rc_37, ct.int32)
    rc_39 = ct.gather(root_coords_arr, (2, 1), check_bounds=True, padding_value=-9999)
    fm_40 = ct.astype(dt_16 == rc_39, ct.int32)
    rc_41 = ct.gather(root_coords_arr, (2, 2), check_bounds=True, padding_value=-9999)
    fm_42 = ct.astype(dt_17 == rc_41, ct.int32)
    fm_43 = fm_38 & fm_40
    fm_44 = fm_43 & fm_42
    find_idx_45 = find_idx_36 * (1 - fm_44) + 2 * fm_44
    # --- Masked gather (abs-prefix, 32^3 node) ---
    bit_idx_46 = d2_12 * 1024 + d2_13 * 32 + d2_14
    word_idx_47 = (bit_idx_46 >> 6) & 511
    bit_pos_48 = ct.astype(bit_idx_46 & 63, ct.uint64)
    tgt_word_49 = ct.astype(ct.gather(upper_masks_arr, (find_idx_45, word_idx_47), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_50 = (tgt_word_49 >> bit_pos_48) & ct.uint64(1)
    is_active_51 = ct.astype(is_active_u_50, ct.int32)
    abs_popc_52 = ct.gather(upper_abs_prefix_arr, (find_idx_45, word_idx_47), check_bounds=True, padding_value=0)
    pmask_53 = tgt_word_49 & ((ct.uint64(1) << bit_pos_48) - ct.uint64(1))
    m1_u64 = ct.uint64(0x5555555555555555)
    m2_u64 = ct.uint64(0x3333333333333333)
    m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)
    h01_u64 = ct.uint64(0x0101010101010101)
    pc_54 = pmask_53 - ((pmask_53 >> ct.uint64(1)) & m1_u64)
    pc_55 = (pc_54 & m2_u64) + ((pc_54 >> ct.uint64(2)) & m2_u64)
    pc_56 = (pc_55 + (pc_55 >> ct.uint64(4))) & m4_u64
    pc_57 = ct.astype((pc_56 * h01_u64) >> ct.uint64(56), ct.int32)
    masked_idx_58 = (abs_popc_52 + pc_57) * is_active_51 + (-1) * (1 - is_active_51)
    # --- Masked gather (abs-prefix, 16^3 node) ---
    bit_idx_59 = d1_9 * 256 + d1_10 * 16 + d1_11
    word_idx_60 = (bit_idx_59 >> 6) & 63
    bit_pos_61 = ct.astype(bit_idx_59 & 63, ct.uint64)
    tgt_word_62 = ct.astype(ct.gather(lower_masks_arr, (masked_idx_58, word_idx_60), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_63 = (tgt_word_62 >> bit_pos_61) & ct.uint64(1)
    is_active_64 = ct.astype(is_active_u_63, ct.int32)
    abs_popc_65 = ct.gather(lower_abs_prefix_arr, (masked_idx_58, word_idx_60), check_bounds=True, padding_value=0)
    pmask_66 = tgt_word_62 & ((ct.uint64(1) << bit_pos_61) - ct.uint64(1))
    pc_67 = pmask_66 - ((pmask_66 >> ct.uint64(1)) & m1_u64)
    pc_68 = (pc_67 & m2_u64) + ((pc_67 >> ct.uint64(2)) & m2_u64)
    pc_69 = (pc_68 + (pc_68 >> ct.uint64(4))) & m4_u64
    pc_70 = ct.astype((pc_69 * h01_u64) >> ct.uint64(56), ct.int32)
    masked_idx_71 = (abs_popc_65 + pc_70) * is_active_64 + (-1) * (1 - is_active_64)
    # --- Masked gather (abs-prefix, 8^3 node) ---
    bit_idx_72 = d0_6 * 64 + d0_7 * 8 + d0_8
    word_idx_73 = (bit_idx_72 >> 6) & 7
    bit_pos_74 = ct.astype(bit_idx_72 & 63, ct.uint64)
    tgt_word_75 = ct.astype(ct.gather(leaf_masks_arr, (masked_idx_71, word_idx_73), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_76 = (tgt_word_75 >> bit_pos_74) & ct.uint64(1)
    is_active_77 = ct.astype(is_active_u_76, ct.int32)
    abs_popc_78 = ct.gather(leaf_abs_prefix_arr, (masked_idx_71, word_idx_73), check_bounds=True, padding_value=0)
    pmask_79 = tgt_word_75 & ((ct.uint64(1) << bit_pos_74) - ct.uint64(1))
    pc_80 = pmask_79 - ((pmask_79 >> ct.uint64(1)) & m1_u64)
    pc_81 = (pc_80 & m2_u64) + ((pc_80 >> ct.uint64(2)) & m2_u64)
    pc_82 = (pc_81 + (pc_81 >> ct.uint64(4))) & m4_u64
    pc_83 = ct.astype((pc_82 * h01_u64) >> ct.uint64(56), ct.int32)
    masked_idx_84 = (abs_popc_78 + pc_83) * is_active_77 + (-1) * (1 - is_active_77)
    
    out_85 = ct.astype(masked_idx_84, ct.int32)
    ct.scatter(result_arr, qidx_2, out_85, check_bounds=True)
