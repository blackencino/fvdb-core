import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: parts = Decompose(Input("query"), Const([3, 4]))
# leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
# leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_offsets"), leaf_idx))
# voxel_idx = Gather(leaf, field(parts, "level_0"))
# voxel_idx
# Tile input: query (rank=3), TILE=256

@ct.kernel
def gen_masked_cig_hw(query_arr, lower_arr, leaf_masks_arr, leaf_offsets_arr, result_arr, TILE: ct.Constant[int]):
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
    # --- Masked gather: u64 bitmask check + popcount ---
    bit_idx_17 = d0_6 * 64 + d0_7 * 8 + d0_8
    word_idx_18 = (bit_idx_17 >> 6) & 7
    bit_pos_19 = ct.astype(bit_idx_17 & 63, ct.uint64)
    tgt_word_20 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, word_idx_18), check_bounds=True, padding_value=0), ct.uint64)
    is_active_u_21 = (tgt_word_20 >> bit_pos_19) & ct.uint64(1)
    is_active_22 = ct.astype(is_active_u_21, ct.int32)
    pmask_23 = tgt_word_20 & ((ct.uint64(1) << bit_pos_19) - ct.uint64(1))
    m1_u64 = ct.uint64(0x5555555555555555)
    m2_u64 = ct.uint64(0x3333333333333333)
    m4_u64 = ct.uint64(0x0F0F0F0F0F0F0F0F)
    h01_u64 = ct.uint64(0x0101010101010101)
    pc_24 = pmask_23 - ((pmask_23 >> ct.uint64(1)) & m1_u64)
    pc_25 = (pc_24 & m2_u64) + ((pc_24 >> ct.uint64(2)) & m2_u64)
    pc_26 = (pc_25 + (pc_25 >> ct.uint64(4))) & m4_u64
    pc_27 = ct.astype((pc_26 * h01_u64) >> ct.uint64(56), ct.int32)
    mw_28 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, 0), check_bounds=True, padding_value=0), ct.uint64)
    pc_29 = mw_28 - ((mw_28 >> ct.uint64(1)) & m1_u64)
    pc_30 = (pc_29 & m2_u64) + ((pc_29 >> ct.uint64(2)) & m2_u64)
    pc_31 = (pc_30 + (pc_30 >> ct.uint64(4))) & m4_u64
    pc_32 = ct.astype((pc_31 * h01_u64) >> ct.uint64(56), ct.int32)
    pt_33 = pc_32 * (word_idx_18 > 0)
    mw_34 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, 1), check_bounds=True, padding_value=0), ct.uint64)
    pc_35 = mw_34 - ((mw_34 >> ct.uint64(1)) & m1_u64)
    pc_36 = (pc_35 & m2_u64) + ((pc_35 >> ct.uint64(2)) & m2_u64)
    pc_37 = (pc_36 + (pc_36 >> ct.uint64(4))) & m4_u64
    pc_38 = ct.astype((pc_37 * h01_u64) >> ct.uint64(56), ct.int32)
    pt_39 = pc_38 * (word_idx_18 > 1)
    mw_40 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, 2), check_bounds=True, padding_value=0), ct.uint64)
    pc_41 = mw_40 - ((mw_40 >> ct.uint64(1)) & m1_u64)
    pc_42 = (pc_41 & m2_u64) + ((pc_41 >> ct.uint64(2)) & m2_u64)
    pc_43 = (pc_42 + (pc_42 >> ct.uint64(4))) & m4_u64
    pc_44 = ct.astype((pc_43 * h01_u64) >> ct.uint64(56), ct.int32)
    pt_45 = pc_44 * (word_idx_18 > 2)
    mw_46 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, 3), check_bounds=True, padding_value=0), ct.uint64)
    pc_47 = mw_46 - ((mw_46 >> ct.uint64(1)) & m1_u64)
    pc_48 = (pc_47 & m2_u64) + ((pc_47 >> ct.uint64(2)) & m2_u64)
    pc_49 = (pc_48 + (pc_48 >> ct.uint64(4))) & m4_u64
    pc_50 = ct.astype((pc_49 * h01_u64) >> ct.uint64(56), ct.int32)
    pt_51 = pc_50 * (word_idx_18 > 3)
    mw_52 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, 4), check_bounds=True, padding_value=0), ct.uint64)
    pc_53 = mw_52 - ((mw_52 >> ct.uint64(1)) & m1_u64)
    pc_54 = (pc_53 & m2_u64) + ((pc_53 >> ct.uint64(2)) & m2_u64)
    pc_55 = (pc_54 + (pc_54 >> ct.uint64(4))) & m4_u64
    pc_56 = ct.astype((pc_55 * h01_u64) >> ct.uint64(56), ct.int32)
    pt_57 = pc_56 * (word_idx_18 > 4)
    mw_58 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, 5), check_bounds=True, padding_value=0), ct.uint64)
    pc_59 = mw_58 - ((mw_58 >> ct.uint64(1)) & m1_u64)
    pc_60 = (pc_59 & m2_u64) + ((pc_59 >> ct.uint64(2)) & m2_u64)
    pc_61 = (pc_60 + (pc_60 >> ct.uint64(4))) & m4_u64
    pc_62 = ct.astype((pc_61 * h01_u64) >> ct.uint64(56), ct.int32)
    pt_63 = pc_62 * (word_idx_18 > 5)
    mw_64 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, 6), check_bounds=True, padding_value=0), ct.uint64)
    pc_65 = mw_64 - ((mw_64 >> ct.uint64(1)) & m1_u64)
    pc_66 = (pc_65 & m2_u64) + ((pc_65 >> ct.uint64(2)) & m2_u64)
    pc_67 = (pc_66 + (pc_66 >> ct.uint64(4))) & m4_u64
    pc_68 = ct.astype((pc_67 * h01_u64) >> ct.uint64(56), ct.int32)
    pt_69 = pc_68 * (word_idx_18 > 6)
    mw_70 = ct.astype(ct.gather(leaf_masks_arr, (gath_15, 7), check_bounds=True, padding_value=0), ct.uint64)
    pc_71 = mw_70 - ((mw_70 >> ct.uint64(1)) & m1_u64)
    pc_72 = (pc_71 & m2_u64) + ((pc_71 >> ct.uint64(2)) & m2_u64)
    pc_73 = (pc_72 + (pc_72 >> ct.uint64(4))) & m4_u64
    pc_74 = ct.astype((pc_73 * h01_u64) >> ct.uint64(56), ct.int32)
    pt_75 = pc_74 * (word_idx_18 > 7)
    full_popc_76 = pt_33 + pt_39 + pt_45 + pt_51 + pt_57 + pt_63 + pt_69 + pt_75
    total_popc_77 = full_popc_76 + pc_27
    masked_idx_78 = (mbase_16 + total_popc_77) * is_active_22 + (-1) * (1 - is_active_22)
    
    out_79 = ct.astype(masked_idx_78, ct.int32)
    ct.scatter(result_arr, qidx_2, out_79, check_bounds=True)
