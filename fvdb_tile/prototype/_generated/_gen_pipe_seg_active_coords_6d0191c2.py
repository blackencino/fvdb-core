import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: active_keys = Gather(Input("hash_map"), Input("occupied"))
# active_masks = Gather(Input("output_masks"), Input("occupied"))
# active_coords = HierarchicalKeyDecode(active_keys, Const([4, 5]))
# active_coords
# Tile input: output_masks (rank=1), TILE=256

@ct.kernel
def seg_active_coords_6d0191c2(output_masks_arr, hash_map_arr, occupied_arr, result_arr, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    idx_1 = ct.arange(256, dtype=ct.int32)
    qidx_2 = bid * 256 + idx_1
    
    qi_3 = ct.gather(output_masks_arr, (qidx_2, 0), check_bounds=True, padding_value=0)
    
    
    gath_4 = ct.gather(hash_map_arr, occupied_arr, check_bounds=True, padding_value=-1)
    gath_5 = ct.gather(qi_3, occupied_arr, check_bounds=True, padding_value=-1)
    # HierarchicalKeyDecode: bit_widths=[4, 5]
    dk_6 = ct.astype(gath_4, ct.int64)
    dx_7 = ct.int64(0)
    dy_8 = ct.int64(0)
    dz_9 = ct.int64(0)
    dlin_10 = dk_6 & ct.int64(4095)
    dlz_11 = dlin_10 % ct.int64(16)
    dly_12 = (dlin_10 / ct.int64(16)) % ct.int64(16)
    dlx_13 = dlin_10 / ct.int64(256)
    dx_14 = dx_7 | dlx_13
    dy_15 = dy_8 | dly_12
    dz_16 = dz_9 | dlz_11
    dlin_17 = (dk_6 >> ct.int64(12)) & ct.int64(32767)
    dlz_18 = dlin_17 % ct.int64(32)
    dly_19 = (dlin_17 / ct.int64(32)) % ct.int64(32)
    dlx_20 = dlin_17 / ct.int64(1024)
    dx_21 = dx_14 | (dlx_20 << ct.int64(4))
    dy_22 = dy_15 | (dly_19 << ct.int64(4))
    dz_23 = dz_16 | (dlz_18 << ct.int64(4))
    drlin_24 = dk_6 >> ct.int64(27)
    drz_25 = drlin_24 & ct.int64(1023)
    dry_26 = (drlin_24 >> ct.int64(10)) & ct.int64(1023)
    drx_27 = drlin_24 >> ct.int64(20)
    fx_28 = ct.astype(dx_21 | (drx_27 << ct.int64(9)), ct.int32)
    fy_29 = ct.astype(dy_22 | (dry_26 << ct.int64(9)), ct.int32)
    fz_30 = ct.astype(dz_23 | (drz_25 << ct.int64(9)), ct.int32)
    
    out_31 = ct.astype(fx_28, ct.int32)
    ct.scatter(result_arr, qidx_2, out_31, check_bounds=True)
