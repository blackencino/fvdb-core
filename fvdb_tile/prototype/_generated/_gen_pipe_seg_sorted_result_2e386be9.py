import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: sorted_result = HierarchicalKeyDecode(Input("sorted_keys"), Const([3, 4, 5]))
# sorted_result
# Tile input: sorted_keys (rank=0), TILE=256

@ct.kernel
def seg_sorted_result_2e386be9(sorted_keys_arr, result_arr, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    idx_1 = ct.arange(256, dtype=ct.int32)
    qidx_2 = bid * 256 + idx_1
    
    qi_3 = ct.gather(sorted_keys_arr, qidx_2, check_bounds=True, padding_value=0)
    
    
    # HierarchicalKeyDecode: bit_widths=[3, 4, 5]
    dk_4 = ct.astype(qi_3, ct.int64)
    dx_5 = ct.int64(0)
    dy_6 = ct.int64(0)
    dz_7 = ct.int64(0)
    dlin_8 = dk_4 & ct.int64(511)
    dlz_9 = dlin_8 % ct.int64(8)
    dly_10 = (dlin_8 / ct.int64(8)) % ct.int64(8)
    dlx_11 = dlin_8 / ct.int64(64)
    dx_12 = dx_5 | dlx_11
    dy_13 = dy_6 | dly_10
    dz_14 = dz_7 | dlz_9
    dlin_15 = (dk_4 >> ct.int64(9)) & ct.int64(4095)
    dlz_16 = dlin_15 % ct.int64(16)
    dly_17 = (dlin_15 / ct.int64(16)) % ct.int64(16)
    dlx_18 = dlin_15 / ct.int64(256)
    dx_19 = dx_12 | (dlx_18 << ct.int64(3))
    dy_20 = dy_13 | (dly_17 << ct.int64(3))
    dz_21 = dz_14 | (dlz_16 << ct.int64(3))
    dlin_22 = (dk_4 >> ct.int64(21)) & ct.int64(32767)
    dlz_23 = dlin_22 % ct.int64(32)
    dly_24 = (dlin_22 / ct.int64(32)) % ct.int64(32)
    dlx_25 = dlin_22 / ct.int64(1024)
    dx_26 = dx_19 | (dlx_25 << ct.int64(7))
    dy_27 = dy_20 | (dly_24 << ct.int64(7))
    dz_28 = dz_21 | (dlz_23 << ct.int64(7))
    drlin_29 = dk_4 >> ct.int64(36)
    drz_30 = drlin_29 & ct.int64(1023)
    dry_31 = (drlin_29 >> ct.int64(10)) & ct.int64(1023)
    drx_32 = drlin_29 >> ct.int64(20)
    fx_33 = ct.astype(dx_26 | (drx_32 << ct.int64(12)), ct.int32)
    fy_34 = ct.astype(dy_27 | (dry_31 << ct.int64(12)), ct.int32)
    fz_35 = ct.astype(dz_28 | (drz_30 << ct.int64(12)), ct.int32)
    
    out_36 = ct.astype(fx_33, ct.int32)
    ct.scatter(result_arr, qidx_2, out_36, check_bounds=True)
