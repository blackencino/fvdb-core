import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: d = Each(Input("c"), v => Sub(v, Const(1)))
# d
# Tile input: c (rank=0), TILE=256

@ct.kernel
def seg_d_8c876c65(c_arr, result_arr, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    idx_1 = ct.arange(256, dtype=ct.int32)
    qidx_2 = bid * 256 + idx_1
    
    qi_3 = ct.gather(c_arr, qidx_2, check_bounds=True, padding_value=0)
    
    
    # Each: v => ...
    sub_4 = qi_3 - 1
    
    out_5 = ct.astype(sub_4, ct.int32)
    ct.scatter(result_arr, qidx_2, out_5, check_bounds=True)
