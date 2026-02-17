import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: shifted = ShiftLeft(Input("x"), Const(2))
# result = ShiftRight(shifted, Const(1))
# result
# Tile input: x (rank=0), TILE=256

@ct.kernel
def seg_result_d91f593d(x_arr, result_arr, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    idx_1 = ct.arange(256, dtype=ct.int32)
    qidx_2 = bid * 256 + idx_1
    
    qi_3 = ct.gather(x_arr, qidx_2, check_bounds=True, padding_value=0)
    
    
    shl_4 = qi_3 << 2
    shr_5 = shl_4 >> 1
    
    out_6 = ct.astype(shr_5, ct.int32)
    ct.scatter(result_arr, qidx_2, out_6, check_bounds=True)
