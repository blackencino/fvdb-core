import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL (tile-parallel) ---
# Source: a = Map(Input("x"), v => Add(v, Const(10)))
# a
# Tile input: x (rank=0), TILE=256

@ct.kernel
def seg_a_b3101389(x_arr, result_arr, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    idx_1 = ct.arange(256, dtype=ct.int32)
    qidx_2 = bid * 256 + idx_1
    
    qi_3 = ct.gather(x_arr, qidx_2, check_bounds=True, padding_value=0)
    
    
    # Map: v => ...
    add_4 = qi_3 + 10
    
    out_5 = ct.astype(add_4, ct.int32)
    ct.scatter(result_arr, qidx_2, out_5, check_bounds=True)
