import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL ---
# Source: pred = Map(Input("offsets"), o => GE(Gather(Input("mask"), Add(Input("coord"), o)), Const(0)))
# pred
# Batch input: coord (dim=3), Map input: offsets (elem_rank=3)
# Tile size: 8 (next power-of-two >= 6)

@ct.kernel
def gen_pred_e2e(mask_arr, coord_arr, offsets_arr, result_arr, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    
    bc_1 = ct.gather(coord_arr, (bid, 0))
    bc_2 = ct.gather(coord_arr, (bid, 1))
    bc_3 = ct.gather(coord_arr, (bid, 2))
    
    idx_4 = ct.arange(8, dtype=ct.int32)
    mi_5 = ct.gather(offsets_arr, (idx_4, 0), check_bounds=True, padding_value=0)
    mi_6 = ct.gather(offsets_arr, (idx_4, 1), check_bounds=True, padding_value=0)
    mi_7 = ct.gather(offsets_arr, (idx_4, 2), check_bounds=True, padding_value=0)
    
    # Map: o => ...
    add_8 = bc_1 + mi_5
    add_9 = bc_2 + mi_6
    add_10 = bc_3 + mi_7
    gath_11 = ct.gather(mask_arr, (add_8, add_9, add_10), check_bounds=True, padding_value=-1)
    ge_12 = gath_11 >= 0
    
    out_13 = ct.astype(ge_12, ct.int32)
    ct.scatter(result_arr, (bid, idx_4), out_13, check_bounds=True)
