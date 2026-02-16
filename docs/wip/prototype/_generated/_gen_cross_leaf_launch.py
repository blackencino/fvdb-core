import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# --- Generated from DSL ---
# Source: pred = Map(Input("offsets"), o => GE(Gather(Gather(Input("leaf_arr"), Gather(Input("lower"), field(Decompose(Add(Input("coord"), o), Const([3, 4])), "level_1"))), field(Decompose(Add(Input("coord"), o), Const([3, 4])), "level_0")), Const(0)))
# pred
# Batch input: coord (dim=3), Map input: offsets (elem_rank=3)
# Tile size: 8 (next power-of-two >= 6)

@ct.kernel
def cross_leaf_launch(coord_arr, offsets_arr, lower_arr, leaf_arr_arr, result_arr, TILE: ct.Constant[int]):
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
    # Decompose: bit_widths=[3, 4]
    d0_11 = add_8 & 7
    d0_12 = add_9 & 7
    d0_13 = add_10 & 7
    d1_14 = (add_8 >> 3) & 15
    d1_15 = (add_9 >> 3) & 15
    d1_16 = (add_10 >> 3) & 15
    dt_17 = add_8 >> 7
    dt_18 = add_9 >> 7
    dt_19 = add_10 >> 7
    gath_20 = ct.gather(lower_arr, (d1_14, d1_15, d1_16), check_bounds=True, padding_value=-1)
    add_21 = bc_1 + mi_5
    add_22 = bc_2 + mi_6
    add_23 = bc_3 + mi_7
    # Decompose: bit_widths=[3, 4]
    d0_24 = add_21 & 7
    d0_25 = add_22 & 7
    d0_26 = add_23 & 7
    d1_27 = (add_21 >> 3) & 15
    d1_28 = (add_22 >> 3) & 15
    d1_29 = (add_23 >> 3) & 15
    dt_30 = add_21 >> 7
    dt_31 = add_22 >> 7
    dt_32 = add_23 >> 7
    # Fused chained gather: 4D index
    gath_33 = ct.gather(leaf_arr_arr, (gath_20, d0_24, d0_25, d0_26), check_bounds=True, padding_value=-1)
    ge_34 = gath_33 >= 0
    
    out_35 = ct.astype(ge_34, ct.int32)
    ct.scatter(result_arr, (bid, idx_4), out_35, check_bounds=True)
