# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
DSL prototype tests: programs are strings, parsed into AST, type-checked,
and executed against torch data.

Three programs:
  1. Where + Gather pipeline
  2. Neighbor finding (jagged emergence)
  3. Two-level hierarchical chain
"""

import torch

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# =========================================================================
# Program 1: Where + Gather
# =========================================================================

WHERE_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
idx = Gather(Input("leaf"), active)
idx
"""


def test_where_program():
    """Parse + type-check + execute the Where pipeline from a string."""

    # ---- SETUP ----
    gen = torch.Generator().manual_seed(42)
    leaf_data = torch.randint(-1, 10, (8, 8, 8), generator=gen, dtype=torch.int32)
    leaf = Value.from_tensor(leaf_data, ScalarType.I32)

    # ---- PROGRAM ----
    types, result = run(WHERE_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    print("Program 1 types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    assert types["mask"] == Type(Shape(Static(8), Static(8), Static(8)), ScalarType.BOOL)
    assert types["active"] == Type(Shape(Dynamic()), coord_type(3))
    assert types["idx"] == Type(Shape(Dynamic()), ScalarType.I32)

    # Data checks
    expected_count = int(torch.sum(leaf_data >= 0).item())
    assert result.data.shape[0] == expected_count
    assert torch.all(result.data >= 0)

    # Verify gathered values match manual lookup
    expected_coords = torch.nonzero(leaf_data >= 0).to(torch.int32)
    expected_vals = leaf_data[
        expected_coords[:, 0], expected_coords[:, 1], expected_coords[:, 2]
    ]
    torch.testing.assert_close(result.data, expected_vals, atol=0, rtol=0)

    print(f"  -> {expected_count} active voxels, all values correct")


# =========================================================================
# Program 2: Neighbor finding
# =========================================================================

NEIGHBOR_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
nbrs = Each(active, a => Map(Input("offsets"), o => Add(a, o)))
filtered = Each(nbrs, cs => Gather(cs, Where(Map(cs, c => And(InBounds(c, Const(0), Const(8)), Gather(mask, c))))))
filtered
"""


def test_neighbor_program():
    """Neighbor finding as a DSL string. Verify jagged type emerges."""

    # ---- SETUP ----
    gen = torch.Generator().manual_seed(42)
    leaf_data = torch.randint(-1, 10, (8, 8, 8), generator=gen, dtype=torch.int32)
    leaf = Value.from_tensor(leaf_data, ScalarType.I32)

    offsets_data = torch.tensor(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=torch.int32,
    )
    offsets = Value(
        Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
        offsets_data,
    )

    # ---- PROGRAM ----
    types, result = run(NEIGHBOR_PROGRAM, {"leaf": leaf, "offsets": offsets})

    # ---- EXTRACTION ----
    print("Program 2 types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    # Key assertion: filtered should be (*) over (~) over (3) over i32
    assert result.type.iteration_shape == Shape(Dynamic())
    inner = result.type.element_type
    assert isinstance(inner, Type)
    assert isinstance(inner.iteration_shape.extents[0], Jagged), (
        f"Expected jagged inner, got {inner.iteration_shape.extents[0]!r}"
    )

    # Verify against brute-force reference
    mask_data = leaf_data >= 0
    active_coords = torch.nonzero(mask_data).to(torch.int32)

    assert len(result.data) == len(active_coords)

    neighbor_counts = []
    for i, coord in enumerate(active_coords):
        nbrs = coord.unsqueeze(0) + offsets_data
        valid = [
            nb
            for nb in nbrs
            if torch.all(nb >= 0).item()
            and torch.all(nb < 8).item()
            and mask_data[nb[0].item(), nb[1].item(), nb[2].item()].item()
        ]
        ref = torch.stack(valid).to(torch.int32) if valid else torch.empty((0, 3), dtype=torch.int32)

        actual = result.data[i].data
        torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        neighbor_counts.append(len(ref))

    neighbor_counts = torch.tensor(neighbor_counts)
    print(
        f"  -> {len(active_coords)} active, "
        f"neighbors min={neighbor_counts.min().item()} max={neighbor_counts.max().item()} "
        f"mean={neighbor_counts.float().mean().item():.1f}"
    )
    assert neighbor_counts.min().item() != neighbor_counts.max().item(), "Expected jagged"


# =========================================================================
# Program 3: Two-level hierarchical chain
# =========================================================================

CHAIN_PROGRAM = """
parts = Decompose(Input("coord"), Const([3, 4]))
leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
leaf_node = Gather(Input("leaf_arr"), leaf_idx)
voxel_idx = Gather(leaf_node, field(parts, "level_0"))
voxel_idx
"""


def test_chain_program():
    """Two-level hierarchical lookup as a DSL string.

    Uses a single lower node (no top-level indexing needed), so the chain is:
      Decompose -> Gather(lower, level_1) -> Gather(leaf_arr, leaf_idx) -> Gather(leaf_node, level_0)
    """

    # ---- SETUP ----
    torch.manual_seed(42)

    # Build a small two-level grid
    lower_data = torch.full((16, 16, 16), -1, dtype=torch.int32)
    leaf_blocks = []

    active_lower_coords = [(2, 3, 4), (5, 1, 7), (10, 10, 10)]
    for i, (lx, ly, lz) in enumerate(active_lower_coords):
        lower_data[lx, ly, lz] = i
        leaf = torch.full((8, 8, 8), -1, dtype=torch.int32)
        for v in range(50):
            vx, vy, vz = v // 16, (v // 4) % 4, v % 4
            if vx < 8 and vy < 8 and vz < 8:
                leaf[vx, vy, vz] = i * 100 + v
        leaf_blocks.append(leaf)

    leaf_data = torch.stack(leaf_blocks)  # (3, 8, 8, 8)

    # A coord that maps to active_lower_coords[1]=(5,1,7), leaf voxel (2,1,3)
    global_coord = torch.tensor([5 * 8 + 2, 1 * 8 + 1, 7 * 8 + 3], dtype=torch.int32)

    lower_val = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data,
    )
    leaf_arr_val = Value(
        Type(Shape(Static(len(leaf_blocks))), Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32)),
        leaf_data,
    )
    coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), global_coord)

    # ---- PROGRAM ----
    types, result = run(CHAIN_PROGRAM, {
        "coord": coord_val,
        "lower": lower_val,
        "leaf_arr": leaf_arr_val,
    })

    # ---- EXTRACTION ----
    print("Program 3 types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    # Manual reference
    ll = (global_coord >> 3) & 15
    vl = global_coord & 7
    expected_leaf_idx = lower_data[ll[0].item(), ll[1].item(), ll[2].item()]
    assert expected_leaf_idx >= 0, "Expected active lower entry"
    expected_voxel_idx = leaf_data[
        expected_leaf_idx.item(), vl[0].item(), vl[1].item(), vl[2].item()
    ]
    assert expected_voxel_idx >= 0, "Expected active voxel"

    actual_voxel = int(result.data)
    assert actual_voxel == expected_voxel_idx.item(), (
        f"Chain mismatch: {actual_voxel} vs {expected_voxel_idx.item()}"
    )
    print(f"  coord {global_coord} -> voxel_idx {actual_voxel} (expected {expected_voxel_idx.item()})")
    print(f"  -> chain fully executed from DSL string")


# =========================================================================

if __name__ == "__main__":
    print("=== Program 1: Where + Gather ===")
    test_where_program()

    print("\n=== Program 2: Neighbor finding ===")
    test_neighbor_program()

    print("\n=== Program 3: Two-level chain ===")
    test_chain_program()

    print("\nAll DSL tests passed.")
