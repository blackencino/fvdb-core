# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Two-level hierarchical grid.

EXPRESSION phases use DSL strings.

Proves:
  - Hierarchical index chain (output of one Gather feeds into the next)
  - Coordinate decomposition as a value-level primitive
  - Swappable indexing strategy (3D vs morton) with identical chain structure
  - Composition of hierarchical lookups with iteration patterns (Each, Where)
"""

import torch

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.ops import Value, morton3d
from fvdb_tile.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# ---------------------------------------------------------------------------
# DSL programs
# ---------------------------------------------------------------------------

CHAIN_PROGRAM = """
parts = Decompose(Input("coord"), Const([3, 4]))
leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
leaf_node = Gather(Input("leaf_arr"), leaf_idx)
voxel_idx = Gather(leaf_node, field(parts, "level_0"))
voxel_idx
"""

BATCH_CHAIN_PROGRAM = """
parts = Decompose(Input("coord"), Const([3, 4]))
leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
leaf_node = Gather(Input("leaf_arr"), leaf_idx)
voxel_idx = Gather(leaf_node, field(parts, "level_0"))
voxel_idx
"""

BATCH_ACTIVE_PROGRAM = """
leaves = cut(Input("leaf_flat"), Const(512))
leaves_3d = Each(leaves, leaf => reshape(leaf, Const([8, 8, 8])))
active = Each(leaves_3d, leaf => Where(Map(leaf, x => GE(x, Const(0)))))
active
"""


# ---------------------------------------------------------------------------
# Test data builder
# ---------------------------------------------------------------------------

def _build_two_level_grid(n_lower=3, seed=42):
    """Build a small two-level grid for testing."""
    gen = torch.Generator().manual_seed(seed)

    L = n_lower
    lower_data = torch.full((L, 16, 16, 16), -1, dtype=torch.int32)

    leaf_blocks = []
    next_leaf_idx = 0
    next_voxel_idx = 0

    for li in range(L):
        n_active = torch.randint(100, 500, (1,), generator=gen).item()
        positions = torch.randperm(16 * 16 * 16, generator=gen)[:n_active]
        for pos in positions:
            pos_val = pos.item()
            lx = pos_val // (16 * 16)
            ly = (pos_val // 16) % 16
            lz = pos_val % 16
            lower_data[li, lx, ly, lz] = next_leaf_idx

            leaf = torch.full((8, 8, 8), -1, dtype=torch.int32)
            n_voxels = torch.randint(50, 400, (1,), generator=gen).item()
            voxel_positions = torch.randperm(512, generator=gen)[:n_voxels]
            for vp in voxel_positions:
                vp_val = vp.item()
                vx = vp_val // 64
                vy = (vp_val // 8) % 8
                vz = vp_val % 8
                leaf[vx, vy, vz] = next_voxel_idx
                next_voxel_idx += 1
            leaf_blocks.append(leaf)
            next_leaf_idx += 1

    K = len(leaf_blocks)
    leaf_data = torch.stack(leaf_blocks)

    return lower_data, leaf_data, next_voxel_idx


# =========================================================================
# Scenario 1: Two-level grid with 3D indexing, chained Gather
# =========================================================================

def test_3d_chain_single_coord():
    """Look up a single global coordinate through two levels via DSL."""

    # ---- SETUP ----
    lower_data, leaf_data, _ = _build_two_level_grid(n_lower=2)

    # Pick a coord that resolves to an active voxel
    active_lower_entries = torch.nonzero(lower_data[0] >= 0)
    ll = active_lower_entries[0]
    leaf_idx_expected = lower_data[0, ll[0].item(), ll[1].item(), ll[2].item()]
    active_leaf_entries = torch.nonzero(leaf_data[leaf_idx_expected] >= 0)
    vl = active_leaf_entries[0]

    global_coord = torch.tensor([
        ll[0].item() * 8 + vl[0].item(),
        ll[1].item() * 8 + vl[1].item(),
        ll[2].item() * 8 + vl[2].item(),
    ], dtype=torch.int32)

    coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), global_coord)
    lower_val = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data[0],
    )
    leaf_arr_val = Value(
        Type(Shape(Static(leaf_data.shape[0])),
             Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32)),
        leaf_data,
    )

    # ---- EXPRESSION ----
    types, result = run(CHAIN_PROGRAM, {
        "coord": coord_val, "lower": lower_val, "leaf_arr": leaf_arr_val,
    })

    # ---- EXTRACTION ----
    print("3D chain types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    vl_ref = global_coord & 7
    ll_ref = (global_coord >> 3) & 15
    expected_leaf = lower_data[0, ll_ref[0].item(), ll_ref[1].item(), ll_ref[2].item()]
    expected_voxel = leaf_data[expected_leaf, vl_ref[0].item(), vl_ref[1].item(), vl_ref[2].item()] if expected_leaf >= 0 else -1

    actual = int(result.data)
    exp = expected_voxel.item() if isinstance(expected_voxel, torch.Tensor) else expected_voxel
    assert actual == exp, f"{actual} vs {exp}"
    assert actual >= 0
    print(f"  global {global_coord} -> voxel_idx {actual}")


def test_3d_chain_batch():
    """Batch lookup: many global coords through two levels via DSL."""

    # ---- SETUP ----
    lower_data, leaf_data, _ = _build_two_level_grid(n_lower=1)
    N = 50
    gen = torch.Generator().manual_seed(99)
    global_coords = torch.randint(0, 128, (N, 3), generator=gen, dtype=torch.int32)

    lower_val = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data[0],
    )
    leaf_arr_val = Value(
        Type(Shape(Static(leaf_data.shape[0])),
             Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32)),
        leaf_data,
    )

    # ---- EXPRESSION (one coord at a time through the DSL) ----
    for i in range(N):
        coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), global_coords[i])
        _, result = run(BATCH_CHAIN_PROGRAM, {
            "coord": coord_val, "lower": lower_val, "leaf_arr": leaf_arr_val,
        })

        # ---- EXTRACTION ----
        gc = global_coords[i]
        ll = (gc >> 3) & 15
        vl = gc & 7
        expected_leaf = lower_data[0, ll[0].item(), ll[1].item(), ll[2].item()]
        expected_voxel = leaf_data[expected_leaf, vl[0].item(), vl[1].item(), vl[2].item()] if expected_leaf >= 0 else -1
        actual = int(result.data)
        exp = expected_voxel.item() if isinstance(expected_voxel, torch.Tensor) else expected_voxel
        assert actual == exp, f"Coord {gc}: {actual} vs {exp}"

    print(f"3D chain batch: {N} lookups verified")


# =========================================================================
# Scenario 2: Morton-linearized nodes, identical results
# =========================================================================

MORTON_CHAIN_PROGRAM = """
parts = Decompose(Input("coord"), Const([3, 4]))
lower_idx = Morton3d(field(parts, "level_1"))
leaf_idx = Gather(Input("lower_m"), lower_idx)
leaf_local_m = Morton3d(field(parts, "level_0"))
leaf_node = Gather(Input("leaf_arr_m"), leaf_idx)
voxel_idx = Gather(leaf_node, leaf_local_m)
voxel_idx
"""


def _morton_lut(dim: int) -> torch.Tensor:
    """Pre-compute a (dim, dim, dim) -> morton index lookup table."""
    coords = torch.stack(torch.meshgrid(
        torch.arange(dim), torch.arange(dim), torch.arange(dim), indexing="ij"
    ), dim=-1).reshape(-1, 3).to(torch.int32)
    return morton3d(coords)


def test_morton_chain_batch():
    """Same grid morton-linearized, verify identical results via DSL."""

    # ---- SETUP ----
    lower_data, leaf_data, _ = _build_two_level_grid(n_lower=1)

    # Morton-linearize (vectorized)
    lut_16 = _morton_lut(16)
    lower_morton = torch.full((4096,), -1, dtype=torch.int32)
    lower_morton[lut_16.long()] = lower_data[0].reshape(-1)

    K = leaf_data.shape[0]
    lut_8 = _morton_lut(8)
    leaf_morton = torch.full((K, 512), -1, dtype=torch.int32)
    leaf_morton[:, lut_8.long()] = leaf_data.reshape(K, -1)

    lower_m_val = Value(Type(Shape(Static(4096)), ScalarType.I32), lower_morton)
    leaf_m_val = Value(
        Type(Shape(Static(K)), Type(Shape(Static(512)), ScalarType.I32)),
        leaf_morton,
    )

    N = 50
    gen = torch.Generator().manual_seed(99)
    global_coords = torch.randint(0, 128, (N, 3), generator=gen, dtype=torch.int32)

    # ---- EXPRESSION ----
    for i in range(N):
        coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), global_coords[i])
        _, result = run(MORTON_CHAIN_PROGRAM, {
            "coord": coord_val, "lower_m": lower_m_val, "leaf_arr_m": leaf_m_val,
        })

        # ---- EXTRACTION ----
        gc = global_coords[i]
        ll = (gc >> 3) & 15
        vl = gc & 7
        expected_leaf = lower_data[0, ll[0].item(), ll[1].item(), ll[2].item()]
        expected_voxel = leaf_data[expected_leaf, vl[0].item(), vl[1].item(), vl[2].item()] if expected_leaf >= 0 else -1
        actual = int(result.data)
        exp = expected_voxel.item() if isinstance(expected_voxel, torch.Tensor) else expected_voxel
        assert actual == exp, f"Morton coord {gc}: {actual} vs {exp}"

    print(f"Morton chain batch: {N} lookups match 3D reference exactly")


# =========================================================================
# Scenario 3: Batch active-voxel lookup through two levels
# =========================================================================

def test_batch_active_voxel_lookup():
    """Where over leaves, then verify active voxels via DSL."""

    # ---- SETUP ----
    lower_data, leaf_data, _ = _build_two_level_grid(n_lower=1)
    K = leaf_data.shape[0]
    n_test = min(5, K)
    leaf_subset = leaf_data[:n_test]
    leaf_flat = Value.from_tensor(leaf_subset.reshape(n_test * 512), ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(BATCH_ACTIVE_PROGRAM, {"leaf_flat": leaf_flat})

    # ---- EXTRACTION ----
    print(f"active_per_leaf type: {result.type}")
    assert result.type.iteration_shape == Shape(Static(n_test))
    inner = result.type.element_type
    assert isinstance(inner, Type)
    assert isinstance(inner.iteration_shape.extents[0], Jagged)

    total_active = 0
    for li in range(n_test):
        expected = torch.nonzero(leaf_subset[li].reshape(8, 8, 8) >= 0).to(torch.int32)
        actual = result.data[li].data
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        total_active += len(expected)

    counts = [result.data[li].data.shape[0] for li in range(n_test)]
    print(f"  {n_test} leaves, active counts: {counts}, total: {total_active}")
    assert len(set(counts)) > 1, "Expected varying counts (jagged)"


# =========================================================================

if __name__ == "__main__":
    print("=== Scenario 1: 3D chain (DSL) ===")
    test_3d_chain_single_coord()
    test_3d_chain_batch()

    print("\n=== Scenario 2: Morton chain (DSL) ===")
    test_morton_chain_batch()

    print("\n=== Scenario 3: Batch active lookup (DSL) ===")
    test_batch_active_voxel_lookup()

    print("\nAll test_two_level tests passed.")
