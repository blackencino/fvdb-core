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

import numpy as np

from docs.wip.prototype.dsl_eval import run
from docs.wip.prototype.ops import Value
from docs.wip.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)
from docs.wip.prototype.ops import morton3d as np_morton3d


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
    np.random.seed(seed)

    L = n_lower
    lower_data = np.full((L, 16, 16, 16), -1, dtype=np.int32)

    leaf_blocks = []
    next_leaf_idx = 0
    next_voxel_idx = 0

    for li in range(L):
        n_active = np.random.randint(100, 500)
        positions = np.random.choice(16 * 16 * 16, n_active, replace=False)
        for pos in positions:
            lx = pos // (16 * 16)
            ly = (pos // 16) % 16
            lz = pos % 16
            lower_data[li, lx, ly, lz] = next_leaf_idx

            leaf = np.full((8, 8, 8), -1, dtype=np.int32)
            n_voxels = np.random.randint(50, 400)
            voxel_positions = np.random.choice(512, n_voxels, replace=False)
            for vp in voxel_positions:
                vx = vp // 64
                vy = (vp // 8) % 8
                vz = vp % 8
                leaf[vx, vy, vz] = next_voxel_idx
                next_voxel_idx += 1
            leaf_blocks.append(leaf)
            next_leaf_idx += 1

    K = len(leaf_blocks)
    leaf_data = np.stack(leaf_blocks)

    return lower_data, leaf_data, next_voxel_idx


# =========================================================================
# Scenario 1: Two-level grid with 3D indexing, chained Gather
# =========================================================================

def test_3d_chain_single_coord():
    """Look up a single global coordinate through two levels via DSL."""

    # ---- SETUP ----
    lower_data, leaf_data, _ = _build_two_level_grid(n_lower=2)

    # Pick a coord that resolves to an active voxel
    active_lower_entries = np.argwhere(lower_data[0] >= 0)
    ll = active_lower_entries[0]
    leaf_idx_expected = lower_data[0, ll[0], ll[1], ll[2]]
    active_leaf_entries = np.argwhere(leaf_data[leaf_idx_expected] >= 0)
    vl = active_leaf_entries[0]

    global_coord = np.array([
        ll[0] * 8 + vl[0],
        ll[1] * 8 + vl[1],
        ll[2] * 8 + vl[2],
    ], dtype=np.int32)

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
    expected_leaf = lower_data[0, ll_ref[0], ll_ref[1], ll_ref[2]]
    expected_voxel = leaf_data[expected_leaf, vl_ref[0], vl_ref[1], vl_ref[2]] if expected_leaf >= 0 else -1

    actual = int(result.data)
    assert actual == expected_voxel, f"{actual} vs {expected_voxel}"
    assert actual >= 0
    print(f"  global {global_coord} -> voxel_idx {actual}")


def test_3d_chain_batch():
    """Batch lookup: many global coords through two levels via DSL."""

    # ---- SETUP ----
    lower_data, leaf_data, _ = _build_two_level_grid(n_lower=1)
    N = 50
    np.random.seed(99)
    global_coords = np.random.randint(0, 128, (N, 3)).astype(np.int32)

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
        expected_leaf = lower_data[0, ll[0], ll[1], ll[2]]
        expected_voxel = leaf_data[expected_leaf, vl[0], vl[1], vl[2]] if expected_leaf >= 0 else -1
        actual = int(result.data)
        assert actual == expected_voxel, f"Coord {gc}: {actual} vs {expected_voxel}"

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


def test_morton_chain_batch():
    """Same grid morton-linearized, verify identical results via DSL."""

    # ---- SETUP ----
    lower_data, leaf_data, _ = _build_two_level_grid(n_lower=1)

    # Morton-linearize
    lower_morton = np.full((4096,), -1, dtype=np.int32)
    for x in range(16):
        for y in range(16):
            for z in range(16):
                m = int(np_morton3d(np.array([x, y, z], dtype=np.int32)))
                lower_morton[m] = lower_data[0, x, y, z]

    K = leaf_data.shape[0]
    leaf_morton = np.full((K, 512), -1, dtype=np.int32)
    for k in range(K):
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    m = int(np_morton3d(np.array([x, y, z], dtype=np.int32)))
                    leaf_morton[k, m] = leaf_data[k, x, y, z]

    lower_m_val = Value(Type(Shape(Static(4096)), ScalarType.I32), lower_morton)
    leaf_m_val = Value(
        Type(Shape(Static(K)), Type(Shape(Static(512)), ScalarType.I32)),
        leaf_morton,
    )

    N = 50
    np.random.seed(99)
    global_coords = np.random.randint(0, 128, (N, 3)).astype(np.int32)

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
        expected_leaf = lower_data[0, ll[0], ll[1], ll[2]]
        expected_voxel = leaf_data[expected_leaf, vl[0], vl[1], vl[2]] if expected_leaf >= 0 else -1
        actual = int(result.data)
        assert actual == expected_voxel, f"Morton coord {gc}: {actual} vs {expected_voxel}"

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
    leaf_flat = Value.from_numpy(leaf_subset.reshape(n_test * 512), ScalarType.I32)

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
        expected = np.argwhere(leaf_subset[li].reshape(8, 8, 8) >= 0).astype(np.int32)
        actual = result.data[li].data
        np.testing.assert_array_equal(actual, expected)
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
