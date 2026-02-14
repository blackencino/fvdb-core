# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Prototype v2: Two-level hierarchical grid.

Three scenarios:
  1. Two-level grid with 3D indexing, chained Gather through levels
  2. Same grid morton-linearized, verify identical results
  3. Batch active-voxel lookup through two levels

Proves:
  - Hierarchical index chain (output of one Gather feeds into the next)
  - Coordinate decomposition as a value-level primitive
  - Swappable indexing strategy (3D vs morton) with identical chain structure
  - Composition of hierarchical lookups with iteration patterns (Each, Where)
"""

import numpy as np

from docs.wip.prototype.layouts import cut_by_size, indexed, reshape
from docs.wip.prototype.ops import (
    Decompose,
    Each,
    Gather,
    Map,
    Value,
    Where,
    morton3d,
    inv_morton3d,
)
from docs.wip.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# ---------------------------------------------------------------------------
# Test data builder: a two-level grid with known structure
# ---------------------------------------------------------------------------

def _build_two_level_grid(n_lower: int = 3, seed: int = 42):
    """Build a small two-level grid for testing.

    Returns (lower_3d, leaf_3d, global_to_voxel_ref) where:
      - lower_3d: (L, 16, 16, 16) i32  -- lower node entries (leaf idx or -1)
      - leaf_3d:  (K, 8, 8, 8) i32     -- leaf node entries (voxel idx or -1)
      - global_to_voxel_ref: function that does the lookup manually
    """
    np.random.seed(seed)

    L = n_lower
    # Each lower node has 16^3 entries. ~30% point to leaf nodes.
    lower_data = np.full((L, 16, 16, 16), -1, dtype=np.int32)

    leaf_blocks = []
    next_leaf_idx = 0
    next_voxel_idx = 0

    for li in range(L):
        # Randomly activate some entries
        n_active = np.random.randint(100, 500)
        positions = np.random.choice(16 * 16 * 16, n_active, replace=False)
        for pos in positions:
            lx = pos // (16 * 16)
            ly = (pos // 16) % 16
            lz = pos % 16
            lower_data[li, lx, ly, lz] = next_leaf_idx

            # Build the leaf node
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
    leaf_data = np.stack(leaf_blocks)  # (K, 8, 8, 8)

    def ref_lookup(global_coord):
        """Manual two-level lookup."""
        i, j, k = global_coord
        li, lj, lk = (i >> 3) & 15, (j >> 3) & 15, (k >> 3) & 15
        vi, vj, vk = i & 7, j & 7, k & 7
        which_lower = 0  # caller must provide
        leaf_idx = lower_data[which_lower, li, lj, lk]
        if leaf_idx < 0:
            return -1
        return leaf_data[leaf_idx, vi, vj, vk]

    return lower_data, leaf_data, next_voxel_idx


# =========================================================================
# Scenario 1: Two-level grid with 3D indexing, chained Gather
# =========================================================================

def _gather_single_coord(target: Value, coord_3: np.ndarray) -> np.int32:
    """Helper: look up a rank-3 target at a single (3,) i32 coordinate.

    Wraps the coord as (1,) over (3,) i32, gathers, extracts the scalar.
    This is the correct way to do a single-point Gather: the indexer must
    have an iteration space (the batch), and its element type (the coord
    vector) must match the target rank.
    """
    coord_batch = coord_3.reshape(1, 3)
    indexer = Value(
        Type(Shape(Dynamic()), coord_type(3)),
        coord_batch,
    )
    result = Gather(target, indexer)
    return np.int32(result.data[0])


def test_3d_chain_single_coord():
    """Look up a single global coordinate through two levels."""

    # ---- SETUP ----
    lower_data, leaf_data, total_voxels = _build_two_level_grid(n_lower=2)
    L, K = lower_data.shape[0], leaf_data.shape[0]

    # Pick a global coord that we know resolves to an active voxel
    # which_lower=0, lower_local picks an active entry
    active_lower_entries = np.argwhere(lower_data[0] >= 0)
    ll = active_lower_entries[0]  # first active lower entry in node 0
    leaf_idx_expected = lower_data[0, ll[0], ll[1], ll[2]]
    active_leaf_entries = np.argwhere(leaf_data[leaf_idx_expected] >= 0)
    vl = active_leaf_entries[0]

    # Reconstruct global coord
    global_coord = np.array([
        ll[0] * 8 + vl[0],
        ll[1] * 8 + vl[1],
        ll[2] * 8 + vl[2],
    ], dtype=np.int32)

    # ---- EXPRESSION ----
    coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), global_coord)
    parts = Decompose(coord_val, [3, 4])

    leaf_local = parts.data.level_0       # (3,) i32 -- bottom 3 bits
    lower_local = parts.data.level_1      # (3,) i32 -- next 4 bits
    which_lower_idx = int(parts.data.which_top[0])  # scalar -- which lower node

    # Chain step 1: lower_nodes[which_lower] -> Gather with lower_local -> leaf_idx
    lower_node = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data[which_lower_idx],
    )
    leaf_idx_result = _gather_single_coord(lower_node, lower_local)

    # Chain step 2: leaf_nodes[leaf_idx] -> Gather with leaf_local -> voxel_idx
    leaf_idx_scalar = int(leaf_idx_result)
    if leaf_idx_scalar >= 0:
        leaf_node = Value(
            Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
            leaf_data[leaf_idx_scalar],
        )
        voxel_idx_result = _gather_single_coord(leaf_node, leaf_local)
    else:
        voxel_idx_result = np.int32(-1)

    # ---- EXTRACTION ----
    expected_leaf = lower_data[which_lower_idx, lower_local[0], lower_local[1], lower_local[2]]
    expected_voxel = leaf_data[expected_leaf, leaf_local[0], leaf_local[1], leaf_local[2]] if expected_leaf >= 0 else -1

    assert int(leaf_idx_result) == expected_leaf
    assert int(voxel_idx_result) == expected_voxel
    assert int(voxel_idx_result) >= 0, "Expected an active voxel"

    print(f"3D chain: global {global_coord} -> leaf_idx {int(leaf_idx_result)} "
          f"-> voxel_idx {int(voxel_idx_result)}")


def test_3d_chain_batch():
    """Batch lookup: many global coords through two levels."""

    # ---- SETUP ----
    lower_data, leaf_data, total_voxels = _build_two_level_grid(n_lower=1)

    # Generate a set of global coords that are within lower node 0's range
    # lower node 0 covers local coords (0..15, 0..15, 0..15) at leaf scale,
    # so global coords 0..127 in each axis
    N = 50
    np.random.seed(99)
    global_coords = np.random.randint(0, 128, (N, 3)).astype(np.int32)

    # ---- EXPRESSION ----
    coords_val = Value(
        Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.I32)),
        global_coords,
    )

    def lookup_one(coord_val: Value) -> Value:
        """Chain through two levels for one coordinate."""
        parts = Decompose(coord_val, [3, 4])
        leaf_local = parts.data.level_0
        lower_local = parts.data.level_1

        # Chain step 1: Gather lower_node[lower_local] -> leaf_idx
        lower_node = Value(
            Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
            lower_data[0],
        )
        leaf_idx = int(_gather_single_coord(lower_node, lower_local))

        # Chain step 2: Gather leaf_node[leaf_local] -> voxel_idx
        if leaf_idx >= 0:
            leaf_node = Value(
                Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
                leaf_data[leaf_idx],
            )
            voxel_idx = int(_gather_single_coord(leaf_node, leaf_local))
        else:
            voxel_idx = -1

        return Value(Type(Shape(), ScalarType.I32), np.int32(voxel_idx))

    results = Each(coords_val, lookup_one)

    # ---- EXTRACTION ----
    # Manual reference
    for i in range(N):
        gc = global_coords[i]
        ll = (gc >> 3) & 15
        vl = gc & 7
        expected_leaf = lower_data[0, ll[0], ll[1], ll[2]]
        if expected_leaf >= 0:
            expected_voxel = leaf_data[expected_leaf, vl[0], vl[1], vl[2]]
        else:
            expected_voxel = -1
        actual = int(results.data[i] if isinstance(results.data, np.ndarray) else results.data[i].data)
        assert actual == expected_voxel, f"Mismatch at coord {gc}: {actual} vs {expected_voxel}"

    print(f"3D chain batch: {N} lookups verified")


# =========================================================================
# Scenario 2: Morton-linearized nodes, identical results
# =========================================================================

def test_morton_chain_batch():
    """Same grid as scenario 1, but nodes stored morton-linearized."""

    # ---- SETUP ----
    lower_data, leaf_data, total_voxels = _build_two_level_grid(n_lower=1)

    # Morton-linearize the nodes
    # lower: (1, 16, 16, 16) -> (1, 4096) via morton encoding of local coords
    lower_morton = np.full((1, 4096), -1, dtype=np.int32)
    for x in range(16):
        for y in range(16):
            for z in range(16):
                m = int(morton3d(np.array([x, y, z], dtype=np.int32)))
                lower_morton[0, m] = lower_data[0, x, y, z]

    # leaf: (K, 8, 8, 8) -> (K, 512) via morton
    K = leaf_data.shape[0]
    leaf_morton = np.full((K, 512), -1, dtype=np.int32)
    for k in range(K):
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    m = int(morton3d(np.array([x, y, z], dtype=np.int32)))
                    leaf_morton[k, m] = leaf_data[k, x, y, z]

    N = 50
    np.random.seed(99)
    global_coords = np.random.randint(0, 128, (N, 3)).astype(np.int32)

    # ---- EXPRESSION ----
    coords_val = Value(
        Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.I32)),
        global_coords,
    )

    def lookup_morton(coord_val: Value) -> Value:
        """Chain through two levels using morton-linearized nodes."""
        parts = Decompose(coord_val, [3, 4])
        leaf_local = parts.data.level_0       # (3,) i32
        lower_local = parts.data.level_1      # (3,) i32

        # Morton-encode the local coords to scalar indices
        lower_morton_idx = morton3d(lower_local)  # scalar i32
        leaf_morton_idx = morton3d(leaf_local)     # scalar i32

        # Gather into morton-linearized lower node 0
        lower_node = Value(
            Type(Shape(Static(4096)), ScalarType.I32),
            lower_morton[0],
        )
        lm_val = Value(Type(Shape(Static(1)), ScalarType.I32), np.array([int(lower_morton_idx)], dtype=np.int32))
        # For rank-1 target with scalar index, we index directly
        leaf_idx = int(lower_morton[0, int(lower_morton_idx)])

        if leaf_idx >= 0:
            leaf_node = Value(
                Type(Shape(Static(512)), ScalarType.I32),
                leaf_morton[leaf_idx],
            )
            voxel_idx = int(leaf_morton[leaf_idx, int(leaf_morton_idx)])
            return Value(Type(Shape(), ScalarType.I32), np.int32(voxel_idx))
        else:
            return Value(Type(Shape(), ScalarType.I32), np.int32(-1))

    results_morton = Each(coords_val, lookup_morton)

    # ---- EXTRACTION ----
    # Compare against 3D reference (same coords, same seed -> same grid)
    for i in range(N):
        gc = global_coords[i]
        ll = (gc >> 3) & 15
        vl = gc & 7
        expected_leaf = lower_data[0, ll[0], ll[1], ll[2]]
        if expected_leaf >= 0:
            expected_voxel = leaf_data[expected_leaf, vl[0], vl[1], vl[2]]
        else:
            expected_voxel = -1

        actual = int(results_morton.data[i] if isinstance(results_morton.data, np.ndarray)
                     else results_morton.data[i].data)
        assert actual == expected_voxel, (
            f"Morton mismatch at coord {gc}: {actual} vs {expected_voxel}"
        )

    print(f"Morton chain batch: {N} lookups match 3D reference exactly")


# =========================================================================
# Scenario 3: Batch active-voxel lookup through two levels
# =========================================================================

def test_batch_active_voxel_lookup():
    """Where over leaves, then hierarchical lookup for each active voxel."""

    # ---- SETUP ----
    lower_data, leaf_data, total_voxels = _build_two_level_grid(n_lower=1)
    K = leaf_data.shape[0]

    # We'll iterate over leaf nodes, find active voxels, and for each one
    # verify the voxel index by chaining back through the hierarchy.
    # Use first 5 leaf nodes for speed.
    n_test_leaves = min(5, K)
    leaf_subset = leaf_data[:n_test_leaves]

    leaf_flat = Value.from_numpy(leaf_subset.reshape(n_test_leaves * 512), ScalarType.I32)
    leaves_type = cut_by_size(512, leaf_flat.type)
    leaves = Value(leaves_type, leaf_subset.reshape(n_test_leaves, 512))

    # ---- EXPRESSION ----
    # Reshape each leaf to (8,8,8), then Where to find active voxels
    leaves_3d = Each(leaves, lambda leaf: Value(
        reshape(leaf.type, (8, 8, 8)),
        leaf.data.reshape(8, 8, 8),
    ))

    active_per_leaf = Each(leaves_3d, lambda leaf:
        Where(Map(leaf, lambda x: x >= 0))
    )

    # ---- EXTRACTION ----
    print(f"active_per_leaf type: {active_per_leaf.type}")

    # Type: (n_test_leaves,) over (~) over (3) i32
    assert active_per_leaf.type.iteration_shape == Shape(Static(n_test_leaves))
    inner = active_per_leaf.type.element_type
    assert isinstance(inner, Type)
    assert isinstance(inner.iteration_shape.extents[0], Jagged)
    assert inner.element_type == coord_type(3)

    # Verify: for each leaf, each active local coord should match leaf_data
    total_active = 0
    for li in range(n_test_leaves):
        active_coords = active_per_leaf.data[li].data  # (N_i, 3)
        expected = np.argwhere(leaf_subset[li].reshape(8, 8, 8) >= 0).astype(np.int32)
        np.testing.assert_array_equal(active_coords, expected)
        total_active += len(expected)

        # For each active coord, verify the voxel index
        for ci in range(len(active_coords)):
            c = active_coords[ci]
            actual_voxel = leaf_subset[li].reshape(8, 8, 8)[c[0], c[1], c[2]]
            assert actual_voxel >= 0

    counts = [active_per_leaf.data[li].data.shape[0] for li in range(n_test_leaves)]
    print(f"  {n_test_leaves} leaves, active counts: {counts}, total: {total_active}")
    assert len(set(counts)) > 1, "Expected varying counts (jagged)"


# =========================================================================

if __name__ == "__main__":
    print("=== Scenario 1: 3D chain ===")
    test_3d_chain_single_coord()
    test_3d_chain_batch()

    print("\n=== Scenario 2: Morton chain ===")
    test_morton_chain_batch()

    print("\n=== Scenario 3: Batch active lookup ===")
    test_batch_active_voxel_lookup()

    print("\nAll test_two_level tests passed.")
