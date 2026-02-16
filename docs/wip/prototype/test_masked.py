# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the masked layout: bitmask + popcount sparse access.

Verifies:
  1. MaskedNode constructs correctly in the DSL
  2. Gather through a masked layout uses bitmask check + popcount
  3. Active positions return correct dense indices
  4. Inactive positions return -1
  5. The full CIG ijk_to_index chain works with masked leaves
"""

import numpy as np

from docs.wip.prototype.dsl_eval import run
from docs.wip.prototype.ops import Value
from docs.wip.prototype.types import ScalarType, Shape, Static, Type


def _pack_mask(active_positions):
    """Pack a list of flat positions (0-511) into an (8,) i64 bitmask."""
    words = np.zeros(8, dtype=np.int64)
    for pos in active_positions:
        word_idx = pos >> 6
        bit_pos = pos & 63
        words[word_idx] |= np.int64(1) << np.int64(bit_pos)
    return words


def _popcount_before(words, flat_idx):
    """Reference popcount: count set bits before flat_idx."""
    word_idx = flat_idx >> 6
    bit_pos = flat_idx & 63
    total = 0
    for w in range(word_idx):
        total += bin(int(words[w]) & 0xFFFFFFFFFFFFFFFF).count("1")
    partial = int(words[word_idx]) & ((1 << bit_pos) - 1)
    total += bin(partial & 0xFFFFFFFFFFFFFFFF).count("1")
    return total


# ---------------------------------------------------------------------------
# Test 1: Basic masked Gather
# ---------------------------------------------------------------------------


MASKED_GATHER_PROGRAM = """
leaf = masked(Input("leaf_mask"), Input("leaf_offset"))
idx = Gather(leaf, Input("coord"))
idx
"""


def test_masked_gather_active():
    """Gather from a masked layout at an active position."""
    # Create a leaf with specific active positions
    active_positions = [0, 1, 5, 64, 65, 100, 200, 300, 400, 511]
    mask = _pack_mask(active_positions)
    base_offset = np.int64(1000)

    mask_val = Value(Type(Shape(Static(8)), ScalarType.I64), mask)
    offset_val = Value(Type(Shape(), ScalarType.I64), base_offset)

    # Query position 100: flat_idx = ?
    # pos 100 in (8,8,8): x=1, y=4, z=4 (100 = 1*64 + 4*8 + 4)
    coord = np.array([1, 4, 4], dtype=np.int32)
    coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), coord)

    _, result = run(MASKED_GATHER_PROGRAM, {
        "leaf_mask": mask_val,
        "leaf_offset": offset_val,
        "coord": coord_val,
    })

    # Position 100 is active. Popcount before 100:
    # positions < 100 that are active: 0, 1, 5, 64, 65 -> 5 active
    expected = int(base_offset) + _popcount_before(mask, 100)
    assert expected == 1000 + 5
    actual = int(result.data)
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"  masked_gather_active: coord (1,4,4) -> index {actual} -- PASSED")


def test_masked_gather_inactive():
    """Gather from a masked layout at an inactive position returns -1."""
    active_positions = [0, 100, 511]
    mask = _pack_mask(active_positions)
    base_offset = np.int64(0)

    mask_val = Value(Type(Shape(Static(8)), ScalarType.I64), mask)
    offset_val = Value(Type(Shape(), ScalarType.I64), base_offset)

    # Query position 50: x=0, y=6, z=2 (50 = 0*64 + 6*8 + 2) -- NOT active
    coord = np.array([0, 6, 2], dtype=np.int32)
    coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), coord)

    _, result = run(MASKED_GATHER_PROGRAM, {
        "leaf_mask": mask_val,
        "leaf_offset": offset_val,
        "coord": coord_val,
    })

    actual = int(result.data)
    assert actual == -1, f"Expected -1 for inactive position, got {actual}"
    print(f"  masked_gather_inactive: coord (0,6,2) -> -1 -- PASSED")


# ---------------------------------------------------------------------------
# Test 2: Full CIG chain with masked leaves
# ---------------------------------------------------------------------------

MASKED_CIG_PROGRAM = """
parts = Decompose(Input("query"), Const([3, 4]))
leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
leaf = masked(Gather(Input("leaf_masks"), leaf_idx), Gather(Input("leaf_offsets"), leaf_idx))
voxel_idx = Gather(leaf, field(parts, "level_0"))
voxel_idx
"""


def test_masked_cig_chain():
    """Full ijk_to_index with masked leaves via DSL."""
    # Build a tiny 2-level grid with 2 leaves
    lower_data = np.full((16, 16, 16), -1, dtype=np.int32)
    lower_data[3, 4, 5] = 0  # leaf 0
    lower_data[3, 4, 6] = 1  # leaf 1

    # Leaf 0: active at local positions (0,0,0), (1,2,3), (7,7,7)
    leaf0_positions = [0 * 64 + 0 * 8 + 0, 1 * 64 + 2 * 8 + 3, 7 * 64 + 7 * 8 + 7]
    mask0 = _pack_mask(leaf0_positions)

    # Leaf 1: active at local positions (0,0,0), (0,0,1)
    leaf1_positions = [0, 1]
    mask1 = _pack_mask(leaf1_positions)

    leaf_masks = np.stack([mask0, mask1])  # (2, 8) i64
    leaf_offsets = np.array([0, 3], dtype=np.int64)  # leaf 0 starts at 0, leaf 1 at 3

    # Type declarations
    lower_val = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data,
    )
    masks_val = Value(
        Type(Shape(Static(2)), Type(Shape(Static(8)), ScalarType.I64)),
        leaf_masks,
    )
    offsets_val = Value(
        Type(Shape(Static(2)), ScalarType.I64),
        leaf_offsets,
    )

    # Test 1: Query an active voxel in leaf 0
    # Global coord for lower (3,4,5), local (1,2,3): global = (3*8+1, 4*8+2, 5*8+3) = (25, 34, 43)
    query = np.array([25, 34, 43], dtype=np.int32)
    query_val = Value(Type(Shape(Static(3)), ScalarType.I32), query)

    _, result = run(MASKED_CIG_PROGRAM, {
        "query": query_val,
        "lower": lower_val,
        "leaf_masks": masks_val,
        "leaf_offsets": offsets_val,
    })

    # local (1,2,3) = flat 75. Active positions before 75 in leaf 0: position 0. So index = 0 + 1 = 1.
    expected = 0 + _popcount_before(mask0, 1 * 64 + 2 * 8 + 3)
    actual = int(result.data)
    assert actual == expected, f"Expected {expected}, got {actual}"

    # Test 2: Query an active voxel in leaf 1
    # Global coord for lower (3,4,6), local (0,0,1): global = (24, 32, 49)
    query2 = np.array([24, 32, 49], dtype=np.int32)
    query_val2 = Value(Type(Shape(Static(3)), ScalarType.I32), query2)

    _, result2 = run(MASKED_CIG_PROGRAM, {
        "query": query_val2,
        "lower": lower_val,
        "leaf_masks": masks_val,
        "leaf_offsets": offsets_val,
    })

    # local (0,0,1) = flat 1. Active positions before 1 in leaf 1: position 0. So index = 3 + 1 = 4.
    expected2 = 3 + _popcount_before(mask1, 1)
    actual2 = int(result2.data)
    assert actual2 == expected2, f"Expected {expected2}, got {actual2}"

    # Test 3: Query an inactive position
    # Global coord for lower (3,4,5), local (5,5,5): global = (29, 37, 45)
    query3 = np.array([29, 37, 45], dtype=np.int32)
    query_val3 = Value(Type(Shape(Static(3)), ScalarType.I32), query3)

    _, result3 = run(MASKED_CIG_PROGRAM, {
        "query": query_val3,
        "lower": lower_val,
        "leaf_masks": masks_val,
        "leaf_offsets": offsets_val,
    })

    actual3 = int(result3.data)
    assert actual3 == -1, f"Expected -1 for inactive voxel, got {actual3}"

    # Test 4: Query a coord in an empty lower node
    query4 = np.array([0, 0, 0], dtype=np.int32)
    query_val4 = Value(Type(Shape(Static(3)), ScalarType.I32), query4)

    _, result4 = run(MASKED_CIG_PROGRAM, {
        "query": query_val4,
        "lower": lower_val,
        "leaf_masks": masks_val,
        "leaf_offsets": offsets_val,
    })

    actual4 = int(result4.data)
    assert actual4 == -1, f"Expected -1 for empty lower node, got {actual4}"

    print(
        f"  masked_cig_chain: active leaf0={actual}, active leaf1={actual2}, "
        f"inactive={actual3}, empty_lower={actual4} -- PASSED"
    )


# =========================================================================

if __name__ == "__main__":
    print("=== Masked layout tests ===")
    test_masked_gather_active()
    test_masked_gather_inactive()
    test_masked_cig_chain()
    print("\nAll masked layout tests passed.")
