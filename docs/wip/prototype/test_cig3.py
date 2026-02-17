# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the 3-level Compressed CIG with root lookup.

Verifies:
  1. build_compressed_cig3 produces correct structure
  2. root_lookup finds upper nodes correctly
  3. Full ijk_to_index chain via numpy reference
  4. DSL evaluator matches numpy reference
"""

import numpy as np
import torch

from docs.wip.prototype.cig import CompressedCIG3, build_compressed_cig3, root_lookup
from docs.wip.prototype.dsl_eval import run as dsl_run
from docs.wip.prototype.ops import Value
from docs.wip.prototype.types import Dynamic, ScalarType, Shape, Static, Type


# ---------------------------------------------------------------------------
# Numpy reference: 3-level ijk_to_index
# ---------------------------------------------------------------------------


def cig3_ijk_to_index_numpy(cig: CompressedCIG3, query_np: np.ndarray) -> np.ndarray:
    """Numpy reference ijk_to_index for 3-level CIG."""
    root_np = cig.root_coords.cpu().numpy()
    u_masks = cig.upper_masks.cpu().numpy()
    u_prefix = cig.upper_prefix.cpu().numpy()
    u_offsets = cig.upper_offsets.cpu().numpy()
    l_masks = cig.lower_masks.cpu().numpy()
    l_prefix = cig.lower_prefix.cpu().numpy()
    l_offsets = cig.lower_offsets.cpu().numpy()
    k_masks = cig.leaf_masks.cpu().numpy()
    k_prefix = cig.leaf_prefix.cpu().numpy()
    k_offsets = cig.leaf_offsets.cpu().numpy()

    N = query_np.shape[0]
    result = np.full(N, -1, dtype=np.int32)

    for i in range(N):
        coord = query_np[i]
        wt = coord >> 12
        ul = (coord >> 7) & 31
        ll = (coord >> 3) & 15
        fl = coord & 7

        # Root: linear scan for matching which_top
        upper_idx = -1
        for r in range(root_np.shape[0]):
            if np.array_equal(root_np[r], wt):
                upper_idx = r
                break
        if upper_idx < 0:
            continue

        # Upper -> lower
        flat_ul = int(ul[0]) * 1024 + int(ul[1]) * 32 + int(ul[2])
        lower_idx = _masked_lookup(u_masks[upper_idx], u_prefix[upper_idx], int(u_offsets[upper_idx]), flat_ul)
        if lower_idx < 0:
            continue

        # Lower -> leaf
        flat_ll = int(ll[0]) * 256 + int(ll[1]) * 16 + int(ll[2])
        leaf_idx = _masked_lookup(l_masks[lower_idx], l_prefix[lower_idx], int(l_offsets[lower_idx]), flat_ll)
        if leaf_idx < 0:
            continue

        # Leaf -> voxel
        flat_fl = int(fl[0]) * 64 + int(fl[1]) * 8 + int(fl[2])
        voxel_idx = _masked_lookup(k_masks[leaf_idx], k_prefix[leaf_idx], int(k_offsets[leaf_idx]), flat_fl)
        result[i] = voxel_idx

    return result


def _masked_lookup(mask_words, prefix, base_offset, flat_idx):
    """Single masked lookup: check bitmask + prefix popcount."""
    word_idx = flat_idx >> 6
    bit_pos = flat_idx & 63
    if word_idx < 0 or word_idx >= len(mask_words):
        return -1
    word = int(mask_words[word_idx])
    if not ((word >> bit_pos) & 1):
        return -1
    cum = int(prefix[word_idx])
    partial = bin(word & ((1 << bit_pos) - 1) & 0xFFFFFFFFFFFFFFFF).count("1")
    return base_offset + cum + partial


# ---------------------------------------------------------------------------
# Test 1: Builder structure
# ---------------------------------------------------------------------------


def test_build_cig3():
    """Build a 3-level CIG and check structural invariants."""
    rng = np.random.RandomState(42)
    # Coordinates spanning multiple upper nodes (different which_top values)
    coords = []
    for _ in range(500):
        x = rng.randint(0, 4096)
        y = rng.randint(0, 4096)
        z = rng.randint(0, 4096)
        coords.append([x, y, z])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))

    cig = build_compressed_cig3(ijk)

    assert cig.root_coords.shape[1] == 3
    assert cig.n_upper == cig.root_coords.shape[0]
    assert cig.upper_masks.shape == (cig.n_upper, 512)
    assert cig.upper_prefix.shape == (cig.n_upper, 512)
    assert cig.lower_masks.shape == (cig.n_lower, 64)
    assert cig.lower_prefix.shape == (cig.n_lower, 64)
    assert cig.leaf_masks.shape == (cig.n_leaves, 8)
    assert cig.leaf_prefix.shape == (cig.n_leaves, 8)

    print(
        f"  build_cig3: {cig.n_active} voxels, {cig.n_leaves} leaves, "
        f"{cig.n_lower} lower, {cig.n_upper} upper, "
        f"{cig.num_bytes / 1024:.1f} KB -- PASSED"
    )


# ---------------------------------------------------------------------------
# Test 2: Root lookup
# ---------------------------------------------------------------------------


def test_root_lookup():
    """Root linear scan finds correct upper node indices."""
    rng = np.random.RandomState(42)
    coords = []
    for _ in range(200):
        coords.append([rng.randint(0, 4096), rng.randint(0, 4096), rng.randint(0, 4096)])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))
    cig = build_compressed_cig3(ijk)

    # Query with grid coordinates (all should match)
    upper_idx = root_lookup(cig.root_coords, ijk)
    assert (upper_idx >= 0).all(), "All grid coords should find a root match"

    # Query with out-of-range coordinates (should return -1)
    bad_queries = torch.full((10, 3), 999999, dtype=torch.int32)
    bad_idx = root_lookup(cig.root_coords, bad_queries)
    assert (bad_idx == -1).all(), "Out-of-range should return -1"

    n_upper = cig.n_upper
    print(f"  root_lookup: {n_upper} upper nodes, all grid hits found, OOB returns -1 -- PASSED")


# ---------------------------------------------------------------------------
# Test 3: Numpy reference ijk_to_index
# ---------------------------------------------------------------------------


def test_cig3_numpy_reference():
    """Numpy reference produces correct results for known coordinates."""
    rng = np.random.RandomState(77)
    coords = []
    for _ in range(300):
        coords.append([rng.randint(0, 4096), rng.randint(0, 4096), rng.randint(0, 4096)])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))
    cig = build_compressed_cig3(ijk)

    query_np = ijk.numpy()
    result = cig3_ijk_to_index_numpy(cig, query_np)

    # All grid coordinates should be active (return >= 0)
    n_active = (result >= 0).sum()
    assert n_active == cig.n_active, f"Expected {cig.n_active} active, got {n_active}"

    # Indices should be unique and in [0, n_active)
    active_indices = result[result >= 0]
    assert len(set(active_indices)) == cig.n_active, "Indices should be unique"
    assert active_indices.min() >= 0
    assert active_indices.max() < cig.n_active

    # Random misses
    miss_coords = np.array([[4095, 4095, 4095]], dtype=np.int32)
    miss_result = cig3_ijk_to_index_numpy(cig, miss_coords)
    # This might or might not be in the grid -- just verify it's deterministic
    print(f"  cig3_numpy_ref: {cig.n_active} voxels, all found, indices unique -- PASSED")


# ---------------------------------------------------------------------------
# Test 4: DSL evaluator for one level
# ---------------------------------------------------------------------------

MASKED_CIG3_LEAF_PROGRAM = """
parts = Decompose(Input("query"), Const([3, 4, 5]))
leaf = masked(Gather(Input("leaf_masks"), Input("leaf_idx")), Gather(Input("leaf_prefix"), Input("leaf_idx")), Gather(Input("leaf_offsets"), Input("leaf_idx")))
voxel_idx = Gather(leaf, field(parts, "level_0"))
voxel_idx
"""


def test_cig3_dsl_leaf_level():
    """DSL evaluator handles a single masked leaf level correctly."""
    rng = np.random.RandomState(99)
    coords = []
    for _ in range(200):
        coords.append([rng.randint(0, 4096), rng.randint(0, 4096), rng.randint(0, 4096)])
    ijk = torch.from_numpy(np.array(coords, dtype=np.int32))
    cig = build_compressed_cig3(ijk)

    # Pick a known active voxel and resolve its leaf manually
    query_np = ijk[0].numpy()
    fl = query_np & 7
    wt = query_np >> 12
    ul = (query_np >> 7) & 31
    ll = (query_np >> 3) & 15

    # Find upper, lower, leaf indices via numpy reference
    ref_result = cig3_ijk_to_index_numpy(cig, query_np.reshape(1, 3))
    assert ref_result[0] >= 0, "First grid coord should be active"

    # Now test just the leaf level via DSL (given a known leaf_idx)
    # We need to find the leaf_idx for this voxel
    flat_ul = int(ul[0]) * 1024 + int(ul[1]) * 32 + int(ul[2])
    flat_ll = int(ll[0]) * 256 + int(ll[1]) * 16 + int(ll[2])

    # Find upper_idx
    root_np = cig.root_coords.cpu().numpy()
    upper_idx = -1
    for r in range(root_np.shape[0]):
        if np.array_equal(root_np[r], wt):
            upper_idx = r
            break
    assert upper_idx >= 0

    lower_idx = _masked_lookup(
        cig.upper_masks.cpu().numpy()[upper_idx],
        cig.upper_prefix.cpu().numpy()[upper_idx],
        int(cig.upper_offsets.cpu().numpy()[upper_idx]),
        flat_ul,
    )
    assert lower_idx >= 0

    leaf_idx = _masked_lookup(
        cig.lower_masks.cpu().numpy()[lower_idx],
        cig.lower_prefix.cpu().numpy()[lower_idx],
        int(cig.lower_offsets.cpu().numpy()[lower_idx]),
        flat_ll,
    )
    assert leaf_idx >= 0

    # Run DSL with the known leaf_idx
    query_val = Value(Type(Shape(Static(3)), ScalarType.I32), query_np)
    leaf_idx_val = Value(Type(Shape(), ScalarType.I32), np.int32(leaf_idx))
    masks_val = Value(
        Type(Shape(Static(cig.n_leaves)), Type(Shape(Static(8)), ScalarType.I64)),
        cig.leaf_masks.cpu().numpy(),
    )
    prefix_val = Value(
        Type(Shape(Static(cig.n_leaves)), Type(Shape(Static(8)), ScalarType.I32)),
        cig.leaf_prefix.cpu().numpy(),
    )
    offsets_val = Value(
        Type(Shape(Static(cig.n_leaves)), ScalarType.I64),
        cig.leaf_offsets.cpu().numpy(),
    )

    _, result = dsl_run(MASKED_CIG3_LEAF_PROGRAM, {
        "query": query_val,
        "leaf_idx": leaf_idx_val,
        "leaf_masks": masks_val,
        "leaf_prefix": prefix_val,
        "leaf_offsets": offsets_val,
    })

    dsl_idx = int(result.data)
    assert dsl_idx == ref_result[0], f"DSL {dsl_idx} != ref {ref_result[0]}"
    print(f"  cig3_dsl_leaf: voxel {query_np} -> idx {dsl_idx}, matches numpy ref -- PASSED")


# =========================================================================

if __name__ == "__main__":
    print("=== 3-level CIG tests ===")
    test_build_cig3()
    print()
    test_root_lookup()
    print()
    test_cig3_numpy_reference()
    print()
    test_cig3_dsl_leaf_level()
    print("\nAll 3-level CIG tests passed.")
