# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
First proof: Where on a single (8,8,8) leaf node.

Each test is structured in three phases:
  1. SETUP         -- create numpy arrays, wrap as typed Values (boilerplate)
  2. EXPRESSION    -- compose operations; this IS the algebra under test
  3. EXTRACTION    -- pull results back to numpy, compare against reference

Validates both type propagation and numerical correctness.
"""

import numpy as np

from docs.wip.prototype.layouts import cut_by_size, indexed, reshape
from docs.wip.prototype.ops import Gather, Map, Value, Where
from docs.wip.prototype.types import Dynamic, ScalarType, Shape, Static, Type, coord_type, tensor_type


def test_map_preserves_shape():
    # ---- SETUP ----
    leaf = Value.from_numpy(np.zeros((8, 8, 8), dtype=np.int32), ScalarType.I32)

    # ---- EXPRESSION ----
    mask = Map(leaf, lambda x: x >= 0)

    # ---- EXTRACTION ----
    assert mask.type.iteration_shape == Shape(Static(8), Static(8), Static(8))
    assert mask.type.element_type == ScalarType.BOOL


def test_where_type():
    # ---- SETUP ----
    leaf = Value.from_numpy(np.ones((8, 8, 8), dtype=np.int32), ScalarType.I32)

    # ---- EXPRESSION ----
    mask = Map(leaf, lambda x: x >= 0)
    active = Where(mask)

    # ---- EXTRACTION ----
    assert active.type.iteration_shape == Shape(Dynamic())
    assert active.type.element_type == coord_type(3)


def test_where_data():
    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    # ---- EXPRESSION ----
    mask = Map(leaf, lambda x: x >= 0)
    active = Where(mask)

    # ---- EXTRACTION ----
    expected_coords = np.argwhere(leaf_data >= 0).astype(np.int32)
    np.testing.assert_array_equal(active.data, expected_coords)


def test_gather_active_indices():
    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    # ---- EXPRESSION ----
    mask = Map(leaf, lambda x: x >= 0)
    active_ijk = Where(mask)
    active_idx = Gather(leaf, active_ijk)

    # ---- EXTRACTION ----
    assert active_idx.type.iteration_shape == Shape(Dynamic())
    assert active_idx.type.element_type == ScalarType.I32
    assert np.all(active_idx.data >= 0)

    expected_coords = np.argwhere(leaf_data >= 0).astype(np.int32)
    expected_vals = leaf_data[tuple(expected_coords[:, i] for i in range(3))]
    np.testing.assert_array_equal(active_idx.data, expected_vals)


def test_gather_features():
    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 5, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)
    max_idx = int(leaf_data.max()) + 1
    features_data = np.arange(max_idx * 4, dtype=np.float32).reshape(max_idx, 4)
    features = Value(
        Type(Shape(Dynamic()), Type(Shape(Static(4)), ScalarType.F32)),
        features_data,
    )

    # ---- EXPRESSION ----
    mask = Map(leaf, lambda x: x >= 0)
    active_ijk = Where(mask)
    active_idx = Gather(leaf, active_ijk)
    active_feat = Gather(features, active_idx)

    # ---- EXTRACTION ----
    assert active_feat.type.iteration_shape == Shape(Dynamic())
    assert active_feat.type.element_type == Type(Shape(Static(4)), ScalarType.F32)
    for i in range(len(active_idx.data)):
        idx = active_idx.data[i]
        np.testing.assert_array_equal(active_feat.data[i], features_data[idx])


def test_full_pipeline():
    """End-to-end: leaf -> mask -> where -> gather indices -> gather features."""

    # ---- SETUP ----
    np.random.seed(123)
    leaf_data = np.random.randint(-1, 8, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)
    max_idx = int(leaf_data.max()) + 1
    C = 3
    feat_data = np.random.randn(max_idx, C).astype(np.float32)
    features = Value(
        Type(Shape(Dynamic()), Type(Shape(Static(C)), ScalarType.F32)),
        feat_data,
    )

    # ---- EXPRESSION ----
    mask = Map(leaf, lambda x: x >= 0)
    active_ijk = Where(mask)
    active_idx = Gather(leaf, active_ijk)
    active_feat = Gather(features, active_idx)

    # ---- EXTRACTION ----
    assert mask.type == Type(Shape(Static(8), Static(8), Static(8)), ScalarType.BOOL)
    assert active_ijk.type.iteration_shape.rank == 1
    assert isinstance(active_ijk.type.iteration_shape.extents[0], Dynamic)
    assert active_ijk.data.shape[0] == np.sum(leaf_data >= 0)
    assert np.all(active_idx.data >= 0)
    assert active_feat.data.shape == (active_idx.data.shape[0], C)

    print(f"Pipeline OK: {active_ijk.data.shape[0]} active voxels, "
          f"features shape {active_feat.data.shape}")


if __name__ == "__main__":
    test_map_preserves_shape()
    test_where_type()
    test_where_data()
    test_gather_active_indices()
    test_gather_features()
    test_full_pipeline()
    print("\nAll test_where tests passed.")
