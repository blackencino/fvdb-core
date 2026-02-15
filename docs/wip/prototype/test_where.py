# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
First proof: Where on a single (8,8,8) leaf node.

EXPRESSION phases use DSL strings executed via run().
SETUP creates numpy data + Value inputs.
EXTRACTION validates types and data against numpy references.
"""

import numpy as np

from docs.wip.prototype.dsl_eval import run
from docs.wip.prototype.ops import Value
from docs.wip.prototype.types import Dynamic, ScalarType, Shape, Static, Type, coord_type


# ---------------------------------------------------------------------------
# DSL programs
# ---------------------------------------------------------------------------

MASK_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
mask
"""

WHERE_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
active
"""

GATHER_IDX_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
idx = Gather(Input("leaf"), active)
idx
"""

FULL_PIPELINE_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
idx = Gather(Input("leaf"), active)
feat = Gather(Input("features"), idx)
feat
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_map_preserves_shape():
    # ---- SETUP ----
    leaf = Value.from_numpy(np.zeros((8, 8, 8), dtype=np.int32), ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(MASK_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    assert types["mask"] == Type(Shape(Static(8), Static(8), Static(8)), ScalarType.BOOL)


def test_where_type():
    # ---- SETUP ----
    leaf = Value.from_numpy(np.ones((8, 8, 8), dtype=np.int32), ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(WHERE_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    assert result.type.iteration_shape == Shape(Dynamic())
    assert result.type.element_type == coord_type(3)


def test_where_data():
    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(WHERE_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    expected_coords = np.argwhere(leaf_data >= 0).astype(np.int32)
    np.testing.assert_array_equal(result.data, expected_coords)


def test_gather_active_indices():
    # ---- SETUP ----
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    leaf = Value.from_numpy(leaf_data, ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(GATHER_IDX_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    assert result.type.iteration_shape == Shape(Dynamic())
    assert result.type.element_type == ScalarType.I32
    assert np.all(result.data >= 0)

    expected_coords = np.argwhere(leaf_data >= 0).astype(np.int32)
    expected_vals = leaf_data[tuple(expected_coords[:, i] for i in range(3))]
    np.testing.assert_array_equal(result.data, expected_vals)


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
    types, result = run(FULL_PIPELINE_PROGRAM, {"leaf": leaf, "features": features})

    # ---- EXTRACTION ----
    assert result.type.iteration_shape == Shape(Dynamic())
    assert result.type.element_type == Type(Shape(Static(4)), ScalarType.F32)
    expected_coords = np.argwhere(leaf_data >= 0).astype(np.int32)
    expected_idx = leaf_data[tuple(expected_coords[:, i] for i in range(3))]
    for i in range(len(expected_idx)):
        np.testing.assert_array_equal(result.data[i], features_data[expected_idx[i]])


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
    types, result = run(FULL_PIPELINE_PROGRAM, {"leaf": leaf, "features": features})

    # ---- EXTRACTION ----
    assert types["mask"] == Type(Shape(Static(8), Static(8), Static(8)), ScalarType.BOOL)
    assert types["active"] == Type(Shape(Dynamic()), coord_type(3))

    # Verify counts and values
    expected_count = np.sum(leaf_data >= 0)
    expected_coords = np.argwhere(leaf_data >= 0).astype(np.int32)
    expected_idx = leaf_data[tuple(expected_coords[:, i] for i in range(3))]
    assert result.data.shape[0] == expected_count

    print(f"Pipeline OK: {expected_count} active voxels, "
          f"features shape {result.data.shape}")


if __name__ == "__main__":
    test_map_preserves_shape()
    test_where_type()
    test_where_data()
    test_gather_active_indices()
    test_gather_features()
    test_full_pipeline()
    print("\nAll test_where tests passed.")
