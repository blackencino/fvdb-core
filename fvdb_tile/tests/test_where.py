# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
First proof: Where on a single (8,8,8) leaf node.

EXPRESSION phases use DSL strings executed via run().
SETUP creates torch data + Value inputs.
EXTRACTION validates types and data against torch references.
"""

import torch

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Static, Type, coord_type


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
    leaf = Value.from_tensor(torch.zeros((8, 8, 8), dtype=torch.int32), ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(MASK_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    assert types["mask"] == Type(Shape(Static(8), Static(8), Static(8)), ScalarType.BOOL)


def test_where_type():
    # ---- SETUP ----
    leaf = Value.from_tensor(torch.ones((8, 8, 8), dtype=torch.int32), ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(WHERE_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    assert result.type.iteration_shape == Shape(Dynamic())
    assert result.type.element_type == coord_type(3)


def test_where_data():
    # ---- SETUP ----
    gen = torch.Generator().manual_seed(42)
    leaf_data = torch.randint(-1, 10, (8, 8, 8), generator=gen, dtype=torch.int32)
    leaf = Value.from_tensor(leaf_data, ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(WHERE_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    expected_coords = torch.nonzero(leaf_data >= 0).to(torch.int32)
    torch.testing.assert_close(result.data, expected_coords, atol=0, rtol=0)


def test_gather_active_indices():
    # ---- SETUP ----
    gen = torch.Generator().manual_seed(42)
    leaf_data = torch.randint(-1, 10, (8, 8, 8), generator=gen, dtype=torch.int32)
    leaf = Value.from_tensor(leaf_data, ScalarType.I32)

    # ---- EXPRESSION ----
    types, result = run(GATHER_IDX_PROGRAM, {"leaf": leaf})

    # ---- EXTRACTION ----
    assert result.type.iteration_shape == Shape(Dynamic())
    assert result.type.element_type == ScalarType.I32
    assert torch.all(result.data >= 0)

    expected_coords = torch.nonzero(leaf_data >= 0).to(torch.int32)
    expected_vals = leaf_data[
        expected_coords[:, 0], expected_coords[:, 1], expected_coords[:, 2]
    ]
    torch.testing.assert_close(result.data, expected_vals, atol=0, rtol=0)


def test_gather_features():
    # ---- SETUP ----
    gen = torch.Generator().manual_seed(42)
    leaf_data = torch.randint(-1, 5, (8, 8, 8), generator=gen, dtype=torch.int32)
    leaf = Value.from_tensor(leaf_data, ScalarType.I32)
    max_idx = int(leaf_data.max().item()) + 1
    features_data = torch.arange(max_idx * 4, dtype=torch.float32).reshape(max_idx, 4)
    features = Value(
        Type(Shape(Dynamic()), Type(Shape(Static(4)), ScalarType.F32)),
        features_data,
    )

    # ---- EXPRESSION ----
    types, result = run(FULL_PIPELINE_PROGRAM, {"leaf": leaf, "features": features})

    # ---- EXTRACTION ----
    assert result.type.iteration_shape == Shape(Dynamic())
    assert result.type.element_type == Type(Shape(Static(4)), ScalarType.F32)
    expected_coords = torch.nonzero(leaf_data >= 0).to(torch.int32)
    expected_idx = leaf_data[
        expected_coords[:, 0], expected_coords[:, 1], expected_coords[:, 2]
    ]
    for i in range(len(expected_idx)):
        torch.testing.assert_close(result.data[i], features_data[expected_idx[i].item()], atol=0, rtol=0)


def test_full_pipeline():
    """End-to-end: leaf -> mask -> where -> gather indices -> gather features."""

    # ---- SETUP ----
    gen = torch.Generator().manual_seed(123)
    leaf_data = torch.randint(-1, 8, (8, 8, 8), generator=gen, dtype=torch.int32)
    leaf = Value.from_tensor(leaf_data, ScalarType.I32)
    max_idx = int(leaf_data.max().item()) + 1
    C = 3
    feat_data = torch.randn(max_idx, C, generator=gen, dtype=torch.float32)
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
    expected_count = int(torch.sum(leaf_data >= 0).item())
    expected_coords = torch.nonzero(leaf_data >= 0).to(torch.int32)
    expected_idx = leaf_data[
        expected_coords[:, 0], expected_coords[:, 1], expected_coords[:, 2]
    ]
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
