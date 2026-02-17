# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for Sort/Unique DSL primitives.

Focus:
  - value semantics (inputs are not mutated)
  - referential transparency (same input -> same output)
  - type behavior (Unique has Dynamic leading extent)
"""

import torch

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Static, Type


SORT_UNIQUE_PROGRAM = """
sorted = Sort(Input("coords"))
unique = Unique(sorted)
unique
"""


def test_sort_unique_coords_correctness_and_types():
    coords_t = torch.tensor(
        [
            [2, 1, 0],
            [0, 0, 1],
            [2, 1, 0],
            [1, 3, 2],
            [0, 0, 1],
        ],
        dtype=torch.int32,
    )
    coords = Value(
        Type(Shape(Static(coords_t.shape[0])), Type(Shape(Static(3)), ScalarType.I32)),
        coords_t.clone(),
    )

    types, out = run(SORT_UNIQUE_PROGRAM, {"coords": coords})
    assert types["sorted"] == Type(Shape(Static(coords_t.shape[0])), Type(Shape(Static(3)), ScalarType.I32))
    assert types["unique"] == Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.I32))

    expected = torch.tensor(
        [
            [0, 0, 1],
            [1, 3, 2],
            [2, 1, 0],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(out.data, expected, atol=0, rtol=0)


def test_sort_unique_value_semantics_and_referential_transparency():
    coords_t = torch.tensor([[3, 0, 0], [1, 0, 0], [3, 0, 0], [2, 0, 0]], dtype=torch.int32)
    before = coords_t.clone()
    coords = Value(
        Type(Shape(Static(coords_t.shape[0])), Type(Shape(Static(3)), ScalarType.I32)),
        coords_t,
    )

    _, out_a = run(SORT_UNIQUE_PROGRAM, {"coords": coords})
    _, out_b = run(SORT_UNIQUE_PROGRAM, {"coords": coords})

    # Input array remains unchanged (immutability at DSL operation boundary).
    torch.testing.assert_close(coords_t, before, atol=0, rtol=0)
    # Referential transparency: repeat run gives identical output.
    torch.testing.assert_close(out_a.data, out_b.data, atol=0, rtol=0)


def test_unique_idempotence_law():
    program_once = """
u = Unique(Input("x"))
u
"""
    program_twice = """
u1 = Unique(Input("x"))
u2 = Unique(u1)
u2
"""
    x_t = torch.tensor([5, 1, 5, 2, 1, 2, 9], dtype=torch.int32)
    x_val = Value(Type(Shape(Static(x_t.shape[0])), ScalarType.I32), x_t.clone())
    _, once = run(program_once, {"x": x_val})
    _, twice = run(program_twice, {"x": x_val})
    torch.testing.assert_close(once.data, twice.data, atol=0, rtol=0)


def test_sort_preserves_multiset():
    x_t = torch.tensor([4, 1, 4, 7, 1, 3, 3, 3], dtype=torch.int32)
    x_val = Value(Type(Shape(Static(x_t.shape[0])), ScalarType.I32), x_t.clone())
    _, out = run('y = Sort(Input("x"))\ny', {"x": x_val})

    in_vals, in_counts = torch.unique(x_t, return_counts=True)
    out_vals, out_counts = torch.unique(out.data, return_counts=True)
    torch.testing.assert_close(in_vals, out_vals, atol=0, rtol=0)
    torch.testing.assert_close(in_counts, out_counts, atol=0, rtol=0)
