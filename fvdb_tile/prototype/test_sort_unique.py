# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for Sort/Unique DSL primitives.

Focus:
  - value semantics (inputs are not mutated)
  - referential transparency (same input -> same output)
  - type behavior (Unique has Dynamic leading extent)
"""

import numpy as np

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Static, Type


SORT_UNIQUE_PROGRAM = """
sorted = Sort(Input("coords"))
unique = Unique(sorted)
unique
"""


def test_sort_unique_coords_correctness_and_types():
    coords_np = np.array(
        [
            [2, 1, 0],
            [0, 0, 1],
            [2, 1, 0],
            [1, 3, 2],
            [0, 0, 1],
        ],
        dtype=np.int32,
    )
    coords = Value(
        Type(Shape(Static(coords_np.shape[0])), Type(Shape(Static(3)), ScalarType.I32)),
        coords_np.copy(),
    )

    types, out = run(SORT_UNIQUE_PROGRAM, {"coords": coords})
    assert types["sorted"] == Type(Shape(Static(coords_np.shape[0])), Type(Shape(Static(3)), ScalarType.I32))
    assert types["unique"] == Type(Shape(Dynamic()), Type(Shape(Static(3)), ScalarType.I32))

    expected = np.array(
        [
            [0, 0, 1],
            [1, 3, 2],
            [2, 1, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(out.data, expected)


def test_sort_unique_value_semantics_and_referential_transparency():
    coords_np = np.array([[3, 0, 0], [1, 0, 0], [3, 0, 0], [2, 0, 0]], dtype=np.int32)
    before = coords_np.copy()
    coords = Value(
        Type(Shape(Static(coords_np.shape[0])), Type(Shape(Static(3)), ScalarType.I32)),
        coords_np,
    )

    _, out_a = run(SORT_UNIQUE_PROGRAM, {"coords": coords})
    _, out_b = run(SORT_UNIQUE_PROGRAM, {"coords": coords})

    # Input array remains unchanged (immutability at DSL operation boundary).
    np.testing.assert_array_equal(coords_np, before)
    # Referential transparency: repeat run gives identical output.
    np.testing.assert_array_equal(out_a.data, out_b.data)


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
    x_np = np.array([5, 1, 5, 2, 1, 2, 9], dtype=np.int32)
    x_val = Value(Type(Shape(Static(x_np.shape[0])), ScalarType.I32), x_np.copy())
    _, once = run(program_once, {"x": x_val})
    _, twice = run(program_twice, {"x": x_val})
    np.testing.assert_array_equal(once.data, twice.data)


def test_sort_preserves_multiset():
    x_np = np.array([4, 1, 4, 7, 1, 3, 3, 3], dtype=np.int32)
    x_val = Value(Type(Shape(Static(x_np.shape[0])), ScalarType.I32), x_np.copy())
    _, out = run('y = Sort(Input("x"))\ny', {"x": x_val})

    in_vals, in_counts = np.unique(x_np, return_counts=True)
    out_vals, out_counts = np.unique(out.data, return_counts=True)
    np.testing.assert_array_equal(in_vals, out_vals)
    np.testing.assert_array_equal(in_counts, out_counts)
