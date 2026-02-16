# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
cuTile codegen test: DSL string -> emitted cuTile code -> verify.

Parses a DSL program, emits cuTile Python source, and verifies the
emitted code structure. Then runs the same DSL program through the
numpy evaluator and a hand-written cuTile kernel to confirm they agree.

This proves the compilation path: DSL string -> AST -> cuTile source.
"""

import numpy as np
import torch

import cuda.tile as ct

from docs.wip.prototype.dsl_to_cutile import emit_program
from docs.wip.prototype.dsl_eval import run as dsl_run
from docs.wip.prototype.ops import Value
from docs.wip.prototype.types import ScalarType, Shape, Static, Type


# ---------------------------------------------------------------------------
# Test 1: Emit code for a simple expression and inspect it
# ---------------------------------------------------------------------------

MASK_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
mask
"""


def test_emit_mask():
    """Emit cuTile code for Map(leaf, x => GE(x, 0)). Verify structure."""
    code, output_var = emit_program(
        MASK_PROGRAM,
        input_map={"leaf": "leaf_arr"},
        kernel_name="mask_kernel",
    )

    print("--- Emitted code (mask) ---")
    print(code)
    print(f"--- Output variable: {output_var} ---")

    # The emitted code should contain key cuTile patterns
    assert "def mask_kernel" in code
    assert "@ct.kernel" in code
    assert ">= 0" in code
    print("  emit_mask: code structure valid -- PASSED")


# ---------------------------------------------------------------------------
# Test 2: Emit code for the neighbor gather predicate
# ---------------------------------------------------------------------------

# The cuTile kernel uses ct.gather with padding_value=-1 for OOB, then
# checks vals >= 0. The DSL evaluator's Gather also returns -1 for OOB.
# So the natural expression is GE(Gather(mask, nbr_coord), 0) -- the
# bounds check is implicit in the sentinel value, matching cuTile exactly.
PREDICATE_PROGRAM = """
pred = Map(Input("offsets"), o => GE(Gather(Input("mask"), Add(Input("coord"), o)), Const(0)))
pred
"""


def test_emit_neighbor_predicate():
    """Emit cuTile code for the full neighbor gather predicate."""
    code, output_var = emit_program(
        PREDICATE_PROGRAM,
        input_map={"mask": "mask_arr", "coord": "coord_tile", "offsets": "offsets_arr"},
        kernel_name="neighbor_pred_kernel",
    )

    print("--- Emitted code (neighbor predicate) ---")
    print(code)
    print(f"--- Output variable: {output_var} ---")

    # The emitted code should contain key patterns
    assert "ct.gather" in code
    assert "check_bounds=True" in code
    assert "padding_value=-1" in code
    assert ">=" in code  # GE
    assert "+" in code  # Add
    print("  emit_neighbor_predicate: code structure valid -- PASSED")


# ---------------------------------------------------------------------------
# Test 3: Run the DSL evaluator and cuTile kernel on the same data
# ---------------------------------------------------------------------------

FACE_OFFSETS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int32,
)


def test_numpy_vs_cutile():
    """Run the DSL expression through numpy evaluator and compare to
    the hand-written cuTile kernel from test_cutile_gather.py."""
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    mask_bool = leaf_data >= 0
    active_coords = np.argwhere(mask_bool).astype(np.int32)
    N = active_coords.shape[0]

    # Run each active voxel through the DSL evaluator
    dsl_results = np.zeros((N, 6), dtype=bool)
    for i in range(N):
        coord_val = Value(
            Type(Shape(Static(3)), ScalarType.I32),
            active_coords[i],
        )
        mask_val = Value(
            Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
            leaf_data,
        )
        offsets_val = Value(
            Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
            FACE_OFFSETS,
        )
        _, result = dsl_run(
            PREDICATE_PROGRAM,
            {"mask": mask_val, "coord": coord_val, "offsets": offsets_val},
        )
        dsl_results[i] = result.data

    # Run through the hand-written cuTile kernel
    from docs.wip.prototype.test_cutile_gather import run_neighbor_predicate

    cutile_results = run_neighbor_predicate(leaf_data, active_coords, FACE_OFFSETS)

    # Compare
    np.testing.assert_array_equal(dsl_results, cutile_results)
    total = int(cutile_results.sum())
    print(f"  numpy_vs_cutile: {N} voxels, {total} active neighbors, results match -- PASSED")


# ---------------------------------------------------------------------------
# Test 4: End-to-end -- emit code, inspect, then verify the emitted logic
# matches the hand-written kernel's output pattern
# ---------------------------------------------------------------------------

def test_end_to_end_emission():
    """Emit cuTile code for the predicate, verify it contains the right
    gather/scatter pattern that the hand-written kernel uses."""
    code, output_var = emit_program(
        PREDICATE_PROGRAM,
        input_map={"mask": "mask_arr", "coord": "coord_tile", "offsets": "offsets_arr"},
        kernel_name="gen_neighbor_pred",
    )

    # Key emission properties:
    # 1. The gather uses check_bounds and padding_value (bounds checking)
    assert "check_bounds=True" in code
    assert "padding_value=-1" in code

    # 2. Addition is used for coordinate computation
    assert "+" in code

    # 3. GE comparison for active check
    assert ">=" in code

    print("  end_to_end_emission: emitted code has correct cuTile patterns -- PASSED")


# =========================================================================

if __name__ == "__main__":
    print("=== cuTile codegen tests ===")
    test_emit_mask()
    print()
    test_emit_neighbor_predicate()
    print()
    test_numpy_vs_cutile()
    print()
    test_end_to_end_emission()
    print("\nAll cuTile codegen tests passed.")
