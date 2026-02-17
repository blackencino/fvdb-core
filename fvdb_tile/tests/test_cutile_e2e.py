# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end cuTile codegen test: DSL string -> compile -> launch -> verify.

Closes the loop: a DSL string program is parsed, type-checked, emitted as
a complete @ct.kernel, written to a .py file (required by cuTile's JIT
which uses inspect.getsource()), imported, launched on the GPU, and
verified against the numpy DSL evaluator.
"""

import importlib
import os
import sys
import tempfile

import numpy as np
import torch

import cuda.tile as ct

from fvdb_tile.prototype.dsl_to_cutile import emit_runnable_kernel
from fvdb_tile.prototype.dsl_eval import run as dsl_run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import ScalarType, Shape, Static, Type

ConstInt = ct.Constant[int]

# Directory for generated kernel files
_GEN_DIR = os.path.join(os.path.dirname(__file__), "_generated")


def _compile_kernel(code: str, kernel_name: str):
    """Write kernel code to a .py file and import the kernel function.

    cuTile's JIT uses inspect.getsource() to parse kernel functions,
    so they must live in actual source files (not exec'd code).
    """
    os.makedirs(_GEN_DIR, exist_ok=True)
    init_path = os.path.join(_GEN_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

    mod_name = f"_gen_{kernel_name}"
    filepath = os.path.join(_GEN_DIR, f"{mod_name}.py")
    with open(filepath, "w") as f:
        f.write(code)

    # Import the module
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return getattr(mod, kernel_name)


# ---------------------------------------------------------------------------
# The DSL program to compile
# ---------------------------------------------------------------------------

PREDICATE_PROGRAM = """
pred = Map(Input("offsets"), o => GE(Gather(Input("mask"), Add(Input("coord"), o)), Const(0)))
pred
"""

# Input type declarations (same as the DSL type system uses)
INPUT_TYPES = {
    "mask": Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
    "coord": Type(Shape(Static(3)), ScalarType.I32),
    "offsets": Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
}

FACE_OFFSETS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# Test 1: Emit and inspect
# ---------------------------------------------------------------------------

def test_emit_runnable():
    """Emit a complete kernel and verify its structure."""
    code, tile_size, map_len = emit_runnable_kernel(
        PREDICATE_PROGRAM,
        INPUT_TYPES,
        batch_input="coord",
        batch_dim=3,
        map_input="offsets",
        map_elem_rank=3,
        kernel_name="gen_neighbor_pred",
    )

    print("--- Emitted runnable kernel ---")
    print(code)
    print(f"--- tile_size={tile_size}, map_len={map_len} ---")

    # Structural checks
    assert "@ct.kernel" in code
    assert "def gen_neighbor_pred(" in code
    assert "ct.bid(0)" in code
    assert "ct.arange(" in code
    assert "ct.gather(" in code
    assert "ct.scatter(" in code
    assert "check_bounds=True" in code

    print("  emit_runnable: structure valid -- PASSED")


# ---------------------------------------------------------------------------
# Test 2: Compile and launch the emitted kernel
# ---------------------------------------------------------------------------

def test_compile_and_launch():
    """Compile the emitted kernel with cuTile JIT and launch it on GPU."""
    code, tile_size, map_len = emit_runnable_kernel(
        PREDICATE_PROGRAM,
        INPUT_TYPES,
        batch_input="coord",
        batch_dim=3,
        map_input="offsets",
        map_elem_rank=3,
        kernel_name="gen_neighbor_pred",
    )

    # Write to file and import (cuTile JIT needs inspect.getsource())
    kernel_fn = _compile_kernel(code, "gen_neighbor_pred")

    # Prepare test data
    np.random.seed(42)
    leaf_data = np.random.randint(-1, 10, (8, 8, 8)).astype(np.int32)
    mask_bool = leaf_data >= 0
    active_coords = np.argwhere(mask_bool).astype(np.int32)
    N = active_coords.shape[0]

    mask_t = torch.from_numpy(leaf_data.copy()).cuda()
    coords_t = torch.from_numpy(active_coords.copy()).cuda()
    offsets_t = torch.from_numpy(FACE_OFFSETS.copy()).cuda()
    result_t = torch.zeros(N, tile_size, dtype=torch.int32, device="cuda")

    # Launch the generated kernel
    grid = (N,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel_fn,
        (mask_t, coords_t, offsets_t, result_t, tile_size),
    )

    # Extract results (only first map_len columns are valid)
    gpu_result = result_t.cpu().numpy()[:, :map_len].astype(bool)

    # Numpy reference
    ref_result = _numpy_reference(leaf_data, active_coords, FACE_OFFSETS)

    np.testing.assert_array_equal(gpu_result, ref_result)
    total = int(ref_result.sum())
    print(f"  compile_and_launch: {N} voxels, {total} active neighbors -- PASSED")


# ---------------------------------------------------------------------------
# Test 3: Full end-to-end -- DSL evaluator vs generated kernel
# ---------------------------------------------------------------------------

def test_dsl_eval_vs_generated():
    """Run the DSL evaluator and the generated kernel, compare results."""
    code, tile_size, map_len = emit_runnable_kernel(
        PREDICATE_PROGRAM,
        INPUT_TYPES,
        batch_input="coord",
        batch_dim=3,
        map_input="offsets",
        map_elem_rank=3,
        kernel_name="gen_pred_e2e",
    )

    kernel_fn = _compile_kernel(code, "gen_pred_e2e")

    # Test data
    np.random.seed(99)
    leaf_data = np.random.randint(-1, 5, (8, 8, 8)).astype(np.int32)
    mask_bool = leaf_data >= 0
    active_coords = np.argwhere(mask_bool).astype(np.int32)
    N = active_coords.shape[0]

    # --- GPU path: generated kernel ---
    mask_t = torch.from_numpy(leaf_data.copy()).cuda()
    coords_t = torch.from_numpy(active_coords.copy()).cuda()
    offsets_t = torch.from_numpy(FACE_OFFSETS.copy()).cuda()
    result_t = torch.zeros(N, tile_size, dtype=torch.int32, device="cuda")

    ct.launch(torch.cuda.current_stream(), (N,), kernel_fn, (mask_t, coords_t, offsets_t, result_t, tile_size))
    gpu_result = result_t.cpu().numpy()[:, :map_len].astype(bool)

    # --- DSL evaluator path ---
    dsl_result = np.zeros((N, 6), dtype=bool)
    for i in range(N):
        coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), active_coords[i])
        mask_val = Value(Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32), leaf_data)
        offsets_val = Value(
            Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
            FACE_OFFSETS,
        )
        _, result = dsl_run(PREDICATE_PROGRAM, {"mask": mask_val, "coord": coord_val, "offsets": offsets_val})
        dsl_result[i] = result.data

    # Compare
    np.testing.assert_array_equal(gpu_result, dsl_result)
    total = int(gpu_result.sum())
    print(f"  dsl_eval_vs_generated: {N} voxels, {total} active neighbors, DSL == GPU -- PASSED")


# ---------------------------------------------------------------------------
# Numpy reference helper
# ---------------------------------------------------------------------------

def _numpy_reference(mask_np, active_coords_np, offsets_np):
    N = active_coords_np.shape[0]
    N_OFF = offsets_np.shape[0]
    result = np.zeros((N, N_OFF), dtype=bool)
    for i in range(N):
        coord = active_coords_np[i]
        for j in range(N_OFF):
            nbr = coord + offsets_np[j]
            in_bounds = np.all(nbr >= 0) and np.all(nbr < 8)
            if in_bounds and mask_np[nbr[0], nbr[1], nbr[2]] >= 0:
                result[i, j] = True
    return result


# =========================================================================

if __name__ == "__main__":
    print("=== cuTile end-to-end codegen ===")
    test_emit_runnable()
    print()
    test_compile_and_launch()
    print()
    test_dsl_eval_vs_generated()
    print("\nAll end-to-end tests passed.")
