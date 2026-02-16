# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Cross-leaf neighbor predicate on GPU: DSL string -> cuTile kernel -> verify.

Extends the v5 end-to-end codegen to hierarchical traversal: for each active
voxel, check 6 face-neighbors via Decompose + chained Gather through a
two-level grid, where neighbors may be in different leaf nodes.

The emitter handles three new patterns:
  - DecomposeNode: bit-shift decomposition per level per axis
  - FieldNode: project a named field from the decomposed struct
  - Chained GatherNode: Gather(Gather(src, i), j) fused into one 4D ct.gather

This is the v6 milestone: DSL string -> GPU execution for cross-leaf traversal.
"""

import importlib
import os

import numpy as np
import torch

import cuda.tile as ct

from docs.wip.prototype.dsl_to_cutile import emit_runnable_kernel
from docs.wip.prototype.dsl_eval import run as dsl_run
from docs.wip.prototype.ops import Value
from docs.wip.prototype.types import ScalarType, Shape, Static, Type

ConstInt = ct.Constant[int]

_GEN_DIR = os.path.join(os.path.dirname(__file__), "_generated")


def _compile_kernel(code: str, kernel_name: str):
    """Write kernel code to a .py file and import the kernel function."""
    os.makedirs(_GEN_DIR, exist_ok=True)
    init_path = os.path.join(_GEN_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

    mod_name = f"_gen_{kernel_name}"
    filepath = os.path.join(_GEN_DIR, f"{mod_name}.py")
    with open(filepath, "w") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return getattr(mod, kernel_name)


# ---------------------------------------------------------------------------
# DSL program: cross-leaf neighbor predicate as a Map expression
# ---------------------------------------------------------------------------

# For each active voxel (batch via ct.bid), check 6 face-neighbors (tile via
# ct.arange) through a two-level grid hierarchy. The chain:
#   1. Add coord + offset -> neighbor global coord
#   2. Decompose into level_0 (leaf-local) and level_1 (lower-node) coords
#   3. Gather from lower array using level_1 -> leaf_idx
#   4. Gather from leaf_arr using (leaf_idx, level_0) -> voxel value (fused)
#   5. GE(voxel_val, 0) -> is_active
#
# Decompose appears twice (once per field extraction). Accepted redundancy --
# trivial cost, and a future CSE pass can eliminate it.

CROSS_LEAF_MAP = (
    'pred = Map(Input("offsets"), o => '
    "GE("
    "Gather("
    'Gather(Input("leaf_arr"), '
    'Gather(Input("lower"), '
    'field(Decompose(Add(Input("coord"), o), Const([3, 4])), "level_1"))), '
    'field(Decompose(Add(Input("coord"), o), Const([3, 4])), "level_0")), '
    "Const(0)))\n"
    "pred"
)

INPUT_TYPES = {
    "coord": Type(Shape(Static(3)), ScalarType.I32),
    "offsets": Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
    "lower": Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
    "leaf_arr": Type(
        Shape(Static(4)),
        Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
    ),
}

FACE_OFFSETS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# Test data builder (same structure as test_cross_leaf.py)
# ---------------------------------------------------------------------------


def _build_grid(seed=42):
    """Build a two-level grid with known active voxels near leaf boundaries."""
    np.random.seed(seed)

    lower_data = np.full((16, 16, 16), -1, dtype=np.int32)

    active_lower = [(3, 4, 5), (3, 4, 6), (3, 5, 5), (4, 4, 5)]
    leaf_blocks = []
    for i, (lx, ly, lz) in enumerate(active_lower):
        lower_data[lx, ly, lz] = i
        leaf = np.full((8, 8, 8), -1, dtype=np.int32)
        n_active = np.random.randint(100, 300)
        positions = np.random.choice(512, n_active, replace=False)
        for idx, pos in enumerate(positions):
            vx = pos // 64
            vy = (pos // 8) % 8
            vz = pos % 8
            leaf[vx, vy, vz] = i * 1000 + idx
        leaf_blocks.append(leaf)

    leaf_data = np.stack(leaf_blocks)
    return lower_data, leaf_data, active_lower


def _global_coord(lower_coord, leaf_coord):
    return np.array(
        [
            lower_coord[0] * 8 + leaf_coord[0],
            lower_coord[1] * 8 + leaf_coord[1],
            lower_coord[2] * 8 + leaf_coord[2],
        ],
        dtype=np.int32,
    )


def _numpy_cross_leaf_reference(lower_data, leaf_data, active_coords, offsets):
    """Numpy reference: hierarchical cross-leaf neighbor predicate."""
    N = active_coords.shape[0]
    N_OFF = offsets.shape[0]
    result = np.zeros((N, N_OFF), dtype=bool)
    for i in range(N):
        coord = active_coords[i]
        for j in range(N_OFF):
            nbr = coord + offsets[j]
            ll = (nbr >> 3) & 15
            vl = nbr & 7
            if np.any(ll < 0) or np.any(ll >= 16):
                continue
            leaf_idx = lower_data[ll[0], ll[1], ll[2]]
            if leaf_idx < 0 or leaf_idx >= leaf_data.shape[0]:
                continue
            if np.any(vl < 0) or np.any(vl >= 8):
                continue
            if leaf_data[leaf_idx, vl[0], vl[1], vl[2]] >= 0:
                result[i, j] = True
    return result


# ---------------------------------------------------------------------------
# Test 1: Emit and inspect the cross-leaf kernel
# ---------------------------------------------------------------------------


def test_emit_cross_leaf():
    """Emit a cross-leaf kernel and verify its structure."""
    code, tile_size, map_len = emit_runnable_kernel(
        CROSS_LEAF_MAP,
        INPUT_TYPES,
        batch_input="coord",
        batch_dim=3,
        map_input="offsets",
        map_elem_rank=3,
        kernel_name="cross_leaf_pred",
    )

    print("--- Emitted cross-leaf kernel ---")
    print(code)
    print(f"--- tile_size={tile_size}, map_len={map_len} ---")

    # Structural checks: new patterns present
    assert "@ct.kernel" in code
    assert "def cross_leaf_pred(" in code
    assert "ct.bid(0)" in code
    assert ">>" in code, "Expected bit-shift from Decompose"
    assert "& 7" in code or "& 15" in code, "Expected bit-mask from Decompose"
    assert code.count("ct.gather(") >= 3, "Expected at least 3 gathers (lower + fused)"
    assert "Fused chained gather" in code, "Expected chain fusion comment"

    print("  emit_cross_leaf: structure valid -- PASSED")
    return code, tile_size, map_len


# ---------------------------------------------------------------------------
# Test 2: Compile and launch the emitted cross-leaf kernel
# ---------------------------------------------------------------------------


def test_compile_and_launch():
    """Compile the emitted cross-leaf kernel and launch it on GPU."""
    lower_data, leaf_data, active_lower = _build_grid()
    K = leaf_data.shape[0]

    # Update INPUT_TYPES with actual K
    input_types = {
        "coord": Type(Shape(Static(3)), ScalarType.I32),
        "offsets": Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
        "lower": Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        "leaf_arr": Type(
            Shape(Static(K)),
            Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
        ),
    }

    code, tile_size, map_len = emit_runnable_kernel(
        CROSS_LEAF_MAP,
        input_types,
        batch_input="coord",
        batch_dim=3,
        map_input="offsets",
        map_elem_rank=3,
        kernel_name="cross_leaf_launch",
    )

    kernel_fn = _compile_kernel(code, "cross_leaf_launch")

    # Collect all active global coords
    all_coords = []
    for li, (lx, ly, lz) in enumerate(active_lower):
        leaf = leaf_data[li]
        active_leaf = np.argwhere(leaf >= 0).astype(np.int32)
        for vl in active_leaf:
            all_coords.append(_global_coord((lx, ly, lz), vl))
    all_coords = np.array(all_coords, dtype=np.int32)
    N = all_coords.shape[0]

    # GPU tensors
    coord_t = torch.from_numpy(all_coords.copy()).cuda()
    offsets_t = torch.from_numpy(FACE_OFFSETS.copy()).cuda()
    lower_t = torch.from_numpy(lower_data.copy()).cuda()
    leaf_arr_t = torch.from_numpy(leaf_data.copy()).cuda()
    result_t = torch.zeros(N, tile_size, dtype=torch.int32, device="cuda")

    # Launch
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        kernel_fn,
        (coord_t, offsets_t, lower_t, leaf_arr_t, result_t, tile_size),
    )

    gpu_result = result_t.cpu().numpy()[:, :map_len].astype(bool)

    # Numpy reference
    ref_result = _numpy_cross_leaf_reference(lower_data, leaf_data, all_coords, FACE_OFFSETS)

    np.testing.assert_array_equal(gpu_result, ref_result)
    total = int(ref_result.sum())
    n_cross = _count_cross_leaf(all_coords, FACE_OFFSETS)
    print(
        f"  compile_and_launch: {N} voxels, {total} active neighbors, "
        f"{n_cross} cross-leaf -- PASSED"
    )


# ---------------------------------------------------------------------------
# Test 3: DSL evaluator vs generated kernel
# ---------------------------------------------------------------------------


def test_dsl_eval_vs_generated():
    """Compare DSL numpy evaluator output to the generated GPU kernel."""
    lower_data, leaf_data, active_lower = _build_grid(seed=99)
    K = leaf_data.shape[0]

    input_types = {
        "coord": Type(Shape(Static(3)), ScalarType.I32),
        "offsets": Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
        "lower": Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        "leaf_arr": Type(
            Shape(Static(K)),
            Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
        ),
    }

    code, tile_size, map_len = emit_runnable_kernel(
        CROSS_LEAF_MAP,
        input_types,
        batch_input="coord",
        batch_dim=3,
        map_input="offsets",
        map_elem_rank=3,
        kernel_name="cross_leaf_e2e",
    )

    kernel_fn = _compile_kernel(code, "cross_leaf_e2e")

    # Collect active coords (test a subset for DSL evaluator speed)
    all_coords = []
    for li, (lx, ly, lz) in enumerate(active_lower):
        leaf = leaf_data[li]
        active_leaf = np.argwhere(leaf >= 0).astype(np.int32)
        for vl in active_leaf:
            all_coords.append(_global_coord((lx, ly, lz), vl))
    all_coords = np.array(all_coords, dtype=np.int32)

    # Use a subset for DSL evaluator (per-coord loop is slow)
    rng = np.random.RandomState(42)
    test_indices = rng.choice(len(all_coords), min(100, len(all_coords)), replace=False)
    test_coords = all_coords[test_indices]
    N_test = len(test_coords)

    # --- GPU path ---
    coord_t = torch.from_numpy(test_coords.copy()).cuda()
    offsets_t = torch.from_numpy(FACE_OFFSETS.copy()).cuda()
    lower_t = torch.from_numpy(lower_data.copy()).cuda()
    leaf_arr_t = torch.from_numpy(leaf_data.copy()).cuda()
    result_t = torch.zeros(N_test, tile_size, dtype=torch.int32, device="cuda")

    ct.launch(
        torch.cuda.current_stream(),
        (N_test,),
        kernel_fn,
        (coord_t, offsets_t, lower_t, leaf_arr_t, result_t, tile_size),
    )
    gpu_result = result_t.cpu().numpy()[:, :map_len].astype(bool)

    # --- DSL evaluator path (per-coord, using the flat cross-leaf program) ---
    FLAT_PROGRAM = """
nbr = Add(Input("coord"), Input("offset"))
parts = Decompose(nbr, Const([3, 4]))
leaf_idx = Gather(Input("lower"), field(parts, "level_1"))
leaf_node = Gather(Input("leaf_arr"), leaf_idx)
voxel_val = Gather(leaf_node, field(parts, "level_0"))
is_active = GE(voxel_val, Const(0))
is_active
"""
    lower_val = Value(
        Type(Shape(Static(16), Static(16), Static(16)), ScalarType.I32),
        lower_data,
    )
    leaf_arr_val = Value(
        Type(
            Shape(Static(K)),
            Type(Shape(Static(8), Static(8), Static(8)), ScalarType.I32),
        ),
        leaf_data,
    )

    dsl_result = np.zeros((N_test, 6), dtype=bool)
    for i in range(N_test):
        coord_val = Value(Type(Shape(Static(3)), ScalarType.I32), test_coords[i])
        for j, off in enumerate(FACE_OFFSETS):
            offset_val = Value(Type(Shape(Static(3)), ScalarType.I32), off)
            _, result = dsl_run(
                FLAT_PROGRAM,
                {
                    "coord": coord_val,
                    "offset": offset_val,
                    "lower": lower_val,
                    "leaf_arr": leaf_arr_val,
                },
            )
            dsl_result[i, j] = bool(result.data)

    # Compare
    np.testing.assert_array_equal(gpu_result, dsl_result)
    total = int(gpu_result.sum())
    print(
        f"  dsl_eval_vs_generated: {N_test} voxels, {total} active neighbors, "
        f"DSL == GPU -- PASSED"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_cross_leaf(coords, offsets):
    """Count neighbor lookups that cross a leaf boundary."""
    n_cross = 0
    for coord in coords:
        coord_lower = (coord >> 3) & 15
        for off in offsets:
            nbr = coord + off
            nbr_lower = (nbr >> 3) & 15
            if not np.array_equal(coord_lower, nbr_lower):
                n_cross += 1
    return n_cross


# =========================================================================

if __name__ == "__main__":
    print("=== cuTile cross-leaf codegen ===")
    test_emit_cross_leaf()
    print()
    test_compile_and_launch()
    print()
    test_dsl_eval_vs_generated()
    print("\nAll cross-leaf GPU tests passed.")
