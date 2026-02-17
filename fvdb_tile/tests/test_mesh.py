# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Triangle mesh exemplar.

EXPRESSION phases use DSL strings where possible.
API TESTS (type-level composition) stay in Python since they test the
layout API itself.

Demonstrates:
  - Geometric structure as layouts over tensors (no custom mesh class)
  - Early vs. late materialisation as scheduling variants (same algorithm,
    same result, different materialisation point)
"""

import torch

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.layouts import cut_by_size, indexed
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import (
    Dynamic,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# ---------------------------------------------------------------------------
# Test mesh: a tetrahedron (4 vertices, 4 triangular faces)
# ---------------------------------------------------------------------------

TETRA_POSITIONS = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 1.0, 0.0],
    [0.5, 0.5, 1.0],
], dtype=torch.float32)

TETRA_FACES = torch.tensor([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3],
], dtype=torch.int32)


# ---------------------------------------------------------------------------
# DSL programs
# ---------------------------------------------------------------------------

# Triangle vertex lookup (no centroid yet).
TRIS_PROGRAM = """
vertices = cut(Input("positions"), Const(3))
face_idx = cut(Input("faces"), Const(3))
tris = Each(face_idx, f => Map(f, i => Gather(vertices, i)))
tris
"""

# Full centroid computation: for each face, gather vertices inline, then
# reduce with Over(Add) and divide by 3. Mean = Over(Add, xs) / Count(xs).
CENTROID_PROGRAM = """
vertices = cut(Input("positions"), Const(3))
face_idx = cut(Input("faces"), Const(3))
centroids = Each(face_idx, f => Div(Over(Add, Map(f, i => Gather(vertices, i))), Const(3)))
centroids
"""


# =========================================================================
# Scenario 1: Type-level mesh composition (API test, stays Python)
# =========================================================================

def test_mesh_types():
    """Verify layout types and that Indexed is correctly rejected.

    API TEST: tests the Python layout API directly, not the algebra.
    """

    # ---- SETUP ----
    V, F = TETRA_POSITIONS.shape[0], TETRA_FACES.shape[0]
    positions_type = Type(Shape(Static(V * 3)), ScalarType.F32)
    faces_type = Type(Shape(Static(F * 3)), ScalarType.I32)

    # ---- EXPRESSION (Python API) ----
    vertices_type = cut_by_size(3, positions_type)
    face_idx_type = cut_by_size(3, faces_type)

    # ---- EXTRACTION ----
    print(f"  vertices:  {vertices_type}")
    print(f"  face_idx:  {face_idx_type}")

    assert vertices_type == Type(Shape(Static(V)), Type(Shape(Static(3)), ScalarType.F32))
    assert face_idx_type == Type(Shape(Static(F)), Type(Shape(Static(3)), ScalarType.I32))

    # Indexed(face_idx, vertices) correctly rejected
    try:
        indexed(face_idx_type, vertices_type)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    print("  Indexed(face_idx, vertices) correctly rejected.")
    print("  Mesh indexing is 3 separate scalar lookups per face.")


# =========================================================================
# Scenario 2: Early vs. late materialisation (both via DSL)
# =========================================================================

def test_mesh_centroids():
    """Face centroids computed entirely in the DSL using Over(Add) + Div."""

    # ---- SETUP ----
    positions = Value(
        Type(Shape(Static(TETRA_POSITIONS.numel())), ScalarType.F32),
        TETRA_POSITIONS.flatten().to(torch.float32),
    )
    faces = Value(
        Type(Shape(Static(TETRA_FACES.numel())), ScalarType.I32),
        TETRA_FACES.flatten().to(torch.int32),
    )

    # ---- EXPRESSION ----
    types, result = run(CENTROID_PROGRAM, {"positions": positions, "faces": faces})

    # ---- EXTRACTION ----
    print("DSL centroid types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    # centroids: (4,) over (3,) f32
    assert result.type.iteration_shape == Shape(Static(4))

    F = TETRA_FACES.shape[0]
    for fi in range(F):
        actual = result.data[fi] if isinstance(result.data[fi], torch.Tensor) else result.data[fi].data
        expected = TETRA_POSITIONS[TETRA_FACES[fi]].mean(dim=0)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    centroid0 = result.data[0] if isinstance(result.data[0], torch.Tensor) else result.data[0].data
    print(f"  4 face centroids verified via DSL")
    print(f"  centroids[0] = {centroid0}")
    print(f"  -> Mean = Div(Over(Add, ...), Const(3)) -- no special primitive needed.")


# =========================================================================
# Scenario 3: DSL mesh program (triangle vertex lookup)
# =========================================================================

def test_mesh_dsl():
    """Face vertex lookup as a DSL string program."""

    # ---- SETUP ----
    positions = Value(
        Type(Shape(Static(TETRA_POSITIONS.numel())), ScalarType.F32),
        TETRA_POSITIONS.flatten().to(torch.float32),
    )
    faces = Value(
        Type(Shape(Static(TETRA_FACES.numel())), ScalarType.I32),
        TETRA_FACES.flatten().to(torch.int32),
    )

    # ---- EXPRESSION ----
    types, result = run(TRIS_PROGRAM, {"positions": positions, "faces": faces})

    # ---- EXTRACTION ----
    print("DSL mesh types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    # tris: (4,) over (3,) over (3,) f32
    assert result.type.iteration_shape == Shape(Static(4))

    for fi in range(4):
        tri_data = result.data[fi]
        actual = tri_data if isinstance(tri_data, torch.Tensor) else tri_data.data
        expected = TETRA_POSITIONS[TETRA_FACES[fi]]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    print(f"  DSL produced correct triangle vertex data for 4 faces")
    print(f"  -> Mesh expressed as layouts over tensors, no custom mesh class.")


# =========================================================================

if __name__ == "__main__":
    print("=== API: Type-level mesh composition ===")
    test_mesh_types()

    print("\n=== Algebra: Centroids via Over+Div (DSL) ===")
    test_mesh_centroids()

    print("\n=== Algebra: Triangle vertex lookup (DSL) ===")
    test_mesh_dsl()

    print("\nAll test_mesh tests passed.")
