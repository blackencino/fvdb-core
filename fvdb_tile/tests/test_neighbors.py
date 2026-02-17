# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Single-leaf neighbor finding.

EXPRESSION phases use DSL strings executed via run().
The jagged extent kind (~) emerges automatically from variable-length
filtering within Each.
"""

import torch

from fvdb_tile.prototype.dsl_eval import run
from fvdb_tile.prototype.ops import Value
from fvdb_tile.prototype.types import (
    Dynamic,
    Jagged,
    ScalarType,
    Shape,
    Static,
    Type,
    coord_type,
)


# ---------------------------------------------------------------------------
# DSL programs
# ---------------------------------------------------------------------------

NEIGHBOR_PROGRAM = """
mask = Map(Input("leaf"), x => GE(x, Const(0)))
active = Where(mask)
nbrs = Each(active, a => Map(Input("offsets"), o => Add(a, o)))
filtered = Each(nbrs, cs => Gather(cs, Where(Map(cs, c => And(InBounds(c, Const(0), Const(8)), Gather(mask, c))))))
filtered
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_bounds(coord, lo=0, hi=8):
    return bool(torch.all(coord >= lo).item() and torch.all(coord < hi).item())


FACE_OFFSETS = torch.tensor(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=torch.int32,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_neighbor_types():
    """Verify type propagation through the neighbor-finding expression."""

    # ---- SETUP ----
    gen = torch.Generator().manual_seed(42)
    leaf_data = torch.randint(-1, 10, (8, 8, 8), generator=gen, dtype=torch.int32)
    leaf = Value.from_tensor(leaf_data, ScalarType.I32)
    offsets = Value(
        Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
        FACE_OFFSETS,
    )

    # ---- EXPRESSION ----
    types, result = run(NEIGHBOR_PROGRAM, {"leaf": leaf, "offsets": offsets})

    # ---- EXTRACTION ----
    print("Neighbor types:")
    for name, ty in types.items():
        print(f"  {name}: {ty}")

    assert types["active"] == Type(Shape(Dynamic()), coord_type(3))
    assert types["nbrs"].iteration_shape == Shape(Dynamic())

    # filtered should have (*) outer, jagged inner
    assert result.type.iteration_shape == Shape(Dynamic())
    inner = result.type.element_type
    assert isinstance(inner, Type)
    assert isinstance(inner.iteration_shape.extents[0], Jagged)


def test_neighbor_data():
    """Verify numerical correctness against brute-force reference."""

    # ---- SETUP ----
    gen = torch.Generator().manual_seed(42)
    leaf_data = torch.randint(-1, 10, (8, 8, 8), generator=gen, dtype=torch.int32)
    leaf = Value.from_tensor(leaf_data, ScalarType.I32)
    offsets = Value(
        Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
        FACE_OFFSETS,
    )

    # ---- EXPRESSION ----
    types, result = run(NEIGHBOR_PROGRAM, {"leaf": leaf, "offsets": offsets})

    # ---- EXTRACTION ----
    mask_data = leaf_data >= 0
    active_coords = torch.nonzero(mask_data).to(torch.int32)

    assert len(result.data) == len(active_coords)

    neighbor_counts = []
    for i, coord in enumerate(active_coords):
        nbrs = coord.unsqueeze(0) + FACE_OFFSETS
        valid = [nb for nb in nbrs if _in_bounds(nb) and mask_data[nb[0].item(), nb[1].item(), nb[2].item()].item()]
        ref = torch.stack(valid).to(torch.int32) if valid else torch.empty((0, 3), dtype=torch.int32)

        actual = result.data[i].data
        torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        neighbor_counts.append(len(ref))

    neighbor_counts = torch.tensor(neighbor_counts)
    print(
        f"Neighbor data OK: {len(active_coords)} active, "
        f"min={neighbor_counts.min().item()} max={neighbor_counts.max().item()} "
        f"mean={neighbor_counts.float().mean().item():.1f}"
    )
    assert neighbor_counts.min().item() != neighbor_counts.max().item(), "Expected jagged"


def test_jagged_type_emerges():
    """The key test: does ~ appear in the type from the DSL?"""

    # ---- SETUP ----
    gen = torch.Generator().manual_seed(99)
    leaf_data = torch.randint(-1, 4, (8, 8, 8), generator=gen, dtype=torch.int32)
    leaf = Value.from_tensor(leaf_data, ScalarType.I32)
    offsets = Value(
        Type(Shape(Static(6)), Type(Shape(Static(3)), ScalarType.I32)),
        FACE_OFFSETS,
    )

    # ---- EXPRESSION ----
    types, result = run(NEIGHBOR_PROGRAM, {"leaf": leaf, "offsets": offsets})

    # ---- EXTRACTION ----
    inner_type = result.type.element_type
    assert isinstance(inner_type, Type)
    leading_extent = inner_type.iteration_shape.extents[0]
    assert isinstance(leading_extent, Jagged), (
        f"Expected jagged (~), got {leading_extent!r}. Full type: {result.type}"
    )
    print(f"Jagged type confirmed: {result.type}")


if __name__ == "__main__":
    test_neighbor_types()
    test_neighbor_data()
    test_jagged_type_emerges()
    print("\nAll test_neighbors tests passed.")
