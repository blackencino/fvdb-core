# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Run all prototype tests in conceptual order.

  v0  -- types, Map, Where, Gather, Each, jagged emergence
  v1  -- multiple leaves (Cut), Indexed layout, Struct + Flip
  v2  -- two-level hierarchical chain, Decompose, morton
  v3  -- micro DSL: string -> parse -> type-check -> execute
"""

import sys


def _run(label: str, fn):
    try:
        fn()
    except Exception as e:
        print(f"  FAIL: {e}", file=sys.stderr)
        raise


def main():
    # -- v0: single leaf fundamentals --
    print("=" * 60)
    print("v0: Single leaf -- Map, Where, Gather, Each")
    print("=" * 60)
    from docs.wip.prototype.test_where import (
        test_map_preserves_shape,
        test_where_type,
        test_where_data,
        test_gather_active_indices,
        test_gather_features,
        test_full_pipeline,
    )
    _run("map_preserves_shape", test_map_preserves_shape)
    _run("where_type", test_where_type)
    _run("where_data", test_where_data)
    _run("gather_active_indices", test_gather_active_indices)
    _run("gather_features", test_gather_features)
    _run("full_pipeline", test_full_pipeline)

    print()
    from docs.wip.prototype.test_neighbors import (
        test_neighbor_types,
        test_neighbor_data,
        test_jagged_type_emerges,
    )
    _run("neighbor_types", test_neighbor_types)
    _run("neighbor_data", test_neighbor_data)
    _run("jagged_type_emerges", test_jagged_type_emerges)

    # -- v1: Cut, Indexed, Struct, Flip --
    print()
    print("=" * 60)
    print("v1: Multiple leaves, Indexed, Struct + Flip")
    print("=" * 60)
    from docs.wip.prototype.test_indexed_flip import (
        test_multi_leaf_cut,
        test_multi_leaf_where,
        test_indexed_type_then_gather,
        test_struct_flip_types,
        test_struct_flip_data,
    )
    _run("multi_leaf_cut", test_multi_leaf_cut)
    _run("multi_leaf_where", test_multi_leaf_where)
    _run("indexed_type_then_gather", test_indexed_type_then_gather)
    _run("struct_flip_types", test_struct_flip_types)
    _run("struct_flip_data", test_struct_flip_data)

    # -- v2: two-level hierarchical chain --
    print()
    print("=" * 60)
    print("v2: Two-level grid -- Decompose, chained Gather, morton")
    print("=" * 60)
    from docs.wip.prototype.test_two_level import (
        test_3d_chain_single_coord,
        test_3d_chain_batch,
        test_morton_chain_batch,
        test_batch_active_voxel_lookup,
    )
    _run("3d_chain_single_coord", test_3d_chain_single_coord)
    _run("3d_chain_batch", test_3d_chain_batch)
    _run("morton_chain_batch", test_morton_chain_batch)
    _run("batch_active_voxel_lookup", test_batch_active_voxel_lookup)

    # -- v3: micro DSL --
    print()
    print("=" * 60)
    print("v3: Micro DSL -- string -> parse -> type-check -> execute")
    print("=" * 60)
    from docs.wip.prototype.test_dsl import (
        test_where_program,
        test_neighbor_program,
        test_chain_program,
    )
    _run("dsl_where_program", test_where_program)
    _run("dsl_neighbor_program", test_neighbor_program)
    _run("dsl_chain_program", test_chain_program)

    # -- mesh exemplar --
    print()
    print("=" * 60)
    print("Mesh: Triangle mesh -- layouts over tensors, scheduling")
    print("=" * 60)
    from docs.wip.prototype.test_mesh import (
        test_mesh_types,
        test_mesh_centroids,
        test_mesh_dsl,
    )
    _run("mesh_types", test_mesh_types)
    _run("mesh_centroids", test_mesh_centroids)
    _run("mesh_dsl", test_mesh_dsl)

    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
