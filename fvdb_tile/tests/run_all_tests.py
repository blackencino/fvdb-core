# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Run all prototype tests in conceptual order.

  v0  -- types, Map, Where, Gather, Each, jagged emergence
  v1  -- multiple leaves (Cut), Indexed layout, Struct + Flip
  v2  -- two-level hierarchical chain, Decompose, morton
  v3  -- micro DSL: string -> parse -> type-check -> execute;
         Sort/Unique primitives; barrier-aware pipeline planning
  mesh -- triangle mesh as layouts over tensors
  v5  -- cross-leaf neighbors via DSL evaluator (numpy)

GPU tests (require `source ~/.venvs/fvdb_cutile/bin/activate`):
  v4  -- test_cutile_smoke.py
  v5  -- test_cutile_e2e.py
  v6  -- test_cutile_cross_leaf.py
  v8+ -- test_cutile_masked_e2e.py, test_cutile_cig3_e2e.py
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
    from fvdb_tile.tests.test_where import (
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
    from fvdb_tile.tests.test_neighbors import (
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
    from fvdb_tile.tests.test_indexed_flip import (
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
    from fvdb_tile.tests.test_two_level import (
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
    from fvdb_tile.tests.test_dsl import (
        test_where_program,
        test_neighbor_program,
        test_chain_program,
    )
    _run("dsl_where_program", test_where_program)
    _run("dsl_neighbor_program", test_neighbor_program)
    _run("dsl_chain_program", test_chain_program)

    print()
    from fvdb_tile.tests.test_sort_unique import (
        test_sort_unique_coords_correctness_and_types,
        test_sort_unique_value_semantics_and_referential_transparency,
        test_unique_idempotence_law,
        test_sort_preserves_multiset,
    )
    _run("dsl_sort_unique_types", test_sort_unique_coords_correctness_and_types)
    _run("dsl_sort_unique_semantics", test_sort_unique_value_semantics_and_referential_transparency)
    _run("dsl_unique_idempotence", test_unique_idempotence_law)
    _run("dsl_sort_multiset_preservation", test_sort_preserves_multiset)

    print()
    from fvdb_tile.tests.test_pipeline import (
        test_pipeline_partitions_collectives,
        test_pipeline_marks_nested_barrier,
        test_pipeline_executable_matches_direct_run,
        test_pipeline_executable_preserves_input_immutability,
    )
    _run("pipeline_partition_collectives", test_pipeline_partitions_collectives)
    _run("pipeline_nested_barrier", test_pipeline_marks_nested_barrier)
    _run("pipeline_exec_matches_direct", test_pipeline_executable_matches_direct_run)
    _run("pipeline_exec_input_immutable", test_pipeline_executable_preserves_input_immutability)

    # -- mesh exemplar --
    print()
    print("=" * 60)
    print("Mesh: Triangle mesh -- layouts over tensors, scheduling")
    print("=" * 60)
    from fvdb_tile.tests.test_mesh import (
        test_mesh_types,
        test_mesh_centroids,
        test_mesh_dsl,
    )
    _run("mesh_types", test_mesh_types)
    _run("mesh_centroids", test_mesh_centroids)
    _run("mesh_dsl", test_mesh_dsl)

    # -- v5: cross-leaf neighbors (numpy DSL evaluator) --
    print()
    print("=" * 60)
    print("v5: Cross-leaf neighbors -- Decompose + chained Gather (DSL)")
    print("=" * 60)
    from fvdb_tile.tests.test_cross_leaf import (
        test_single_coord_cross_leaf,
        test_cross_boundary_neighbor,
        test_batch_cross_leaf,
    )
    _run("single_coord_cross_leaf", test_single_coord_cross_leaf)
    _run("cross_boundary_neighbor", test_cross_boundary_neighbor)
    _run("batch_cross_leaf", test_batch_cross_leaf)

    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
