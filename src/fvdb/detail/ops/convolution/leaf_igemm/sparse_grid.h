// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// sparse_grid.h -- SparseGrid concept and NanoVDB adapter.
//
// Defines the four-operation interface from the formal specification:
//   leaf_count()            -> int
//   leaf_origin(leaf_id)    -> coord3/dcoord3
//   leaf_value(leaf_id, p)  -> ElemIdx  (0 = inactive)
//   coord_to_idx(coord)     -> ElemIdx  (0 = inactive)
//
// The NanoVDB adapter wraps fVDB's GridBatchImpl::Accessor and NanoVDB
// tree to satisfy this concept. It is the only code that knows about
// NanoVDB's OnIndexGrid, LeafNodeType, and tree accessor.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_SPARSE_GRID_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_SPARSE_GRID_H

#include "im2col_map.h"

#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// nanovdb_sparse_grid: concrete SparseGrid wrapping fVDB's NanoVDB trees
// =============================================================================
//
// Wraps a single grid from a GridBatchImpl (identified by batch_idx).
// All methods are __device__ only.
//
// Convention: leaf_value and coord_to_idx return 0 for inactive voxels.
// NanoVDB's OnIndexGrid stores 1-based values for active voxels, so we
// use the raw value directly (0 is already the inactive sentinel in
// OnIndexGrid for unset positions).
//
// The voxel_offset is added to produce a globally unique feature index
// across the batch.

template <typename ElemIdx>
struct nanovdb_sparse_grid {
    using grid_type = nanovdb::OnIndexGrid;
    using tree_type = typename grid_type::TreeType;
    using leaf_type = typename tree_type::LeafNodeType;
    using acc_type  = nanovdb::ReadAccessor<nanovdb::ValueOnIndex>;

    grid_type const *grid_ptr;
    int64_t voxel_offset;
    int64_t cum_leaf_offset;

    __device__ int
    leaf_count() const {
        return static_cast<int>(grid_ptr->tree().nodeCount(0));
    }

    __device__ dcoord3
    leaf_origin(int leaf_id) const {
        auto const &leaf = grid_ptr->tree().template getFirstNode<0>()[leaf_id];
        auto const orig  = leaf.origin();
        return {orig[0], orig[1], orig[2]};
    }

    // Return the global feature index for position p within leaf leaf_id.
    // p is a linear offset in [0, 512).
    // Returns 0 if the voxel is inactive.
    __device__ ElemIdx
    leaf_value(int leaf_id, int p) const {
        auto const &leaf = grid_ptr->tree().template getFirstNode<0>()[leaf_id];
        if (!leaf.isActive(p)) {
            return static_cast<ElemIdx>(0);
        }
        // NanoVDB OnIndexGrid stores 1-based values. We keep them 1-based
        // so that 0 remains the inactive sentinel. The caller subtracts 1
        // and adds voxel_offset when computing the feature address.
        //
        // global_feature_index = voxel_offset + (leaf.getValue(p) - 1)
        //
        // We encode this as: voxel_offset + leaf.getValue(p)
        // The -1 is folded into the base pointer at the call site to
        // keep the gather_map entries as simple offsets.
        return static_cast<ElemIdx>(voxel_offset + leaf.getValue(p));
    }

    // Return the global feature index for an arbitrary world coordinate.
    // This is the expensive tree traversal (the only one).
    // Returns 0 if the coordinate is inactive.
    __device__ ElemIdx
    coord_to_idx(dcoord3 coord) const {
        auto acc = grid_ptr->getAccessor();
        nanovdb::Coord const c(coord.x, coord.y, coord.z);
        if (!acc.isActive(c)) {
            return static_cast<ElemIdx>(0);
        }
        return static_cast<ElemIdx>(voxel_offset + acc.getValue(c));
    }
};

// =============================================================================
// Factory: create a nanovdb_sparse_grid from GridBatchImpl::Accessor
// =============================================================================

template <typename ElemIdx>
__device__ nanovdb_sparse_grid<ElemIdx>
make_sparse_grid(GridBatchImpl::Accessor const &acc, int64_t batch_idx) {
    auto const *grid = acc.grid(batch_idx);
    return {grid, acc.voxelOffset(batch_idx), acc.leafOffset(batch_idx)};
}

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_SPARSE_GRID_H
