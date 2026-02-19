// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// index_maps.cuh -- Phase 1: cooperative construction of gather/scatter maps.
//
// This is the ONLY code that touches the SparseGrid interface. After this
// phase completes, the algorithm operates on flat index buffers in shared
// memory and never touches the tree structure again.
//
// The gather_map and scatter_map are built cooperatively by all threads
// in the threadblock, then a __syncthreads() barrier ensures visibility
// before the GEMM phases begin.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_INDEX_MAPS_CUH
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_INDEX_MAPS_CUH

#include "conv_geometry.h"
#include "conv_variant.h"
#include "im2col_map.h"
#include "sparse_grid.h"

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// build_scatter_map: fill scatter_map from the iteration grid's leaf
// =============================================================================
//
// scatter_map[p] = iter_grid.leaf_value(leaf_id, p)
//   for p in [0, LEAF^3)
//
// A value of 0 means "inactive -- don't write".

template <geometry_like Geom, typename ElemIdx, typename SparseGrid>
__device__ void
build_scatter_map(SparseGrid const &iter_grid, int leaf_id, ElemIdx *scatter_map) {
    constexpr int LEAF_VOL = Geom::leaf_vol();
    int const tid          = threadIdx.x;
    int const nthreads     = blockDim.x;

    for (int p = tid; p < LEAF_VOL; p += nthreads) {
        scatter_map[p] = iter_grid.leaf_value(leaf_id, p);
    }
}

// =============================================================================
// build_gather_map: fill gather_map from the gather grid's tree traversal
// =============================================================================
//
// gather_map[h] = gather_grid.coord_to_idx(halo_origin + decode3(h, HALO))
//   for h in [0, HALO_VOL)
//
// halo_origin = leaf_origin * S + OFFSET   (forward direction)
//
// A value of 0 means "inactive -- treat as zero input".

template <geometry_like Geom, typename ElemIdx, typename SparseGrid>
__device__ void
build_gather_map(SparseGrid const &gather_grid, dcoord3 halo_origin, ElemIdx *gather_map) {
    constexpr int HALO_VOL = Geom::halo_vol();
    int const tid          = threadIdx.x;
    int const nthreads     = blockDim.x;

    dcoord3 const halo_dims = to_dcoord3(Geom::halo());

    for (int h = tid; h < HALO_VOL; h += nthreads) {
        dcoord3 const h3    = ddecode3(h, halo_dims);
        dcoord3 const coord = halo_origin + h3;
        gather_map[h]       = gather_grid.coord_to_idx(coord);
    }
}

// =============================================================================
// build_index_maps: unified entry point for Phase 1
// =============================================================================
//
// Dispatches to the correct grid roles based on the variant.
// After this function returns AND a __syncthreads(), the scatter_map
// and gather_map in shared memory are ready for the GEMM phases.
//
// Template parameters:
//   Geom    : geometry_like
//   Variant : conv_variant
//
// For forward / weight_grad:
//   iter_grid   = output_grid  (we iterate output leaves)
//   gather_grid = input_grid   (we gather from input)
//
// For input_grad / transposed_conv:
//   iter_grid   = input_grid   (we iterate input leaves)
//   gather_grid = output_grid  (we gather from output)

template <geometry_like Geom, conv_variant Variant, typename ElemIdx, typename SparseGrid>
__device__ void
build_index_maps(SparseGrid const &iter_grid,
                 SparseGrid const &gather_grid,
                 int leaf_id,
                 ElemIdx *scatter_map,
                 ElemIdx *gather_map) {
    // Build the scatter map from the iteration grid's leaf.
    build_scatter_map<Geom>(iter_grid, leaf_id, scatter_map);

    // Compute halo origin in the gather grid's coordinate space.
    dcoord3 const leaf_origin = iter_grid.leaf_origin(leaf_id);
    dcoord3 const S           = to_dcoord3(Geom::S());
    dcoord3 const off         = to_dcoord3(Geom::offset());
    dcoord3 const halo_origin = leaf_origin * S + off;

    // Build the gather map via tree traversal.
    build_gather_map<Geom>(gather_grid, halo_origin, gather_map);
}

// =============================================================================
// build_cluster_predicates: construct boolean masks for a cluster's halo
// =============================================================================
//
// For each position in the cluster's halo, check whether the corresponding
// gather_map entry is non-zero (active). This is done once per cluster and
// shared across all blocks within the cluster.

template <geometry_like Geom, tiling_like Tiling, typename ElemIdx>
__device__ void
build_cluster_predicates(ElemIdx const *gather_map,
                         ElemIdx const *scatter_map,
                         dcoord3 cluster_orig,
                         bool *gather_pred,
                         bool *scatter_pred) {
    constexpr int CL_HALO_VOL = Tiling::cl_halo_vol();
    constexpr int CL_VOL      = Tiling::cl_vol();
    int const tid              = threadIdx.x;
    int const nthreads         = blockDim.x;

    dcoord3 const S         = to_dcoord3(Geom::S());
    dcoord3 const cl_halo   = to_dcoord3(Tiling::cl_halo());
    dcoord3 const cl_extent = to_dcoord3(Tiling::cl_extent());
    dcoord3 const halo_dims = to_dcoord3(Geom::halo());

    // Gather predicates: check gather_map at cluster-halo positions.
    for (int h = tid; h < CL_HALO_VOL; h += nthreads) {
        dcoord3 const h3       = ddecode3(h, cl_halo);
        dcoord3 const full_pos = cluster_orig * S + h3;
        int const full_idx     = dencode3(full_pos, halo_dims);
        gather_pred[h]         = (gather_map[full_idx] != static_cast<ElemIdx>(0));
    }

    // Scatter predicates: check scatter_map at cluster-extent positions.
    dcoord3 const leaf3 = to_dcoord3(Geom::leaf3());
    for (int p = tid; p < CL_VOL; p += nthreads) {
        dcoord3 const p3       = ddecode3(p, cl_extent);
        dcoord3 const full_pos = cluster_orig + p3;
        int const full_idx     = dencode3(full_pos, leaf3);
        scatter_pred[p]        = (scatter_map[full_idx] != static_cast<ElemIdx>(0));
    }
}

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_INDEX_MAPS_CUH
