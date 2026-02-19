// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// kernel.cuh -- Leaf-level iGEMM convolution kernel template.
//
// Orchestrates the three phases of the algorithm:
//   Phase 1: Build gather/scatter index maps in shared memory
//   Phase 2: Construct predicate masks per cluster
//   Phase 3: Pipelined GEMM with predicated gather + scatter epilogue
//
// One threadblock per output leaf. All phases run within a single kernel
// launch so the index maps stay in fast shared memory.
//
// Template parameters:
//   Geom    : geometry_like   (spatial parameters)
//   Types   : types_like      (scalar types + MMA atom)
//   Tiling  : tiling_like     (block/cluster decomposition)
//   Variant : conv_variant    (forward / backward / etc.)
//
// Target architecture: SM80 (Ampere).
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_KERNEL_CUH
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_KERNEL_CUH

#include "conv_geometry.h"
#include "conv_tiling.h"
#include "conv_types.h"
#include "conv_variant.h"
#include "gather_tensor.h"
#include "im2col_map.h"
#include "index_maps.cuh"
#include "predicated_mainloop.h"
#include "scatter_epilogue.h"
#include "sparse_grid.h"

#include <cute/tensor.hpp>

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// Kernel parameters (passed from host to device)
// =============================================================================

template <typename Types>
struct KernelParams {
    using ElemWt   = typename Types::ElemWt;
    using ElemFeat = typename Types::ElemFeat;
    using ElemOut  = typename Types::ElemOut;
    using ElemIdx  = typename Types::ElemIdx;

    // Grid accessors (device pointers, set by host)
    GridBatchImpl::Accessor iter_acc;
    GridBatchImpl::Accessor gather_acc;
    int64_t iter_batch_idx;
    int64_t gather_batch_idx;

    // Feature / weight / output arrays
    ElemFeat const *features;
    ElemWt const *weights;
    ElemOut *output;

    // Channel dimensions (runtime)
    int C_in;
    int C_out;
};

// =============================================================================
// SharedStorage: union of index maps + GEMM buffers + predicates
// =============================================================================
//
// The index maps persist across all clusters and blocks within the leaf.
// The GEMM buffers and predicates are reused per cluster/block.

template <geometry_like Geom, tiling_like Tiling, types_like Types>
struct SharedStorage {
    using ElemIdx = typename Types::ElemIdx;

    // ---- Persistent across the entire leaf ----
    ElemIdx gather_map[Geom::halo_vol()];
    ElemIdx scatter_map[Geom::leaf_vol()];

    // ---- Reused per cluster ----
    bool gather_pred[Tiling::cl_halo_vol()];
    bool scatter_pred[Tiling::cl_vol()];

    // ---- GEMM shared memory is managed separately via the mainloop's
    //      SharedStorage, allocated from the same smem pool but after
    //      the index maps. The mainloop uses its own char* offset. ----
};

// =============================================================================
// leaf_igemm_kernel: the __global__ kernel
// =============================================================================

template <geometry_like Geom, types_like Types, tiling_like Tiling, conv_variant Variant>
__global__ void __launch_bounds__(256)
    leaf_igemm_kernel(KernelParams<Types> params) {
    using ElemIdx  = typename Types::ElemIdx;
    using ElemFeat = typename Types::ElemFeat;
    using ElemWt   = typename Types::ElemWt;
    using ElemAcc  = typename Types::ElemAcc;
    using ElemOut  = typename Types::ElemOut;

    // Each threadblock processes one leaf from the iteration grid.
    int const leaf_id = blockIdx.x;

    // Construct SparseGrid adapters for both grids.
    auto iter_grid   = make_sparse_grid<ElemIdx>(params.iter_acc, params.iter_batch_idx);
    auto gather_grid = make_sparse_grid<ElemIdx>(params.gather_acc, params.gather_batch_idx);

    // ---- Shared memory ----
    extern __shared__ char smem_raw[];
    auto &smem = *reinterpret_cast<SharedStorage<Geom, Tiling, Types> *>(smem_raw);

    // ========================================================================
    // PHASE 1: Build index maps
    // ========================================================================

    build_index_maps<Geom, Variant>(iter_grid, gather_grid, leaf_id, smem.scatter_map,
                                    smem.gather_map);
    __syncthreads();

    // ========================================================================
    // PHASE 2-3: Tiled GEMM with predicated gather/scatter
    // ========================================================================

    int const C_in    = params.C_in;
    int const C_out   = params.C_out;
    int const contract = C_in * Geom::kern_vol();

    // The M dimension is output channels (for forward).
    // For input_grad, M = C_in. For weight_grad, M = C_out.
    // This is selected at the call site via the dispatch; here we
    // parameterize generically.
    int const M_total = C_out; // TODO: variant-dependent

    constexpr int TILE_M   = Tiling::tile_m();
    constexpr int BLK_VOL  = Tiling::blk_vol();

    // Accumulator buffer in registers (conceptually TILE_M x BLK_VOL).
    // For simplicity in this initial implementation, we use a local array.
    // A production version would use CuTe's register-resident fragments.
    ElemAcc accum[TILE_M * BLK_VOL];

    dcoord3 const B_shape = to_dcoord3(Tiling::B());
    dcoord3 const leaf3d  = to_dcoord3(Geom::leaf3());

    for (int m_tile = 0; m_tile < M_total; m_tile += TILE_M) {

        for (int cl_id = 0; cl_id < Tiling::ncl_tot(); ++cl_id) {

            // Build cluster predicates.
            dcoord3 const cl_coord = ddecode3(cl_id, to_dcoord3(Tiling::ncl()));
            dcoord3 const cl_orig  = cl_coord * to_dcoord3(Tiling::cl_extent());

            build_cluster_predicates<Geom, Tiling>(smem.gather_map, smem.scatter_map, cl_orig,
                                                   smem.gather_pred, smem.scatter_pred);
            __syncthreads();

            // Iterate over blocks within the cluster.
            constexpr int BLOCKS_PER_CL = prod(Tiling::CL());
            for (int local_blk = 0; local_blk < BLOCKS_PER_CL; ++local_blk) {
                dcoord3 const blk_in_cl = ddecode3(local_blk, to_dcoord3(Tiling::CL()));
                dcoord3 const block_orig = cl_orig + blk_in_cl * B_shape;

                // Zero the accumulator.
                for (int i = 0; i < TILE_M * BLK_VOL; ++i) {
                    accum[i] = ElemAcc(0);
                }

                // ---- Inner GEMM loop over contraction tiles ----
                // In the full CUTLASS-integrated version, this is handled by
                // the predicated mainloop (CollectiveMma). For now, we show
                // the logical structure with explicit loads.
                //
                // TODO: Replace this with the CuTe ComposedLayout GEMM
                // pipeline from predicated_mainloop.h once the full
                // integration is in place.

                int const tid      = threadIdx.x;
                int const nthreads = blockDim.x;

                for (int ck_tile = 0; ck_tile < contract; ck_tile += Tiling::tile_ck()) {
                    // Each thread accumulates its portion.
                    for (int idx = tid; idx < TILE_M * BLK_VOL; idx += nthreads) {
                        int const m = idx / BLK_VOL;
                        int const n = idx % BLK_VOL;

                        ElemAcc partial = ElemAcc(0);

                        for (int ck_local = 0;
                             ck_local < Tiling::tile_ck() && (ck_tile + ck_local) < contract;
                             ++ck_local) {
                            int const ck = ck_tile + ck_local;

                            // im2col: map (ck, n) -> (voxel_idx, channel)
                            auto [voxel_idx, c] =
                                im2col_map<Geom, Tiling, Variant>::apply(ck, n, block_orig,
                                                                        smem.gather_map,
                                                                        Geom::kern_vol());
                            // B-matrix element (predicated)
                            ElemFeat b_val = ElemFeat(0);
                            if (voxel_idx != ElemIdx(0)) {
                                int64_t const feat_addr =
                                    static_cast<int64_t>(voxel_idx) * C_in + c;
                                b_val = params.features[feat_addr];
                            }

                            // A-matrix element (filter weight)
                            int64_t const wt_addr =
                                static_cast<int64_t>(m_tile + m) * contract + ck;
                            ElemWt const a_val = params.weights[wt_addr];

                            partial += static_cast<ElemAcc>(a_val) * static_cast<ElemAcc>(b_val);
                        }

                        accum[idx] += partial;
                    }
                    __syncthreads();
                }

                // ---- Scatter epilogue ----
                // Compute scatter_pred for this specific block (subset of cluster pred).
                // For simplicity, use the scatter_map directly.
                bool block_scatter_pred[BLK_VOL];
                for (int n = 0; n < BLK_VOL; ++n) {
                    dcoord3 const v       = ddecode3(n, B_shape);
                    dcoord3 const out_pos = block_orig + v;
                    int const out_lin     = dencode3(out_pos, leaf3d);
                    block_scatter_pred[n] = (smem.scatter_map[out_lin] != ElemIdx(0));
                }

                scatter_write<Variant>(accum, TILE_M, BLK_VOL, smem.scatter_map, block_orig,
                                       B_shape, leaf3d, params.output, C_out, m_tile,
                                       block_scatter_pred);
                __syncthreads();
            }
        }
    }
}

// =============================================================================
// Launch wrapper
// =============================================================================

template <geometry_like Geom, types_like Types, tiling_like Tiling, conv_variant Variant>
void
launch_leaf_igemm(KernelParams<Types> const &params, int num_leaves, cudaStream_t stream) {
    if (num_leaves == 0) {
        return;
    }

    constexpr int THREADS = 256;
    constexpr size_t SMEM = sizeof(SharedStorage<Geom, Tiling, Types>);

    // If smem exceeds baseline 48 KB, request dynamic shared memory.
    if constexpr (SMEM > SMEM_BASELINE_BYTES) {
        cudaFuncSetAttribute(leaf_igemm_kernel<Geom, Types, Tiling, Variant>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(SMEM));
    }

    leaf_igemm_kernel<Geom, Types, Tiling, Variant>
        <<<num_leaves, THREADS, SMEM, stream>>>(params);
}

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_KERNEL_CUH
