// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// scatter_epilogue.h -- Predicated scatter-write epilogue.
//
// After the GEMM accumulates results in registers, this epilogue:
//   1. Optionally converts Elem_Acc -> Elem_Out (e.g. f32 -> f16).
//   2. Predicated scatter-write to global memory via scatter_map,
//      skipping inactive output voxels.
//   3. For weight_grad variant: atomicAdd to a global accumulator
//      instead of direct scatter.
//
// This is NOT a CUTLASS epilogue template (those are deeply coupled to
// CUTLASS's epilogue visitor pattern). Instead, this is a standalone
// device function called after the mainloop completes, operating on
// the register-resident accumulator fragments.
//
// Target architecture: SM80 (Ampere).
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_SCATTER_EPILOGUE_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_SCATTER_EPILOGUE_H

#include "conv_variant.h"
#include "im2col_map.h"

#include <cuda_runtime.h>

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// Type conversion helpers
// =============================================================================

template <typename To, typename From>
__device__ __forceinline__ To
convert_type(From val) {
    return static_cast<To>(val);
}

// =============================================================================
// scatter_epilogue: write accumulator to global memory
// =============================================================================
//
// Template parameters:
//   Geom     : geometry_like
//   Tiling   : tiling_like
//   Variant  : conv_variant
//   ElemAcc  : accumulator element type (e.g. float)
//   ElemOut  : output storage type (e.g. float or half_t)
//   ElemIdx  : index type (e.g. uint32_t)
//
// For scatter_variant (forward, input_grad, transposed_conv):
//   output[scatter_map[block_orig + v] * C_out + m_offset + m] = convert(accum[m][n])
//
// For atomic_variant (weight_grad):
//   atomicAdd(&grad_weights[...], accum[m][n])

template <conv_variant Variant, typename ElemAcc, typename ElemOut, typename ElemIdx>
__device__ void
scatter_write(ElemAcc const *accum_buf,
              int tile_m,
              int blk_vol,
              ElemIdx const *scatter_map,
              dcoord3 block_orig,
              dcoord3 block_shape,
              dcoord3 leaf3,
              ElemOut *output,
              int C_out,
              int m_offset,
              bool const *scatter_pred) {

    int const tid      = threadIdx.x;
    int const nthreads = blockDim.x;
    int const total    = tile_m * blk_vol;

    for (int idx = tid; idx < total; idx += nthreads) {
        int const m = idx / blk_vol;
        int const n = idx % blk_vol;

        // Check scatter predicate for this output voxel.
        if (!scatter_pred[n]) {
            continue;
        }

        // Compute the output voxel's leaf-local position.
        dcoord3 const v         = ddecode3(n, block_shape);
        dcoord3 const out_pos   = block_orig + v;
        int const out_lin       = dencode3(out_pos, leaf3);
        ElemIdx const global_idx = scatter_map[out_lin];

        if (global_idx == static_cast<ElemIdx>(0)) {
            continue;
        }

        // NanoVDB stores 1-based indices; the base pointer is adjusted so
        // that global_idx can be used directly as an offset.
        int64_t const addr = static_cast<int64_t>(global_idx) * C_out + m_offset + m;

        ElemAcc const val = accum_buf[m * blk_vol + n];

        if constexpr (variant_traits<Variant>::atomic_accum()) {
            atomicAdd(&output[addr], convert_type<ElemOut>(val));
        } else {
            output[addr] = convert_type<ElemOut>(val);
        }
    }
}

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_SCATTER_EPILOGUE_H
