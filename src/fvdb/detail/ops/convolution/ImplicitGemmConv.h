// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ImplicitGemmConv.h -- Leaf-fused sparse convolution.
//
// Single kernel launch: one CUDA block per output NanoVDB leaf.
// Each block builds gather/scatter index maps in shared memory by
// probing the input tree, then runs the full convolution for that
// leaf without touching the tree again.
//
// No topology pre-pass.  No intermediate buffers.  No global index arrays.
//
// Current restrictions:
//   - Sm80+ only (runtime check)
//   - fp16 or fp32 features and weights
//   - C_in, C_out must be positive multiples of 32
//   - Kernel sizes: 3x3x3, 5x5x5 (templatized, others error)
//   - Stride must be (1,1,1) (sub-manifold convolution)
//   - Forward only (no backward)
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_IMPLICITGEMMCONV_H
#define FVDB_DETAIL_OPS_CONVOLUTION_IMPLICITGEMMCONV_H

#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// Leaf-fused sparse convolution (forward only).
///
/// @param features     Input features, shape [feature_total_voxels, C_in].
/// @param weights      Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param feature_grid Input grid (NanoVDB ValueOnIndex on device).
/// @param output_grid  Output grid (NanoVDB ValueOnIndex on device).
/// @param kernel_size  Spatial kernel dimensions (must be 3x3x3 or 5x5x5).
/// @param stride       Convolution stride (must be 1x1x1).
/// @return             Output features, shape [output_total_voxels, C_out].
torch::Tensor implicitGemmConv(torch::Tensor features,
                               torch::Tensor weights,
                               GridBatchImpl const &feature_grid,
                               GridBatchImpl const &output_grid,
                               nanovdb::Coord kernel_size,
                               nanovdb::Coord stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_IMPLICITGEMMCONV_H
