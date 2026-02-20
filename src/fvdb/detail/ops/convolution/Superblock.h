// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Superblock.h -- Superblock GEMM sparse convolution.
//
// Assigns one CUDA block per superblock (N consecutive NanoVDB leaves).
// Phase A cooperatively builds halo maps and compacts active voxels into
// shared memory.  Phase B runs a tiled GEMM over only the compacted active
// voxels using CuTe TiledMMA tensor cores.
//
// Supports forward, backward (input gradient + weight gradient), and
// transposed forward in a single unified kernel template.
//
// No topology pre-pass.  No intermediate buffers.  No global index arrays.
//
// Restrictions:
//   - Sm80+ only (runtime check)
//   - fp16 or fp32 features and weights
//   - C_in, C_out must be positive multiples of 32
//   - Kernel sizes: 3x3x3, 5x5x5, 7x7x7 (templatized, others error)
//   - Stride must be (1,1,1) (sub-manifold convolution)
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_SUPERBLOCK_H
#define FVDB_DETAIL_OPS_CONVOLUTION_SUPERBLOCK_H

#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// Superblock GEMM sparse convolution (forward).
///
/// @param features     Input features, shape [feature_total_voxels, C_in].
/// @param weights      Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param feature_grid Input grid (NanoVDB ValueOnIndex on device).
/// @param output_grid  Output grid (NanoVDB ValueOnIndex on device).
/// @param kernel_size  Spatial kernel dimensions (3x3x3, 5x5x5, or 7x7x7).
/// @param stride       Convolution stride (must be 1x1x1).
/// @return             Output features, shape [output_total_voxels, C_out].
torch::Tensor superblockConv(torch::Tensor features,
                             torch::Tensor weights,
                             GridBatchImpl const &feature_grid,
                             GridBatchImpl const &output_grid,
                             nanovdb::Coord kernel_size,
                             nanovdb::Coord stride);

/// Backward pass for superblock sparse convolution.
///
/// @param grad_output  Gradient w.r.t. output, shape [output_total_voxels, C_out].
/// @param features     Input features from forward pass, shape [feature_total_voxels, C_in].
/// @param weights      Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param feature_grid Input grid.
/// @param output_grid  Output grid.
/// @param kernel_size  Spatial kernel dimensions.
/// @param stride       Convolution stride (must be 1x1x1).
/// @return             Tuple of (grad_features [feature_total_voxels, C_in],
///                               grad_weights  [C_out, C_in, k0, k1, k2]).
std::tuple<torch::Tensor, torch::Tensor>
superblockConvBackward(torch::Tensor grad_output,
                       torch::Tensor features,
                       torch::Tensor weights,
                       GridBatchImpl const &feature_grid,
                       GridBatchImpl const &output_grid,
                       nanovdb::Coord kernel_size,
                       nanovdb::Coord stride);

/// Superblock GEMM transposed sparse convolution (forward direction of transpose).
///
/// @param features     Input features, shape [source_total_voxels, C_in].
/// @param weights      Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param source_grid  Source grid (the grid features live on).
/// @param target_grid  Target grid (output grid for the transposed conv).
/// @param kernel_size  Spatial kernel dimensions.
/// @param stride       Convolution stride (must be 1x1x1).
/// @return             Output features, shape [target_total_voxels, C_out].
torch::Tensor superblockConvTranspose(torch::Tensor features,
                                      torch::Tensor weights,
                                      GridBatchImpl const &source_grid,
                                      GridBatchImpl const &target_grid,
                                      nanovdb::Coord kernel_size,
                                      nanovdb::Coord stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_SUPERBLOCK_H
