// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file PredGatherIGemmBackward.h
/// @brief Backward pass for predicated-gather implicit-GEMM sparse convolution.
///
/// Leaf-local no-kmap backward that walks NanoVDB topology directly,
/// avoiding global k-map construction. Uses FP32 shared-memory GEMM with
/// atomic fan-in for both dgrad and wgrad.
///
/// Constraints (same as forward):
///   - CUDA only, float32 only
///   - Input / output channels must be multiples of 32
///   - Uniform kernel sizes {3, 5, 7} and uniform strides {1, 2}
///   - Batch size 1
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_PREDGATHERIGEMMBACKWARD_H
#define FVDB_DETAIL_OPS_CONVOLUTION_PREDGATHERIGEMMBACKWARD_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Backward pass of predicated-gather IGEMM sparse convolution.
///
/// Returns gradients for features and weights.  Tensors use standard fVDB
/// layout (0-indexed features, weights as [K, C, T, R, S]).  NanoVDB 1-based
/// ValueOnIndex indices are converted to 0-based inline (matching the
/// forward pass's IndexedGather<-1> convention).
///
/// @param grad_output   Upstream gradient, shape [N_out, K], float32, on CUDA.
/// @param features      Input features (saved from forward), shape [N_in, C].
/// @param weights       Kernel weights, shape [K, C, ks, ks, ks], float32.
/// @param feature_grid  NanoVDB grid batch for the input (feature) voxels.
/// @param output_grid   NanoVDB grid batch for the output voxels.
/// @param kernel_size   Uniform spatial kernel extent (3, 5, or 7).
/// @param stride        Uniform convolution stride (1 or 2).
/// @return {grad_features [N_in, C], grad_weights [K, C, ks, ks, ks]}.
std::tuple<torch::Tensor, torch::Tensor>
predGatherIGemmSparseConvBackward(torch::Tensor grad_output,
                                  torch::Tensor features,
                                  torch::Tensor weights,
                                  GridBatchImpl const &feature_grid,
                                  GridBatchImpl const &output_grid,
                                  int kernel_size,
                                  int stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_PREDGATHERIGEMMBACKWARD_H
