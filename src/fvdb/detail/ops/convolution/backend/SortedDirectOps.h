// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// SortedDirectOps.h
//
// Sorted Direct Sparse Convolution
// ================================
//
// This is a simplified, general-purpose sparse convolution implementation that:
// - Works with any kernel size, stride, and dtype
// - Eliminates the CPU loop present in GatherScatter
// - Uses a single CUDA kernel launch per forward pass
// - Reuses kernel weights via shared memory within each kernel position segment
//
// Algorithm Overview:
// ------------------
// The neighbor_map is already sorted by kernel position (from construction).
// We compute segment_starts as a prefix sum of neighbor_sizes.
// Each CUDA block handles one kernel position, loading its weights into shared
// memory and processing all edges in that segment.
//
// Data Structures:
// ---------------
// neighbor_map: [P, 2] = [in_idx, out_idx], sorted by kernel position
// segment_starts: [K+1] = prefix sum of neighbor_sizes (computed via cumsum)
// kernel: [K, c, o] = K weight matrices
// in_feat: [N, c] = input features
// out_feat: [M, o] = output features (initialized to zero)
//
// Python Fallback Implementation (for reference/CPU):
// ---------------------------------------------------
/*
def sparse_conv_sorted_direct_pytorch(in_feat, kernel, neighbor_map, segment_starts, M):
    """
    Memory-efficient PyTorch fallback implementation.

    Args:
        in_feat: [N, c] input features
        kernel: [K, c, o] weight matrices (one per kernel position)
        neighbor_map: [P, 2] = [in_idx, out_idx], sorted by kernel position
        segment_starts: [K+1] segment boundaries (prefix sum of sizes)
        M: number of output voxels

    Returns:
        out_feat: [M, o] output features
    """
    K, c, o = kernel.shape
    out_feat = torch.zeros(M, o, dtype=in_feat.dtype, device=in_feat.device)

    for k in range(K):
        seg_start = segment_starts[k].item()
        seg_end = segment_starts[k + 1].item()
        n_k = seg_end - seg_start

        if n_k == 0:
            continue

        in_idx = neighbor_map[seg_start:seg_end, 0]
        out_idx = neighbor_map[seg_start:seg_end, 1]

        gathered = in_feat[in_idx]          # (n_k, c) - memory: n_k * c
        transformed = gathered @ kernel[k]  # (n_k, o) - memory: n_k * o
        out_feat.index_add_(0, out_idx, transformed)

    return out_feat


# Memory Analysis:
# ----------------
# Peak intermediate memory: max(n_k) * (c + o)
# where n_k is edges per kernel position (typically P / K)
#
# For P = 1M edges, K = 27, c = o = 128:
#   max(n_k) ≈ P/K ≈ 37K
#   Peak ≈ 37K * 256 * 4 bytes ≈ 38 MB (acceptable)
#
# This is viable as a fallback/reference implementation.
*/

#ifndef FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SORTEDDIRECTOPS_H
#define FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SORTEDDIRECTOPS_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// Dispatch the sorted direct sparse convolution forward pass.
///
/// This implementation:
/// - Processes all kernel positions in a single CUDA kernel launch
/// - Uses shared memory to cache kernel weights per segment
/// - Supports arbitrary kernel sizes, strides, and dtypes
/// - Handles channel tiling for large c and o
///
/// @param in_feat      Input features [N, c]
/// @param out_feat     Output features [M, o] (must be pre-initialized to zero)
/// @param kernel       Kernel weights [K, c, o] where K = kernel volume
/// @param neighbor_map Edge list [P, 2] = [in_idx, out_idx], sorted by kernel position
/// @param segment_starts Segment boundaries [K+1] from prefix sum of neighbor_sizes
/// @param transpose    If true, swap gather/scatter roles for transpose conv
template <c10::DeviceType DeviceTag>
void dispatchSortedDirectConv(torch::Tensor in_feat,
                              torch::Tensor out_feat,
                              torch::Tensor kernel,
                              torch::Tensor neighbor_map,
                              torch::Tensor segment_starts,
                              bool transpose);

/// Dispatch the sorted direct sparse convolution backward pass.
///
/// @param in_feat          Input features from forward [N, c]
/// @param grad_in_feat     Gradient w.r.t. input features [N, c] (output)
/// @param grad_out_feat    Gradient w.r.t. output features [M, o] (input)
/// @param kernel           Kernel weights [K, c, o]
/// @param grad_kernel      Gradient w.r.t. kernel [K, c, o] (output)
/// @param neighbor_map     Edge list [P, 2]
/// @param segment_starts   Segment boundaries [K+1]
/// @param transpose        If true, swap roles for transpose conv backward
template <c10::DeviceType DeviceTag>
void dispatchSortedDirectConvGrad(torch::Tensor in_feat,
                                  torch::Tensor grad_in_feat,
                                  torch::Tensor grad_out_feat,
                                  torch::Tensor kernel,
                                  torch::Tensor grad_kernel,
                                  torch::Tensor neighbor_map,
                                  torch::Tensor segment_starts,
                                  bool transpose);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SORTEDDIRECTOPS_H
