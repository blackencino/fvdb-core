// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// LeafIGemmAltConv.h -- Leaf-level implicit GEMM sparse convolution (Alt).
//
// GPU-only (SM80 Ampere). One threadblock per output leaf, fusing
// topology densification and implicit GEMM into a single kernel launch.
// Uses CuTe ComposedLayout for virtual im2col and a predicated CUTLASS
// CollectiveMma for the pipelined GEMM, following the Sifakis reference
// implementation.
//
// Current restrictions:
//   - SM80+ only (Ampere and newer)
//   - fp32 features and weights (fp16 types defined, not yet instantiated)
//   - Forward only
//   - Kernel size deduced from weights tensor shape
//   - C_in, C_out must be multiples of 8
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAFIGEMMALTCONV_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAFIGEMMALTCONV_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// Leaf-level implicit GEMM forward sparse convolution (Alt backend).
///
/// @param features     Input features, shape [total_input_voxels, C_in].
/// @param weights      Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param input_grid   Input sparse grid (features are indexed into this).
/// @param output_grid  Output sparse grid (output voxels are defined by this).
/// @return             Output features, shape [total_output_voxels, C_out].
torch::Tensor
leafIGemmAltConv(torch::Tensor features,
                 torch::Tensor weights,
                 GridBatchImpl const &input_grid,
                 GridBatchImpl const &output_grid);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAFIGEMMALTCONV_H
