// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ImplicitGemmConv.h -- CUTLASS 2.x fused gather-GEMM-scatter sparse convolution.
//
// GPU-only (Sm80+: Ampere through Blackwell). Reuses the GatherScatterDefault
// topology (CSR offset segments) and replaces the sequential per-offset
// torch::mm + separate gather/scatter kernels with a loop of fused
// gather-GEMM-scatter launches via CUTLASS DefaultGemmUniversal with
// GatherA / ScatterD enabled.
//
// Zero intermediate buffers: features are read directly through indirect
// indexing, and outputs are scatter-written in the epilogue.
//
// Current restrictions:
//   - Sm80+ only (runtime check, graceful fallback)
//   - fp16 or fp32 features and weights
//   - C_in, C_out must be multiples of 8 (fp16) or 4 (fp32)
//   - Forward only (no backward)
//   - Arbitrary kernel sizes (loops over K_vol offsets)
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_IMPLICITGEMMCONV_H
#define FVDB_DETAIL_OPS_CONVOLUTION_IMPLICITGEMMCONV_H

#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// CUTLASS 2.x fused gather-GEMM-scatter forward sparse convolution.
///
/// @param features  Input features, shape [feature_total_voxels, Cin].
/// @param weights   Kernel weights, shape [Cout, Cin, k0, k1, k2].
/// @param topo      Precomputed topology from gatherScatterDefaultSparseConvTopology.
/// @return          Output features, shape [output_total_voxels, Cout].
torch::Tensor
implicitGemmConv(torch::Tensor features,
                 torch::Tensor weights,
                 GatherScatterDefaultTopology const &topo);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_IMPLICITGEMMCONV_H
