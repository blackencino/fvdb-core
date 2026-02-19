// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CutlassGroupedGemm.h -- CUTLASS grouped-GEMM sparse convolution (forward only).
//
// GPU-only. Reuses the GatherScatterDefault topology (CSR offset segments)
// and replaces the sequential per-offset torch::mm loop with a single
// CUTLASS grouped GEMM launch (fp16 tensor cores, fp32 accumulate).
//
// Current restrictions (to be lifted once performance is validated):
//   - fp16 features and weights only
//   - Channel counts (Cin, Cout) must be multiples of 32
//   - CUDA only (no CPU fallback)
//   - Forward only (no backward, no transposed)
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_CUTLASSGROUPEDGEMM_H
#define FVDB_DETAIL_OPS_CONVOLUTION_CUTLASSGROUPEDGEMM_H

#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// CUTLASS grouped-GEMM forward sparse convolution (fp16 in, fp16 out).
///
/// @param features  Input features, shape [feature_total_voxels, Cin], fp16.
/// @param weights   Kernel weights, shape [Cout, Cin, k0, k1, k2], fp16.
/// @param topo      Precomputed topology from gatherScatterDefaultSparseConvTopology.
/// @return          Output features, shape [output_total_voxels, Cout], fp16.
torch::Tensor
cutlassGroupedGemmConv(torch::Tensor features,
                       torch::Tensor weights,
                       GatherScatterDefaultTopology const &topo);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_CUTLASSGROUPEDGEMM_H
