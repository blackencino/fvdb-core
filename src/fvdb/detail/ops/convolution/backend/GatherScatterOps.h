// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_GATHERSCATTEROPS_H
#define FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_GATHERSCATTEROPS_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

// struct GatherScatterOp : public DeviceDispatchOp<GatherScatterOp> {
//     static void execute(CudaTag tag,
//                         torch::Tensor inFeat,
//                         torch::Tensor outFeat,
//                         torch::Tensor kernel,
//                         torch::Tensor nbMap,
//                         torch::Tensor nbSizes,
//                         bool middleAcceleration);
//     static void execute(CpuTag,
//                         torch::Tensor inFeat,
//                         torch::Tensor outFeat,
//                         torch::Tensor kernel,
//                         torch::Tensor nbMap,
//                         torch::Tensor nbSizes,
//                         bool middleAcceleration);
// };

// inline void
// gatherScatter(torch::Tensor inFeat,
//               torch::Tensor outFeat,
//               torch::Tensor kernel,
//               torch::Tensor nbMap,
//               torch::Tensor nbSizes,
//               bool middleAcceleration) {
//     GatherScatterOp::apply(
//         inFeat.device(), inFeat, outFeat, kernel, nbMap, nbSizes, middleAcceleration);
// }

// struct GatherScatterGradOp : public DeviceDispatchOp<GatherScatterOp> {
//     static void execute(CudaTag,
//                         torch::Tensor inFeat,
//                         torch::Tensor outFeat,
//                         torch::Tensor kernel,
//                         torch::Tensor nbMap,
//                         torch::Tensor nbSizes,
//                         bool middleAcceleration);
//     static void execute(CpuTag,
//                         torch::Tensor inFeat,
//                         torch::Tensor outFeat,
//                         torch::Tensor kernel,
//                         torch::Tensor nbMap,
//                         torch::Tensor nbSizes,
//                         bool middleAcceleration);
// };

// inline void
// gatherScatterGrad(torch::Tensor inFeat,
//                   torch::Tensor outFeat,
//                   torch::Tensor kernel,
//                   torch::Tensor nbMap,
//                   torch::Tensor nbSizes,
//                   bool middleAcceleration) {
//     GatherScatterGradOp::apply(
//         inFeat.device(), inFeat, outFeat, kernel, nbMap, nbSizes, middleAcceleration);
// }

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_GATHERSCATTEROPS_H
