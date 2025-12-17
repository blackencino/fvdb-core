// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_CONVOLUTION_CONVGATHERSCATTERBACKENDAG_H
#define FVDB_DETAIL_AUTOGRAD_CONVOLUTION_CONVGATHERSCATTERBACKENDAG_H

#include <fvdb/GridBatch.h>
#include <fvdb/detail/autograd/Common.h>
#include <fvdb/detail/autograd/convolution/BackendConfig.h>

#include <nanovdb/math/Math.h>

#include <torch/autograd.h>
#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace autograd {
namespace convolution {

struct GatherScatterAutograd : public Function<GatherScatterAutograd> {
    struct Topology {
        torch::Tensor neighborMap;
        torch::Tensor neighborSizes;
        int64_t sourceTotalVoxelCount = 0;
        int64_t targetTotalVoxelCount = 0;
        nanovdb::Vec3i kernelSize     = nanovdb::Vec3i{1, 1, 1};
        nanovdb::Vec3i stride         = nanovdb::Vec3i{1, 1, 1};

        Topology to(torch::Device device) const {
            return Topology{
                .neighborMap = neighborMap.to(device),
                .neighborSizes = neighborSizes.to(device),
                .sourceTotalVoxelCount = sourceTotalVoxelCount,
                .targetTotalVoxelCount = targetTotalVoxelCount,
                .kernelSize = kernelSize,
                .stride = stride,
            };
        }
    };

    static Topology topologyFromBackendConfig(BackendConfig config);

    static variable_list forward(AutogradContext *ctx,
                                 torch::Tensor inFeatures,
                                 torch::Tensor kernels,
                                 Topology topology);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace convolution
} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_CONVOLUTION_CONVGATHERSCATTERBACKENDAG_H
