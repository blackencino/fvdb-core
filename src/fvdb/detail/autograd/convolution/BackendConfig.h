// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_CONVOLUTION_BACKENDCONFIG_H
#define FVDB_DETAIL_AUTOGRAD_CONVOLUTION_BACKENDCONFIG_H

#include <fvdb/GridBatch.h>
#include <fvdb/detail/autograd/Common.h>

#include <nanovdb/math/Math.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace autograd {
namespace convolution {

struct BackendConfig {
    GridBatch sourceGrid;
    GridBatch targetGrid;
    nanovdb::Vec3i kernelSize;
    nanovdb::Vec3i stride;

    BackendConfig to(torch::Device device) const {
        return BackendConfig{
            .sourceGrid = sourceGrid.to(device),
            .targetGrid = targetGrid.to(device),
            .kernelSize = kernelSize,
            .stride = stride,
        };
    }
};

} // namespace convolution
} // namespace autograd
} // namespace detail
} // namespace fvdb


#endif // FVDB_DETAIL_AUTOGRAD_CONVOLUTION_BACKENDCONFIG_H
