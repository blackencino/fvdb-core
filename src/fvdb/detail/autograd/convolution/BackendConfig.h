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

    BackendConfig
    to(torch::Device device) const {
        BackendConfig ret;
        ret.sourceGrid = sourceGrid.to(device);
        // Make sure that if the target grid is aliasing the source grid, we preserve that
        // relationship. This won't solve the problem of duplication of these grids if they're
        // referenced externally, though.
        if (sourceGrid.address() == targetGrid.address()) {
            ret.targetGrid = ret.sourceGrid;
        } else {
            ret.targetGrid = targetGrid.to(device);
        }
        ret.kernelSize = kernelSize;
        ret.stride     = stride;
        return ret;
    }
};

} // namespace convolution
} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_CONVOLUTION_BACKENDCONFIG_H
