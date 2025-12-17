// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_CONVOLUTIONBACKENDS_H
#define FVDB_CONVOLUTIONBACKENDS_H

#include <fvdb/GridBatch.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/autograd/convolution/BackendConfig.h>
#include <fvdb/detail/autograd/convolution/GatherScatter.h>

#include <nanovdb/math/Math.h>

#include <torch/types.h>

#include <map>
#include <string>

namespace fvdb {

struct ConvBackendGatherScatter {
    detail::autograd::convolution::BackendConfig config;
    detail::autograd::convolution::GatherScatterAutograd::Topology topology;

    static ConvBackendGatherScatter create(GridBatch sourceGrid,
                                           GridBatch targetGrid,
                                           nanovdb::Vec3i kernelSize,
                                           nanovdb::Vec3i stride,
                                           std::map<std::string, std::string> const &expertConfig);

    ConvBackendGatherScatter to(torch::Device device) const;

    JaggedTensor execute(JaggedTensor const &input, torch::Tensor weights) const;
};

} // namespace fvdb

#endif
