
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/ConvolutionBackends.h>

namespace fvdb {

using namespace detail::autograd::convolution;

ConvBackendGatherScatter
ConvBackendGatherScatter::create(GridBatch sourceGrid,
                                 GridBatch targetGrid,
                                 nanovdb::Vec3i kernelSize,
                                 nanovdb::Vec3i stride,
                                 std::map<std::string, std::string> const &expertConfig) {
    BackendConfig const config{
        .sourceGrid = sourceGrid,
        .targetGrid = targetGrid,
        .kernelSize = kernelSize,
        .stride     = stride,
    };

    auto const topology = GatherScatterAutograd::topologyFromBackendConfig(config);

    return ConvBackendGatherScatter{
        .config   = config,
        .topology = topology,
    };
}

ConvBackendGatherScatter
ConvBackendGatherScatter::to(torch::Device device) const {
    return ConvBackendGatherScatter{
        .config   = config.to(device),
        .topology = topology.to(device),
    };
}

JaggedTensor
ConvBackendGatherScatter::execute(JaggedTensor const &input, torch::Tensor weights) const {
    auto flat = GatherScatterAutograd::apply(input.jdata(), weights, topology)[0];
    return input.jagged_like(flat);
}

} // namespace fvdb
