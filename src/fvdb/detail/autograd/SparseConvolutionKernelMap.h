// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_SPARSECONVOLUTIONKERNELMAP_H
#define FVDB_DETAIL_AUTOGRAD_SPARSECONVOLUTIONKERNELMAP_H

#include <fvdb/SparseConvPackInfo.h>
#include <fvdb/detail/autograd/Common.h>

#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct SparseConvolutionKernelMap : public torch::autograd::Function<SparseConvolutionKernelMap> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    struct Topology {
        torch::Tensor neighborMap;
        torch::Tensor neighborSizes;
        int64_t sourceTotalVoxelCount = 0;
        int64_t targetTotalVoxelCount = 0;
        nanovdb::Vec3i kernelSize     = nanovdb::Vec3i{1, 1, 1};
        nanovdb::Vec3i stride         = nanovdb::Vec3i{1, 1, 1};
        bool tranposed                = false;
        bool useME                    = false;
    };

    static variable_list
    forward(AutogradContext *ctx, Variable inFeatures, Variable kernels, Topology topology);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_SPARSECONVOLUTIONKERNELMAP_H
