// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_CONVOLUTION_CONVGATHERSCATTERBACKENDAG_H
#define FVDB_DETAIL_AUTOGRAD_CONVOLUTION_CONVGATHERSCATTERBACKENDAG_H

#include <fvdb/GridBatch.h>
#include <fvdb/detail/autograd/Common.h>

#include <nanovdb/math/Math.h>

#include <torch/autograd.h>
#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct ConvGatherScatterBackendAG : public Function<ConvGatherScatterBackendAG> {
    struct Topology {
        torch::Tensor neighborMap;
        torch::Tensor neighborSizes;
        int64_t sourceTotalVoxelCount = 0;
        int64_t targetTotalVoxelCount = 0;
        nanovdb::Vec3i kernelSize     = nanovdb::Vec3i{1, 1, 1};
        nanovdb::Vec3i stride         = nanovdb::Vec3i{1, 1, 1};
    };

    static Topology topologyFromGridBatches(GridBatch sourceGrid,
                                            GridBatch targetGrid,
                                            nanovdb::Vec3i kernelSize,
                                            nanovdb::Vec3i stride);

    static variable_list forward(AutogradContext *ctx,
                                 torch::Tensor inFeatures,
                                 torch::Tensor kernels,
                                 Topology topology);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

#if 0

// =============================================================================
// Immutable Convolution Backend Base Class
// =============================================================================

struct ConvolutionConfig {
    GridBatch sourceGrid;
    GridBatch targetGrid;
    nanovdb::Vec3i kernelSize = nanovdb::Vec3i{1, 1, 1};
    nanovdb::Vec3i stride = nanovdb::Vec3i{1, 1, 1};
    bool useTF32 = false;
};

class ConvolutionBackend {
protected:
    GridBatch mSourceGrid;
    GridBatch mTargetGrid;
    nanovdb::Vec3i mKernelSize = nanovdb::Vec3i{1, 1, 1};
    nanovdb::Vec3i mStride = nanovdb::Vec3i{1, 1, 1};
    bool mUseTF32 = false;

public:
    ConvolutionBackend(GridBatch sourceGrid,
                       GridBatch targetGrid,
                       nanovdb::Vec3i kernelSize,
                       nanovdb::Vec3i stride,
                       bool useTF32);

    virtual ~ConvolutionBackend() = default;

    GridBatch sourceGrid() const { return mSourceGrid; }
    GridBatch targetGrid() const { return mTargetGrid; }
    nanovdb::Vec3i kernelSize() const { return mKernelSize; }
    nanovdb::Vec3i stride() const { return mStride; }
    bool useTF32() const { return mUseTF32; }

    torch::Device device() const { return mSourceGrid.device(); }

    int64_t kernelVolume() const {
        return static_cast<int64_t>(mKernelSize[0]) *
               static_cast<int64_t>(mKernelSize[1]) *
               static_cast<int64_t>(mKernelSize[2]);
    }

    virtual JaggedTensor forward(JaggedTensor const& input, torch::Tensor const& weights) const = 0;

    virtual std::unique_ptr<ConvolutionBackend> to(torch::Device device) const = 0;
};

struct GatherScatterForwardMapping {
    torch::Tensor neighborMap;
    torch::Tensor neighborSizes;

    explicit GatherScatterForwardMapping(ConvolutionBackend const& backend);

    GatherScatterForwardMapping to(torch::Device device) const;
};

class GatherScatterBackend : public ConvolutionBackend {
protected:
    GatherScatterForwardMapping mForwardMapping;



public:
GatherScatterBackend(GridBatch sourceGrid,
    GridBatch targetGrid,
    nanovdb::Vec3i kernelSize,
    nanovdb::Vec3i stride,
    bool useTF32,
    GatherScatterForwardMapping forwardMapping);

    GatherScatterBackend(GridBatch sourceGrid,
                         GridBatch targetGrid,
                         nanovdb::Vec3i kernelSize,
                         nanovdb::Vec3i stride,
                         bool useTF32);


    std::unique_ptr<ConvolutionBackend> to(torch::Device device) const override;
    JaggedTensor forward(JaggedTensor const& input, torch::Tensor const& weights) const override;
};

class GatherScatterTransposeBackend : public ConvolutionBackend {
protected:
    GatherScatterForwardMapping mForwardMapping;
public:
    GatherScatterTransposeBackend(GridBatch sourceGrid,
                                  GridBatch targetGrid,
                                  nanovdb::Vec3i kernelSize,
                                  nanovdb::Vec3i stride,
                                  bool useTF32,
                                  GatherScatterForwardMapping forwardMapping);

    GatherScatterTransposeBackend(GridBatch sourceGrid,
                                  GridBatch targetGrid,
                                  nanovdb::Vec3i kernelSize,
                                  nanovdb::Vec3i stride,
                                  bool useTF32);

    explicit GatherScatterTransposeBackend(GatherScatterBackend const& untransposedBackend);


    std::unique_ptr<ConvolutionBackend> to(torch::Device device) const override;

    JaggedTensor forward(JaggedTensor const& input, torch::Tensor const& weights) const override;
};

#endif

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_CONVOLUTION_CONVGATHERSCATTERBACKENDAG_H
