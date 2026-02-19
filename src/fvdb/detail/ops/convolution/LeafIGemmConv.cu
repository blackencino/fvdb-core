// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// LeafIGemmConv.cu -- Entry point for leaf-level implicit GEMM convolution.
//
// Initial instantiation: 3x3x3 kernel, stride 1, dilation 1, forward only.
// fp32 and fp16 element types.
//

#include "LeafIGemmConv.h"
#include "leaf_igemm/kernel.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

namespace fvdb {
namespace detail {
namespace ops {

// ============================================================================
// SM80 capability check
// ============================================================================

static bool
deviceSupportsSm80(torch::Device device) {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device.index());
    return major >= 8;
}

// ============================================================================
// Typed forward dispatcher
// ============================================================================

template <typename Types>
static torch::Tensor
leafIGemmConvTyped(torch::Tensor features,
                   torch::Tensor weights,
                   GridBatchImpl const &input_grid,
                   GridBatchImpl const &output_grid,
                   torch::ScalarType output_dtype) {
    using namespace leaf_igemm;

    using ElemWt   = typename Types::ElemWt;
    using ElemFeat = typename Types::ElemFeat;
    using ElemAcc  = typename Types::ElemAcc;
    using ElemOut  = typename Types::ElemOut;
    using ElemIdx  = typename Types::ElemIdx;

    // Geometry: 3x3x3, stride 1, dilation 1
    using Geom   = geom_3x3x3_s1;
    using Tiling = tiling_default<Geom, Types>;

    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const C_in  = features.size(1);
    int64_t const C_out = weights.size(0);

    // For batched grids, process each grid in the batch.
    // Initial implementation: single grid (batch_idx = 0).
    auto iter_acc   = output_grid.deviceAccessor();
    auto gather_acc = input_grid.deviceAccessor();

    int64_t const total_output_voxels = output_grid.totalVoxels();
    int64_t const num_leaves          = output_grid.totalLeaves();

    // Allocate output: accumulate in ElemAcc, then convert.
    auto accum_opts = torch::dtype(c10::CppTypeToScalarType<ElemAcc>::value).device(device);
    auto output     = torch::zeros({total_output_voxels, C_out}, accum_opts);

    if (total_output_voxels == 0 || num_leaves == 0) {
        return output.to(output_dtype);
    }

    // Reshape weights: [C_out, C_in, k0, k1, k2] -> [C_out, k0*k1*k2*C_in]
    // The kernel expects weights as [M, CONTRACT] where CONTRACT = C_in * KERN_VOL.
    auto W = weights.reshape({C_out, -1}).contiguous();

    // Build kernel parameters.
    // For single-grid batch, batch_idx = 0 for both grids.
    KernelParams<Types> params{
        iter_acc,
        gather_acc,
        0,
        0,
        reinterpret_cast<ElemFeat const *>(features.data_ptr()),
        reinterpret_cast<ElemWt const *>(W.data_ptr()),
        reinterpret_cast<ElemOut *>(output.data_ptr()),
        static_cast<int>(C_in),
        static_cast<int>(C_out),
    };

    launch_leaf_igemm<Geom, Types, Tiling, conv_variant::forward>(
        params, static_cast<int>(num_leaves), stream);

    return output.to(output_dtype);
}

// ============================================================================
// Entry point
// ============================================================================

torch::Tensor
leafIGemmConv(torch::Tensor features,
              torch::Tensor weights,
              GridBatchImpl const &input_grid,
              GridBatchImpl const &output_grid) {
    TORCH_CHECK(features.is_cuda(), "leafIGemmConv: features must be on CUDA");
    TORCH_CHECK(
        deviceSupportsSm80(features.device()),
        "leafIGemmConv: requires SM80+ (Ampere or newer)");

    TORCH_CHECK(features.dim() == 2, "leafIGemmConv: features must be 2D [N, C_in]");
    TORCH_CHECK(features.is_contiguous(), "leafIGemmConv: features must be contiguous");

    TORCH_CHECK(weights.dim() == 5,
                "leafIGemmConv: weights must be 5D [C_out, C_in, k0, k1, k2]");
    TORCH_CHECK(features.size(1) == weights.size(1),
                "leafIGemmConv: C_in mismatch between features and weights");
    TORCH_CHECK(weights.size(2) == 3 && weights.size(3) == 3 && weights.size(4) == 3,
                "leafIGemmConv: only 3x3x3 kernels supported in this implementation");
    TORCH_CHECK(features.device() == weights.device(),
                "leafIGemmConv: features and weights must be on same device");
    TORCH_CHECK(features.scalar_type() == weights.scalar_type(),
                "leafIGemmConv: features and weights must have same dtype");

    if (features.scalar_type() == torch::kFloat32) {
        return leafIGemmConvTyped<leaf_igemm::types_f32>(
            features, weights, input_grid, output_grid, torch::kFloat32);
    } else if (features.scalar_type() == torch::kFloat16) {
        return leafIGemmConvTyped<leaf_igemm::types_f16>(
            features, weights, input_grid, output_grid, torch::kFloat16);
    } else {
        TORCH_CHECK(false,
                    "leafIGemmConv: unsupported dtype ", features.scalar_type(),
                    ". Supported: fp32, fp16.");
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
