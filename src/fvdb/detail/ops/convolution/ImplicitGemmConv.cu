// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ImplicitGemmConv.cu -- CUTLASS 2.x fused gather-GEMM-scatter sparse convolution.
//
// For each kernel offset k (0 .. K_vol-1), one CUTLASS kernel launch fuses:
//   - Gather: A-matrix rows read from features[gather_indices[i], :]
//   - GEMM:   Sm80 tensor-core MMA (Ampere through Blackwell)
//   - Scatter: D-matrix rows written to output[scatter_indices[i], :]
//
// Zero intermediate buffers. Memory usage is input + output + weights only.
//

#include <fvdb/detail/ops/convolution/ImplicitGemmConv.h>

// NOTE: torch / ATen / c10 headers MUST precede CUTLASS headers.
// See CutlassGroupedGemm.cu for the full explanation (CCCL version mismatch
// between local nvcc 13.1 and conda CUDA 12.9 toolkit headers).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/gemm.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

// ============================================================================
// Sm80+ capability check
// ============================================================================

static bool
deviceSupportsImplicitGemm(torch::Device device) {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device.index());
    return major >= 8;
}

// ============================================================================
// CUTLASS type definitions -- explicit specializations per element type
// ============================================================================

template <typename Element> struct ImplicitGemmTypes;

template <> struct ImplicitGemmTypes<cutlass::half_t> {
    using ElementA           = cutlass::half_t;
    using ElementB           = cutlass::half_t;
    using ElementC           = float;
    using ElementAccumulator = float;
    using ElementCompute     = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;

    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementC,
                                                     128 / cutlass::sizeof_bits<ElementC>::value,
                                                     ElementAccumulator,
                                                     ElementCompute>;

    using Gemm = cutlass::gemm::device::GemmUniversal<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,
        AlignmentA,
        AlignmentB,
        cutlass::arch::OpMultiplyAdd,
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone,
        true,
        false,
        true>;
};

template <> struct ImplicitGemmTypes<float> {
    using ElementA           = float;
    using ElementB           = float;
    using ElementC           = float;
    using ElementAccumulator = float;
    using ElementCompute     = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 4;
    static constexpr int AlignmentB = 4;

    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementC,
                                                     128 / cutlass::sizeof_bits<ElementC>::value,
                                                     ElementAccumulator,
                                                     ElementCompute>;

    using Gemm = cutlass::gemm::device::GemmUniversal<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 16>,
        cutlass::gemm::GemmShape<64, 64, 16>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,
        AlignmentA,
        AlignmentB,
        cutlass::arch::OpMultiplyAdd,
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone,
        true,
        false,
        true>;
};

// ============================================================================
// Per-offset GEMM runner
// ============================================================================

template <typename Types>
static void
runOneOffset(void const *features_ptr,
             void const *weights_k_ptr,
             void *output_ptr,
             int const *gather_ptr,
             int const *scatter_ptr,
             int M_k,
             int N,
             int K,
             float alpha,
             float beta,
             cudaStream_t stream) {
    using Gemm     = typename Types::Gemm;
    using ElementA = typename Types::ElementA;
    using ElementB = typename Types::ElementB;
    using ElementC = typename Types::ElementC;

    cutlass::gemm::GemmCoord problem_size(M_k, N, K);

    typename Types::EpilogueOp::Params epilogue_params(alpha, beta);

    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
                                  problem_size,
                                  1,
                                  epilogue_params,
                                  reinterpret_cast<ElementA const *>(features_ptr),
                                  reinterpret_cast<ElementB const *>(weights_k_ptr),
                                  reinterpret_cast<ElementC const *>(output_ptr),
                                  reinterpret_cast<ElementC *>(output_ptr),
                                  0,
                                  0,
                                  0,
                                  0,
                                  K,
                                  N,
                                  N,
                                  N,
                                  gather_ptr,
                                  nullptr,
                                  scatter_ptr);

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "implicitGemmConv: CUTLASS can_implement failed for M_k=",
                M_k,
                " N=",
                N,
                " K=",
                K,
                ": ",
                cutlass::cutlassGetStatusString(status));

    size_t workspace_bytes = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({std::max(static_cast<int64_t>(workspace_bytes), int64_t{1})},
                                  torch::dtype(torch::kByte).device(torch::kCUDA));

    status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "implicitGemmConv: CUTLASS initialize failed: ",
                cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "implicitGemmConv: CUTLASS run failed: ",
                cutlass::cutlassGetStatusString(status));
}

// ============================================================================
// Typed forward dispatcher
// ============================================================================

template <typename Types>
static torch::Tensor
implicitGemmConvTyped(torch::Tensor features,
                      torch::Tensor weights,
                      GatherScatterDefaultTopology const &topo,
                      torch::ScalarType output_dtype) {
    using ElementA = typename Types::ElementA;
    using ElementC = typename Types::ElementC;

    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const Cin   = features.size(1);
    int64_t const Cout  = weights.size(0);
    int64_t const NB    = topo.output_total_voxels;
    int64_t const K_vol = topo.kernel_volume;

    auto accum_opts = torch::dtype(c10::CppTypeToScalarType<ElementC>::value).device(device);
    auto output     = torch::zeros({NB, Cout}, accum_opts);

    if (NB == 0 || K_vol == 0 || topo.total_pairs == 0) {
        return output.to(output_dtype);
    }

    auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K_vol, Cin, Cout}).contiguous();

    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    for (int64_t k = 0; k < K_vol; ++k) {
        int64_t const start = off_acc[k];
        int64_t const M_k   = off_acc[k + 1] - start;
        if (M_k == 0)
            continue;

        int const *gather_k =
            reinterpret_cast<int const *>(topo.gather_indices.data_ptr<int32_t>() + start);
        int const *scatter_k =
            reinterpret_cast<int const *>(topo.scatter_indices.data_ptr<int32_t>() + start);

        void const *W_k = reinterpret_cast<void const *>(
            reinterpret_cast<ElementA const *>(W.data_ptr()) + k * Cin * Cout);

        runOneOffset<Types>(features.data_ptr(),
                            W_k,
                            output.data_ptr(),
                            gather_k,
                            scatter_k,
                            static_cast<int>(M_k),
                            static_cast<int>(Cout),
                            static_cast<int>(Cin),
                            1.0f,
                            1.0f,
                            stream);
    }

    return output.to(output_dtype);
}

// ============================================================================
// Entry point
// ============================================================================

torch::Tensor
implicitGemmConv(torch::Tensor features,
                 torch::Tensor weights,
                 GatherScatterDefaultTopology const &topo) {
    TORCH_CHECK(features.is_cuda(), "implicitGemmConv: features must be on CUDA");
    TORCH_CHECK(deviceSupportsImplicitGemm(features.device()),
                "implicitGemmConv: requires Sm80+ (Ampere or newer), got compute capability < 8.0");

    TORCH_CHECK(features.dim() == 2, "implicitGemmConv: features must be 2D");
    TORCH_CHECK(features.size(0) == topo.feature_total_voxels,
                "implicitGemmConv: features.size(0) mismatch");
    TORCH_CHECK(features.is_contiguous(), "implicitGemmConv: features must be contiguous");

    TORCH_CHECK(weights.dim() == 5, "implicitGemmConv: weights must be 5D [Cout, Cin, k0, k1, k2]");
    TORCH_CHECK(features.size(1) == weights.size(1),
                "implicitGemmConv: Cin mismatch between features and weights");
    TORCH_CHECK(weights.size(2) == topo.kernel_size[0] && weights.size(3) == topo.kernel_size[1] &&
                    weights.size(4) == topo.kernel_size[2],
                "implicitGemmConv: weights spatial dims must match topology kernel_size");
    TORCH_CHECK(features.device() == weights.device(),
                "implicitGemmConv: features and weights must be on same device");
    TORCH_CHECK(features.scalar_type() == weights.scalar_type(),
                "implicitGemmConv: features and weights must have same dtype");

    int64_t const Cin  = features.size(1);
    int64_t const Cout = weights.size(0);

    if (features.scalar_type() == torch::kFloat16) {
        TORCH_CHECK(Cin > 0 && Cin % 8 == 0,
                    "implicitGemmConv (fp16): Cin must be a positive multiple of 8, got ",
                    Cin);
        TORCH_CHECK(Cout > 0 && Cout % 8 == 0,
                    "implicitGemmConv (fp16): Cout must be a positive multiple of 8, got ",
                    Cout);
        return implicitGemmConvTyped<ImplicitGemmTypes<cutlass::half_t>>(
            features, weights, topo, torch::kFloat16);
    } else if (features.scalar_type() == torch::kFloat32) {
        TORCH_CHECK(Cin > 0 && Cin % 4 == 0,
                    "implicitGemmConv (fp32): Cin must be a positive multiple of 4, got ",
                    Cin);
        TORCH_CHECK(Cout > 0 && Cout % 4 == 0,
                    "implicitGemmConv (fp32): Cout must be a positive multiple of 4, got ",
                    Cout);
        return implicitGemmConvTyped<ImplicitGemmTypes<float>>(
            features, weights, topo, torch::kFloat32);
    } else {
        TORCH_CHECK(false,
                    "implicitGemmConv: unsupported dtype ",
                    features.scalar_type(),
                    ". Supported: fp16, fp32.");
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
