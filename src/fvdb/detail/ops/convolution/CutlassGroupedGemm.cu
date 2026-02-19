// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CutlassGroupedGemm.cu -- CUTLASS grouped-GEMM sparse convolution (forward only).
//
// Replaces the sequential per-offset torch::mm loop from GatherScatterDefault
// with a single CUTLASS grouped GEMM launch, plus vectorised gather and
// single-launch atomic scatter-add.
//
// Pipeline (forward):
//   1. Gather features into contiguous [total_pairs, Cin] fp16 buffer
//   2. CUTLASS grouped GEMM (all offsets in one launch, fp16 in, fp32 accum)
//   3. Atomic scatter-add GEMM output into fp32 accumulator (single launch)
//   4. Cast fp32 accumulator to fp16 output
//

#include <fvdb/detail/ops/convolution/CutlassGroupedGemm.h>

// NOTE: torch / ATen / c10 headers MUST precede CUTLASS headers.
// nvcc 13.1 causes CUTLASS 4.x to use <cccl/cuda/std/...> from the local
// CUDA toolkit, but those headers resolve internal <cuda/std/...> includes
// against the conda CUDA 12.9 copies (via -isystem), which lack the
// _LIBCUDACXX_HAS_SPACESHIP_OPERATOR macro.  Including torch first loads
// the 12.9 <cuda/std/utility> and sets its include guard, preventing the
// mismatched 13.1 cccl/ variant from ever being processed.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

// ============================================================================
// CUTLASS type configuration
// ============================================================================

// fp16 A * fp16 B -> fp32 C, RowMajor everything, sm80 tensor cores.
using CutlassElementA    = cutlass::half_t;
using CutlassElementB    = cutlass::half_t;
using CutlassElementC    = float;
using CutlassAccumulator = float;

using CutlassLayoutA = cutlass::layout::RowMajor;
using CutlassLayoutB = cutlass::layout::RowMajor;
using CutlassLayoutC = cutlass::layout::RowMajor;

static constexpr int kAlignmentA = 8; // 8 x half = 16 bytes
static constexpr int kAlignmentB = 8;

using CutlassEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    CutlassElementC,
    128 / cutlass::sizeof_bits<CutlassElementC>::value, // 4 floats = 16 bytes
    CutlassAccumulator,
    CutlassAccumulator>;

using CutlassGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    CutlassElementA,
    CutlassLayoutA,
    cutlass::ComplexTransform::kNone,
    kAlignmentA,
    CutlassElementB,
    CutlassLayoutB,
    cutlass::ComplexTransform::kNone,
    kAlignmentB,
    CutlassElementC,
    CutlassLayoutC,
    CutlassAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,   // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,    // MMA instruction
    CutlassEpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4                                       // pipeline stages
    >::GemmKernel;

using CutlassGemmGrouped = cutlass::gemm::device::GemmGrouped<CutlassGemmKernel>;

// Narrow-N variant: 128x64 threadblock for GEMM N-dimension <= 64.
using CutlassGemmKernelNarrow = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    CutlassElementA,
    CutlassLayoutA,
    cutlass::ComplexTransform::kNone,
    kAlignmentA,
    CutlassElementB,
    CutlassLayoutB,
    cutlass::ComplexTransform::kNone,
    kAlignmentB,
    CutlassElementC,
    CutlassLayoutC,
    CutlassAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>, // narrower N-tile
    cutlass::gemm::GemmShape<64, 32, 32>,  // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,   // MMA instruction
    CutlassEpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4                                      // pipeline stages
    >::GemmKernel;

using CutlassGemmGroupedNarrow = cutlass::gemm::device::GemmGrouped<CutlassGemmKernelNarrow>;

// ============================================================================
// CUDA kernel launch helpers
// ============================================================================

static constexpr int kBlockSize = 256;

static int
gridSizeFor(int64_t total_elements) {
    return static_cast<int>(
        std::min((total_elements + kBlockSize - 1) / kBlockSize, static_cast<int64_t>(4096)));
}

// ============================================================================
// Kernel: gather fp16 features into contiguous buffer (vectorised)
// ============================================================================
//
// dst[i, :] = src[indices[i], :]
// C must be a multiple of 32, so C/8 >= 4 and all float4 accesses are
// naturally 16-byte aligned.

__global__ void
gatherHalfKernel(at::Half const *__restrict__ src,    // [NA, C]
                 at::Half *__restrict__ dst,          // [TP, C]
                 int32_t const *__restrict__ indices, // [TP]
                 int64_t total_vecs,                  // TP * (C / 8)
                 int64_t C_vec) {                     // C / 8
    auto const *src4 = reinterpret_cast<float4 const *>(src);
    auto *dst4       = reinterpret_cast<float4 *>(dst);

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_vecs;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t const pos     = idx / C_vec;
        int64_t const v       = idx % C_vec;
        int32_t const src_row = indices[pos];
        dst4[idx]             = src4[static_cast<int64_t>(src_row) * C_vec + v];
    }
}

// ============================================================================
// Kernel: atomic scatter-add fp32 GEMM output into fp32 accumulator
// ============================================================================

__global__ void
scatterAddF32Kernel(float const *__restrict__ src,       // [TP, C]
                    int32_t const *__restrict__ indices, // [TP]
                    int64_t TP,
                    int64_t C,
                    float *__restrict__ dst)             // [NB, C]
{
    int64_t const total = TP * C;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t const pos     = idx / C;
        int64_t const c       = idx % C;
        int32_t const dst_row = indices[pos];
        atomicAdd(&dst[static_cast<int64_t>(dst_row) * C + c], src[idx]);
    }
}

// ============================================================================
// CUTLASS grouped GEMM runner
// ============================================================================

template <typename GemmGroupedT>
static void
runCutlassGroupedGemm(torch::Tensor h_problem_sizes, // [num_groups, 3] int32, host
                      int num_groups,
                      torch::Tensor d_problem_sizes, // [num_groups, 3] int32, device
                      int64_t *d_pack,               // [8 * num_groups] int64, device
                      torch::Device device,
                      cudaStream_t stream,
                      char const *label) {
    int threadblock_count = GemmGroupedT::sufficient(
        reinterpret_cast<cutlass::gemm::GemmCoord *>(h_problem_sizes.data_ptr<int32_t>()),
        num_groups);

    typename GemmGroupedT::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);

    typename GemmGroupedT::Arguments args(
        reinterpret_cast<cutlass::gemm::GemmCoord *>(d_problem_sizes.data_ptr<int32_t>()),
        num_groups,
        threadblock_count,
        epilogue,
        reinterpret_cast<CutlassElementA **>(d_pack + 0 * num_groups),
        reinterpret_cast<CutlassElementB **>(d_pack + 1 * num_groups),
        reinterpret_cast<CutlassElementC **>(d_pack + 2 * num_groups),
        reinterpret_cast<CutlassElementC **>(d_pack + 3 * num_groups),
        d_pack + 4 * num_groups,
        d_pack + 5 * num_groups,
        d_pack + 6 * num_groups,
        d_pack + 7 * num_groups,
        reinterpret_cast<cutlass::gemm::GemmCoord *>(h_problem_sizes.data_ptr<int32_t>()));

    size_t workspace_bytes = GemmGroupedT::get_workspace_size(args);
    auto workspace = torch::empty({std::max(static_cast<int64_t>(workspace_bytes), int64_t{1})},
                                  torch::dtype(torch::kByte).device(device));

    GemmGroupedT gemm_op;
    cutlass::Status status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                label,
                ": CUTLASS initialize failed: ",
                cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                label,
                ": CUTLASS run failed: ",
                cutlass::cutlassGetStatusString(status));
}

// ============================================================================
// Forward sparse convolution via CUTLASS grouped GEMM
// ============================================================================

torch::Tensor
cutlassGroupedGemmConv(torch::Tensor features,
                       torch::Tensor weights,
                       GatherScatterDefaultTopology const &topo) {
    // ---- Precondition checks ----
    TORCH_CHECK(features.dim() == 2, "cutlassGroupedGemmConv: features must be 2D");
    TORCH_CHECK(features.size(0) == topo.feature_total_voxels,
                "cutlassGroupedGemmConv: features.size(0) mismatch");
    TORCH_CHECK(features.scalar_type() == torch::kFloat16,
                "cutlassGroupedGemmConv: features must be fp16");
    TORCH_CHECK(features.is_contiguous(), "cutlassGroupedGemmConv: features must be contiguous");
    TORCH_CHECK(features.is_cuda(), "cutlassGroupedGemmConv: features must be on CUDA");

    TORCH_CHECK(weights.dim() == 5,
                "cutlassGroupedGemmConv: weights must be 5D [Cout, Cin, k0, k1, k2]");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat16,
                "cutlassGroupedGemmConv: weights must be fp16");
    TORCH_CHECK(features.size(1) == weights.size(1),
                "cutlassGroupedGemmConv: Cin mismatch between features and weights");
    TORCH_CHECK(weights.size(2) == topo.kernel_size[0] && weights.size(3) == topo.kernel_size[1] &&
                    weights.size(4) == topo.kernel_size[2],
                "cutlassGroupedGemmConv: weights spatial dims must match topology kernel_size");
    TORCH_CHECK(features.device() == weights.device(),
                "cutlassGroupedGemmConv: features and weights must be on same device");

    int64_t const Cin  = features.size(1);
    int64_t const Cout = weights.size(0);
    TORCH_CHECK(Cin > 0 && Cin % 32 == 0,
                "cutlassGroupedGemmConv: Cin must be a positive multiple of 32, got ",
                Cin);
    TORCH_CHECK(Cout > 0 && Cout % 32 == 0,
                "cutlassGroupedGemmConv: Cout must be a positive multiple of 32, got ",
                Cout);

    // ---- Device / stream setup ----
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const NB = topo.output_total_voxels;
    int64_t const K  = topo.kernel_volume;
    int64_t const TP = topo.total_pairs;

    // ---- Allocate output (fp32 accumulator, cast to fp16 at end) ----
    auto output_f32 = torch::zeros({NB, Cout}, torch::dtype(torch::kFloat32).device(device));
    if (NB == 0 || K == 0 || TP == 0) {
        return output_f32.to(torch::kFloat16);
    }

    // ---- Reshape weights: [Cout, Cin, k0, k1, k2] -> [K, Cin, Cout] row-major ----
    auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, Cin, Cout}).contiguous();

    // ---- Phase 1: Gather features into [TP, Cin] fp16 buffer (vectorised) ----
    auto buf_A = torch::empty({TP, Cin}, torch::dtype(torch::kFloat16).device(device));
    {
        int64_t const C_vec      = Cin / 8;
        int64_t const total_vecs = TP * C_vec;
        gatherHalfKernel<<<gridSizeFor(total_vecs), kBlockSize, 0, stream>>>(
            features.data_ptr<at::Half>(),
            buf_A.data_ptr<at::Half>(),
            topo.gather_indices.data_ptr<int32_t>(),
            total_vecs,
            C_vec);
    }

    // ---- Phase 2: CUTLASS grouped GEMM ----
    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    int num_groups = 0;
    for (int64_t k = 0; k < K; ++k) {
        if (off_acc[k + 1] > off_acc[k])
            ++num_groups;
    }

    auto buf_C = torch::empty({TP, Cout}, torch::dtype(torch::kFloat32).device(device));

    if (num_groups > 0) {
        auto h_problem_sizes = torch::empty({num_groups, 3}, torch::kInt32);
        auto h_packed        = torch::empty({8, num_groups}, torch::kInt64);

        auto ps = h_problem_sizes.accessor<int32_t, 2>();
        auto pk = h_packed.accessor<int64_t, 2>();

        uintptr_t const base_A = reinterpret_cast<uintptr_t>(buf_A.data_ptr<at::Half>());
        uintptr_t const base_B = reinterpret_cast<uintptr_t>(W.data_ptr<at::Half>());
        uintptr_t const base_C = reinterpret_cast<uintptr_t>(buf_C.data_ptr<float>());

        int g = 0;
        for (int64_t k = 0; k < K; ++k) {
            int64_t const start = off_acc[k];
            int64_t const Mk    = off_acc[k + 1] - start;
            if (Mk == 0)
                continue;

            ps[g][0] = static_cast<int32_t>(Mk);
            ps[g][1] = static_cast<int32_t>(Cout);
            ps[g][2] = static_cast<int32_t>(Cin);

            pk[0][g] = static_cast<int64_t>(base_A +
                                            static_cast<uintptr_t>(start * Cin * sizeof(at::Half)));
            pk[1][g] = static_cast<int64_t>(
                base_B + static_cast<uintptr_t>(k * Cin * Cout * sizeof(at::Half)));
            pk[2][g] =
                static_cast<int64_t>(base_C + static_cast<uintptr_t>(start * Cout * sizeof(float)));
            pk[3][g] = pk[2][g]; // D = C (beta=0, in-place)

            pk[4][g] = Cin;      // ldA: A [Mk, Cin]
            pk[5][g] = Cout;     // ldB: B [Cin, Cout]
            pk[6][g] = Cout;     // ldC: C [Mk, Cout]
            pk[7][g] = Cout;     // ldD: D [Mk, Cout]
            ++g;
        }

        auto d_problem_sizes = h_problem_sizes.to(device).contiguous();
        auto d_packed        = h_packed.to(device).contiguous();

        if (Cout <= 64) {
            runCutlassGroupedGemm<CutlassGemmGroupedNarrow>(h_problem_sizes,
                                                            num_groups,
                                                            d_problem_sizes,
                                                            d_packed.data_ptr<int64_t>(),
                                                            device,
                                                            stream,
                                                            "cutlassGroupedGemmConv");
        } else {
            runCutlassGroupedGemm<CutlassGemmGrouped>(h_problem_sizes,
                                                      num_groups,
                                                      d_problem_sizes,
                                                      d_packed.data_ptr<int64_t>(),
                                                      device,
                                                      stream,
                                                      "cutlassGroupedGemmConv");
        }
    }

    // ---- Phase 3: Scatter-add GEMM output into accumulator (single launch) ----
    {
        int64_t const total = TP * Cout;
        scatterAddF32Kernel<<<gridSizeFor(total), kBlockSize, 0, stream>>>(
            buf_C.data_ptr<float>(),
            topo.scatter_indices.data_ptr<int32_t>(),
            TP,
            Cout,
            output_f32.data_ptr<float>());
    }

    // ---- Cast fp32 accumulator to fp16 output ----
    return output_f32.to(torch::kFloat16);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
