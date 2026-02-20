// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ImplicitGemmConv.cu -- Leaf-fused sparse convolution.
//
// One CUDA block per output NanoVDB leaf.  Each block:
//   Phase 1: Cooperatively builds gather/scatter index maps in shared memory
//            by probing the NanoVDB tree.  After __syncthreads the tree is
//            never touched again.
//   Phase 2: Computes the full convolution for this leaf's voxels using the
//            shared-memory maps for indirect feature reads and output writes.
//
// No topology pre-pass.  No intermediate buffers.  No global index arrays.
// Single kernel launch for the entire convolution.
//

#include <fvdb/detail/ops/convolution/ImplicitGemmConv.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <nanovdb/NanoVDB.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

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
// LeafConvGeometry -- compile-time spatial parameters
// ============================================================================

template <int T_, int R_, int S_> struct LeafConvGeometry {
    static constexpr int T       = T_;
    static constexpr int R       = R_;
    static constexpr int S       = S_;
    static constexpr int LeafDim = 8;
    static constexpr int Hx      = T + LeafDim - 1;
    static constexpr int Hy      = R + LeafDim - 1;
    static constexpr int Hz      = S + LeafDim - 1;
    static constexpr int HaloVol = Hx * Hy * Hz;
    static constexpr int LeafVol = LeafDim * LeafDim * LeafDim;
    static constexpr int KernVol = T * R * S;
    static constexpr int Dx      = -(T / 2);
    static constexpr int Dy      = -(R / 2);
    static constexpr int Dz      = -(S / 2);
};

// ============================================================================
// MMA configuration per scalar type
// ============================================================================

template <typename Scalar> struct MmaConfig;

template <> struct MmaConfig<float> {
    using MmaElement = cute::tfloat32_t;
    using MmaAtom    = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    static constexpr int TILE_K = 8;
};

template <> struct MmaConfig<c10::Half> {
    using MmaElement = cute::half_t;
    using MmaAtom    = cute::SM80_16x8x16_F32F16F16F32_TN;
    static constexpr int TILE_K = 16;
};

// ============================================================================
// Shared memory layout
// ============================================================================

template <typename Geom> struct LeafConvSmem {
    int64_t gather_map[Geom::HaloVol];
    int64_t scatter_map[Geom::LeafVol];
    static constexpr int TILE_M         = 32;
    static constexpr int TILE_N         = 32;
    static constexpr int TILE_BUF_BYTES = TILE_M * TILE_N * sizeof(float);
    alignas(16) char tile_buf[TILE_BUF_BYTES];
};

// ============================================================================
// Leaf-fused convolution kernel
// ============================================================================
//
// Grid:   <<<num_output_leaves, 128, sizeof(LeafConvSmem<Geom>)>>>
// Each block processes one output leaf.
// Phase 1 builds gather/scatter maps; Phase 2 uses CuTe TiledMMA (tensor cores).

template <typename Geom, typename Scalar>
__global__ void
leaf_fused_conv_kernel(GridBatchImpl::Accessor feat_acc,
                       GridBatchImpl::Accessor out_acc,
                       Scalar const *__restrict__ features,
                       Scalar const *__restrict__ weights,
                       Scalar *__restrict__ output,
                       int C_in,
                       int C_out) {
    using namespace cute;
    using Cfg        = MmaConfig<Scalar>;
    using MmaElement = typename Cfg::MmaElement;

    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = Cfg::TILE_K;

    using TiledMma =
        TiledMMA<MMA_Atom<typename Cfg::MmaAtom>, Layout<Shape<_2, _2, _1>>,
                 Tile<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>>;

    extern __shared__ char smem_raw[];
    auto &smem = *reinterpret_cast<LeafConvSmem<Geom> *>(smem_raw);

    int const leaf_id  = blockIdx.x;
    int const tid      = threadIdx.x;
    int const nthreads = blockDim.x;

    auto const *out_grid  = out_acc.grid(0);
    auto const *feat_grid = feat_acc.grid(0);
    int64_t const feat_vo = feat_acc.voxelOffset(0);
    int64_t const out_vo  = out_acc.voxelOffset(0);

    // ---- Phase 1: build scatter map (unchanged) ----
    auto const &out_leaf = out_grid->tree().template getFirstNode<0>()[leaf_id];
    for (int v = tid; v < Geom::LeafVol; v += nthreads) {
        if (out_leaf.isActive(v)) {
            smem.scatter_map[v] = out_vo + static_cast<int64_t>(out_leaf.getValue(v)) - 1;
        } else {
            smem.scatter_map[v] = -1;
        }
    }

    // ---- Phase 1: build gather map (unchanged) ----
    auto feat_tree_acc       = feat_grid->getAccessor();
    nanovdb::Coord origin    = out_leaf.origin();
    nanovdb::Coord halo_base = origin.offsetBy(Geom::Dx, Geom::Dy, Geom::Dz);

    for (int h = tid; h < Geom::HaloVol; h += nthreads) {
        int const hi = h / (Geom::Hy * Geom::Hz);
        int const hj = (h / Geom::Hz) % Geom::Hy;
        int const hk = h % Geom::Hz;
        nanovdb::Coord coord = halo_base.offsetBy(hi, hj, hk);
        if (feat_tree_acc.isActive(coord)) {
            smem.gather_map[h] = feat_vo + static_cast<int64_t>(feat_tree_acc.getValue(coord)) - 1;
        } else {
            smem.gather_map[h] = -1;
        }
    }

    __syncthreads();

    // ---- Phase 2: tiled MMA convolution ----
    //
    // GEMM: C[C_out, LeafVol] = W[C_out, K_total] * F[K_total, LeafVol]
    // where K_total = C_in * KernVol.
    //
    // Weight layout (from launcher): [K_total, C_out] row-major (stride-1 along C_out).
    // Feature loads use gather_map for spatial indirection.
    // Output writes use scatter_map.

    auto const *feat_mma = reinterpret_cast<MmaElement const *>(features);
    auto const *wt_mma   = reinterpret_cast<MmaElement const *>(weights);

    int const K_total = C_in * Geom::KernVol;

    auto *tile_A_ptr = reinterpret_cast<MmaElement *>(smem.tile_buf);
    auto *tile_B_ptr = reinterpret_cast<MmaElement *>(smem.tile_buf + TILE_M * TILE_K * sizeof(MmaElement));

    auto sA = make_tensor(make_smem_ptr(tile_A_ptr),
                          make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                      make_stride(Int<1>{}, Int<TILE_M>{})));
    auto sB = make_tensor(make_smem_ptr(tile_B_ptr),
                          make_layout(make_shape(Int<TILE_N>{}, Int<TILE_K>{}),
                                      make_stride(Int<1>{}, Int<TILE_N>{})));

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sB);
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);

    constexpr int AB_ELEMS = TILE_M * TILE_K;
    constexpr int BB_ELEMS = TILE_N * TILE_K;

    for (int m0 = 0; m0 < C_out; m0 += TILE_M) {
        for (int n0 = 0; n0 < Geom::LeafVol; n0 += TILE_N) {
            auto accum = partition_fragment_C(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
            clear(accum);

            for (int k0 = 0; k0 < K_total; k0 += TILE_K) {
                // ---- Load weight tile A[TILE_M, TILE_K] into smem ----
                for (int i = tid; i < AB_ELEMS; i += nthreads) {
                    int m_local = i % TILE_M;
                    int k_local = i / TILE_M;
                    tile_A_ptr[i] = wt_mma[(k0 + k_local) * C_out + (m0 + m_local)];
                }

                // ---- Load feature tile B[TILE_N, TILE_K] into smem (gathered) ----
                for (int i = tid; i < BB_ELEMS; i += nthreads) {
                    int n_local = i % TILE_N;
                    int k_local = i / TILE_N;

                    int k        = k0 + k_local;
                    int kern_pos = k / C_in;
                    int cin      = k % C_in;
                    int di       = kern_pos / (Geom::R * Geom::S);
                    int dj       = (kern_pos / Geom::S) % Geom::R;
                    int dk       = kern_pos % Geom::S;

                    int voxel   = n0 + n_local;
                    int vi      = voxel >> 6;
                    int vj      = (voxel >> 3) & 7;
                    int vk      = voxel & 7;
                    int halo_idx =
                        (vi + di) * Geom::Hy * Geom::Hz + (vj + dj) * Geom::Hz + (vk + dk);

                    int64_t g_idx  = smem.gather_map[halo_idx];
                    tile_B_ptr[i] = (g_idx >= 0) ? feat_mma[g_idx * C_in + cin] : MmaElement{};
                }

                __syncthreads();

                // ---- smem -> register fragments ----
#pragma unroll
                for (int j = 0; j < size(tCrA); ++j) {
                    tCrA(j) = tCsA(j);
                }
#pragma unroll
                for (int j = 0; j < size(tCrB); ++j) {
                    tCrB(j) = tCsB(j);
                }

                // ---- Tensor-core MMA ----
                gemm(tiled_mma, accum, tCrA, tCrB, accum);

                __syncthreads();
            }

            // ---- Epilogue: accumulator -> smem -> gmem (with scatter) ----
            auto *smem_C = reinterpret_cast<float *>(smem.tile_buf);
            auto sC      = make_tensor(
                make_smem_ptr(smem_C),
                make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                 make_stride(Int<1>{}, Int<TILE_M>{})));

            auto smem_copy_c     = make_tiled_copy_C(Copy_Atom<UniversalCopy<uint32_t>, float>{}, tiled_mma);
            auto smem_thr_copy_c = smem_copy_c.get_thread_slice(tid);
            auto tCrC_copy       = smem_thr_copy_c.retile_S(accum);
            auto tCsC            = smem_thr_copy_c.partition_D(sC);
            copy(smem_copy_c, tCrC_copy, tCsC);

            __syncthreads();

            for (int i = tid; i < TILE_M * TILE_N; i += nthreads) {
                int m_local = i % TILE_M;
                int n_local = i / TILE_M;
                int cout    = m0 + m_local;
                int voxel   = n0 + n_local;
                if (cout < C_out && voxel < Geom::LeafVol) {
                    int64_t s_idx = smem.scatter_map[voxel];
                    if (s_idx >= 0) {
                        output[s_idx * C_out + cout] = static_cast<Scalar>(smem_C[i]);
                    }
                }
            }

            __syncthreads();
        }
    }
}

// ============================================================================
// Typed launcher
// ============================================================================

template <typename Geom, typename Scalar>
static void
launchLeafFusedConv(GridBatchImpl const &feature_grid,
                    GridBatchImpl const &output_grid,
                    torch::Tensor features,
                    torch::Tensor weights,
                    torch::Tensor output,
                    int C_in,
                    int C_out,
                    cudaStream_t stream) {
    auto feat_acc = feature_grid.deviceAccessor();
    auto out_acc  = output_grid.deviceAccessor();

    int64_t const num_leaves = output_grid.totalLeaves();
    if (num_leaves == 0) {
        return;
    }

    constexpr int THREADS   = 128;
    constexpr size_t SMEM   = sizeof(LeafConvSmem<Geom>);
    constexpr size_t SMEM48 = 48u * 1024u;

    if constexpr (SMEM > SMEM48) {
        cudaFuncSetAttribute(leaf_fused_conv_kernel<Geom, Scalar>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(SMEM));
    }

    // Reshape weights from [C_out, C_in, k0, k1, k2] to [K_total, C_out]
    // where K_total = KernVol * C_in, with stride-1 along C_out for coalesced MMA loads.
    auto W = weights.permute({2, 3, 4, 1, 0}).contiguous().view({Geom::KernVol * C_in, C_out});

    leaf_fused_conv_kernel<Geom, Scalar>
        <<<static_cast<int>(num_leaves), THREADS, SMEM, stream>>>(feat_acc,
                                                                  out_acc,
                                                                  features.template data_ptr<Scalar>(),
                                                                  W.template data_ptr<Scalar>(),
                                                                  output.template data_ptr<Scalar>(),
                                                                  C_in,
                                                                  C_out);
}

// ============================================================================
// Kernel-size dispatch
// ============================================================================

template <typename Scalar>
static torch::Tensor
implicitGemmConvDispatched(torch::Tensor features,
                           torch::Tensor weights,
                           GridBatchImpl const &feature_grid,
                           GridBatchImpl const &output_grid,
                           nanovdb::Coord kernel_size,
                           int C_in,
                           int C_out) {
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const NB = output_grid.totalVoxels();
    auto output      = torch::zeros({NB, C_out}, features.options());

    if (NB == 0)
        return output;

    if (kernel_size == nanovdb::Coord(3, 3, 3)) {
        launchLeafFusedConv<LeafConvGeometry<3, 3, 3>, Scalar>(
            feature_grid, output_grid, features, weights, output, C_in, C_out, stream);
    } else if (kernel_size == nanovdb::Coord(5, 5, 5)) {
        launchLeafFusedConv<LeafConvGeometry<5, 5, 5>, Scalar>(
            feature_grid, output_grid, features, weights, output, C_in, C_out, stream);
    } else {
        TORCH_CHECK(false,
                    "implicitGemmConv: unsupported kernel size (",
                    kernel_size[0],
                    ",",
                    kernel_size[1],
                    ",",
                    kernel_size[2],
                    "). Supported: 3x3x3, 5x5x5.");
    }

    return output;
}

// ============================================================================
// Entry point
// ============================================================================

torch::Tensor
implicitGemmConv(torch::Tensor features,
                 torch::Tensor weights,
                 GridBatchImpl const &feature_grid,
                 GridBatchImpl const &output_grid,
                 nanovdb::Coord kernel_size,
                 nanovdb::Coord stride) {
    TORCH_CHECK(features.is_cuda(), "implicitGemmConv: features must be on CUDA");
    TORCH_CHECK(deviceSupportsImplicitGemm(features.device()),
                "implicitGemmConv: requires Sm80+ (Ampere or newer)");

    TORCH_CHECK(features.dim() == 2, "implicitGemmConv: features must be 2D");
    TORCH_CHECK(features.size(0) == feature_grid.totalVoxels(),
                "implicitGemmConv: features.size(0)=",
                features.size(0),
                " must match feature_grid totalVoxels=",
                feature_grid.totalVoxels());
    TORCH_CHECK(features.is_contiguous(), "implicitGemmConv: features must be contiguous");

    TORCH_CHECK(weights.dim() == 5, "implicitGemmConv: weights must be 5D [Cout, Cin, k0, k1, k2]");
    TORCH_CHECK(features.size(1) == weights.size(1),
                "implicitGemmConv: Cin mismatch between features (",
                features.size(1),
                ") and weights (",
                weights.size(1),
                ")");
    TORCH_CHECK(weights.size(2) == kernel_size[0] && weights.size(3) == kernel_size[1] &&
                    weights.size(4) == kernel_size[2],
                "implicitGemmConv: weights spatial dims must match kernel_size");
    TORCH_CHECK(features.device() == weights.device(),
                "implicitGemmConv: features and weights must be on same device");
    TORCH_CHECK(features.scalar_type() == weights.scalar_type(),
                "implicitGemmConv: features and weights must have same dtype");

    TORCH_CHECK(stride[0] == 1 && stride[1] == 1 && stride[2] == 1,
                "implicitGemmConv: only stride=(1,1,1) is supported, got (",
                stride[0],
                ",",
                stride[1],
                ",",
                stride[2],
                ")");

    int64_t const C_in  = features.size(1);
    int64_t const C_out = weights.size(0);

    TORCH_CHECK(C_in > 0 && C_in % 32 == 0,
                "implicitGemmConv: C_in must be a positive multiple of 32, got ",
                C_in);
    TORCH_CHECK(C_out > 0 && C_out % 32 == 0,
                "implicitGemmConv: C_out must be a positive multiple of 32, got ",
                C_out);

    if (features.scalar_type() == torch::kFloat16) {
        auto f16 = features.to(torch::kFloat16).contiguous();
        auto w16 = weights.to(torch::kFloat16).contiguous();
        auto out = implicitGemmConvDispatched<c10::Half>(
            f16, w16, feature_grid, output_grid, kernel_size, C_in, C_out);
        return out.to(torch::kFloat16);
    } else if (features.scalar_type() == torch::kFloat32) {
        return implicitGemmConvDispatched<float>(
            features, weights, feature_grid, output_grid, kernel_size, C_in, C_out);
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
