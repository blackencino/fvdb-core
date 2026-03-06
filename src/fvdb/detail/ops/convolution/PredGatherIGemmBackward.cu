// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// PredGatherIGemmBackward.cu -- Leaf-local no-kmap sparse convolution backward.
//
// SM80 CuTe TF32 tensor-core backward for the PredGatherIGemm forward.
// Replaces the original scalar FP32 GEMM with SM80_16x8x8_F32TF32TF32F32_TN
// MMA atoms with swizzled shared-memory layouts and LDSM register loads.
//
// dgrad: output-leaf-centered with halo pre-loading.
//   Per (cluster, kernel_offset): D[M=64,N=32] += dY[64,K] * W^T[K,32]
//   Atomic scatter-add into dX.
//
// wgrad: output-leaf-centered with per-offset tree probes.
//   Per kernel_offset, accumulated over 8 clusters:
//   D[M=32,N=32] += dY^T[32,spatial=64] * X[32,64]
//   Atomic reduction into dW.
//
// Constraints (same as forward):
//   - CUDA SM80+, float32, TF32 working precision
//   - Input / output channels multiples of 32
//   - Uniform kernel sizes {3, 5, 7}, uniform strides {1, 2}
//   - Batch size 1
//
// TODO: remaining performance bottlenecks
//   1. cp.async pipelining: loads and MMA compute are fully serialized
//      (single-buffered). Adopting the forward's multi-stage
//      MainloopSm80CpAsyncPredGatherB pattern would overlap global loads
//      with tensor-core compute for ~2x on memory-bound tiles.
//   2. Empty-cluster early exit: sparse leaves waste MMA cycles on
//      zero-filled 64-voxel clusters. Skipping clusters whose output
//      rows are all inactive would eliminate ~75% of work at 25%
//      occupancy and ~90% at 10%.
//   3. Kernel-offset fusion for dgrad: the current loop does one
//      separate GEMM per (cluster, kernel_offset). Fusing multiple
//      kernel offsets into the K-reduction dimension (as the forward
//      fuses C*T*R*S) would increase arithmetic intensity per tile and
//      reduce loop overhead.

// NOTE: <torch/types.h> MUST precede CuTe / CUTLASS headers to avoid
// CCCL include-order issues between toolkit versions.
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/PredGatherIGemmBackward.h>

#include <nanovdb/NanoVDB.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {
namespace pred_gather_igemm_backward {

using namespace cute;

// ============================================================================
// Constants
// ============================================================================

constexpr int kLeafDim         = 8;
constexpr int kLeafVoxels      = kLeafDim * kLeafDim * kLeafDim;
constexpr int kClusterDim      = 4;
constexpr int kClusterVoxels   = kClusterDim * kClusterDim * kClusterDim;
constexpr int kClustersPerDim  = kLeafDim / kClusterDim;
constexpr int kClustersPerLeaf = kClustersPerDim * kClustersPerDim * kClustersPerDim; // 8
constexpr int kTileC           = 32;
constexpr int kTileK           = 32;

template <int KernelSize, int Stride> struct HaloGeom {
    static constexpr int CHx         = (kClusterDim - 1) * Stride + KernelSize;
    static constexpr int CHy         = CHx;
    static constexpr int CHz         = CHx;
    static constexpr int kHaloVoxels = CHx * CHy * CHz;
};

// ============================================================================
// MMA and smem layout types (matching the forward's atom choices)
// ============================================================================

using SmemLayoutAtom = decltype(composition(
    Swizzle<1, 2, 3>{}, Layout<Shape<_8, Shape<_4, _2>>, Stride<_4, Stride<_1, _32>>>{}));

using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, tfloat32_t>;
using EpiCopyAtom  = Copy_Atom<UniversalCopy<uint32_t>, float>;

using DgradMma = TiledMMA<MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
                          Layout<Shape<_2, _2, _1>>,
                          Tile<_64, _32, Underscore>>;

using WgradMma = TiledMMA<MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
                          Layout<Shape<_2, _2, _1>>,
                          Tile<_32, _32, Underscore>>;

static constexpr int kBlockThreads = size(DgradMma{});
static_assert(size(DgradMma{}) == size(WgradMma{}));
static_assert(kBlockThreads == 128);

using SmemLayoutDgradA = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(_64{}, _32{})));
using SmemLayoutDgradB = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(_32{}, _32{})));

using SmemLayoutWgradA = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(_32{}, _64{})));
using SmemLayoutWgradB = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(_32{}, _64{})));

// ============================================================================
// Shared storage (dynamic smem)
// ============================================================================

template <int KernelSize, int Stride> struct alignas(128) DgradSmem {
    using H = HaloGeom<KernelSize, Stride>;
    union {
        struct {
            tfloat32_t sA[cosize_v<SmemLayoutDgradA>];
            tfloat32_t sB[cosize_v<SmemLayoutDgradB>];
        };
        float sC[kClusterVoxels * kTileC];
    };
    uint64_t leaf_idx[kLeafVoxels];
    uint64_t halo_idx[H::kHaloVoxels];
    uint64_t out_rows[kClusterVoxels];
    uint64_t in_rows[kClusterVoxels];
};

template <int KernelSize, int Stride> struct alignas(128) WgradSmem {
    union {
        struct {
            tfloat32_t sA[cosize_v<SmemLayoutWgradA>];
            tfloat32_t sB[cosize_v<SmemLayoutWgradB>];
        };
        float sC[kTileK * kTileC];
    };
    uint64_t leaf_idx[kLeafVoxels];
    uint64_t out_rows[kClusterVoxels];
    uint64_t in_rows[kClusterVoxels];
};

// ============================================================================
// Helpers
// ============================================================================

__device__ __forceinline__ int
leafLinearIndex(int x, int y, int z) {
    return x * (kLeafDim * kLeafDim) + y * kLeafDim + z;
}

__device__ __forceinline__ void
clusterIdToBase(int id, int &bx, int &by, int &bz) {
    bx = (id / (kClustersPerDim * kClustersPerDim)) * kClusterDim;
    by = ((id / kClustersPerDim) % kClustersPerDim) * kClusterDim;
    bz = (id % kClustersPerDim) * kClusterDim;
}

__device__ __forceinline__ void
voxelIdToLocal(int n, int &lx, int &ly, int &lz) {
    lx      = n / (kClusterDim * kClusterDim);
    int rem = n % (kClusterDim * kClusterDim);
    ly      = rem / kClusterDim;
    lz      = rem % kClusterDim;
}

template <int KernelSize>
__device__ __forceinline__ int64_t
weightIndex(int oc, int ic, int t, int r, int s, int C) {
    return (((static_cast<int64_t>(oc) * C + ic) * KernelSize + t) * KernelSize + r) * KernelSize +
           s;
}

// ============================================================================
// dgrad kernel
// ============================================================================

template <int KernelSize, int Stride, typename BuildT>
__global__ void __launch_bounds__(kBlockThreads)
predGatherIGemmDgradKernel(const nanovdb::NanoGrid<BuildT> *act_grid,
                           const nanovdb::NanoGrid<BuildT> *out_grid,
                           const float *__restrict__ grad_out,
                           const float *__restrict__ W_perm,
                           float *__restrict__ grad_features,
                           int C,
                           int K) {
    extern __shared__ char smem_buf[];
    auto &smem = *reinterpret_cast<DgradSmem<KernelSize, Stride> *>(smem_buf);

    using H                     = HaloGeom<KernelSize, Stride>;
    constexpr int radius        = KernelSize / 2;
    constexpr int kernel_volume = KernelSize * KernelSize * KernelSize;

    int const leaf_id = static_cast<int>(blockIdx.x);
    int const c_tile  = static_cast<int>(blockIdx.y);
    int const c0      = c_tile * kTileC;
    if (c0 >= C)
        return;

    auto const &out_leaf = out_grid->tree().template getFirstNode<0>()[leaf_id];
    for (int v = threadIdx.x; v < kLeafVoxels; v += kBlockThreads)
        smem.leaf_idx[v] = out_leaf.getValue(v);
    __syncthreads();

    auto const &act_tree  = act_grid->tree();
    auto const out_origin = out_leaf.origin();
    auto const act_origin =
        nanovdb::Coord(out_origin[0] * Stride, out_origin[1] * Stride, out_origin[2] * Stride);

    // ---- MMA / copy setup (reused across all iterations) ----

    DgradMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor accum = partition_fragment_C(tiled_mma, Shape<_64, _32>{});

    Tensor tSA = make_tensor(make_smem_ptr(smem.sA), SmemLayoutDgradA{});
    Tensor tSB = make_tensor(make_smem_ptr(smem.sB), SmemLayoutDgradB{});

    Tensor fragA = thr_mma.partition_fragment_A(tSA);
    Tensor fragB = thr_mma.partition_fragment_B(tSB);

    auto smem_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_A  = smem_copy_A.get_thread_slice(threadIdx.x);
    auto tCsA        = smem_thr_A.partition_S(tSA);
    auto tCrA        = smem_thr_A.retile_D(fragA);

    auto smem_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_B  = smem_copy_B.get_thread_slice(threadIdx.x);
    auto tCsB        = smem_thr_B.partition_S(tSB);
    auto tCrB        = smem_thr_B.retile_D(fragB);

    auto epi_copy = make_tiled_copy_C(EpiCopyAtom{}, tiled_mma);
    auto epi_thr  = epi_copy.get_slice(threadIdx.x);

    // ---- main loop: cluster → kernel_offset → K-tile ----

    for (int cluster_id = 0; cluster_id < kClustersPerLeaf; ++cluster_id) {
        int bx, by, bz;
        clusterIdToBase(cluster_id, bx, by, bz);

        for (int n = threadIdx.x; n < kClusterVoxels; n += kBlockThreads) {
            int lx, ly, lz;
            voxelIdToLocal(n, lx, ly, lz);
            smem.out_rows[n] = smem.leaf_idx[leafLinearIndex(bx + lx, by + ly, bz + lz)];
        }

        auto const halo_origin =
            act_origin.offsetBy(bx * Stride - radius, by * Stride - radius, bz * Stride - radius);
        for (int v = threadIdx.x; v < H::kHaloVoxels; v += kBlockThreads) {
            int hx           = v / (H::CHy * H::CHz);
            int rem          = v % (H::CHy * H::CHz);
            int hy           = rem / H::CHz;
            int hz           = rem % H::CHz;
            smem.halo_idx[v] = act_tree.getValue(halo_origin.offsetBy(hx, hy, hz));
        }
        __syncthreads();

        for (int ko = 0; ko < kernel_volume; ++ko) {
            int const t = ko / (KernelSize * KernelSize);
            int const r = (ko / KernelSize) % KernelSize;
            int const s = ko % KernelSize;

            for (int n = threadIdx.x; n < kClusterVoxels; n += kBlockThreads) {
                uint64_t in_row = 0;
                if (smem.out_rows[n] != 0) {
                    int lx, ly, lz;
                    voxelIdToLocal(n, lx, ly, lz);
                    int hx = lx * Stride + t;
                    int hy = ly * Stride + r;
                    int hz = lz * Stride + s;
                    in_row = smem.halo_idx[hx * H::CHy * H::CHz + hy * H::CHz + hz];
                }
                smem.in_rows[n] = in_row;
            }
            __syncthreads();

            clear(accum);

            int64_t const w_base = static_cast<int64_t>(ko) * K * C;

            for (int k0 = 0; k0 < K; k0 += kTileK) {
                for (int idx = threadIdx.x; idx < kClusterVoxels * kTileK; idx += kBlockThreads) {
                    int n        = idx / kTileK;
                    int kk       = idx % kTileK;
                    uint64_t row = smem.out_rows[n];
                    tSA(n, kk)   = tfloat32_t(
                        row ? grad_out[static_cast<int64_t>(row - 1) * K + k0 + kk] : 0.0f);
                }

                for (int idx = threadIdx.x; idx < kTileC * kTileK; idx += kBlockThreads) {
                    int cc = idx / kTileK;
                    int kk = idx % kTileK;
                    tSB(cc, kk) =
                        tfloat32_t(W_perm[w_base + static_cast<int64_t>(k0 + kk) * C + c0 + cc]);
                }
                __syncthreads();

                copy(smem_copy_A, tCsA, tCrA);
                copy(smem_copy_B, tCsB, tCrB);

                CUTLASS_PRAGMA_UNROLL
                for (int kb = 0; kb < size<2>(fragA); ++kb)
                    cute::gemm(tiled_mma, accum, fragA(_, _, kb), fragB(_, _, kb), accum);
                __syncthreads();
            }

            // epilogue: stage accum → smem, scatter-add to dX
            Tensor tSC = make_tensor(make_smem_ptr(smem.sC), Layout<Shape<_64, _32>>{});
            auto tCrC  = epi_thr.retile_S(accum);
            auto tCsC  = epi_thr.partition_D(tSC);
            copy(epi_copy, tCrC, tCsC);
            __syncthreads();

            for (int idx = threadIdx.x; idx < kClusterVoxels * kTileC; idx += kBlockThreads) {
                int n        = idx / kTileC;
                int c        = idx % kTileC;
                uint64_t row = smem.in_rows[n];
                if (row != 0)
                    atomicAdd(&grad_features[static_cast<int64_t>(row - 1) * C + c0 + c],
                              tSC(n, c));
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// wgrad kernel
// ============================================================================

template <int KernelSize, int Stride, typename BuildT>
__global__ void __launch_bounds__(kBlockThreads)
predGatherIGemmWgradKernel(const nanovdb::NanoGrid<BuildT> *act_grid,
                           const nanovdb::NanoGrid<BuildT> *out_grid,
                           const float *__restrict__ features,
                           const float *__restrict__ grad_out,
                           float *__restrict__ grad_weights,
                           int C,
                           int K) {
    extern __shared__ char smem_buf[];
    auto &smem = *reinterpret_cast<WgradSmem<KernelSize, Stride> *>(smem_buf);

    constexpr int radius        = KernelSize / 2;
    constexpr int kernel_volume = KernelSize * KernelSize * KernelSize;

    int const leaf_id = static_cast<int>(blockIdx.x);
    int const c_tile  = static_cast<int>(blockIdx.y);
    int const k_tile  = static_cast<int>(blockIdx.z);
    int const c0      = c_tile * kTileC;
    int const k0      = k_tile * kTileK;
    if (c0 >= C || k0 >= K)
        return;

    auto const &out_leaf = out_grid->tree().template getFirstNode<0>()[leaf_id];
    for (int v = threadIdx.x; v < kLeafVoxels; v += kBlockThreads)
        smem.leaf_idx[v] = out_leaf.getValue(v);
    __syncthreads();

    auto const &act_tree  = act_grid->tree();
    auto const out_origin = out_leaf.origin();
    auto const act_origin =
        nanovdb::Coord(out_origin[0] * Stride, out_origin[1] * Stride, out_origin[2] * Stride);

    // ---- MMA / copy setup ----

    WgradMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor accum = partition_fragment_C(tiled_mma, Shape<_32, _32>{});

    Tensor tSA = make_tensor(make_smem_ptr(smem.sA), SmemLayoutWgradA{});
    Tensor tSB = make_tensor(make_smem_ptr(smem.sB), SmemLayoutWgradB{});

    Tensor fragA = thr_mma.partition_fragment_A(tSA);
    Tensor fragB = thr_mma.partition_fragment_B(tSB);

    auto smem_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_A  = smem_copy_A.get_thread_slice(threadIdx.x);
    auto tCsA        = smem_thr_A.partition_S(tSA);
    auto tCrA        = smem_thr_A.retile_D(fragA);

    auto smem_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_B  = smem_copy_B.get_thread_slice(threadIdx.x);
    auto tCsB        = smem_thr_B.partition_S(tSB);
    auto tCrB        = smem_thr_B.retile_D(fragB);

    auto epi_copy = make_tiled_copy_C(EpiCopyAtom{}, tiled_mma);
    auto epi_thr  = epi_copy.get_slice(threadIdx.x);

    // ---- main loop: kernel_offset → cluster ----

    for (int ko = 0; ko < kernel_volume; ++ko) {
        int const t = ko / (KernelSize * KernelSize);
        int const r = (ko / KernelSize) % KernelSize;
        int const s = ko % KernelSize;

        clear(accum);

        for (int cluster_id = 0; cluster_id < kClustersPerLeaf; ++cluster_id) {
            int bx, by, bz;
            clusterIdToBase(cluster_id, bx, by, bz);

            for (int n = threadIdx.x; n < kClusterVoxels; n += kBlockThreads) {
                int lx, ly, lz;
                voxelIdToLocal(n, lx, ly, lz);
                smem.out_rows[n] = smem.leaf_idx[leafLinearIndex(bx + lx, by + ly, bz + lz)];
            }
            __syncthreads();

            for (int n = threadIdx.x; n < kClusterVoxels; n += kBlockThreads) {
                uint64_t in_row = 0;
                if (smem.out_rows[n] != 0) {
                    int lx, ly, lz;
                    voxelIdToLocal(n, lx, ly, lz);
                    nanovdb::Coord probe = act_origin.offsetBy((bx + lx) * Stride + (t - radius),
                                                               (by + ly) * Stride + (r - radius),
                                                               (bz + lz) * Stride + (s - radius));
                    in_row               = act_tree.getValue(probe);
                }
                smem.in_rows[n] = in_row;
            }
            __syncthreads();

            // sA[M=32, K=64]: dY^T -- row = output channel, col = voxel
            for (int idx = threadIdx.x; idx < kTileK * kClusterVoxels; idx += kBlockThreads) {
                int kk       = idx / kClusterVoxels;
                int n        = idx % kClusterVoxels;
                uint64_t row = smem.out_rows[n];
                tSA(kk, n) =
                    tfloat32_t(row ? grad_out[static_cast<int64_t>(row - 1) * K + k0 + kk] : 0.0f);
            }

            // sB[N=32, K=64]: X -- row = input channel, col = voxel
            for (int idx = threadIdx.x; idx < kTileC * kClusterVoxels; idx += kBlockThreads) {
                int cc       = idx / kClusterVoxels;
                int n        = idx % kClusterVoxels;
                uint64_t row = smem.in_rows[n];
                tSB(cc, n) =
                    tfloat32_t(row ? features[static_cast<int64_t>(row - 1) * C + c0 + cc] : 0.0f);
            }
            __syncthreads();

            copy(smem_copy_A, tCsA, tCrA);
            copy(smem_copy_B, tCsB, tCrB);

            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < size<2>(fragA); ++kb)
                cute::gemm(tiled_mma, accum, fragA(_, _, kb), fragB(_, _, kb), accum);
            __syncthreads();
        }

        // epilogue: stage accum → smem, atomicAdd to dW
        Tensor tSC = make_tensor(make_smem_ptr(smem.sC), Layout<Shape<_32, _32>>{});
        auto tCrC  = epi_thr.retile_S(accum);
        auto tCsC  = epi_thr.partition_D(tSC);
        copy(epi_copy, tCrC, tCsC);
        __syncthreads();

        for (int idx = threadIdx.x; idx < kTileK * kTileC; idx += kBlockThreads) {
            int kk = idx / kTileC;
            int cc = idx % kTileC;
            atomicAdd(&grad_weights[weightIndex<KernelSize>(k0 + kk, c0 + cc, t, r, s, C)],
                      tSC(kk, cc));
        }
        __syncthreads();
    }
}

// ============================================================================
// Launch
// ============================================================================

template <int KernelSize, int Stride>
std::tuple<torch::Tensor, torch::Tensor>
launchPredGatherIGemmBackward(torch::Tensor grad_output,
                              torch::Tensor features,
                              torch::Tensor weights,
                              GridBatchImpl const &feature_grid,
                              GridBatchImpl const &output_grid) {
    auto *nano_input_grid =
        feature_grid.nanoGridHandle().template deviceGrid<nanovdb::ValueOnIndex>();
    auto *nano_output_grid =
        output_grid.nanoGridHandle().template deviceGrid<nanovdb::ValueOnIndex>();

    TORCH_CHECK(nano_input_grid != nullptr, "Failed to get device input grid");
    TORCH_CHECK(nano_output_grid != nullptr, "Failed to get device output grid");

    int64_t const N_in  = features.size(0);
    int64_t const N_out = grad_output.size(0);
    int64_t const C     = features.size(1);
    int64_t const K     = weights.size(0);

    auto opts = torch::dtype(torch::kFloat32).device(features.device());

    auto grad_features = torch::zeros({N_in, C}, opts);
    auto grad_weights  = torch::zeros_like(weights);

    uint32_t const output_leaf_count = output_grid.numLeavesAt(0);

    if (N_in == 0 || N_out == 0 || C == 0 || K == 0 || output_leaf_count == 0) {
        return {grad_features, grad_weights};
    }

    auto filter_perm = weights.permute({2, 3, 4, 0, 1}).contiguous();

    dim3 const block(kBlockThreads);
    dim3 const dgrad_grid(
        static_cast<unsigned int>(output_leaf_count), static_cast<unsigned int>(C / kTileC), 1u);
    dim3 const wgrad_grid(static_cast<unsigned int>(output_leaf_count),
                          static_cast<unsigned int>(C / kTileC),
                          static_cast<unsigned int>(K / kTileK));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr size_t dgrad_smem_size = sizeof(DgradSmem<KernelSize, Stride>);
    constexpr size_t wgrad_smem_size = sizeof(WgradSmem<KernelSize, Stride>);

    auto dgrad_fn = predGatherIGemmDgradKernel<KernelSize, Stride, nanovdb::ValueOnIndex>;
    auto wgrad_fn = predGatherIGemmWgradKernel<KernelSize, Stride, nanovdb::ValueOnIndex>;

    if (dgrad_smem_size > 48u * 1024u)
        cudaFuncSetAttribute(dgrad_fn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(dgrad_smem_size));
    if (wgrad_smem_size > 48u * 1024u)
        cudaFuncSetAttribute(wgrad_fn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(wgrad_smem_size));

    dgrad_fn<<<dgrad_grid, block, dgrad_smem_size, stream>>>(nano_input_grid,
                                                             nano_output_grid,
                                                             grad_output.data_ptr<float>(),
                                                             filter_perm.data_ptr<float>(),
                                                             grad_features.data_ptr<float>(),
                                                             static_cast<int>(C),
                                                             static_cast<int>(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    wgrad_fn<<<wgrad_grid, block, wgrad_smem_size, stream>>>(nano_input_grid,
                                                             nano_output_grid,
                                                             features.data_ptr<float>(),
                                                             grad_output.data_ptr<float>(),
                                                             grad_weights.data_ptr<float>(),
                                                             static_cast<int>(C),
                                                             static_cast<int>(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_features, grad_weights};
}

// ============================================================================
// Preconditions
// ============================================================================

static void
checkPredGatherIGemmBackwardPreconditions(torch::Tensor grad_output,
                                          torch::Tensor features,
                                          torch::Tensor weights,
                                          GridBatchImpl const &feature_grid,
                                          GridBatchImpl const &output_grid,
                                          int kernel_size,
                                          int stride) {
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    TORCH_CHECK(features.scalar_type() == torch::kFloat32, "features must be float32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");

    TORCH_CHECK(features.dim() == 2, "features must be 2-D [N_in, C]");
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2-D [N_out, K]");
    TORCH_CHECK(weights.dim() == 5, "weights must be 5-D [K, C, T, R, S]");

    TORCH_CHECK(kernel_size == 3 || kernel_size == 5 || kernel_size == 7,
                "PredGatherIGemm backward supports kernel sizes 3, 5, 7; got ",
                kernel_size);
    TORCH_CHECK(
        stride == 1 || stride == 2, "PredGatherIGemm backward supports strides 1, 2; got ", stride);

    int64_t const C = features.size(1);
    int64_t const K = weights.size(0);

    TORCH_CHECK(weights.size(1) == C, "weights C dimension must match features");
    TORCH_CHECK(grad_output.size(1) == K, "grad_output channel dimension must match weights K");
    TORCH_CHECK(weights.size(2) == kernel_size && weights.size(3) == kernel_size &&
                    weights.size(4) == kernel_size,
                "weights spatial dimensions must be ",
                kernel_size,
                "x",
                kernel_size,
                "x",
                kernel_size);
    TORCH_CHECK(C % 32 == 0, "Input channels must be a multiple of 32, got ", C);
    TORCH_CHECK(K % 32 == 0, "Output channels must be a multiple of 32, got ", K);

    TORCH_CHECK(feature_grid.batchSize() == 1,
                "PredGatherIGemm backward currently supports batch size 1");
    TORCH_CHECK(output_grid.batchSize() == 1,
                "PredGatherIGemm backward currently supports batch size 1");

    TORCH_CHECK(feature_grid.totalVoxels() == features.size(0),
                "feature_grid voxel count (",
                feature_grid.totalVoxels(),
                ") must match features row count (",
                features.size(0),
                ")");
    TORCH_CHECK(output_grid.totalVoxels() == grad_output.size(0),
                "output_grid voxel count (",
                output_grid.totalVoxels(),
                ") must match grad_output row count (",
                grad_output.size(0),
                ")");
    TORCH_CHECK(features.device() == weights.device() && features.device() == grad_output.device(),
                "features, weights, and grad_output must be on the same CUDA device");
}

} // namespace pred_gather_igemm_backward

// ============================================================================
// Public entry point
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
predGatherIGemmSparseConvBackward(torch::Tensor grad_output,
                                  torch::Tensor features,
                                  torch::Tensor weights,
                                  GridBatchImpl const &feature_grid,
                                  GridBatchImpl const &output_grid,
                                  int kernel_size,
                                  int stride) {
    using namespace pred_gather_igemm_backward;

    checkPredGatherIGemmBackwardPreconditions(
        grad_output, features, weights, feature_grid, output_grid, kernel_size, stride);

    if (!features.is_contiguous())
        features = features.contiguous();
    if (!grad_output.is_contiguous())
        grad_output = grad_output.contiguous();
    if (!weights.is_contiguous())
        weights = weights.contiguous();

    switch (kernel_size) {
    case 3:
        switch (stride) {
        case 1:
            return launchPredGatherIGemmBackward<3, 1>(
                grad_output, features, weights, feature_grid, output_grid);
        case 2:
            return launchPredGatherIGemmBackward<3, 2>(
                grad_output, features, weights, feature_grid, output_grid);
        default: break;
        }
        break;
    case 5:
        switch (stride) {
        case 1:
            return launchPredGatherIGemmBackward<5, 1>(
                grad_output, features, weights, feature_grid, output_grid);
        case 2:
            return launchPredGatherIGemmBackward<5, 2>(
                grad_output, features, weights, feature_grid, output_grid);
        default: break;
        }
        break;
    case 7:
        switch (stride) {
        case 1:
            return launchPredGatherIGemmBackward<7, 1>(
                grad_output, features, weights, feature_grid, output_grid);
        case 2:
            return launchPredGatherIGemmBackward<7, 2>(
                grad_output, features, weights, feature_grid, output_grid);
        default: break;
        }
        break;
    default: break;
    }

    TORCH_CHECK(false,
                "Unsupported PredGatherIGemm backward configuration: kernel_size=",
                kernel_size,
                ", stride=",
                stride);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
