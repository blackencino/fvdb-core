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
//   1. Kernel-offset fusion for dgrad: the current loop does one
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

// ============================================================================
// CuTe upcast overloads for composed gather layouts (cp.async pipelining)
// ============================================================================

namespace cute {

template <int N, int I, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto
upcast(Shape const &shape, Stride const &stride) {
    if constexpr (is_tuple<Shape>::value) {
        return transform_layout(
            shape, stride, [](auto const &s, auto const &d) { return upcast<N, I>(s, d); });
    } else if constexpr (is_scaled_basis<Stride>::value) {
        if constexpr (Stride::mode() == I) {
            return make_layout(ceil_div(shape, Int<N>{}), ceil_div(stride, Int<N>{}));
        } else {
            return make_layout(shape, stride);
        }
    } else {
        return upcast<N>(shape, stride);
    }
    CUTE_GCC_UNREACHABLE;
}

template <int N, class OuterShape, class OuterStride, class Offset, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto
upcast(
    ComposedLayout<Layout<OuterShape, OuterStride>, Offset, Layout<Shape, Stride>> const &layout) {
    auto idx =
        find_if(layout.layout_a().stride(), [](auto x) { return is_constant<1, decltype(x)>{}; });
    constexpr int I = decltype(idx)::value;
    auto outer      = upcast<N>(layout.layout_a());
    auto offset =
        as_arithmetic_tuple(replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));
    auto inner = upcast<N, I>(layout.layout_b().shape(), layout.layout_b().stride());
    return composition(outer, offset, inner);
}

} // namespace cute

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

constexpr int kDgradStages = 3;

using SmemLayoutDgradA_Staged =
    decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(_64{}, _32{}, Int<kDgradStages>{})));
using SmemLayoutDgradB_Staged =
    decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(_32{}, _32{}, Int<kDgradStages>{})));

// ============================================================================
// Shared storage (dynamic smem)
// ============================================================================

template <int KernelSize, int Stride> struct alignas(128) DgradSmem {
    using H = HaloGeom<KernelSize, Stride>;
    union {
        struct {
            tfloat32_t sA[cosize_v<SmemLayoutDgradA_Staged>];
            tfloat32_t sB[cosize_v<SmemLayoutDgradB_Staged>];
        };
        float sC[kClusterVoxels * kTileC];
    };
    uint64_t leaf_idx[kLeafVoxels];
    uint64_t halo_idx[H::kHaloVoxels];
    uint64_t out_rows[kClusterVoxels];
    uint64_t in_rows[kClusterVoxels];
    bool out_pred[kClusterVoxels];
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

__device__ __forceinline__ uint8_t
computeActiveClusterMask(const uint64_t *leaf_idx) {
    uint8_t mask = 0;
    for (int cid = 0; cid < kClustersPerLeaf; ++cid) {
        int bx, by, bz;
        clusterIdToBase(cid, bx, by, bz);
        for (int n = 0; n < kClusterVoxels; ++n) {
            int lx, ly, lz;
            voxelIdToLocal(n, lx, ly, lz);
            if (leaf_idx[leafLinearIndex(bx + lx, by + ly, bz + lz)] != 0) {
                mask |= static_cast<uint8_t>(1 << cid);
                break;
            }
        }
    }
    return mask;
}

// ============================================================================
// Gather layout utilities for cp.async pipelining
// ============================================================================

template <class Index, int Offset = 0> struct IndexedGather {
    CUTE_HOST_DEVICE constexpr IndexedGather(Index const *indices = {}) : indices_(indices) {}

    template <typename I>
    CUTE_HOST_DEVICE constexpr Index
    operator()(I i) const {
        return indices_[i] + Index(Offset);
    }

    CUTE_HOST_DEVICE friend void
    print(IndexedGather const &) {
        cute::print("Indexed");
    }

    Index const *indices_;
};

template <class Func, class StrideT> struct CustomStride {
    CUTE_HOST_DEVICE constexpr CustomStride(Func const &func, StrideT const &stride)
        : func_(func), stride_(stride) {}

    template <class I>
    CUTE_HOST_DEVICE constexpr friend auto
    operator*(I i, CustomStride const &s) {
        return s.func_(i) * s.stride_;
    }

    template <class I>
    CUTE_HOST_DEVICE constexpr friend auto
    operator*(CustomStride const &s, I i) {
        return s.func_(i) * s.stride_;
    }

    CUTE_HOST_DEVICE friend void
    print(CustomStride const &s) {
        cute::print("Custom{");
        print(s.func_);
        cute::print(",");
        print(s.stride_);
        cute::print("}");
    }

    template <class Div>
    CUTE_HOST_DEVICE constexpr friend auto
    safe_div(CustomStride const &s, Div const &div) {
        return CustomStride<Func, decltype(safe_div(s.stride_, div))>(s.func_,
                                                                      safe_div(s.stride_, div));
    }

    template <class Shape>
    CUTE_HOST_DEVICE constexpr friend auto
    make_layout(Shape const &shape, CustomStride const &stride) {
        return Layout<Shape, CustomStride>(shape, stride);
    }

    Func func_;
    StrideT stride_;
};

using GmemTiledCopyDgradA = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, tfloat32_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _4>>{}));

using GmemTiledCopyDgradB = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, tfloat32_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _4>>{}));

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

    uint8_t const active_cluster_mask = computeActiveClusterMask(smem.leaf_idx);

    // ---- MMA / copy setup (reused across all iterations) ----

    DgradMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor accum = partition_fragment_C(tiled_mma, Shape<_64, _32>{});

    Tensor tSA = make_tensor(make_smem_ptr(smem.sA), SmemLayoutDgradA_Staged{});
    Tensor tSB = make_tensor(make_smem_ptr(smem.sB), SmemLayoutDgradB_Staged{});

    Tensor fragA = thr_mma.partition_fragment_A(tSA(_, _, 0));
    Tensor fragB = thr_mma.partition_fragment_B(tSB(_, _, 0));

    auto smem_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_A  = smem_copy_A.get_thread_slice(threadIdx.x);
    auto tCsA        = smem_thr_A.partition_S(tSA);
    auto tCrA_view   = smem_thr_A.retile_D(fragA);

    auto smem_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_B  = smem_copy_B.get_thread_slice(threadIdx.x);
    auto tCsB        = smem_thr_B.partition_S(tSB);
    auto tCrB_view   = smem_thr_B.retile_D(fragB);

    auto epi_copy = make_tiled_copy_C(EpiCopyAtom{}, tiled_mma);
    auto epi_thr  = epi_copy.get_slice(threadIdx.x);

    GmemTiledCopyDgradA gmem_copy_A;
    GmemTiledCopyDgradB gmem_copy_B;
    auto gmem_thr_A = gmem_copy_A.get_slice(threadIdx.x);
    auto gmem_thr_B = gmem_copy_B.get_slice(threadIdx.x);

    auto tAsA = gmem_thr_A.partition_D(tSA);
    auto tBsB = gmem_thr_B.partition_D(tSB);

    auto K_BLOCK_MAX = size<2>(fragA);

    // ---- main loop: cluster → kernel_offset → pipelined K-tiles ----

    for (int cluster_id = 0; cluster_id < kClustersPerLeaf; ++cluster_id) {
        if (!(active_cluster_mask & (1 << cluster_id)))
            continue;

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

        for (int n = threadIdx.x; n < kClusterVoxels; n += kBlockThreads)
            smem.out_pred[n] = (smem.out_rows[n] != 0);
        __syncthreads();

        // Gathered gmem tensor for grad_out: address(n,k) = (out_rows[n]-1)*K + k
        auto gA_inner = make_layout(
            make_shape(Int<kClusterVoxels>{}, K),
            make_stride(E<0>{}, E<1>{}));
        auto gA_outer = make_layout(
            make_shape(_1{}, _1{}),
            make_stride(
                CustomStride{IndexedGather<uint64_t, -1>{smem.out_rows}, K},
                _1{}));
        auto gA_full = make_tensor(
            make_gmem_ptr(grad_out),
            composition(gA_outer, make_arithmetic_tuple(_0{}, _0{}), gA_inner));
        auto gA = local_tile(
            gA_full,
            make_shape(Int<kClusterVoxels>{}, Int<kTileK>{}),
            make_coord(_0{}, _));

        // Predicate tensor (stride _0 in K: pred depends only on voxel index)
        auto sP_full = make_tensor(
            make_smem_ptr(smem.out_pred),
            make_layout(
                make_shape(Int<kClusterVoxels>{}, K),
                make_stride(_1{}, _0{})));
        auto sP = local_tile(
            sP_full,
            make_shape(Int<kClusterVoxels>{}, Int<kTileK>{}),
            make_coord(_0{}, _));

        auto tAgA = gmem_thr_A.partition_S(gA);
        auto tAsP = gmem_thr_A.partition_S(sP);

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

            // Contiguous gmem tensor for weights (permutation [T,R,S,C,K]: K contiguous)
            int64_t const w_base = static_cast<int64_t>(ko) * C * K;
            auto gB_full = make_tensor(
                make_gmem_ptr(W_perm + w_base + static_cast<int64_t>(c0) * K),
                make_layout(make_shape(Int<kTileC>{}, K), make_stride(K, _1{})));
            auto gB = local_tile(
                gB_full,
                make_shape(Int<kTileC>{}, Int<kTileK>{}),
                make_coord(_0{}, _));
            auto tBgB = gmem_thr_B.partition_S(gB);

            clear(accum);

            int k_tile_count = K / kTileK;
            int k_tile       = 0;

            // ---- Prologue: fill Stages-1 pipeline stages ----
            CUTLASS_PRAGMA_UNROLL
            for (int k_pipe = 0; k_pipe < kDgradStages - 1; ++k_pipe) {
                copy_if(gmem_copy_A, tAsP(_, _, _, k_tile), tAgA(_, _, _, k_tile),
                        tAsA(_, _, _, k_pipe));
                copy(gmem_copy_B, tBgB(_, _, _, k_tile), tBsB(_, _, _, k_pipe));
                cp_async_fence();
                --k_tile_count;
                if (k_tile_count > 0)
                    ++k_tile;
            }

            // ---- Pipelined mainloop ----
            int smem_pipe_read  = 0;
            int smem_pipe_write = kDgradStages - 1;

            auto tCsA_p = tCsA(_, _, _, smem_pipe_read);
            auto tCsB_p = tCsB(_, _, _, smem_pipe_read);

            if (K_BLOCK_MAX > 1) {
                cp_async_wait<kDgradStages - 2>();
                __syncthreads();
                copy(smem_copy_A, tCsA_p(_, _, Int<0>{}), tCrA_view(_, _, Int<0>{}));
                copy(smem_copy_B, tCsB_p(_, _, Int<0>{}), tCrB_view(_, _, Int<0>{}));
            }

            CUTLASS_PRAGMA_NO_UNROLL
            while (k_tile_count > -(kDgradStages - 1)) {
                for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                    if (k_block == K_BLOCK_MAX - 1) {
                        tCsA_p = tCsA(_, _, _, smem_pipe_read);
                        tCsB_p = tCsB(_, _, _, smem_pipe_read);
                        cp_async_wait<kDgradStages - 2>();
                        __syncthreads();
                    }

                    auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                    copy(smem_copy_A, tCsA_p(_, _, k_block_next), tCrA_view(_, _, k_block_next));
                    copy(smem_copy_B, tCsB_p(_, _, k_block_next), tCrB_view(_, _, k_block_next));

                    if (k_block == 0) {
                        copy_if(gmem_copy_A, tAsP(_, _, _, k_tile), tAgA(_, _, _, k_tile),
                                tAsA(_, _, _, smem_pipe_write));
                        copy(gmem_copy_B, tBgB(_, _, _, k_tile), tBsB(_, _, _, smem_pipe_write));
                        cp_async_fence();

                        --k_tile_count;
                        if (k_tile_count > 0)
                            ++k_tile;

                        smem_pipe_write = smem_pipe_read;
                        ++smem_pipe_read;
                        smem_pipe_read =
                            (smem_pipe_read == kDgradStages) ? 0 : smem_pipe_read;
                    }

                    cute::gemm(tiled_mma, accum, fragA(_, _, k_block), fragB(_, _, k_block), accum);
                });
            }

            cp_async_wait<0>();
            __syncthreads();

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

    uint8_t const active_cluster_mask = computeActiveClusterMask(smem.leaf_idx);

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
            if (!(active_cluster_mask & (1 << cluster_id)))
                continue;

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

    auto filter_perm = weights.permute({2, 3, 4, 1, 0}).contiguous();

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
