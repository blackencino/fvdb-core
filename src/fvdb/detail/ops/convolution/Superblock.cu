#if 0

// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Superblock.cu -- Adaptive Superblock GEMM sparse convolution.
//
// Dynamically toggles between a dense bounding-box halo map (for dense blocks)
// and an exact targeted sparse neighbor list (for sparse blocks) inside shared
// memory. This completely resolves sparsity penalties while outperforming
// global hash map solutions by maintaining 100% local compute density.
//

#include <fvdb/detail/ops/convolution/Superblock.h>

#include <nanovdb/NanoVDB.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

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
deviceSupportsSuperblock(torch::Device device) {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device.index());
    return major >= 8;
}

// ============================================================================
// ConvOp -- which operation the kernel performs
// ============================================================================

enum class ConvOp { Forward, InputGrad, WeightGrad, TransposedFwd };

// ============================================================================
// SuperblockGeometry -- compile-time spatial parameters
// ============================================================================

template <int T_, int R_, int S_> struct SuperblockGeometry {
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
// CompactEntry -- one active voxel in the compact list
// ============================================================================

struct CompactEntry {
    int32_t scatter_idx;
    uint8_t leaf_in_block;
    uint8_t _pad;
    uint16_t local_voxel;
};

static_assert(sizeof(CompactEntry) == 8, "CompactEntry must be 8 bytes");

// ============================================================================
// MMA configuration per scalar type
// ============================================================================

template <typename Scalar> struct SbMmaConfig;

template <> struct SbMmaConfig<float> {
    using MmaElement = cute::tfloat32_t;
    using MmaAtom    = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    static constexpr int MMA_K = 8;
    static constexpr bool USE_VECTORIZED_PIPELINE = true;

    using SmemLayoutAtom = decltype(cute::composition(
        cute::Swizzle<1, 2, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::Shape<cute::_4, cute::_2>>,
                     cute::Stride<cute::_4, cute::Stride<cute::_1, cute::_32>>>{}));
    using SmemCopyAtom =
        cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, MmaElement>;
    using GmemTiledCopyA = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>,
                        MmaElement>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>,
                     cute::Stride<cute::_8, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_4>>{}));
    static constexpr int GMEM_VEC_K = 4;
};

template <> struct SbMmaConfig<c10::Half> {
    using MmaElement = cute::half_t;
    using MmaAtom    = cute::SM80_16x8x16_F32F16F16F32_TN;
    static constexpr int MMA_K = 16;
    static constexpr bool USE_VECTORIZED_PIPELINE = false;
    static constexpr int GMEM_VEC_K = 1;
};

// ============================================================================
// Tile and pipeline constants
// ============================================================================

static constexpr int SB_TILE_M = 32;
static constexpr int SB_TILE_N = 32;
static constexpr int SB_TILE_K = 32;
static constexpr int SB_STAGES = 3;

// ============================================================================
// Adaptive Shared memory layout
// ============================================================================

template <typename Geom, int N_LEAVES> struct SuperblockSmem {
    union {
        int32_t halo_maps[N_LEAVES][Geom::HaloVol];
        int32_t sparse_neighbors[(N_LEAVES * Geom::HaloVol) / Geom::KernVol][Geom::KernVol];
    };
    CompactEntry compact_list[N_LEAVES * Geom::LeafVol];
    nanovdb::Coord leaf_origins[N_LEAVES];
    int compact_count;

    static constexpr int MAINLOOP_BYTES =
        (SB_TILE_M + SB_TILE_N) * SB_TILE_K * SB_STAGES * 4;
    static constexpr int EPILOGUE_BYTES = SB_TILE_M * SB_TILE_N * sizeof(float);
    static constexpr int MMA_BUF_BYTES =
        MAINLOOP_BYTES > EPILOGUE_BYTES ? MAINLOOP_BYTES : EPILOGUE_BYTES;
    alignas(16) char mma_buf[MMA_BUF_BYTES];
};

// ============================================================================
// cp.async helpers
// ============================================================================

__device__ __forceinline__ void
sb_cp_async_zfill_4(void *smem_dst, const void *gmem_src, bool pred) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 : : "r"(smem_addr), "l"(gmem_src), "r"(pred ? 4 : 0) : "memory");
}

__device__ __forceinline__ void
sb_cp_async_zfill_16(void *smem_dst, const void *gmem_src, bool pred) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 : : "r"(smem_addr), "l"(gmem_src), "r"(pred ? 16 : 0) : "memory");
}

template <typename T>
__device__ __forceinline__ void
sb_store_or_cp_async_4(T *smem_dst, T const *gmem_src, bool pred) {
    if constexpr (sizeof(T) >= 4) {
        sb_cp_async_zfill_4(smem_dst, gmem_src, pred);
    } else {
        *smem_dst = pred ? *gmem_src : T{};
    }
}

// ============================================================================
// Phase B helpers: async tile loading
// ============================================================================

template <typename MmaElement>
__device__ void
load_a_tile_async(MmaElement const *__restrict__ weights, MmaElement *stage_A,
                  int C_fast, int m0, int k0, int K_total, int tid, int nthreads) {
    constexpr int elems = SB_TILE_M * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int m_local = i % SB_TILE_M;
        int k_local = i / SB_TILE_M;
        int m       = m0 + m_local;
        int k       = k0 + k_local;
        bool valid  = (m < C_fast && k < K_total);
        auto const *src = valid
            ? &weights[static_cast<int64_t>(m) * K_total + k]
            : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(&stage_A[i], src, valid);
    }
}

template <typename Geom, bool FlipKernel, bool IsSparse, typename MmaElement, typename SmemTensor>
__device__ void
load_b_tile_vec16(int32_t const *__restrict__ halo_ptr,
                  int32_t const *__restrict__ sparse_ptr,
                  CompactEntry const *__restrict__ compact_list,
                  MmaElement const *__restrict__ data,
                  int C_channel, int voxel_start, int k0, int N_active, int K_total,
                  SmemTensor sB, int pipe_stage, int tid, int nthreads) {
    constexpr int VEC = 4;
    constexpr int groups = SB_TILE_N * (SB_TILE_K / VEC);
    for (int g = tid; g < groups; g += nthreads) {
        int n_local = g / (SB_TILE_K / VEC);
        int k_group = g % (SB_TILE_K / VEC);
        int k_base  = k_group * VEC;
        int n       = voxel_start + n_local;
        int k       = k0 + k_base;

        auto *dst = &sB(n_local, k_base, pipe_stage);

        if (n >= N_active || k >= K_total) {
            sb_cp_async_zfill_16(dst, nullptr, false);
            continue;
        }

        int kern_pos = k / C_channel;
        int c        = k % C_channel;

        int32_t src;
        if constexpr (IsSparse) {
            src = sparse_ptr[n * Geom::KernVol + kern_pos];
        } else {
            auto entry = compact_list[n];
            int vi = entry.local_voxel >> 6;
            int vj = (entry.local_voxel >> 3) & 7;
            int vk = entry.local_voxel & 7;

            int di = kern_pos / (Geom::R * Geom::S);
            int dj = (kern_pos / Geom::S) % Geom::R;
            int dk = kern_pos % Geom::S;

            if constexpr (FlipKernel) {
                di = Geom::T - 1 - di;
                dj = Geom::R - 1 - dj;
                dk = Geom::S - 1 - dk;
            }

            int halo_idx = (vi + di) * Geom::Hy * Geom::Hz + (vj + dj) * Geom::Hz + (vk + dk);
            src = halo_ptr[entry.leaf_in_block * Geom::HaloVol + halo_idx];
        }

        bool active  = (src >= 0);
        auto const *ptr = active
            ? &data[static_cast<int64_t>(src) * C_channel + c]
            : reinterpret_cast<MmaElement const *>(0);
        sb_cp_async_zfill_16(dst, ptr, active);
    }
}

template <typename Geom, bool FlipKernel, bool IsSparse, typename MmaElement>
__device__ void
load_b_tile_async(int32_t const *__restrict__ halo_ptr,
                  int32_t const *__restrict__ sparse_ptr,
                  CompactEntry const *__restrict__ compact_list,
                  MmaElement const *__restrict__ data,
                  int C_channel, int voxel_start, int k0, int N_active, int K_total,
                  MmaElement *stage_B, int tid, int nthreads) {
    constexpr int elems = SB_TILE_N * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int n_local = i % SB_TILE_N;
        int k_local = i / SB_TILE_N;
        int n       = voxel_start + n_local;
        int k       = k0 + k_local;

        if (n >= N_active || k >= K_total) {
            sb_store_or_cp_async_4(&stage_B[i],
                                   reinterpret_cast<MmaElement const *>(0), false);
            continue;
        }

        int kern_pos = k / C_channel;
        int c        = k % C_channel;

        int32_t src;
        if constexpr (IsSparse) {
            src = sparse_ptr[n * Geom::KernVol + kern_pos];
        } else {
            auto entry = compact_list[n];
            int vi = entry.local_voxel >> 6;
            int vj = (entry.local_voxel >> 3) & 7;
            int vk = entry.local_voxel & 7;

            int di = kern_pos / (Geom::R * Geom::S);
            int dj = (kern_pos / Geom::S) % Geom::R;
            int dk = kern_pos % Geom::S;

            if constexpr (FlipKernel) {
                di = Geom::T - 1 - di;
                dj = Geom::R - 1 - dj;
                dk = Geom::S - 1 - dk;
            }

            int halo_idx = (vi + di) * Geom::Hy * Geom::Hz + (vj + dj) * Geom::Hz + (vk + dk);
            src = halo_ptr[entry.leaf_in_block * Geom::HaloVol + halo_idx];
        }

        bool active  = (src >= 0);
        auto const *ptr = active
            ? &data[static_cast<int64_t>(src) * C_channel + c]
            : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(&stage_B[i], ptr, active);
    }
}

template <typename MmaElement>
__device__ void
load_a_tile_wgrad_async(MmaElement const *__restrict__ grad_output,
                        CompactEntry const *__restrict__ compact_list,
                        MmaElement *stage_A, int C_out, int m0, int k0, int N_active, int tid, int nthreads) {
    constexpr int elems = SB_TILE_M * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int m_local = i % SB_TILE_M;
        int k_local = i / SB_TILE_M;
        int m       = m0 + m_local;
        int k       = k0 + k_local;
        bool valid  = (m < C_out && k < N_active);
        auto const *src = valid
            ? &grad_output[static_cast<int64_t>(compact_list[k].scatter_idx) * C_out + m]
            : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(&stage_A[i], src, valid);
    }
}

template <typename Geom, bool IsSparse, typename MmaElement>
__device__ void
load_b_tile_wgrad_async(int32_t const *__restrict__ halo_ptr,
                        int32_t const *__restrict__ sparse_ptr,
                        CompactEntry const *__restrict__ compact_list,
                        MmaElement const *__restrict__ data,
                        int C_channel, int ktotal_start, int voxel_start, int N_active, int K_total,
                        MmaElement *stage_B, int tid, int nthreads) {
    constexpr int elems = SB_TILE_N * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int n_local = i % SB_TILE_N;
        int k_local = i / SB_TILE_N;

        int ktotal_pos = ktotal_start + n_local;
        int voxel_idx  = voxel_start + k_local;

        if (ktotal_pos >= K_total || voxel_idx >= N_active) {
            sb_store_or_cp_async_4(&stage_B[i],
                                   reinterpret_cast<MmaElement const *>(0), false);
            continue;
        }

        int kern_pos = ktotal_pos / C_channel;
        int c        = ktotal_pos % C_channel;

        int32_t src;
        if constexpr (IsSparse) {
            src = sparse_ptr[voxel_idx * Geom::KernVol + kern_pos];
        } else {
            auto entry = compact_list[voxel_idx];
            int vi = entry.local_voxel >> 6;
            int vj = (entry.local_voxel >> 3) & 7;
            int vk = entry.local_voxel & 7;

            int di = kern_pos / (Geom::R * Geom::S);
            int dj = (kern_pos / Geom::S) % Geom::R;
            int dk = kern_pos % Geom::S;

            int halo_idx = (vi + di) * Geom::Hy * Geom::Hz + (vj + dj) * Geom::Hz + (vk + dk);
            src = halo_ptr[entry.leaf_in_block * Geom::HaloVol + halo_idx];
        }

        bool active  = (src >= 0);
        auto const *ptr = active
            ? &data[static_cast<int64_t>(src) * C_channel + c]
            : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(&stage_B[i], ptr, active);
    }
}

// Swizzled-aware wgrad A-tile loader: writes through CuTe smem tensor.
template <typename MmaElement, typename SmemTensor>
__device__ void
load_a_tile_wgrad_swizzled(MmaElement const *__restrict__ grad_output,
                           CompactEntry const *__restrict__ compact_list,
                           SmemTensor sA, int pipe, int C_out,
                           int m0, int k0, int N_active, int tid, int nthreads) {
    constexpr int elems = SB_TILE_M * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int m_local = i % SB_TILE_M;
        int k_local = i / SB_TILE_M;
        int m       = m0 + m_local;
        int k       = k0 + k_local;
        auto *dst   = &sA(m_local, k_local, pipe);
        bool valid  = (m < C_out && k < N_active);
        auto const *src = valid
            ? &grad_output[static_cast<int64_t>(compact_list[k].scatter_idx) * C_out + m]
            : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(dst, src, valid);
    }
}

// Swizzled-aware wgrad B-tile loader: writes through CuTe smem tensor.
template <typename Geom, bool IsSparse, typename MmaElement, typename SmemTensor>
__device__ void
load_b_tile_wgrad_swizzled(int32_t const *__restrict__ halo_ptr,
                           int32_t const *__restrict__ sparse_ptr,
                           CompactEntry const *__restrict__ compact_list,
                           MmaElement const *__restrict__ data,
                           int C_channel, int ktotal_start, int voxel_start,
                           int N_active, int K_total,
                           SmemTensor sB, int pipe, int tid, int nthreads) {
    constexpr int elems = SB_TILE_N * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int n_local = i % SB_TILE_N;
        int k_local = i / SB_TILE_N;

        int ktotal_pos = ktotal_start + n_local;
        int voxel_idx  = voxel_start + k_local;
        auto *dst      = &sB(n_local, k_local, pipe);

        if (ktotal_pos >= K_total || voxel_idx >= N_active) {
            sb_store_or_cp_async_4(dst,
                                   reinterpret_cast<MmaElement const *>(0), false);
            continue;
        }

        int kern_pos = ktotal_pos / C_channel;
        int c        = ktotal_pos % C_channel;

        int32_t src;
        if constexpr (IsSparse) {
            src = sparse_ptr[voxel_idx * Geom::KernVol + kern_pos];
        } else {
            auto entry = compact_list[voxel_idx];
            int vi = entry.local_voxel >> 6;
            int vj = (entry.local_voxel >> 3) & 7;
            int vk = entry.local_voxel & 7;

            int di = kern_pos / (Geom::R * Geom::S);
            int dj = (kern_pos / Geom::S) % Geom::R;
            int dk = kern_pos % Geom::S;

            int halo_idx = (vi + di) * Geom::Hy * Geom::Hz + (vj + dj) * Geom::Hz + (vk + dk);
            src = halo_ptr[entry.leaf_in_block * Geom::HaloVol + halo_idx];
        }

        bool active  = (src >= 0);
        auto const *ptr = active
            ? &data[static_cast<int64_t>(src) * C_channel + c]
            : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(dst, ptr, active);
    }
}

// ============================================================================
// Phase B Executable Block
// ============================================================================

template <typename Geom, typename Scalar, int N_LEAVES, ConvOp Op, bool IsSparse>
__device__ __forceinline__ void
superblock_gemm_phase(
    Scalar const *__restrict__ data_B,
    Scalar const *__restrict__ data_A_extra,
    Scalar const *__restrict__ weights,
    Scalar *__restrict__ output,
    int C_fast,
    int C_slow,
    SuperblockSmem<Geom, N_LEAVES>& smem,
    int N_active
) {
    using namespace cute;
    using Cfg        = SbMmaConfig<Scalar>;
    using MmaElement = typename Cfg::MmaElement;

    int const tid      = threadIdx.x;
    int const nthreads = blockDim.x;

    constexpr int TILE_M = SB_TILE_M;
    constexpr int TILE_N = SB_TILE_N;
    constexpr int TILE_K = SB_TILE_K;
    constexpr int MMA_K  = Cfg::MMA_K;
    constexpr int STAGES = SB_STAGES;
    constexpr bool Flip  = (Op == ConvOp::InputGrad || Op == ConvOp::TransposedFwd);

    using TiledMma = TiledMMA<MMA_Atom<typename Cfg::MmaAtom>, Layout<Shape<_2, _2, _1>>,
                              Tile<Int<TILE_M>, Int<TILE_N>, Int<MMA_K>>>;

    int const K_total = C_slow * Geom::KernVol;

    auto const *data_B_mma  = reinterpret_cast<MmaElement const *>(data_B);
    auto const *weights_mma = reinterpret_cast<MmaElement const *>(weights);
    auto const *extra_mma   = reinterpret_cast<MmaElement const *>(data_A_extra);

    constexpr int STAGE_A_ELEMS = TILE_M * TILE_K;
    constexpr int STAGE_B_ELEMS = TILE_N * TILE_K;
    auto *smem_A_base = reinterpret_cast<MmaElement *>(smem.mma_buf);
    auto *smem_B_base = smem_A_base + STAGE_A_ELEMS * STAGES;

    int32_t const *halo_ptr   = &smem.halo_maps[0][0];
    int32_t const *sparse_ptr = &smem.sparse_neighbors[0][0];

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    if constexpr (Cfg::USE_VECTORIZED_PIPELINE) {
        using SmemLayoutA = decltype(tile_to_shape(typename Cfg::SmemLayoutAtom{},
                                                   make_shape(Int<TILE_M>{}, Int<TILE_K>{}, Int<STAGES>{})));
        using SmemLayoutB = decltype(tile_to_shape(typename Cfg::SmemLayoutAtom{},
                                                   make_shape(Int<TILE_N>{}, Int<TILE_K>{}, Int<STAGES>{})));

        Tensor sA = make_tensor(make_smem_ptr(smem_A_base), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(smem_B_base), SmemLayoutB{});

        typename Cfg::GmemTiledCopyA gmem_tiled_copy_A;
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(tid);
        auto tAsA = gmem_thr_copy_A.partition_D(sA);

        auto smem_tiled_copy_A = make_tiled_copy_A(typename Cfg::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tid);
        auto tCsA = smem_thr_copy_A.partition_S(sA);
        auto tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
        auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

        auto smem_tiled_copy_B = make_tiled_copy_B(typename Cfg::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(tid);
        auto tCsB = smem_thr_copy_B.partition_S(sB);
        auto tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
        auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

        auto K_BLOCK_MAX = size<2>(tCrA);

        if constexpr (Op == ConvOp::Forward || Op == ConvOp::InputGrad || Op == ConvOp::TransposedFwd) {
            for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
                for (int n0 = 0; n0 < N_active; n0 += TILE_N) {
                    auto accum = partition_fragment_C(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
                    clear(accum);

                    int const k_tile_total = (K_total + TILE_K - 1) / TILE_K;
                    int k_tile_count = k_tile_total;
                    int k_iter       = 0;

                    int const pf = (STAGES - 1 < k_tile_count) ? STAGES - 1 : k_tile_count;
                    for (int pipe = 0; pipe < pf; ++pipe) {
                        int k0 = k_iter * TILE_K;
                        auto gA = make_tensor(
                            make_gmem_ptr(weights_mma + static_cast<int64_t>(m0) * K_total + k0),
                            make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), make_stride(K_total, Int<1>{})));
                        copy(gmem_tiled_copy_A, gmem_thr_copy_A.partition_S(gA), tAsA(_, _, _, pipe));
                        load_b_tile_vec16<Geom, Flip, IsSparse, MmaElement>(
                            halo_ptr, sparse_ptr, smem.compact_list, data_B_mma, C_slow, n0, k0,
                            N_active, K_total, sB, pipe, tid, nthreads);
                        cp_async_fence();
                        ++k_iter;
                        --k_tile_count;
                    }

                    int smem_pipe_read  = 0;
                    int smem_pipe_write = STAGES - 1;

                    cp_async_wait<STAGES - 2>();
                    __syncthreads();

                    Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
                    Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

                    if (K_BLOCK_MAX > 1) {
                        copy(smem_tiled_copy_A, tCsA_p(_, _, Int<0>{}), tCrA_copy_view(_, _, Int<0>{}));
                        copy(smem_tiled_copy_B, tCsB_p(_, _, Int<0>{}), tCrB_copy_view(_, _, Int<0>{}));
                    }

                    CUTLASS_PRAGMA_NO_UNROLL
                    while (k_tile_count > -(STAGES - 1)) {
                        for_each(make_int_sequence<decltype(K_BLOCK_MAX)::value>{}, [&](auto k_block) {
                            if (k_block == K_BLOCK_MAX - 1) {
                                tCsA_p = tCsA(_, _, _, smem_pipe_read);
                                tCsB_p = tCsB(_, _, _, smem_pipe_read);
                                cp_async_wait<STAGES - 2>();
                                __syncthreads();
                            }

                            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                            copy(smem_tiled_copy_A, tCsA_p(_, _, k_block_next), tCrA_copy_view(_, _, k_block_next));
                            copy(smem_tiled_copy_B, tCsB_p(_, _, k_block_next), tCrB_copy_view(_, _, k_block_next));

                            if (k_block == 0) {
                                if (k_tile_count > 0) {
                                    int k0 = k_iter * TILE_K;
                                    auto gA = make_tensor(
                                        make_gmem_ptr(weights_mma + static_cast<int64_t>(m0) * K_total + k0),
                                        make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), make_stride(K_total, Int<1>{})));
                                    copy(gmem_tiled_copy_A, gmem_thr_copy_A.partition_S(gA), tAsA(_, _, _, smem_pipe_write));
                                    load_b_tile_vec16<Geom, Flip, IsSparse, MmaElement>(
                                        halo_ptr, sparse_ptr, smem.compact_list, data_B_mma, C_slow, n0, k0,
                                        N_active, K_total, sB, smem_pipe_write, tid, nthreads);
                                }
                                cp_async_fence();
                                --k_tile_count;
                                if (k_tile_count > 0) ++k_iter;

                                smem_pipe_write = smem_pipe_read;
                                ++smem_pipe_read;
                                smem_pipe_read = (smem_pipe_read == STAGES) ? 0 : smem_pipe_read;
                            }

                            cute::transform(tCrA(_, _, k_block), identity{});
                            cute::transform(tCrB(_, _, k_block), identity{});
                            gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
                        });
                    }

                    cp_async_wait<0>();
                    __syncthreads();

                    auto *smem_C = reinterpret_cast<float *>(smem.mma_buf);
                    auto sC = make_tensor(make_smem_ptr(smem_C), make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_stride(Int<1>{}, Int<TILE_M>{})));
                    auto smem_copy_c = make_tiled_copy_C(Copy_Atom<UniversalCopy<uint32_t>, float>{}, tiled_mma);
                    auto smem_thr_copy_c = smem_copy_c.get_thread_slice(tid);
                    copy(smem_copy_c, smem_thr_copy_c.retile_S(accum), smem_thr_copy_c.partition_D(sC));

                    __syncthreads();

                    for (int i = tid; i < TILE_M * TILE_N; i += nthreads) {
                        int m_local = i % TILE_M;
                        int n_local = i / TILE_M;
                        int cout    = m0 + m_local;
                        int voxel   = n0 + n_local;
                        if (cout < C_fast && voxel < N_active) {
                            int32_t s_idx = smem.compact_list[voxel].scatter_idx;
                            output[static_cast<int64_t>(s_idx) * C_fast + cout] = static_cast<Scalar>(smem_C[i]);
                        }
                    }
                    __syncthreads();
                }
            }
        } else if constexpr (Op == ConvOp::WeightGrad) {
            for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
                for (int n0 = 0; n0 < K_total; n0 += TILE_N) {
                    auto accum = partition_fragment_C(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
                    clear(accum);

                    int const k_tile_total = (N_active + TILE_K - 1) / TILE_K;
                    int k_tile_count = k_tile_total;
                    int k_iter       = 0;

                    int const pf = (STAGES - 1 < k_tile_count) ? STAGES - 1 : k_tile_count;
                    for (int pipe = 0; pipe < pf; ++pipe) {
                        int k0 = k_iter * TILE_K;
                        load_a_tile_wgrad_swizzled<MmaElement>(extra_mma, smem.compact_list, sA, pipe, C_fast, m0, k0, N_active, tid, nthreads);
                        load_b_tile_wgrad_swizzled<Geom, IsSparse, MmaElement>(halo_ptr, sparse_ptr, smem.compact_list, data_B_mma, C_slow, n0, k0, N_active, K_total, sB, pipe, tid, nthreads);
                        cute::cp_async_fence();
                        ++k_iter;
                        --k_tile_count;
                    }

                    int smem_pipe_read  = 0;
                    int smem_pipe_write = STAGES - 1;

                    cp_async_wait<STAGES - 2>();
                    __syncthreads();

                    Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
                    Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

                    if (K_BLOCK_MAX > 1) {
                        copy(smem_tiled_copy_A, tCsA_p(_, _, Int<0>{}), tCrA_copy_view(_, _, Int<0>{}));
                        copy(smem_tiled_copy_B, tCsB_p(_, _, Int<0>{}), tCrB_copy_view(_, _, Int<0>{}));
                    }

                    CUTLASS_PRAGMA_NO_UNROLL
                    while (k_tile_count > -(STAGES - 1)) {
                        for_each(make_int_sequence<decltype(K_BLOCK_MAX)::value>{}, [&](auto k_block) {
                            if (k_block == K_BLOCK_MAX - 1) {
                                tCsA_p = tCsA(_, _, _, smem_pipe_read);
                                tCsB_p = tCsB(_, _, _, smem_pipe_read);
                                cp_async_wait<STAGES - 2>();
                                __syncthreads();
                            }

                            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                            copy(smem_tiled_copy_A, tCsA_p(_, _, k_block_next), tCrA_copy_view(_, _, k_block_next));
                            copy(smem_tiled_copy_B, tCsB_p(_, _, k_block_next), tCrB_copy_view(_, _, k_block_next));

                            if (k_block == 0) {
                                if (k_tile_count > 0) {
                                    int k0 = k_iter * TILE_K;
                                    load_a_tile_wgrad_swizzled<MmaElement>(extra_mma, smem.compact_list, sA, smem_pipe_write, C_fast, m0, k0, N_active, tid, nthreads);
                                    load_b_tile_wgrad_swizzled<Geom, IsSparse, MmaElement>(halo_ptr, sparse_ptr, smem.compact_list, data_B_mma, C_slow, n0, k0, N_active, K_total, sB, smem_pipe_write, tid, nthreads);
                                }
                                cp_async_fence();
                                --k_tile_count;
                                if (k_tile_count > 0) ++k_iter;

                                smem_pipe_write = smem_pipe_read;
                                ++smem_pipe_read;
                                smem_pipe_read = (smem_pipe_read == STAGES) ? 0 : smem_pipe_read;
                            }

                            cute::transform(tCrA(_, _, k_block), identity{});
                            cute::transform(tCrB(_, _, k_block), identity{});
                            gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
                        });
                    }

                    cp_async_wait<0>();
                    __syncthreads();

                    auto *smem_C = reinterpret_cast<float *>(smem.mma_buf);
                    auto sC = make_tensor(make_smem_ptr(smem_C), make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_stride(Int<1>{}, Int<TILE_M>{})));
                    auto smem_copy_c = make_tiled_copy_C(Copy_Atom<UniversalCopy<uint32_t>, float>{}, tiled_mma);
                    auto smem_thr_copy_c = smem_copy_c.get_thread_slice(tid);
                    copy(smem_copy_c, smem_thr_copy_c.retile_S(accum), smem_thr_copy_c.partition_D(sC));

                    __syncthreads();

                    auto *output_f = reinterpret_cast<float *>(output);
                    for (int i = tid; i < TILE_M * TILE_N; i += nthreads) {
                        int m_local = i % TILE_M;
                        int n_local = i / TILE_M;
                        int cout    = m0 + m_local;
                        int kpos    = n0 + n_local;
                        if (cout < C_fast && kpos < K_total) {
                            atomicAdd(&output_f[static_cast<int64_t>(cout) * K_total + kpos], smem_C[i]);
                        }
                    }
                    __syncthreads();
                }
            }
        }
    } else {
        // Legacy fp16 implementation (No ldmatrix pipelines)
        auto get_A_stage = [&](int s) -> MmaElement * { return smem_A_base + s * STAGE_A_ELEMS; };
        auto get_B_stage = [&](int s) -> MmaElement * { return smem_B_base + s * STAGE_B_ELEMS; };
        auto make_sA = [](MmaElement *ptr) { return make_tensor(make_smem_ptr(ptr), make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), make_stride(Int<1>{}, Int<TILE_M>{}))); };
        auto make_sB = [](MmaElement *ptr) { return make_tensor(make_smem_ptr(ptr), make_layout(make_shape(Int<TILE_N>{}, Int<TILE_K>{}), make_stride(Int<1>{}, Int<TILE_N>{}))); };

        auto tCrA = thr_mma.partition_fragment_A(make_sA(get_A_stage(0)));
        auto tCrB = thr_mma.partition_fragment_B(make_sB(get_B_stage(0)));
        constexpr int K_BLOCK_MAX_L = TILE_K / MMA_K;

        auto copy_smem_to_reg_kb = [&](int stage, int kb) {
            auto tCsA_s = thr_mma.partition_A(make_sA(get_A_stage(stage)));
            auto tCsB_s = thr_mma.partition_B(make_sB(get_B_stage(stage)));
#pragma unroll
            for (int j = 0; j < size(tCrA(_, _, 0)); ++j) tCrA(_, _, kb)(j) = tCsA_s(_, _, kb)(j);
#pragma unroll
            for (int j = 0; j < size(tCrB(_, _, 0)); ++j) tCrB(_, _, kb)(j) = tCsB_s(_, _, kb)(j);
        };

        if constexpr (Op == ConvOp::Forward || Op == ConvOp::InputGrad || Op == ConvOp::TransposedFwd) {
            for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
                for (int n0 = 0; n0 < N_active; n0 += TILE_N) {
                    auto accum = partition_fragment_C(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
                    clear(accum);

                    int const k_tile_total = (K_total + TILE_K - 1) / TILE_K;
                    int k_tile_count = k_tile_total;
                    int k_iter       = 0;
                    int const pf = (STAGES - 1 < k_tile_count) ? STAGES - 1 : k_tile_count;
                    for (int pipe = 0; pipe < pf; ++pipe) {
                        int k0 = k_iter * TILE_K;
                        load_a_tile_async<MmaElement>(weights_mma, get_A_stage(pipe), C_fast, m0, k0, K_total, tid, nthreads);
                        load_b_tile_async<Geom, Flip, IsSparse, MmaElement>(halo_ptr, sparse_ptr, smem.compact_list, data_B_mma, C_slow, n0, k0, N_active, K_total, get_B_stage(pipe), tid, nthreads);
                        cute::cp_async_fence();
                        ++k_iter;
                        --k_tile_count;
                    }

                    int smem_pipe_read  = 0;
                    int smem_pipe_write = STAGES - 1;
                    int current_read_stage = 0;

                    cute::cp_async_wait<STAGES - 2>();
                    __syncthreads();
                    copy_smem_to_reg_kb(current_read_stage, 0);

                    while (k_tile_count > -(STAGES - 1)) {
#pragma unroll
                        for (int kb = 0; kb < K_BLOCK_MAX_L; ++kb) {
                            if (kb == K_BLOCK_MAX_L - 1) {
                                current_read_stage = smem_pipe_read;
                                cute::cp_async_wait<STAGES - 2>();
                                __syncthreads();
                            }
                            int kb_next = (kb + 1) % K_BLOCK_MAX_L;
                            copy_smem_to_reg_kb(current_read_stage, kb_next);
                            if (kb == 0) {
                                if (k_tile_count > 0) {
                                    int k0 = k_iter * TILE_K;
                                    load_a_tile_async<MmaElement>(weights_mma, get_A_stage(smem_pipe_write), C_fast, m0, k0, K_total, tid, nthreads);
                                    load_b_tile_async<Geom, Flip, IsSparse, MmaElement>(halo_ptr, sparse_ptr, smem.compact_list, data_B_mma, C_slow, n0, k0, N_active, K_total, get_B_stage(smem_pipe_write), tid, nthreads);
                                }
                                cute::cp_async_fence();
                                --k_tile_count;
                                if (k_tile_count > 0) ++k_iter;
                                smem_pipe_write = smem_pipe_read;
                                ++smem_pipe_read;
                                if (smem_pipe_read == STAGES) smem_pipe_read = 0;
                            }
                            gemm(tiled_mma, accum, tCrA(_, _, kb), tCrB(_, _, kb), accum);
                        }
                    }

                    cute::cp_async_wait<0>();
                    __syncthreads();

                    auto *smem_C = reinterpret_cast<float *>(smem.mma_buf);
                    auto sC = make_tensor(make_smem_ptr(smem_C), make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_stride(Int<1>{}, Int<TILE_M>{})));
                    auto smem_copy_c = make_tiled_copy_C(Copy_Atom<UniversalCopy<uint32_t>, float>{}, tiled_mma);
                    auto smem_thr_copy_c = smem_copy_c.get_thread_slice(tid);
                    copy(smem_copy_c, smem_thr_copy_c.retile_S(accum), smem_thr_copy_c.partition_D(sC));
                    __syncthreads();

                    for (int i = tid; i < TILE_M * TILE_N; i += nthreads) {
                        int m_local = i % TILE_M;
                        int n_local = i / TILE_M;
                        int cout    = m0 + m_local;
                        int voxel   = n0 + n_local;
                        if (cout < C_fast && voxel < N_active) {
                            int32_t s_idx = smem.compact_list[voxel].scatter_idx;
                            output[static_cast<int64_t>(s_idx) * C_fast + cout] = static_cast<Scalar>(smem_C[i]);
                        }
                    }
                    __syncthreads();
                }
            }
        } else if constexpr (Op == ConvOp::WeightGrad) {
            for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
                for (int n0 = 0; n0 < K_total; n0 += TILE_N) {
                    auto accum = partition_fragment_C(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
                    clear(accum);

                    int const k_tile_total = (N_active + TILE_K - 1) / TILE_K;
                    int k_tile_count = k_tile_total;
                    int k_iter       = 0;

                    int const pf = (STAGES - 1 < k_tile_count) ? STAGES - 1 : k_tile_count;
                    for (int pipe = 0; pipe < pf; ++pipe) {
                        int k0 = k_iter * TILE_K;
                        load_a_tile_wgrad_async<MmaElement>(extra_mma, smem.compact_list, get_A_stage(pipe), C_fast, m0, k0, N_active, tid, nthreads);
                        load_b_tile_wgrad_async<Geom, IsSparse, MmaElement>(halo_ptr, sparse_ptr, smem.compact_list, data_B_mma, C_slow, n0, k0, N_active, K_total, get_B_stage(pipe), tid, nthreads);
                        cute::cp_async_fence();
                        ++k_iter;
                        --k_tile_count;
                    }

                    int smem_pipe_read  = 0;
                    int smem_pipe_write = STAGES - 1;
                    int current_read_stage = 0;

                    cute::cp_async_wait<STAGES - 2>();
                    __syncthreads();
                    copy_smem_to_reg_kb(current_read_stage, 0);

                    while (k_tile_count > -(STAGES - 1)) {
#pragma unroll
                        for (int kb = 0; kb < K_BLOCK_MAX_L; ++kb) {
                            if (kb == K_BLOCK_MAX_L - 1) {
                                current_read_stage = smem_pipe_read;
                                cute::cp_async_wait<STAGES - 2>();
                                __syncthreads();
                            }
                            int kb_next = (kb + 1) % K_BLOCK_MAX_L;
                            copy_smem_to_reg_kb(current_read_stage, kb_next);
                            if (kb == 0) {
                                if (k_tile_count > 0) {
                                    int k0 = k_iter * TILE_K;
                                    load_a_tile_wgrad_async<MmaElement>(extra_mma, smem.compact_list, get_A_stage(smem_pipe_write), C_fast, m0, k0, N_active, tid, nthreads);
                                    load_b_tile_wgrad_async<Geom, IsSparse, MmaElement>(halo_ptr, sparse_ptr, smem.compact_list, data_B_mma, C_slow, n0, k0, N_active, K_total, get_B_stage(smem_pipe_write), tid, nthreads);
                                }
                                cute::cp_async_fence();
                                --k_tile_count;
                                if (k_tile_count > 0) ++k_iter;
                                smem_pipe_write = smem_pipe_read;
                                ++smem_pipe_read;
                                if (smem_pipe_read == STAGES) smem_pipe_read = 0;
                            }
                            gemm(tiled_mma, accum, tCrA(_, _, kb), tCrB(_, _, kb), accum);
                        }
                    }

                    cute::cp_async_wait<0>();
                    __syncthreads();

                    auto *smem_C = reinterpret_cast<float *>(smem.mma_buf);
                    auto sC = make_tensor(make_smem_ptr(smem_C), make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_stride(Int<1>{}, Int<TILE_M>{})));
                    auto smem_copy_c = make_tiled_copy_C(Copy_Atom<UniversalCopy<uint32_t>, float>{}, tiled_mma);
                    auto smem_thr_copy_c = smem_copy_c.get_thread_slice(tid);
                    copy(smem_copy_c, smem_thr_copy_c.retile_S(accum), smem_thr_copy_c.partition_D(sC));
                    __syncthreads();

                    auto *output_f = reinterpret_cast<float *>(output);
                    for (int i = tid; i < TILE_M * TILE_N; i += nthreads) {
                        int m_local = i % TILE_M;
                        int n_local = i / TILE_M;
                        int cout    = m0 + m_local;
                        int kpos    = n0 + n_local;
                        if (cout < C_fast && kpos < K_total) {
                            atomicAdd(&output_f[static_cast<int64_t>(cout) * K_total + kpos], smem_C[i]);
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}

// ============================================================================
// Unified Adaptive Superblock convolution kernel
// ============================================================================

template <typename Geom, typename Scalar, int N_LEAVES, ConvOp Op>
__global__ void __launch_bounds__(128, 1)
superblock_conv_kernel(GridBatchImpl::Accessor primary_acc,
                       GridBatchImpl::Accessor secondary_acc,
                       Scalar const *__restrict__ data_B,
                       Scalar const *__restrict__ data_A_extra,
                       Scalar const *__restrict__ weights,
                       Scalar *__restrict__ output,
                       int C_fast,
                       int C_slow,
                       int total_primary_leaves) {
    extern __shared__ char smem_raw[];
    auto &smem = *reinterpret_cast<SuperblockSmem<Geom, N_LEAVES> *>(smem_raw);

    int const tid      = threadIdx.x;
    int const nthreads = blockDim.x;
    int const sb_id    = blockIdx.x;

    int const first_leaf = sb_id * N_LEAVES;
    int const n_leaves = (first_leaf + N_LEAVES <= total_primary_leaves)
                       ? N_LEAVES : (total_primary_leaves - first_leaf);

    if (n_leaves <= 0) return;

    auto const *pri_grid = primary_acc.grid(0);
    auto const *sec_grid = secondary_acc.grid(0);
    int32_t const pri_vo = static_cast<int32_t>(primary_acc.voxelOffset(0));
    int32_t const sec_vo = static_cast<int32_t>(secondary_acc.voxelOffset(0));

    // ================================================================
    // Phase A: Adaptive Topology build + Compaction
    // ================================================================

    if (tid == 0) smem.compact_count = 0;
    __syncthreads();

    auto sec_tree_acc = sec_grid->getAccessor();

    for (int li = 0; li < n_leaves; ++li) {
        auto const &leaf = pri_grid->tree().template getFirstNode<0>()[first_leaf + li];

        // Cache node origins locally
        if (tid == 0) smem.leaf_origins[li] = leaf.origin();

        for (int v = tid; v < Geom::LeafVol; v += nthreads) {
            if (leaf.isActive(v)) {
                int pos = atomicAdd(&smem.compact_count, 1);
                CompactEntry entry;
                entry.scatter_idx   = static_cast<int32_t>(leaf.getValue(v) - 1) + pri_vo;
                entry.leaf_in_block = static_cast<uint8_t>(li);
                entry._pad          = 0;
                entry.local_voxel   = static_cast<uint16_t>(v);
                smem.compact_list[pos] = entry;
            }
        }
    }
    __syncthreads();

    int const N_active = smem.compact_count;
    if (N_active == 0) return;

    // Adaptive Sparsity Heuristic:
    // If targeted active-voxel queries cost fewer memory cycles than populating
    // the 3D local bounding box (which has dense empty padding), we flip to sparse mode.
    bool is_sparse = (N_active * Geom::KernVol <= n_leaves * Geom::HaloVol);

    if (is_sparse) {
        // Parallelizing outer-loop by voxel maximizes thread-local NanoVDB accessor Cache Hits.
        // A single thread queries the 27 neighbors contiguously.
        for (int v = tid; v < N_active; v += nthreads) {
            auto entry = smem.compact_list[v];
            int li     = entry.leaf_in_block;
            int vi     = entry.local_voxel >> 6;
            int vj     = (entry.local_voxel >> 3) & 7;
            int vk     = entry.local_voxel & 7;

            for (int k_offset = 0; k_offset < Geom::KernVol; ++k_offset) {
                int di = k_offset / (Geom::R * Geom::S);
                int dj = (k_offset / Geom::S) % Geom::R;
                int dk = k_offset % Geom::S;

                constexpr bool Flip = (Op == ConvOp::InputGrad || Op == ConvOp::TransposedFwd);
                if constexpr (Flip) {
                    di = Geom::T - 1 - di;
                    dj = Geom::R - 1 - dj;
                    dk = Geom::S - 1 - dk;
                }

                nanovdb::Coord coord = smem.leaf_origins[li].offsetBy(vi + di + Geom::Dx, vj + dj + Geom::Dy, vk + dk + Geom::Dz);
                auto val = sec_tree_acc.getValue(coord);
                smem.sparse_neighbors[v][k_offset] = (val > 0) ? static_cast<int32_t>(val - 1) + sec_vo : -1;
            }
        }
    } else {
        for (int li = 0; li < n_leaves; ++li) {
            nanovdb::Coord halo_base = smem.leaf_origins[li].offsetBy(Geom::Dx, Geom::Dy, Geom::Dz);
            for (int h = tid; h < Geom::HaloVol; h += nthreads) {
                int hi = h / (Geom::Hy * Geom::Hz);
                int hj = (h / Geom::Hz) % Geom::Hy;
                int hk = h % Geom::Hz;
                nanovdb::Coord coord = halo_base.offsetBy(hi, hj, hk);
                auto val             = sec_tree_acc.getValue(coord);
                smem.halo_maps[li][h] = (val > 0) ? static_cast<int32_t>(val - 1) + sec_vo : -1;
            }
        }
    }
    __syncthreads();

    // ================================================================
    // Phase B: Execute Block MMA
    // ================================================================
    if (is_sparse) {
        superblock_gemm_phase<Geom, Scalar, N_LEAVES, Op, true>(
            data_B, data_A_extra, weights, output, C_fast, C_slow, smem, N_active);
    } else {
        superblock_gemm_phase<Geom, Scalar, N_LEAVES, Op, false>(
            data_B, data_A_extra, weights, output, C_fast, C_slow, smem, N_active);
    }
}

// ============================================================================
// Typed launcher
// ============================================================================

template <typename Geom, typename Scalar, int N_LEAVES, ConvOp Op>
static void
launchSuperblockConv(GridBatchImpl const &primary_grid, GridBatchImpl const &secondary_grid,
                     torch::Tensor data_B, torch::Tensor data_A_extra, torch::Tensor weights_permuted, torch::Tensor output,
                     int C_fast, int C_slow, cudaStream_t stream) {
    auto pri_acc = primary_grid.deviceAccessor();
    auto sec_acc = secondary_grid.deviceAccessor();

    int64_t const num_leaves = primary_grid.totalLeaves();
    if (num_leaves == 0) return;

    int const num_blocks = static_cast<int>((num_leaves + N_LEAVES - 1) / N_LEAVES);

    constexpr int THREADS = 128;
    constexpr size_t SMEM = sizeof(SuperblockSmem<Geom, N_LEAVES>);
    constexpr size_t SMEM48 = 48u * 1024u;

    if constexpr (SMEM > SMEM48) {
        cudaFuncSetAttribute(superblock_conv_kernel<Geom, Scalar, N_LEAVES, Op>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SMEM));
    }

    Scalar const *data_B_ptr   = data_B.data_ptr<Scalar>();
    Scalar const *extra_ptr    = data_A_extra.defined() ? data_A_extra.data_ptr<Scalar>() : nullptr;
    Scalar const *weights_ptr  = weights_permuted.defined() ? weights_permuted.data_ptr<Scalar>() : nullptr;

    Scalar *output_ptr;
    if constexpr (Op == ConvOp::WeightGrad) output_ptr = reinterpret_cast<Scalar *>(output.data_ptr<float>());
    else output_ptr = output.data_ptr<Scalar>();

    superblock_conv_kernel<Geom, Scalar, N_LEAVES, Op><<<num_blocks, THREADS, SMEM, stream>>>(
        pri_acc, sec_acc, data_B_ptr, extra_ptr, weights_ptr, output_ptr,
        C_fast, C_slow, static_cast<int>(num_leaves));
}

// ============================================================================
// Per-operation launcher wrappers
// ============================================================================

template <typename Geom, typename Scalar, int N_LEAVES>
static void
launchForward(GridBatchImpl const &feature_grid, GridBatchImpl const &output_grid, torch::Tensor features, torch::Tensor weights, torch::Tensor output, int C_in, int C_out, cudaStream_t stream) {
    auto W = weights.permute({0, 2, 3, 4, 1}).contiguous().view({C_out, Geom::KernVol * C_in});
    launchSuperblockConv<Geom, Scalar, N_LEAVES, ConvOp::Forward>(output_grid, feature_grid, features, torch::Tensor(), W, output, C_out, C_in, stream);
}

template <typename Geom, typename Scalar, int N_LEAVES>
static void
launchInputGrad(GridBatchImpl const &feature_grid, GridBatchImpl const &output_grid, torch::Tensor grad_output, torch::Tensor weights, torch::Tensor grad_features, int C_in, int C_out, cudaStream_t stream) {
    auto W = weights.permute({1, 2, 3, 4, 0}).contiguous().view({C_in, Geom::KernVol * C_out});
    launchSuperblockConv<Geom, Scalar, N_LEAVES, ConvOp::InputGrad>(feature_grid, output_grid, grad_output, torch::Tensor(), W, grad_features, C_in, C_out, stream);
}

template <typename Geom, typename Scalar, int N_LEAVES>
static void
launchWeightGrad(GridBatchImpl const &feature_grid, GridBatchImpl const &output_grid, torch::Tensor features, torch::Tensor grad_output, torch::Tensor grad_weights_flat, int C_in, int C_out, cudaStream_t stream) {
    launchSuperblockConv<Geom, Scalar, N_LEAVES, ConvOp::WeightGrad>(output_grid, feature_grid, features, grad_output, torch::Tensor(), grad_weights_flat, C_out, C_in, stream);
}

template <typename Geom, typename Scalar, int N_LEAVES>
static void
launchTransposedFwd(GridBatchImpl const &source_grid, GridBatchImpl const &target_grid, torch::Tensor features, torch::Tensor weights, torch::Tensor output, int C_in, int C_out, cudaStream_t stream) {
    auto W = weights.permute({0, 2, 3, 4, 1}).contiguous().view({C_out, Geom::KernVol * C_in});
    launchSuperblockConv<Geom, Scalar, N_LEAVES, ConvOp::TransposedFwd>(target_grid, source_grid, features, torch::Tensor(), W, output, C_out, C_in, stream);
}

// ============================================================================
// Kernel-size + N_LEAVES dispatch
// ============================================================================

template <typename Scalar>
static torch::Tensor
superblockConvDispatched(torch::Tensor features, torch::Tensor weights, GridBatchImpl const &feature_grid, GridBatchImpl const &output_grid, nanovdb::Coord kernel_size, int C_in, int C_out) {
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const NB = output_grid.totalVoxels();
    auto output      = torch::zeros({NB, C_out}, features.options());
    if (NB == 0) return output;

    if (kernel_size == nanovdb::Coord(3, 3, 3)) launchForward<SuperblockGeometry<3, 3, 3>, Scalar, 4>(feature_grid, output_grid, features, weights, output, C_in, C_out, stream);
    else if (kernel_size == nanovdb::Coord(5, 5, 5)) launchForward<SuperblockGeometry<5, 5, 5>, Scalar, 4>(feature_grid, output_grid, features, weights, output, C_in, C_out, stream);
    else if (kernel_size == nanovdb::Coord(7, 7, 7)) launchForward<SuperblockGeometry<7, 7, 7>, Scalar, 2>(feature_grid, output_grid, features, weights, output, C_in, C_out, stream);
    else TORCH_CHECK(false, "AdaptiveSuperblock: unsupported kernel size. Supported: 3x3x3, 5x5x5, 7x7x7.");
    return output;
}

template <typename Scalar>
static std::tuple<torch::Tensor, torch::Tensor>
superblockConvBackwardDispatched(torch::Tensor grad_output, torch::Tensor features, torch::Tensor weights, GridBatchImpl const &feature_grid, GridBatchImpl const &output_grid, nanovdb::Coord kernel_size, int C_in, int C_out) {
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const NF = feature_grid.totalVoxels();
    auto grad_features = torch::zeros({NF, C_in}, features.options());

    int K_vol          = kernel_size[0] * kernel_size[1] * kernel_size[2];
    int K_total        = C_in * K_vol;
    auto grad_weights_flat = torch::zeros({static_cast<int64_t>(C_out), static_cast<int64_t>(K_total)}, features.options().dtype(torch::kFloat32));

    auto launchIGrad = [&](auto geom_tag, auto n_leaves_tag) {
        using Geom = decltype(geom_tag); constexpr int N_LEAVES = decltype(n_leaves_tag)::value;
        launchInputGrad<Geom, Scalar, N_LEAVES>(feature_grid, output_grid, grad_output, weights, grad_features, C_in, C_out, stream);
    };

    auto launchWGrad = [&](auto geom_tag, auto n_leaves_tag) {
        using Geom = decltype(geom_tag); constexpr int N_LEAVES = decltype(n_leaves_tag)::value;
        launchWeightGrad<Geom, Scalar, N_LEAVES>(feature_grid, output_grid, features, grad_output, grad_weights_flat, C_in, C_out, stream);
    };

    if (kernel_size == nanovdb::Coord(3, 3, 3)) { launchIGrad(SuperblockGeometry<3, 3, 3>{}, std::integral_constant<int, 4>{}); launchWGrad(SuperblockGeometry<3, 3, 3>{}, std::integral_constant<int, 4>{}); }
    else if (kernel_size == nanovdb::Coord(5, 5, 5)) { launchIGrad(SuperblockGeometry<5, 5, 5>{}, std::integral_constant<int, 4>{}); launchWGrad(SuperblockGeometry<5, 5, 5>{}, std::integral_constant<int, 4>{}); }
    else if (kernel_size == nanovdb::Coord(7, 7, 7)) { launchIGrad(SuperblockGeometry<7, 7, 7>{}, std::integral_constant<int, 2>{}); launchWGrad(SuperblockGeometry<7, 7, 7>{}, std::integral_constant<int, 2>{}); }
    else TORCH_CHECK(false, "AdaptiveSuperblockBackward: unsupported kernel size.");

    auto grad_weights = grad_weights_flat.view({C_out, kernel_size[0], kernel_size[1], kernel_size[2], C_in}).permute({0, 4, 1, 2, 3}).contiguous();
    return {grad_features, grad_weights};
}

template <typename Scalar>
static torch::Tensor
superblockConvTransposeDispatched(torch::Tensor features, torch::Tensor weights, GridBatchImpl const &source_grid, GridBatchImpl const &target_grid, nanovdb::Coord kernel_size, int C_in, int C_out) {
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const NT = target_grid.totalVoxels();
    auto output      = torch::zeros({NT, C_out}, features.options());
    if (NT == 0) return output;

    if (kernel_size == nanovdb::Coord(3, 3, 3)) launchTransposedFwd<SuperblockGeometry<3, 3, 3>, Scalar, 4>(source_grid, target_grid, features, weights, output, C_in, C_out, stream);
    else if (kernel_size == nanovdb::Coord(5, 5, 5)) launchTransposedFwd<SuperblockGeometry<5, 5, 5>, Scalar, 4>(source_grid, target_grid, features, weights, output, C_in, C_out, stream);
    else if (kernel_size == nanovdb::Coord(7, 7, 7)) launchTransposedFwd<SuperblockGeometry<7, 7, 7>, Scalar, 2>(source_grid, target_grid, features, weights, output, C_in, C_out, stream);
    else TORCH_CHECK(false, "AdaptiveSuperblockTranspose: unsupported kernel size.");
    return output;
}

// ============================================================================
// Input validation helper
// ============================================================================

static void
validateSuperblockInputs(torch::Tensor features, torch::Tensor weights, GridBatchImpl const &feature_grid, nanovdb::Coord kernel_size, nanovdb::Coord stride, char const *func_name) {
    TORCH_CHECK(features.is_cuda() && deviceSupportsSuperblock(features.device()), func_name, ": requires Sm80+ (Ampere or newer)");
    TORCH_CHECK(features.dim() == 2 && features.size(0) == feature_grid.totalVoxels() && features.is_contiguous(), func_name, ": invalid feature tensor");
    TORCH_CHECK(weights.dim() == 5 && features.size(1) == weights.size(1) && features.device() == weights.device() && features.scalar_type() == weights.scalar_type(), func_name, ": invalid weights");
    TORCH_CHECK(weights.size(2) == kernel_size[0] && weights.size(3) == kernel_size[1] && weights.size(4) == kernel_size[2], func_name, ": kernel dim mismatch");
    TORCH_CHECK(stride[0] == 1 && stride[1] == 1 && stride[2] == 1, func_name, ": only stride=(1,1,1) is supported");
    TORCH_CHECK(features.size(1) > 0 && features.size(1) % 32 == 0 && weights.size(0) > 0 && weights.size(0) % 32 == 0, func_name, ": channels must be multiple of 32");
}

// ============================================================================
// Entry points
// ============================================================================

torch::Tensor
superblockConv(torch::Tensor features, torch::Tensor weights, GridBatchImpl const &feature_grid, GridBatchImpl const &output_grid, nanovdb::Coord kernel_size, nanovdb::Coord stride) {
    validateSuperblockInputs(features, weights, feature_grid, kernel_size, stride, "superblockConv");
    int64_t C_in = features.size(1), C_out = weights.size(0);
    if (features.scalar_type() == torch::kFloat16) {
        return superblockConvDispatched<c10::Half>(features.to(torch::kFloat16).contiguous(), weights.to(torch::kFloat16).contiguous(), feature_grid, output_grid, kernel_size, C_in, C_out).to(torch::kFloat16);
    } else if (features.scalar_type() == torch::kFloat32) {
        return superblockConvDispatched<float>(features, weights, feature_grid, output_grid, kernel_size, C_in, C_out);
    } else TORCH_CHECK(false, "superblockConv: unsupported dtype");
}

std::tuple<torch::Tensor, torch::Tensor>
superblockConvBackward(torch::Tensor grad_output, torch::Tensor features, torch::Tensor weights, GridBatchImpl const &feature_grid, GridBatchImpl const &output_grid, nanovdb::Coord kernel_size, nanovdb::Coord stride) {
    validateSuperblockInputs(features, weights, feature_grid, kernel_size, stride, "superblockConvBackward");
    TORCH_CHECK(grad_output.is_cuda() && grad_output.dim() == 2 && grad_output.size(0) == output_grid.totalVoxels() && grad_output.size(1) == weights.size(0) && grad_output.is_contiguous(), "superblockConvBackward: invalid grad_output");
    int64_t C_in = features.size(1), C_out = weights.size(0);
    if (features.scalar_type() == torch::kFloat16) {
        auto [gf, gw] = superblockConvBackwardDispatched<c10::Half>(grad_output.to(torch::kFloat16).contiguous(), features.to(torch::kFloat16).contiguous(), weights.to(torch::kFloat16).contiguous(), feature_grid, output_grid, kernel_size, C_in, C_out);
        return {gf.to(torch::kFloat16), gw.to(torch::kFloat16)};
    } else if (features.scalar_type() == torch::kFloat32) {
        return superblockConvBackwardDispatched<float>(grad_output, features, weights, feature_grid, output_grid, kernel_size, C_in, C_out);
    } else TORCH_CHECK(false, "superblockConvBackward: unsupported dtype");
}

torch::Tensor
superblockConvTranspose(torch::Tensor features, torch::Tensor weights, GridBatchImpl const &source_grid, GridBatchImpl const &target_grid, nanovdb::Coord kernel_size, nanovdb::Coord stride) {
    validateSuperblockInputs(features, weights, source_grid, kernel_size, stride, "superblockConvTranspose");
    int64_t C_in = features.size(1), C_out = weights.size(0);
    if (features.scalar_type() == torch::kFloat16) {
        return superblockConvTransposeDispatched<c10::Half>(features.to(torch::kFloat16).contiguous(), weights.to(torch::kFloat16).contiguous(), source_grid, target_grid, kernel_size, C_in, C_out).to(torch::kFloat16);
    } else if (features.scalar_type() == torch::kFloat32) {
        return superblockConvTransposeDispatched<float>(features, weights, source_grid, target_grid, kernel_size, C_in, C_out);
    } else TORCH_CHECK(false, "superblockConvTranspose: unsupported dtype");
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#else

// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Superblock.cu -- Superblock GEMM sparse convolution.
//
// One CUDA block per superblock (N consecutive NanoVDB leaves).
//   Phase A: Cooperatively builds N halo maps and compacts active voxels
//            into a single compact list in shared memory.
//   Phase B: Tiled MMA GEMM over only the compacted active voxels.
//   Phase C: Epilogue -- scatter writes (forward/igrad/transpose) or
//            atomicAdd to weight gradient.
//
// All four operations (forward, input gradient, weight gradient, transposed
// forward) are handled by a single kernel template parameterized by ConvOp.
//

#include <fvdb/detail/ops/convolution/Superblock.h>

#include <nanovdb/NanoVDB.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

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
deviceSupportsSuperblock(torch::Device device) {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device.index());
    return major >= 8;
}

// ============================================================================
// ConvOp -- which operation the kernel performs
// ============================================================================

enum class ConvOp { Forward, InputGrad, WeightGrad, TransposedFwd };

// ============================================================================
// SuperblockGeometry -- compile-time spatial parameters
// ============================================================================

template <int T_, int R_, int S_> struct SuperblockGeometry {
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
// CompactEntry -- one active voxel in the compact list
// ============================================================================

struct CompactEntry {
    int32_t scatter_idx;
    uint8_t leaf_in_block;
    uint8_t _pad;
    uint16_t local_voxel;
};

static_assert(sizeof(CompactEntry) == 8, "CompactEntry must be 8 bytes");

// ============================================================================
// MMA configuration per scalar type
// ============================================================================

template <typename Scalar> struct SbMmaConfig;

template <> struct SbMmaConfig<float> {
    using MmaElement                              = cute::tfloat32_t;
    using MmaAtom                                 = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    static constexpr int MMA_K                    = 8;
    static constexpr bool USE_VECTORIZED_PIPELINE = true;

    using SmemLayoutAtom            = decltype(cute::composition(
        cute::Swizzle<1, 2, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::Shape<cute::_4, cute::_2>>,
                                cute::Stride<cute::_4, cute::Stride<cute::_1, cute::_32>>>{}));
    using SmemCopyAtom              = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, MmaElement>;
    using GmemTiledCopyA            = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, MmaElement>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_8, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_4>>{}));
    static constexpr int GMEM_VEC_K = 4;
};

template <> struct SbMmaConfig<c10::Half> {
    using MmaElement                              = cute::half_t;
    using MmaAtom                                 = cute::SM80_16x8x16_F32F16F16F32_TN;
    static constexpr int MMA_K                    = 16;
    static constexpr bool USE_VECTORIZED_PIPELINE = false;
    static constexpr int GMEM_VEC_K               = 1;
};

// ============================================================================
// Tile and pipeline constants
// ============================================================================

static constexpr int SB_TILE_M = 32;
static constexpr int SB_TILE_N = 32;
static constexpr int SB_TILE_K = 32;
static constexpr int SB_STAGES = 3;

// ============================================================================
// Shared memory layout
// ============================================================================

template <typename Geom, int N_LEAVES> struct SuperblockSmem {
    int32_t halo_maps[N_LEAVES][Geom::HaloVol];
    CompactEntry compact_list[N_LEAVES * Geom::LeafVol];
    int compact_count;
    // 3-stage pipeline buffers (A+B) unioned with epilogue C buffer.
    // Sized for the largest MmaElement (tfloat32_t = 4 bytes).
    static constexpr int MAINLOOP_BYTES = (SB_TILE_M + SB_TILE_N) * SB_TILE_K * SB_STAGES * 4;
    static constexpr int EPILOGUE_BYTES = SB_TILE_M * SB_TILE_N * sizeof(float);
    static constexpr int MMA_BUF_BYTES =
        MAINLOOP_BYTES > EPILOGUE_BYTES ? MAINLOOP_BYTES : EPILOGUE_BYTES;
    alignas(16) char mma_buf[MMA_BUF_BYTES];
};

// ============================================================================
// cp.async helpers
// ============================================================================

__device__ __forceinline__ void
sb_cp_async_zfill_4(void *smem_dst, const void *gmem_src, bool pred) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_src), "r"(pred ? 4 : 0)
                 : "memory");
}

__device__ __forceinline__ void
sb_cp_async_zfill_16(void *smem_dst, const void *gmem_src, bool pred) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_src), "r"(pred ? 16 : 0)
                 : "memory");
}

template <typename T>
__device__ __forceinline__ void
sb_store_or_cp_async_4(T *smem_dst, T const *gmem_src, bool pred) {
    if constexpr (sizeof(T) >= 4) {
        sb_cp_async_zfill_4(smem_dst, gmem_src, pred);
    } else {
        *smem_dst = pred ? *gmem_src : T{};
    }
}

// ============================================================================
// Phase B helpers: async tile loading
// ============================================================================

// Load weight tile A into a pipeline stage via 4-byte cp.async (legacy path).
// Weights: [C_fast, K_total] row-major (stride-1 along K_total = K dim).
template <typename MmaElement>
__device__ void
load_a_tile_async(MmaElement const *__restrict__ weights,
                  MmaElement *stage_A,
                  int C_fast,
                  int m0,
                  int k0,
                  int K_total,
                  int tid,
                  int nthreads) {
    constexpr int elems = SB_TILE_M * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int m_local     = i % SB_TILE_M;
        int k_local     = i / SB_TILE_M;
        int m           = m0 + m_local;
        int k           = k0 + k_local;
        bool valid      = (m < C_fast && k < K_total);
        auto const *src = valid ? &weights[static_cast<int64_t>(m) * K_total + k]
                                : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(&stage_A[i], src, valid);
    }
}

// Load B tile for forward / input-grad / transposed-fwd via 4-byte cp.async
// (legacy path). Double indirection: compact_list -> halo_map -> feature addr.
template <typename Geom, bool FlipKernel, typename MmaElement>
__device__ void
load_b_tile_async(int32_t const *__restrict__ halo_maps,
                  CompactEntry const *__restrict__ compact_list,
                  MmaElement const *__restrict__ data,
                  int C_channel,
                  int voxel_start,
                  int k0,
                  int N_active,
                  int K_total,
                  MmaElement *stage_B,
                  int tid,
                  int nthreads) {
    constexpr int elems = SB_TILE_N * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int n_local = i % SB_TILE_N;
        int k_local = i / SB_TILE_N;
        int n       = voxel_start + n_local;
        int k       = k0 + k_local;

        if (n >= N_active || k >= K_total) {
            sb_store_or_cp_async_4(&stage_B[i], reinterpret_cast<MmaElement const *>(0), false);
            continue;
        }

        auto entry = compact_list[n];
        int vi     = entry.local_voxel >> 6;
        int vj     = (entry.local_voxel >> 3) & 7;
        int vk     = entry.local_voxel & 7;

        int kern_pos = k / C_channel;
        int c        = k % C_channel;
        int di       = kern_pos / (Geom::R * Geom::S);
        int dj       = (kern_pos / Geom::S) % Geom::R;
        int dk       = kern_pos % Geom::S;

        if constexpr (FlipKernel) {
            di = Geom::T - 1 - di;
            dj = Geom::R - 1 - dj;
            dk = Geom::S - 1 - dk;
        }

        int halo_idx    = (vi + di) * Geom::Hy * Geom::Hz + (vj + dj) * Geom::Hz + (vk + dk);
        int32_t src     = halo_maps[entry.leaf_in_block * Geom::HaloVol + halo_idx];
        bool active     = (src >= 0);
        auto const *ptr = active ? &data[static_cast<int64_t>(src) * C_channel + c]
                                 : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(&stage_B[i], ptr, active);
    }
}

// Load B tile via 128-bit cp.async (vectorized path for tf32 forward).
// Groups of 4 K-elements per load; smem offsets computed via CuTe swizzled layout.
template <typename Geom, bool FlipKernel, typename MmaElement, typename SmemTensor>
__device__ void
load_b_tile_vec16(int32_t const *__restrict__ halo_maps,
                  CompactEntry const *__restrict__ compact_list,
                  MmaElement const *__restrict__ data,
                  int C_channel,
                  int voxel_start,
                  int k0,
                  int N_active,
                  int K_total,
                  SmemTensor sB,
                  int pipe_stage,
                  int tid,
                  int nthreads) {
    constexpr int VEC    = 4;
    constexpr int groups = SB_TILE_N * (SB_TILE_K / VEC);
    for (int g = tid; g < groups; g += nthreads) {
        int n_local = g / (SB_TILE_K / VEC);
        int k_group = g % (SB_TILE_K / VEC);
        int k_base  = k_group * VEC;
        int n       = voxel_start + n_local;
        int k       = k0 + k_base;

        auto *dst = &sB(n_local, k_base, pipe_stage);

        if (n >= N_active || k >= K_total) {
            sb_cp_async_zfill_16(dst, nullptr, false);
            continue;
        }

        auto entry = compact_list[n];
        int vi     = entry.local_voxel >> 6;
        int vj     = (entry.local_voxel >> 3) & 7;
        int vk     = entry.local_voxel & 7;

        int kern_pos = k / C_channel;
        int c        = k % C_channel;
        int di       = kern_pos / (Geom::R * Geom::S);
        int dj       = (kern_pos / Geom::S) % Geom::R;
        int dk       = kern_pos % Geom::S;

        if constexpr (FlipKernel) {
            di = Geom::T - 1 - di;
            dj = Geom::R - 1 - dj;
            dk = Geom::S - 1 - dk;
        }

        int halo_idx    = (vi + di) * Geom::Hy * Geom::Hz + (vj + dj) * Geom::Hz + (vk + dk);
        int32_t src     = halo_maps[entry.leaf_in_block * Geom::HaloVol + halo_idx];
        bool active     = (src >= 0);
        auto const *ptr = active ? &data[static_cast<int64_t>(src) * C_channel + c]
                                 : reinterpret_cast<MmaElement const *>(0);
        sb_cp_async_zfill_16(dst, ptr, active);
    }
}

// Load A tile for weight gradient via 4-byte cp.async.
// grad_output gathered by scatter_idx. A[TILE_M, TILE_K]: M=C_out, K=N_active.
template <typename MmaElement>
__device__ void
load_a_tile_wgrad_async(MmaElement const *__restrict__ grad_output,
                        CompactEntry const *__restrict__ compact_list,
                        MmaElement *stage_A,
                        int C_out,
                        int m0,
                        int k0,
                        int N_active,
                        int tid,
                        int nthreads) {
    constexpr int elems = SB_TILE_M * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int m_local = i % SB_TILE_M;
        int k_local = i / SB_TILE_M;
        int m       = m0 + m_local;
        int k       = k0 + k_local;
        bool valid  = (m < C_out && k < N_active);
        auto const *src =
            valid ? &grad_output[static_cast<int64_t>(compact_list[k].scatter_idx) * C_out + m]
                  : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(&stage_A[i], src, valid);
    }
}

// Load B tile for weight gradient via 4-byte cp.async.
// sB[n_local, k_local]: n indexes K_total (free), k indexes N_active (reduction).
template <typename Geom, typename MmaElement>
__device__ void
load_b_tile_wgrad_async(int32_t const *__restrict__ halo_maps,
                        CompactEntry const *__restrict__ compact_list,
                        MmaElement const *__restrict__ data,
                        int C_channel,
                        int ktotal_start,
                        int voxel_start,
                        int N_active,
                        int K_total,
                        MmaElement *stage_B,
                        int tid,
                        int nthreads) {
    constexpr int elems = SB_TILE_N * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int n_local = i % SB_TILE_N;
        int k_local = i / SB_TILE_N;

        int ktotal_pos = ktotal_start + n_local;
        int voxel_idx  = voxel_start + k_local;

        if (ktotal_pos >= K_total || voxel_idx >= N_active) {
            sb_store_or_cp_async_4(&stage_B[i], reinterpret_cast<MmaElement const *>(0), false);
            continue;
        }

        auto entry = compact_list[voxel_idx];
        int vi     = entry.local_voxel >> 6;
        int vj     = (entry.local_voxel >> 3) & 7;
        int vk     = entry.local_voxel & 7;

        int kern_pos = ktotal_pos / C_channel;
        int c        = ktotal_pos % C_channel;
        int di       = kern_pos / (Geom::R * Geom::S);
        int dj       = (kern_pos / Geom::S) % Geom::R;
        int dk       = kern_pos % Geom::S;

        int halo_idx    = (vi + di) * Geom::Hy * Geom::Hz + (vj + dj) * Geom::Hz + (vk + dk);
        int32_t src     = halo_maps[entry.leaf_in_block * Geom::HaloVol + halo_idx];
        bool active     = (src >= 0);
        auto const *ptr = active ? &data[static_cast<int64_t>(src) * C_channel + c]
                                 : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(&stage_B[i], ptr, active);
    }
}

// Swizzled-aware wgrad A-tile loader: writes through CuTe smem tensor.
template <typename MmaElement, typename SmemTensor>
__device__ void
load_a_tile_wgrad_swizzled(MmaElement const *__restrict__ grad_output,
                           CompactEntry const *__restrict__ compact_list,
                           SmemTensor sA,
                           int pipe,
                           int C_out,
                           int m0,
                           int k0,
                           int N_active,
                           int tid,
                           int nthreads) {
    constexpr int elems = SB_TILE_M * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int m_local = i % SB_TILE_M;
        int k_local = i / SB_TILE_M;
        int m       = m0 + m_local;
        int k       = k0 + k_local;
        auto *dst   = &sA(m_local, k_local, pipe);
        bool valid  = (m < C_out && k < N_active);
        auto const *src =
            valid ? &grad_output[static_cast<int64_t>(compact_list[k].scatter_idx) * C_out + m]
                  : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(dst, src, valid);
    }
}

// Swizzled-aware wgrad B-tile loader: writes through CuTe smem tensor.
template <typename Geom, typename MmaElement, typename SmemTensor>
__device__ void
load_b_tile_wgrad_swizzled(int32_t const *__restrict__ halo_maps,
                           CompactEntry const *__restrict__ compact_list,
                           MmaElement const *__restrict__ data,
                           int C_channel,
                           int ktotal_start,
                           int voxel_start,
                           int N_active,
                           int K_total,
                           SmemTensor sB,
                           int pipe,
                           int tid,
                           int nthreads) {
    constexpr int elems = SB_TILE_N * SB_TILE_K;
    for (int i = tid; i < elems; i += nthreads) {
        int n_local = i % SB_TILE_N;
        int k_local = i / SB_TILE_N;

        int ktotal_pos = ktotal_start + n_local;
        int voxel_idx  = voxel_start + k_local;
        auto *dst      = &sB(n_local, k_local, pipe);

        if (ktotal_pos >= K_total || voxel_idx >= N_active) {
            sb_store_or_cp_async_4(dst, reinterpret_cast<MmaElement const *>(0), false);
            continue;
        }

        int kern_pos = ktotal_pos / C_channel;
        int c        = ktotal_pos % C_channel;

        auto entry = compact_list[voxel_idx];
        int vi     = entry.local_voxel >> 6;
        int vj     = (entry.local_voxel >> 3) & 7;
        int vk     = entry.local_voxel & 7;

        int di = kern_pos / (Geom::R * Geom::S);
        int dj = (kern_pos / Geom::S) % Geom::R;
        int dk = kern_pos % Geom::S;

        int halo_idx = (vi + di) * Geom::Hy * Geom::Hz + (vj + dj) * Geom::Hz + (vk + dk);
        int32_t src  = halo_maps[entry.leaf_in_block * Geom::HaloVol + halo_idx];

        bool active     = (src >= 0);
        auto const *ptr = active ? &data[static_cast<int64_t>(src) * C_channel + c]
                                 : reinterpret_cast<MmaElement const *>(0);
        sb_store_or_cp_async_4(dst, ptr, active);
    }
}

// ============================================================================
// Unified superblock convolution kernel
// ============================================================================

template <typename Geom, typename Scalar, int N_LEAVES, ConvOp Op>
__global__ void __launch_bounds__(128, 1)
superblock_conv_kernel(GridBatchImpl::Accessor primary_acc,
                       GridBatchImpl::Accessor secondary_acc,
                       Scalar const *__restrict__ data_B,
                       Scalar const *__restrict__ data_A_extra,
                       Scalar const *__restrict__ weights,
                       Scalar *__restrict__ output,
                       int C_fast,
                       int C_slow,
                       int total_primary_leaves) {
    using namespace cute;
    using Cfg        = SbMmaConfig<Scalar>;
    using MmaElement = typename Cfg::MmaElement;

    constexpr int TILE_M = SB_TILE_M;
    constexpr int TILE_N = SB_TILE_N;
    constexpr int TILE_K = SB_TILE_K;
    constexpr int MMA_K  = Cfg::MMA_K;
    constexpr int STAGES = SB_STAGES;

    constexpr bool Flip = (Op == ConvOp::InputGrad || Op == ConvOp::TransposedFwd);

    using TiledMma = TiledMMA<MMA_Atom<typename Cfg::MmaAtom>,
                              Layout<Shape<_2, _2, _1>>,
                              Tile<Int<TILE_M>, Int<TILE_N>, Int<MMA_K>>>;

    extern __shared__ char smem_raw[];
    auto &smem = *reinterpret_cast<SuperblockSmem<Geom, N_LEAVES> *>(smem_raw);

    int const tid      = threadIdx.x;
    int const nthreads = blockDim.x;
    int const sb_id    = blockIdx.x;

    int const first_leaf = sb_id * N_LEAVES;
    int const n_leaves   = (first_leaf + N_LEAVES <= total_primary_leaves)
                               ? N_LEAVES
                               : (total_primary_leaves - first_leaf);

    if (n_leaves <= 0)
        return;

    auto const *pri_grid = primary_acc.grid(0);
    auto const *sec_grid = secondary_acc.grid(0);
    int32_t const pri_vo = static_cast<int32_t>(primary_acc.voxelOffset(0));
    int32_t const sec_vo = static_cast<int32_t>(secondary_acc.voxelOffset(0));

    // ================================================================
    // Phase A: Topology build + compaction
    // ================================================================

    if (tid == 0) {
        smem.compact_count = 0;
    }
    __syncthreads();

    auto sec_tree_acc = sec_grid->getAccessor();

    for (int li = 0; li < n_leaves; ++li) {
        auto const &leaf         = pri_grid->tree().template getFirstNode<0>()[first_leaf + li];
        nanovdb::Coord origin    = leaf.origin();
        nanovdb::Coord halo_base = origin.offsetBy(Geom::Dx, Geom::Dy, Geom::Dz);

        for (int h = tid; h < Geom::HaloVol; h += nthreads) {
            int hi                = h / (Geom::Hy * Geom::Hz);
            int hj                = (h / Geom::Hz) % Geom::Hy;
            int hk                = h % Geom::Hz;
            nanovdb::Coord coord  = halo_base.offsetBy(hi, hj, hk);
            auto val              = sec_tree_acc.getValue(coord);
            smem.halo_maps[li][h] = (val > 0) ? static_cast<int32_t>(val - 1) + sec_vo : -1;
        }
        __syncthreads();

        for (int v = tid; v < Geom::LeafVol; v += nthreads) {
            if (leaf.isActive(v)) {
                int pos = atomicAdd(&smem.compact_count, 1);
                CompactEntry entry;
                entry.scatter_idx      = static_cast<int32_t>(leaf.getValue(v) - 1) + pri_vo;
                entry.leaf_in_block    = static_cast<uint8_t>(li);
                entry._pad             = 0;
                entry.local_voxel      = static_cast<uint16_t>(v);
                smem.compact_list[pos] = entry;
            }
        }
        __syncthreads();
    }

    int const N_active = smem.compact_count;
    if (N_active == 0)
        return;

    // ================================================================
    // Phase B: Pipelined compacted dense GEMM
    // ================================================================

    int const K_total = C_slow * Geom::KernVol;

    auto const *data_B_mma  = reinterpret_cast<MmaElement const *>(data_B);
    auto const *weights_mma = reinterpret_cast<MmaElement const *>(weights);
    auto const *extra_mma   = reinterpret_cast<MmaElement const *>(data_A_extra);

    constexpr int STAGE_A_ELEMS = TILE_M * TILE_K;
    constexpr int STAGE_B_ELEMS = TILE_N * TILE_K;
    auto *smem_A_base           = reinterpret_cast<MmaElement *>(smem.mma_buf);
    auto *smem_B_base           = smem_A_base + STAGE_A_ELEMS * STAGES;

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    if constexpr (Cfg::USE_VECTORIZED_PIPELINE) {
        // ==============================================================
        // Vectorized path: swizzled smem + 128-bit cp.async + ldmatrix
        // ==============================================================

        using SmemLayoutA =
            decltype(tile_to_shape(typename Cfg::SmemLayoutAtom{},
                                   make_shape(Int<TILE_M>{}, Int<TILE_K>{}, Int<STAGES>{})));
        using SmemLayoutB =
            decltype(tile_to_shape(typename Cfg::SmemLayoutAtom{},
                                   make_shape(Int<TILE_N>{}, Int<TILE_K>{}, Int<STAGES>{})));

        Tensor sA = make_tensor(make_smem_ptr(smem_A_base), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(smem_B_base), SmemLayoutB{});

        // -- A tile: CuTe gmem -> smem copy (128-bit cp.async) --
        typename Cfg::GmemTiledCopyA gmem_tiled_copy_A;
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(tid);
        auto tAsA            = gmem_thr_copy_A.partition_D(sA);

        // -- Smem -> register: ldmatrix --
        auto smem_tiled_copy_A = make_tiled_copy_A(typename Cfg::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(tid);
        auto tCsA              = smem_thr_copy_A.partition_S(sA);
        auto tCrA              = thr_mma.partition_fragment_A(sA(_, _, 0));
        auto tCrA_copy_view    = smem_thr_copy_A.retile_D(tCrA);

        auto smem_tiled_copy_B = make_tiled_copy_B(typename Cfg::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(tid);
        auto tCsB              = smem_thr_copy_B.partition_S(sB);
        auto tCrB              = thr_mma.partition_fragment_B(sB(_, _, 0));
        auto tCrB_copy_view    = smem_thr_copy_B.retile_D(tCrB);

        auto K_BLOCK_MAX = size<2>(tCrA);

        if constexpr (Op == ConvOp::Forward || Op == ConvOp::InputGrad ||
                      Op == ConvOp::TransposedFwd) {
            for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
                for (int n0 = 0; n0 < N_active; n0 += TILE_N) {
                    auto accum =
                        partition_fragment_C(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
                    clear(accum);

                    int const k_tile_total = (K_total + TILE_K - 1) / TILE_K;
                    int k_tile_count       = k_tile_total;
                    int k_iter             = 0;

                    // -- Prefetch Stages-1 stages --
                    int const pf = (STAGES - 1 < k_tile_count) ? STAGES - 1 : k_tile_count;
                    for (int pipe = 0; pipe < pf; ++pipe) {
                        int k0 = k_iter * TILE_K;
                        // A tile via CuTe copy
                        auto gA = make_tensor(
                            make_gmem_ptr(weights_mma + static_cast<int64_t>(m0) * K_total + k0),
                            make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                        make_stride(K_total, Int<1>{})));
                        auto tAgA = gmem_thr_copy_A.partition_S(gA);
                        copy(gmem_tiled_copy_A, tAgA, tAsA(_, _, _, pipe));
                        // B tile via manual 128-bit cp.async
                        load_b_tile_vec16<Geom, Flip, MmaElement>(&smem.halo_maps[0][0],
                                                                  smem.compact_list,
                                                                  data_B_mma,
                                                                  C_slow,
                                                                  n0,
                                                                  k0,
                                                                  N_active,
                                                                  K_total,
                                                                  sB,
                                                                  pipe,
                                                                  tid,
                                                                  nthreads);
                        cp_async_fence();
                        ++k_iter;
                        --k_tile_count;
                    }

                    int smem_pipe_read  = 0;
                    int smem_pipe_write = STAGES - 1;

                    // Wait for first stage, prefetch k_block 0
                    cp_async_wait<STAGES - 2>();
                    __syncthreads();

                    Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
                    Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

                    if (K_BLOCK_MAX > 1) {
                        copy(smem_tiled_copy_A,
                             tCsA_p(_, _, Int<0>{}),
                             tCrA_copy_view(_, _, Int<0>{}));
                        copy(smem_tiled_copy_B,
                             tCsB_p(_, _, Int<0>{}),
                             tCrB_copy_view(_, _, Int<0>{}));
                    }

                    // -- Pipelined main loop --
                    CUTLASS_PRAGMA_NO_UNROLL
                    while (k_tile_count > -(STAGES - 1)) {
                        for_each(
                            make_int_sequence<decltype(K_BLOCK_MAX)::value>{}, [&](auto k_block) {
                                if (k_block == K_BLOCK_MAX - 1) {
                                    tCsA_p = tCsA(_, _, _, smem_pipe_read);
                                    tCsB_p = tCsB(_, _, _, smem_pipe_read);
                                    cp_async_wait<STAGES - 2>();
                                    __syncthreads();
                                }

                                auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                                copy(smem_tiled_copy_A,
                                     tCsA_p(_, _, k_block_next),
                                     tCrA_copy_view(_, _, k_block_next));
                                copy(smem_tiled_copy_B,
                                     tCsB_p(_, _, k_block_next),
                                     tCrB_copy_view(_, _, k_block_next));

                                if (k_block == 0) {
                                    if (k_tile_count > 0) {
                                        int k0  = k_iter * TILE_K;
                                        auto gA = make_tensor(
                                            make_gmem_ptr(weights_mma +
                                                          static_cast<int64_t>(m0) * K_total + k0),
                                            make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                                        make_stride(K_total, Int<1>{})));
                                        copy(gmem_tiled_copy_A,
                                             gmem_thr_copy_A.partition_S(gA),
                                             tAsA(_, _, _, smem_pipe_write));
                                        load_b_tile_vec16<Geom, Flip, MmaElement>(
                                            &smem.halo_maps[0][0],
                                            smem.compact_list,
                                            data_B_mma,
                                            C_slow,
                                            n0,
                                            k0,
                                            N_active,
                                            K_total,
                                            sB,
                                            smem_pipe_write,
                                            tid,
                                            nthreads);
                                    }
                                    cp_async_fence();
                                    --k_tile_count;
                                    if (k_tile_count > 0)
                                        ++k_iter;

                                    smem_pipe_write = smem_pipe_read;
                                    ++smem_pipe_read;
                                    smem_pipe_read =
                                        (smem_pipe_read == STAGES) ? 0 : smem_pipe_read;
                                }

                                cute::transform(tCrA(_, _, k_block), identity{});
                                cute::transform(tCrB(_, _, k_block), identity{});
                                gemm(tiled_mma,
                                     accum,
                                     tCrA(_, _, k_block),
                                     tCrB(_, _, k_block),
                                     accum);
                            });
                    }

                    cp_async_wait<0>();
                    __syncthreads();

                    // Epilogue: accumulator -> smem -> gmem (scatter)
                    auto *smem_C = reinterpret_cast<float *>(smem.mma_buf);
                    auto sC      = make_tensor(make_smem_ptr(smem_C),
                                          make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                                      make_stride(Int<1>{}, Int<TILE_M>{})));

                    auto smem_copy_c =
                        make_tiled_copy_C(Copy_Atom<UniversalCopy<uint32_t>, float>{}, tiled_mma);
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
                        if (cout < C_fast && voxel < N_active) {
                            int32_t s_idx = smem.compact_list[voxel].scatter_idx;
                            output[static_cast<int64_t>(s_idx) * C_fast + cout] =
                                static_cast<Scalar>(smem_C[i]);
                        }
                    }

                    __syncthreads();
                }
            }
        } else if constexpr (Op == ConvOp::WeightGrad) {
            for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
                for (int n0 = 0; n0 < K_total; n0 += TILE_N) {
                    auto accum =
                        partition_fragment_C(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
                    clear(accum);

                    int const k_tile_total = (N_active + TILE_K - 1) / TILE_K;
                    int k_tile_count       = k_tile_total;
                    int k_iter             = 0;

                    int const pf = (STAGES - 1 < k_tile_count) ? STAGES - 1 : k_tile_count;
                    for (int pipe = 0; pipe < pf; ++pipe) {
                        int k0 = k_iter * TILE_K;
                        load_a_tile_wgrad_swizzled<MmaElement>(extra_mma,
                                                               smem.compact_list,
                                                               sA,
                                                               pipe,
                                                               C_fast,
                                                               m0,
                                                               k0,
                                                               N_active,
                                                               tid,
                                                               nthreads);
                        load_b_tile_wgrad_swizzled<Geom, MmaElement>(&smem.halo_maps[0][0],
                                                                     smem.compact_list,
                                                                     data_B_mma,
                                                                     C_slow,
                                                                     n0,
                                                                     k0,
                                                                     N_active,
                                                                     K_total,
                                                                     sB,
                                                                     pipe,
                                                                     tid,
                                                                     nthreads);
                        cute::cp_async_fence();
                        ++k_iter;
                        --k_tile_count;
                    }

                    int smem_pipe_read  = 0;
                    int smem_pipe_write = STAGES - 1;

                    cp_async_wait<STAGES - 2>();
                    __syncthreads();

                    Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
                    Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

                    if (K_BLOCK_MAX > 1) {
                        copy(smem_tiled_copy_A,
                             tCsA_p(_, _, Int<0>{}),
                             tCrA_copy_view(_, _, Int<0>{}));
                        copy(smem_tiled_copy_B,
                             tCsB_p(_, _, Int<0>{}),
                             tCrB_copy_view(_, _, Int<0>{}));
                    }

                    CUTLASS_PRAGMA_NO_UNROLL
                    while (k_tile_count > -(STAGES - 1)) {
                        for_each(
                            make_int_sequence<decltype(K_BLOCK_MAX)::value>{}, [&](auto k_block) {
                                if (k_block == K_BLOCK_MAX - 1) {
                                    tCsA_p = tCsA(_, _, _, smem_pipe_read);
                                    tCsB_p = tCsB(_, _, _, smem_pipe_read);
                                    cp_async_wait<STAGES - 2>();
                                    __syncthreads();
                                }

                                auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                                copy(smem_tiled_copy_A,
                                     tCsA_p(_, _, k_block_next),
                                     tCrA_copy_view(_, _, k_block_next));
                                copy(smem_tiled_copy_B,
                                     tCsB_p(_, _, k_block_next),
                                     tCrB_copy_view(_, _, k_block_next));

                                if (k_block == 0) {
                                    if (k_tile_count > 0) {
                                        int k0 = k_iter * TILE_K;
                                        load_a_tile_wgrad_swizzled<MmaElement>(extra_mma,
                                                                               smem.compact_list,
                                                                               sA,
                                                                               smem_pipe_write,
                                                                               C_fast,
                                                                               m0,
                                                                               k0,
                                                                               N_active,
                                                                               tid,
                                                                               nthreads);
                                        load_b_tile_wgrad_swizzled<Geom, MmaElement>(
                                            &smem.halo_maps[0][0],
                                            smem.compact_list,
                                            data_B_mma,
                                            C_slow,
                                            n0,
                                            k0,
                                            N_active,
                                            K_total,
                                            sB,
                                            smem_pipe_write,
                                            tid,
                                            nthreads);
                                    }
                                    cp_async_fence();
                                    --k_tile_count;
                                    if (k_tile_count > 0)
                                        ++k_iter;

                                    smem_pipe_write = smem_pipe_read;
                                    ++smem_pipe_read;
                                    smem_pipe_read =
                                        (smem_pipe_read == STAGES) ? 0 : smem_pipe_read;
                                }

                                cute::transform(tCrA(_, _, k_block), identity{});
                                cute::transform(tCrB(_, _, k_block), identity{});
                                gemm(tiled_mma,
                                     accum,
                                     tCrA(_, _, k_block),
                                     tCrB(_, _, k_block),
                                     accum);
                            });
                    }

                    cp_async_wait<0>();
                    __syncthreads();

                    auto *smem_C = reinterpret_cast<float *>(smem.mma_buf);
                    auto sC      = make_tensor(make_smem_ptr(smem_C),
                                          make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                                      make_stride(Int<1>{}, Int<TILE_M>{})));

                    auto smem_copy_c =
                        make_tiled_copy_C(Copy_Atom<UniversalCopy<uint32_t>, float>{}, tiled_mma);
                    auto smem_thr_copy_c = smem_copy_c.get_thread_slice(tid);
                    auto tCrC_copy       = smem_thr_copy_c.retile_S(accum);
                    auto tCsC            = smem_thr_copy_c.partition_D(sC);
                    copy(smem_copy_c, tCrC_copy, tCsC);

                    __syncthreads();

                    auto *output_f = reinterpret_cast<float *>(output);
                    for (int i = tid; i < TILE_M * TILE_N; i += nthreads) {
                        int m_local = i % TILE_M;
                        int n_local = i / TILE_M;
                        int cout    = m0 + m_local;
                        int kpos    = n0 + n_local;
                        if (cout < C_fast && kpos < K_total) {
                            atomicAdd(&output_f[static_cast<int64_t>(cout) * K_total + kpos],
                                      smem_C[i]);
                        }
                    }

                    __syncthreads();
                }
            }
        }
    } else {
        // ==============================================================
        // Legacy path (fp16): 4-byte cp.async, simple smem, elem copy
        // ==============================================================
        auto get_A_stage = [&](int s) -> MmaElement * { return smem_A_base + s * STAGE_A_ELEMS; };
        auto get_B_stage = [&](int s) -> MmaElement * { return smem_B_base + s * STAGE_B_ELEMS; };
        auto make_sA     = [](MmaElement *ptr) {
            return make_tensor(make_smem_ptr(ptr),
                               make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                           make_stride(Int<1>{}, Int<TILE_M>{})));
        };
        auto make_sB = [](MmaElement *ptr) {
            return make_tensor(make_smem_ptr(ptr),
                               make_layout(make_shape(Int<TILE_N>{}, Int<TILE_K>{}),
                                           make_stride(Int<1>{}, Int<TILE_N>{})));
        };

        auto tCrA                   = thr_mma.partition_fragment_A(make_sA(get_A_stage(0)));
        auto tCrB                   = thr_mma.partition_fragment_B(make_sB(get_B_stage(0)));
        constexpr int K_BLOCK_MAX_L = TILE_K / MMA_K;

        auto copy_smem_to_reg_kb = [&](int stage, int kb) {
            auto tCsA_s = thr_mma.partition_A(make_sA(get_A_stage(stage)));
            auto tCsB_s = thr_mma.partition_B(make_sB(get_B_stage(stage)));
#pragma unroll
            for (int j = 0; j < size(tCrA(_, _, 0)); ++j)
                tCrA(_, _, kb)(j) = tCsA_s(_, _, kb)(j);
#pragma unroll
            for (int j = 0; j < size(tCrB(_, _, 0)); ++j)
                tCrB(_, _, kb)(j) = tCsB_s(_, _, kb)(j);
        };

        if constexpr (Op == ConvOp::Forward || Op == ConvOp::InputGrad ||
                      Op == ConvOp::TransposedFwd) {
            for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
                for (int n0 = 0; n0 < N_active; n0 += TILE_N) {
                    auto accum =
                        partition_fragment_C(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
                    clear(accum);

                    int const k_tile_total = (K_total + TILE_K - 1) / TILE_K;
                    int k_tile_count       = k_tile_total;
                    int k_iter             = 0;

                    int const pf = (STAGES - 1 < k_tile_count) ? STAGES - 1 : k_tile_count;
                    for (int pipe = 0; pipe < pf; ++pipe) {
                        int k0 = k_iter * TILE_K;
                        load_a_tile_async<MmaElement>(
                            weights_mma, get_A_stage(pipe), C_fast, m0, k0, K_total, tid, nthreads);
                        load_b_tile_async<Geom, Flip, MmaElement>(&smem.halo_maps[0][0],
                                                                  smem.compact_list,
                                                                  data_B_mma,
                                                                  C_slow,
                                                                  n0,
                                                                  k0,
                                                                  N_active,
                                                                  K_total,
                                                                  get_B_stage(pipe),
                                                                  tid,
                                                                  nthreads);
                        cute::cp_async_fence();
                        ++k_iter;
                        --k_tile_count;
                    }

                    int smem_pipe_read     = 0;
                    int smem_pipe_write    = STAGES - 1;
                    int current_read_stage = 0;

                    cute::cp_async_wait<STAGES - 2>();
                    __syncthreads();
                    copy_smem_to_reg_kb(current_read_stage, 0);

                    while (k_tile_count > -(STAGES - 1)) {
#pragma unroll
                        for (int kb = 0; kb < K_BLOCK_MAX_L; ++kb) {
                            if (kb == K_BLOCK_MAX_L - 1) {
                                current_read_stage = smem_pipe_read;
                                cute::cp_async_wait<STAGES - 2>();
                                __syncthreads();
                            }
                            int kb_next = (kb + 1) % K_BLOCK_MAX_L;
                            copy_smem_to_reg_kb(current_read_stage, kb_next);
                            if (kb == 0) {
                                if (k_tile_count > 0) {
                                    int k0 = k_iter * TILE_K;
                                    load_a_tile_async<MmaElement>(weights_mma,
                                                                  get_A_stage(smem_pipe_write),
                                                                  C_fast,
                                                                  m0,
                                                                  k0,
                                                                  K_total,
                                                                  tid,
                                                                  nthreads);
                                    load_b_tile_async<Geom, Flip, MmaElement>(
                                        &smem.halo_maps[0][0],
                                        smem.compact_list,
                                        data_B_mma,
                                        C_slow,
                                        n0,
                                        k0,
                                        N_active,
                                        K_total,
                                        get_B_stage(smem_pipe_write),
                                        tid,
                                        nthreads);
                                }
                                cute::cp_async_fence();
                                --k_tile_count;
                                if (k_tile_count > 0)
                                    ++k_iter;
                                smem_pipe_write = smem_pipe_read;
                                ++smem_pipe_read;
                                if (smem_pipe_read == STAGES)
                                    smem_pipe_read = 0;
                            }
                            gemm(tiled_mma, accum, tCrA(_, _, kb), tCrB(_, _, kb), accum);
                        }
                    }

                    cute::cp_async_wait<0>();
                    __syncthreads();

                    auto *smem_C = reinterpret_cast<float *>(smem.mma_buf);
                    auto sC      = make_tensor(make_smem_ptr(smem_C),
                                          make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                                      make_stride(Int<1>{}, Int<TILE_M>{})));
                    auto smem_copy_c =
                        make_tiled_copy_C(Copy_Atom<UniversalCopy<uint32_t>, float>{}, tiled_mma);
                    auto smem_thr_copy_c = smem_copy_c.get_thread_slice(tid);
                    copy(smem_copy_c,
                         smem_thr_copy_c.retile_S(accum),
                         smem_thr_copy_c.partition_D(sC));
                    __syncthreads();

                    for (int i = tid; i < TILE_M * TILE_N; i += nthreads) {
                        int m_local = i % TILE_M;
                        int n_local = i / TILE_M;
                        int cout    = m0 + m_local;
                        int voxel   = n0 + n_local;
                        if (cout < C_fast && voxel < N_active) {
                            int32_t s_idx = smem.compact_list[voxel].scatter_idx;
                            output[static_cast<int64_t>(s_idx) * C_fast + cout] =
                                static_cast<Scalar>(smem_C[i]);
                        }
                    }
                    __syncthreads();
                }
            }
        } else if constexpr (Op == ConvOp::WeightGrad) {
            for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
                for (int n0 = 0; n0 < K_total; n0 += TILE_N) {
                    auto accum =
                        partition_fragment_C(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
                    clear(accum);

                    int const k_tile_total = (N_active + TILE_K - 1) / TILE_K;
                    int k_tile_count       = k_tile_total;
                    int k_iter             = 0;

                    int const pf = (STAGES - 1 < k_tile_count) ? STAGES - 1 : k_tile_count;
                    for (int pipe = 0; pipe < pf; ++pipe) {
                        int k0 = k_iter * TILE_K;
                        load_a_tile_wgrad_async<MmaElement>(extra_mma,
                                                            smem.compact_list,
                                                            get_A_stage(pipe),
                                                            C_fast,
                                                            m0,
                                                            k0,
                                                            N_active,
                                                            tid,
                                                            nthreads);
                        load_b_tile_wgrad_async<Geom, MmaElement>(&smem.halo_maps[0][0],
                                                                  smem.compact_list,
                                                                  data_B_mma,
                                                                  C_slow,
                                                                  n0,
                                                                  k0,
                                                                  N_active,
                                                                  K_total,
                                                                  get_B_stage(pipe),
                                                                  tid,
                                                                  nthreads);
                        cute::cp_async_fence();
                        ++k_iter;
                        --k_tile_count;
                    }

                    int smem_pipe_read     = 0;
                    int smem_pipe_write    = STAGES - 1;
                    int current_read_stage = 0;

                    cute::cp_async_wait<STAGES - 2>();
                    __syncthreads();
                    copy_smem_to_reg_kb(current_read_stage, 0);

                    while (k_tile_count > -(STAGES - 1)) {
#pragma unroll
                        for (int kb = 0; kb < K_BLOCK_MAX_L; ++kb) {
                            if (kb == K_BLOCK_MAX_L - 1) {
                                current_read_stage = smem_pipe_read;
                                cute::cp_async_wait<STAGES - 2>();
                                __syncthreads();
                            }
                            int kb_next = (kb + 1) % K_BLOCK_MAX_L;
                            copy_smem_to_reg_kb(current_read_stage, kb_next);
                            if (kb == 0) {
                                if (k_tile_count > 0) {
                                    int k0 = k_iter * TILE_K;
                                    load_a_tile_wgrad_async<MmaElement>(
                                        extra_mma,
                                        smem.compact_list,
                                        get_A_stage(smem_pipe_write),
                                        C_fast,
                                        m0,
                                        k0,
                                        N_active,
                                        tid,
                                        nthreads);
                                    load_b_tile_wgrad_async<Geom, MmaElement>(
                                        &smem.halo_maps[0][0],
                                        smem.compact_list,
                                        data_B_mma,
                                        C_slow,
                                        n0,
                                        k0,
                                        N_active,
                                        K_total,
                                        get_B_stage(smem_pipe_write),
                                        tid,
                                        nthreads);
                                }
                                cute::cp_async_fence();
                                --k_tile_count;
                                if (k_tile_count > 0)
                                    ++k_iter;
                                smem_pipe_write = smem_pipe_read;
                                ++smem_pipe_read;
                                if (smem_pipe_read == STAGES)
                                    smem_pipe_read = 0;
                            }
                            gemm(tiled_mma, accum, tCrA(_, _, kb), tCrB(_, _, kb), accum);
                        }
                    }

                    cute::cp_async_wait<0>();
                    __syncthreads();

                    auto *smem_C = reinterpret_cast<float *>(smem.mma_buf);
                    auto sC      = make_tensor(make_smem_ptr(smem_C),
                                          make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                                      make_stride(Int<1>{}, Int<TILE_M>{})));
                    auto smem_copy_c =
                        make_tiled_copy_C(Copy_Atom<UniversalCopy<uint32_t>, float>{}, tiled_mma);
                    auto smem_thr_copy_c = smem_copy_c.get_thread_slice(tid);
                    copy(smem_copy_c,
                         smem_thr_copy_c.retile_S(accum),
                         smem_thr_copy_c.partition_D(sC));
                    __syncthreads();

                    auto *output_f = reinterpret_cast<float *>(output);
                    for (int i = tid; i < TILE_M * TILE_N; i += nthreads) {
                        int m_local = i % TILE_M;
                        int n_local = i / TILE_M;
                        int cout    = m0 + m_local;
                        int kpos    = n0 + n_local;
                        if (cout < C_fast && kpos < K_total) {
                            atomicAdd(&output_f[static_cast<int64_t>(cout) * K_total + kpos],
                                      smem_C[i]);
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}

// ============================================================================
// Typed launcher
// ============================================================================

template <typename Geom, typename Scalar, int N_LEAVES, ConvOp Op>
static void
launchSuperblockConv(GridBatchImpl const &primary_grid,
                     GridBatchImpl const &secondary_grid,
                     torch::Tensor data_B,
                     torch::Tensor data_A_extra,
                     torch::Tensor weights_permuted,
                     torch::Tensor output,
                     int C_fast,
                     int C_slow,
                     cudaStream_t stream) {
    auto pri_acc = primary_grid.deviceAccessor();
    auto sec_acc = secondary_grid.deviceAccessor();

    int64_t const num_leaves = primary_grid.totalLeaves();
    if (num_leaves == 0)
        return;

    int const num_blocks = static_cast<int>((num_leaves + N_LEAVES - 1) / N_LEAVES);

    constexpr int THREADS   = 128;
    constexpr size_t SMEM   = sizeof(SuperblockSmem<Geom, N_LEAVES>);
    constexpr size_t SMEM48 = 48u * 1024u;

    if constexpr (SMEM > SMEM48) {
        cudaFuncSetAttribute(superblock_conv_kernel<Geom, Scalar, N_LEAVES, Op>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(SMEM));
    }

    Scalar const *data_B_ptr = data_B.data_ptr<Scalar>();
    Scalar const *extra_ptr  = data_A_extra.defined() ? data_A_extra.data_ptr<Scalar>() : nullptr;
    Scalar const *weights_ptr =
        weights_permuted.defined() ? weights_permuted.data_ptr<Scalar>() : nullptr;

    // WeightGrad output is always float32 (for atomicAdd), so we reinterpret
    // it through the Scalar* parameter -- the kernel's WeightGrad epilogue
    // immediately casts back to float* and never dereferences it as Scalar*.
    Scalar *output_ptr;
    if constexpr (Op == ConvOp::WeightGrad) {
        output_ptr = reinterpret_cast<Scalar *>(output.data_ptr<float>());
    } else {
        output_ptr = output.data_ptr<Scalar>();
    }

    superblock_conv_kernel<Geom, Scalar, N_LEAVES, Op>
        <<<num_blocks, THREADS, SMEM, stream>>>(pri_acc,
                                                sec_acc,
                                                data_B_ptr,
                                                extra_ptr,
                                                weights_ptr,
                                                output_ptr,
                                                C_fast,
                                                C_slow,
                                                static_cast<int>(num_leaves));
}

// ============================================================================
// Per-operation launcher wrappers (handle weight permutation)
// ============================================================================

template <typename Geom, typename Scalar, int N_LEAVES>
static void
launchForward(GridBatchImpl const &feature_grid,
              GridBatchImpl const &output_grid,
              torch::Tensor features,
              torch::Tensor weights,
              torch::Tensor output,
              int C_in,
              int C_out,
              cudaStream_t stream) {
    // [C_out, C_in, k0, k1, k2] -> [C_out, KernVol*C_in] row-major
    // (stride-1 along K_total for 128-bit cp.async vectorization)
    auto W     = weights.permute({0, 2, 3, 4, 1}).contiguous().view({C_out, Geom::KernVol * C_in});
    auto dummy = torch::Tensor();
    launchSuperblockConv<Geom, Scalar, N_LEAVES, ConvOp::Forward>(
        output_grid, feature_grid, features, dummy, W, output, C_out, C_in, stream);
}

template <typename Geom, typename Scalar, int N_LEAVES>
static void
launchInputGrad(GridBatchImpl const &feature_grid,
                GridBatchImpl const &output_grid,
                torch::Tensor grad_output,
                torch::Tensor weights,
                torch::Tensor grad_features,
                int C_in,
                int C_out,
                cudaStream_t stream) {
    // [C_out, C_in, k0, k1, k2] -> [C_in, KernVol*C_out] row-major
    // (stride-1 along K_total for 128-bit cp.async vectorization)
    auto W     = weights.permute({1, 2, 3, 4, 0}).contiguous().view({C_in, Geom::KernVol * C_out});
    auto dummy = torch::Tensor();
    launchSuperblockConv<Geom, Scalar, N_LEAVES, ConvOp::InputGrad>(
        feature_grid, output_grid, grad_output, dummy, W, grad_features, C_in, C_out, stream);
}

template <typename Geom, typename Scalar, int N_LEAVES>
static void
launchWeightGrad(GridBatchImpl const &feature_grid,
                 GridBatchImpl const &output_grid,
                 torch::Tensor features,
                 torch::Tensor grad_output,
                 torch::Tensor grad_weights_flat,
                 int C_in,
                 int C_out,
                 cudaStream_t stream) {
    auto dummy = torch::Tensor();
    // grad_weights_flat is [C_out, K_total] = [C_out, C_in * KernVol]
    launchSuperblockConv<Geom, Scalar, N_LEAVES, ConvOp::WeightGrad>(output_grid,
                                                                     feature_grid,
                                                                     features,
                                                                     grad_output,
                                                                     dummy,
                                                                     grad_weights_flat,
                                                                     C_out,
                                                                     C_in,
                                                                     stream);
}

template <typename Geom, typename Scalar, int N_LEAVES>
static void
launchTransposedFwd(GridBatchImpl const &source_grid,
                    GridBatchImpl const &target_grid,
                    torch::Tensor features,
                    torch::Tensor weights,
                    torch::Tensor output,
                    int C_in,
                    int C_out,
                    cudaStream_t stream) {
    // Same weight layout as forward: [C_out, KernVol*C_in] row-major, but
    // with flipped kernel (handled by the FlipKernel template flag).
    auto W     = weights.permute({0, 2, 3, 4, 1}).contiguous().view({C_out, Geom::KernVol * C_in});
    auto dummy = torch::Tensor();
    launchSuperblockConv<Geom, Scalar, N_LEAVES, ConvOp::TransposedFwd>(
        target_grid, source_grid, features, dummy, W, output, C_out, C_in, stream);
}

// ============================================================================
// Kernel-size + N_LEAVES dispatch
// ============================================================================

template <typename Scalar>
static torch::Tensor
superblockConvDispatched(torch::Tensor features,
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
        launchForward<SuperblockGeometry<3, 3, 3>, Scalar, 4>(
            feature_grid, output_grid, features, weights, output, C_in, C_out, stream);
    } else if (kernel_size == nanovdb::Coord(5, 5, 5)) {
        launchForward<SuperblockGeometry<5, 5, 5>, Scalar, 4>(
            feature_grid, output_grid, features, weights, output, C_in, C_out, stream);
    } else if (kernel_size == nanovdb::Coord(7, 7, 7)) {
        launchForward<SuperblockGeometry<7, 7, 7>, Scalar, 2>(
            feature_grid, output_grid, features, weights, output, C_in, C_out, stream);
    } else {
        TORCH_CHECK(false,
                    "superblockConv: unsupported kernel size (",
                    kernel_size[0],
                    ",",
                    kernel_size[1],
                    ",",
                    kernel_size[2],
                    "). Supported: 3x3x3, 5x5x5, 7x7x7.");
    }
    return output;
}

template <typename Scalar>
static std::tuple<torch::Tensor, torch::Tensor>
superblockConvBackwardDispatched(torch::Tensor grad_output,
                                 torch::Tensor features,
                                 torch::Tensor weights,
                                 GridBatchImpl const &feature_grid,
                                 GridBatchImpl const &output_grid,
                                 nanovdb::Coord kernel_size,
                                 int C_in,
                                 int C_out) {
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const NF   = feature_grid.totalVoxels();
    auto grad_features = torch::zeros({NF, C_in}, features.options());

    int K_vol   = kernel_size[0] * kernel_size[1] * kernel_size[2];
    int K_total = C_in * K_vol;
    auto grad_weights_flat =
        torch::zeros({static_cast<int64_t>(C_out), static_cast<int64_t>(K_total)},
                     features.options().dtype(torch::kFloat32));

    auto launchIGrad = [&](auto geom_tag, auto n_leaves_tag) {
        using Geom             = decltype(geom_tag);
        constexpr int N_LEAVES = decltype(n_leaves_tag)::value;
        launchInputGrad<Geom, Scalar, N_LEAVES>(
            feature_grid, output_grid, grad_output, weights, grad_features, C_in, C_out, stream);
    };

    auto launchWGrad = [&](auto geom_tag, auto n_leaves_tag) {
        using Geom             = decltype(geom_tag);
        constexpr int N_LEAVES = decltype(n_leaves_tag)::value;
        launchWeightGrad<Geom, Scalar, N_LEAVES>(feature_grid,
                                                 output_grid,
                                                 features,
                                                 grad_output,
                                                 grad_weights_flat,
                                                 C_in,
                                                 C_out,
                                                 stream);
    };

    if (kernel_size == nanovdb::Coord(3, 3, 3)) {
        launchIGrad(SuperblockGeometry<3, 3, 3>{}, std::integral_constant<int, 4>{});
        launchWGrad(SuperblockGeometry<3, 3, 3>{}, std::integral_constant<int, 4>{});
    } else if (kernel_size == nanovdb::Coord(5, 5, 5)) {
        launchIGrad(SuperblockGeometry<5, 5, 5>{}, std::integral_constant<int, 4>{});
        launchWGrad(SuperblockGeometry<5, 5, 5>{}, std::integral_constant<int, 4>{});
    } else if (kernel_size == nanovdb::Coord(7, 7, 7)) {
        launchIGrad(SuperblockGeometry<7, 7, 7>{}, std::integral_constant<int, 2>{});
        launchWGrad(SuperblockGeometry<7, 7, 7>{}, std::integral_constant<int, 2>{});
    } else {
        TORCH_CHECK(false,
                    "superblockConvBackward: unsupported kernel size (",
                    kernel_size[0],
                    ",",
                    kernel_size[1],
                    ",",
                    kernel_size[2],
                    "). Supported: 3x3x3, 5x5x5, 7x7x7.");
    }

    // Reshape flat grad_weights [C_out, K_total] -> [C_out, C_in, k0, k1, k2]
    auto grad_weights =
        grad_weights_flat.view({C_out, kernel_size[0], kernel_size[1], kernel_size[2], C_in})
            .permute({0, 4, 1, 2, 3})
            .contiguous();

    return {grad_features, grad_weights};
}

template <typename Scalar>
static torch::Tensor
superblockConvTransposeDispatched(torch::Tensor features,
                                  torch::Tensor weights,
                                  GridBatchImpl const &source_grid,
                                  GridBatchImpl const &target_grid,
                                  nanovdb::Coord kernel_size,
                                  int C_in,
                                  int C_out) {
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const NT = target_grid.totalVoxels();
    auto output      = torch::zeros({NT, C_out}, features.options());
    if (NT == 0)
        return output;

    if (kernel_size == nanovdb::Coord(3, 3, 3)) {
        launchTransposedFwd<SuperblockGeometry<3, 3, 3>, Scalar, 4>(
            source_grid, target_grid, features, weights, output, C_in, C_out, stream);
    } else if (kernel_size == nanovdb::Coord(5, 5, 5)) {
        launchTransposedFwd<SuperblockGeometry<5, 5, 5>, Scalar, 4>(
            source_grid, target_grid, features, weights, output, C_in, C_out, stream);
    } else if (kernel_size == nanovdb::Coord(7, 7, 7)) {
        launchTransposedFwd<SuperblockGeometry<7, 7, 7>, Scalar, 2>(
            source_grid, target_grid, features, weights, output, C_in, C_out, stream);
    } else {
        TORCH_CHECK(false,
                    "superblockConvTranspose: unsupported kernel size (",
                    kernel_size[0],
                    ",",
                    kernel_size[1],
                    ",",
                    kernel_size[2],
                    "). Supported: 3x3x3, 5x5x5, 7x7x7.");
    }
    return output;
}

// ============================================================================
// Input validation helper
// ============================================================================

static void
validateSuperblockInputs(torch::Tensor features,
                         torch::Tensor weights,
                         GridBatchImpl const &feature_grid,
                         nanovdb::Coord kernel_size,
                         nanovdb::Coord stride,
                         char const *func_name) {
    TORCH_CHECK(features.is_cuda(), func_name, ": features must be on CUDA");
    TORCH_CHECK(deviceSupportsSuperblock(features.device()),
                func_name,
                ": requires Sm80+ (Ampere or newer)");
    TORCH_CHECK(features.dim() == 2, func_name, ": features must be 2D");
    TORCH_CHECK(features.size(0) == feature_grid.totalVoxels(),
                func_name,
                ": features.size(0)=",
                features.size(0),
                " must match feature_grid totalVoxels=",
                feature_grid.totalVoxels());
    TORCH_CHECK(features.is_contiguous(), func_name, ": features must be contiguous");
    TORCH_CHECK(weights.dim() == 5, func_name, ": weights must be 5D [Cout, Cin, k0, k1, k2]");
    TORCH_CHECK(features.size(1) == weights.size(1),
                func_name,
                ": Cin mismatch between features (",
                features.size(1),
                ") and weights (",
                weights.size(1),
                ")");
    TORCH_CHECK(weights.size(2) == kernel_size[0] && weights.size(3) == kernel_size[1] &&
                    weights.size(4) == kernel_size[2],
                func_name,
                ": weights spatial dims must match kernel_size");
    TORCH_CHECK(features.device() == weights.device(),
                func_name,
                ": features and weights must be on same device");
    TORCH_CHECK(features.scalar_type() == weights.scalar_type(),
                func_name,
                ": features and weights must have same dtype");
    TORCH_CHECK(stride[0] == 1 && stride[1] == 1 && stride[2] == 1,
                func_name,
                ": only stride=(1,1,1) is supported, got (",
                stride[0],
                ",",
                stride[1],
                ",",
                stride[2],
                ")");

    int64_t const C_in  = features.size(1);
    int64_t const C_out = weights.size(0);
    TORCH_CHECK(C_in > 0 && C_in % 32 == 0,
                func_name,
                ": C_in must be a positive multiple of 32, got ",
                C_in);
    TORCH_CHECK(C_out > 0 && C_out % 32 == 0,
                func_name,
                ": C_out must be a positive multiple of 32, got ",
                C_out);
}

// ============================================================================
// Entry points
// ============================================================================

torch::Tensor
superblockConv(torch::Tensor features,
               torch::Tensor weights,
               GridBatchImpl const &feature_grid,
               GridBatchImpl const &output_grid,
               nanovdb::Coord kernel_size,
               nanovdb::Coord stride) {
    validateSuperblockInputs(
        features, weights, feature_grid, kernel_size, stride, "superblockConv");

    int64_t const C_in  = features.size(1);
    int64_t const C_out = weights.size(0);

    if (features.scalar_type() == torch::kFloat16) {
        auto f16 = features.to(torch::kFloat16).contiguous();
        auto w16 = weights.to(torch::kFloat16).contiguous();
        auto out = superblockConvDispatched<c10::Half>(
            f16, w16, feature_grid, output_grid, kernel_size, C_in, C_out);
        return out.to(torch::kFloat16);
    } else if (features.scalar_type() == torch::kFloat32) {
        return superblockConvDispatched<float>(
            features, weights, feature_grid, output_grid, kernel_size, C_in, C_out);
    } else {
        TORCH_CHECK(false,
                    "superblockConv: unsupported dtype ",
                    features.scalar_type(),
                    ". Supported: fp16, fp32.");
    }
}

std::tuple<torch::Tensor, torch::Tensor>
superblockConvBackward(torch::Tensor grad_output,
                       torch::Tensor features,
                       torch::Tensor weights,
                       GridBatchImpl const &feature_grid,
                       GridBatchImpl const &output_grid,
                       nanovdb::Coord kernel_size,
                       nanovdb::Coord stride) {
    validateSuperblockInputs(
        features, weights, feature_grid, kernel_size, stride, "superblockConvBackward");
    TORCH_CHECK(grad_output.is_cuda(), "superblockConvBackward: grad_output must be on CUDA");
    TORCH_CHECK(grad_output.dim() == 2, "superblockConvBackward: grad_output must be 2D");
    TORCH_CHECK(grad_output.size(0) == output_grid.totalVoxels(),
                "superblockConvBackward: grad_output.size(0)=",
                grad_output.size(0),
                " must match output_grid totalVoxels=",
                output_grid.totalVoxels());
    TORCH_CHECK(grad_output.size(1) == weights.size(0),
                "superblockConvBackward: grad_output.size(1)=",
                grad_output.size(1),
                " must match C_out=",
                weights.size(0));
    TORCH_CHECK(grad_output.is_contiguous(),
                "superblockConvBackward: grad_output must be contiguous");

    int64_t const C_in  = features.size(1);
    int64_t const C_out = weights.size(0);

    if (features.scalar_type() == torch::kFloat16) {
        auto f16      = features.to(torch::kFloat16).contiguous();
        auto w16      = weights.to(torch::kFloat16).contiguous();
        auto go16     = grad_output.to(torch::kFloat16).contiguous();
        auto [gf, gw] = superblockConvBackwardDispatched<c10::Half>(
            go16, f16, w16, feature_grid, output_grid, kernel_size, C_in, C_out);
        return {gf.to(torch::kFloat16), gw.to(torch::kFloat16)};
    } else if (features.scalar_type() == torch::kFloat32) {
        return superblockConvBackwardDispatched<float>(
            grad_output, features, weights, feature_grid, output_grid, kernel_size, C_in, C_out);
    } else {
        TORCH_CHECK(false,
                    "superblockConvBackward: unsupported dtype ",
                    features.scalar_type(),
                    ". Supported: fp16, fp32.");
    }
}

torch::Tensor
superblockConvTranspose(torch::Tensor features,
                        torch::Tensor weights,
                        GridBatchImpl const &source_grid,
                        GridBatchImpl const &target_grid,
                        nanovdb::Coord kernel_size,
                        nanovdb::Coord stride) {
    TORCH_CHECK(features.is_cuda(), "superblockConvTranspose: features must be on CUDA");
    TORCH_CHECK(deviceSupportsSuperblock(features.device()),
                "superblockConvTranspose: requires Sm80+ (Ampere or newer)");
    TORCH_CHECK(features.dim() == 2, "superblockConvTranspose: features must be 2D");
    TORCH_CHECK(features.size(0) == source_grid.totalVoxels(),
                "superblockConvTranspose: features.size(0)=",
                features.size(0),
                " must match source_grid totalVoxels=",
                source_grid.totalVoxels());
    TORCH_CHECK(features.is_contiguous(), "superblockConvTranspose: features must be contiguous");
    TORCH_CHECK(weights.dim() == 5, "superblockConvTranspose: weights must be 5D");
    TORCH_CHECK(features.size(1) == weights.size(1), "superblockConvTranspose: Cin mismatch");
    TORCH_CHECK(features.device() == weights.device(),
                "superblockConvTranspose: features and weights must be on same device");
    TORCH_CHECK(features.scalar_type() == weights.scalar_type(),
                "superblockConvTranspose: features and weights must have same dtype");
    TORCH_CHECK(stride[0] == 1 && stride[1] == 1 && stride[2] == 1,
                "superblockConvTranspose: only stride=(1,1,1) supported");

    int64_t const C_in  = features.size(1);
    int64_t const C_out = weights.size(0);
    TORCH_CHECK(C_in > 0 && C_in % 32 == 0,
                "superblockConvTranspose: C_in must be a positive multiple of 32");
    TORCH_CHECK(C_out > 0 && C_out % 32 == 0,
                "superblockConvTranspose: C_out must be a positive multiple of 32");

    if (features.scalar_type() == torch::kFloat16) {
        auto f16 = features.to(torch::kFloat16).contiguous();
        auto w16 = weights.to(torch::kFloat16).contiguous();
        auto out = superblockConvTransposeDispatched<c10::Half>(
            f16, w16, source_grid, target_grid, kernel_size, C_in, C_out);
        return out.to(torch::kFloat16);
    } else if (features.scalar_type() == torch::kFloat32) {
        return superblockConvTransposeDispatched<float>(
            features, weights, source_grid, target_grid, kernel_size, C_in, C_out);
    } else {
        TORCH_CHECK(false,
                    "superblockConvTranspose: unsupported dtype ",
                    features.scalar_type(),
                    ". Supported: fp16, fp32.");
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif
