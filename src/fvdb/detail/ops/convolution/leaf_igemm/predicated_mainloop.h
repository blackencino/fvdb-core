// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// predicated_mainloop.h -- SM80 pipelined GEMM mainloop with predicated
//                          B-matrix loads.
//
// This is the single modification to a stock CUTLASS SM80 mainloop that
// makes the GEMM sparse-aware: B-matrix global-to-shared copies use
// copy_if gated by a predicate tensor in shared memory, so that inactive
// voxels (gather_map entry == 0) produce zero instead of issuing a global
// memory read.
//
// A-matrix (filter weights) loads are unpredicated -- the filter is always
// fully present.
//
// Adapted from sifakis/openvdb sm80_mma_multistage_custom.hpp.
// Target architecture: SM80 (Ampere).
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_PREDICATED_MAINLOOP_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_PREDICATED_MAINLOOP_H

#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_mma_decl.hpp>
#include <cutlass/gemm/collective/sm80_mma_multistage.hpp>

#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// Dispatch policy tag for the predicated mainloop
// =============================================================================
//
// This is a new dispatch policy that CUTLASS's CollectiveMma will
// specialize on. It is identical to MainloopSm80CpAsyncUnpredicated
// except for the tag type, which selects our custom specialization.

template <int Stages_>
struct MainloopSm80Predicated {
    static constexpr int Stages = Stages_;
    using ArchTag               = cutlass::arch::Sm80;
    using Schedule              = cutlass::gemm::KernelMultistage;
    using ClusterShape          = cute::Shape<cute::_1, cute::_1, cute::_1>;
};

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

// =============================================================================
// CollectiveMma specialization for MainloopSm80Predicated
// =============================================================================
//
// Must be in the cutlass::gemm::collective namespace for CUTLASS's
// template machinery to find it.

namespace cutlass::gemm::collective {

using namespace cute;

template <int Stages, class TileShape_, class ElementA_, class StrideA_, class ElementB_,
          class StrideB_, class TiledMma_, class GmemTiledCopyA_, class SmemLayoutAtomA_,
          class SmemCopyAtomA_, class TransformA_, class GmemTiledCopyB_, class SmemLayoutAtomB_,
          class SmemCopyAtomB_, class TransformB_>
struct CollectiveMma<fvdb::detail::leaf_igemm::MainloopSm80Predicated<Stages>, TileShape_,
                     ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_, GmemTiledCopyA_,
                     SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_,
                     SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_> {
    //
    // Type Aliases
    //
    using DispatchPolicy    = MainloopSm80CpAsyncUnpredicated<Stages>;
    using TileShape         = TileShape_;
    using ElementA          = ElementA_;
    using StrideA           = StrideA_;
    using ElementB          = ElementB_;
    using StrideB           = StrideB_;
    using TiledMma          = TiledMma_;
    using ElementAccumulator = typename TiledMma::ValTypeC;
    using GmemTiledCopyA    = GmemTiledCopyA_;
    using GmemTiledCopyB    = GmemTiledCopyB_;
    using SmemLayoutAtomA   = SmemLayoutAtomA_;
    using SmemLayoutAtomB   = SmemLayoutAtomB_;
    using SmemCopyAtomA     = SmemCopyAtomA_;
    using SmemCopyAtomB     = SmemCopyAtomB_;
    using TransformA        = TransformA_;
    using TransformB        = TransformB_;
    using ArchTag           = typename DispatchPolicy::ArchTag;
    using CtaShape_MNK      = TileShape;

    static_assert(cute::rank(SmemLayoutAtomA{}) == 2,
                  "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    static_assert(cute::rank(SmemLayoutAtomB{}) == 2,
                  "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<Stages>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<Stages>{})));

    static_assert(DispatchPolicy::Stages >= 2,
                  "CpAsync mainloop must have at least 2 pipeline stages.");

    struct SharedStorage {
        cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
        cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    };

    struct Arguments {
        ElementA const *ptr_A;
        StrideA dA;
        ElementB const *ptr_B;
        StrideB dB;
    };

    using Params = Arguments;

    CollectiveMma() = default;

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const &, Arguments const &args, void *) {
        return args;
    }

    // ---- The pipelined mainloop with predicated B-loads ---------------------
    //
    // Signature differs from stock CUTLASS: takes an extra TensorP (sP)
    // which is the boolean predicate tensor in shared memory.
    //
    // A-loads:  unpredicated copy  (filter weights are always present)
    // B-loads:  copy_if            (zero-fill for inactive input voxels)

    template <class FrgTensorD, class TensorA, class TensorB, class TensorP, class FrgTensorC,
              class KTileIterator, class ResidueMNK>
    CUTLASS_DEVICE void
    operator()(FrgTensorD &accum, TensorA gA, TensorB gB, TensorP sP, FrgTensorC const &src_accum,
               KTileIterator k_tile_iter, int k_tile_count, ResidueMNK residue_mnk, int thread_idx,
               char *smem_buf) {
        using namespace cute;

        static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
        static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
        static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");
        static_assert(is_smem<TensorP>::value, "Predicate tensor must be smem resident.");
        static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

        SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
        Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

        // Thread partitions for global-to-shared copies
        GmemTiledCopyA gmem_tiled_copy_A;
        GmemTiledCopyB gmem_tiled_copy_B;
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

        Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
        Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
        Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
        Tensor tBsB = gmem_thr_copy_B.partition_D(sB);
        Tensor tBsP = gmem_thr_copy_B.partition_S(sP);

        (void)residue_mnk;

        // ---- PREFETCH: fill all pipeline stages except the last ----

        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < Stages - 1; ++k_pipe) {
            copy(gmem_tiled_copy_A, tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, k_pipe));
            copy_if(gmem_tiled_copy_B, tBsP(_, _, _, *k_tile_iter), tBgB(_, _, _, *k_tile_iter),
                    tBsB(_, _, _, k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) {
                ++k_tile_iter;
            }
        }

        // ---- MMA partitioning ----

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA  = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB  = thr_mma.partition_fragment_B(sB(_, _, 0));

        // Shared-to-register copy atoms
        auto smem_tiled_copy_A   = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A     = smem_tiled_copy_A.get_thread_slice(thread_idx);
        Tensor tCsA              = smem_thr_copy_A.partition_S(sA);
        Tensor tCrA_copy_view    = smem_thr_copy_A.retile_D(tCrA);

        auto smem_tiled_copy_B   = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B     = smem_tiled_copy_B.get_thread_slice(thread_idx);
        Tensor tCsB              = smem_thr_copy_B.partition_S(sB);
        Tensor tCrB_copy_view    = smem_thr_copy_B.retile_D(tCrB);

        // ---- PIPELINED MAIN LOOP ----

        int smem_pipe_read  = 0;
        int smem_pipe_write = Stages - 1;

        Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
        Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

        auto K_BLOCK_MAX = size<2>(tCrA);

        // Prefetch first register tile
        if (K_BLOCK_MAX > 1) {
            cp_async_wait<Stages - 2>();
            __syncthreads();
            copy(smem_tiled_copy_A, tCsA_p(_, _, Int<0>{}), tCrA_copy_view(_, _, Int<0>{}));
            copy(smem_tiled_copy_B, tCsB_p(_, _, Int<0>{}), tCrB_copy_view(_, _, Int<0>{}));
        }

        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > -(Stages - 1)) {
            for_each(make_int_sequence<decltype(K_BLOCK_MAX)::value>{}, [&](auto k_block) {
                if (k_block == K_BLOCK_MAX - 1) {
                    tCsA_p = tCsA(_, _, _, smem_pipe_read);
                    tCsB_p = tCsB(_, _, _, smem_pipe_read);
                    cp_async_wait<Stages - 2>();
                    __syncthreads();
                }

                auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                copy(smem_tiled_copy_A, tCsA_p(_, _, k_block_next),
                     tCrA_copy_view(_, _, k_block_next));
                copy(smem_tiled_copy_B, tCsB_p(_, _, k_block_next),
                     tCrB_copy_view(_, _, k_block_next));

                if (k_block == 0) {
                    copy(gmem_tiled_copy_A, tAgA(_, _, _, *k_tile_iter),
                         tAsA(_, _, _, smem_pipe_write));
                    copy_if(gmem_tiled_copy_B, tBsP(_, _, _, *k_tile_iter),
                            tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, smem_pipe_write));
                    cp_async_fence();

                    --k_tile_count;
                    if (k_tile_count > 0) {
                        ++k_tile_iter;
                    }

                    smem_pipe_write = smem_pipe_read;
                    ++smem_pipe_read;
                    smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
                }

                cute::transform(tCrA(_, _, k_block), TransformA{});
                cute::transform(tCrB(_, _, k_block), TransformB{});
                cute::gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrB(_, _, k_block), src_accum);
            });
        }

        cp_async_wait<0>();
        __syncthreads();
    }
};

} // namespace cutlass::gemm::collective

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_PREDICATED_MAINLOOP_H
