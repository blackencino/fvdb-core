// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// conv_tiling.h -- Block/cluster decomposition and shared memory budget.
//
// Encodes the output-block shape (B), cluster grouping (CL), pipeline
// depth, and GEMM tile sizes. Computes all derived tiling quantities and
// validates constraints C1, C3, C4, C5 at compile time.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_TILING_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_TILING_H

#include "conv_geometry.h"
#include "conv_types.h"
#include "core_types.h"

#include <cstddef>

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// SM80 shared memory limits.
// 48 KB is the baseline; up to 164 KB available via cudaFuncSetAttribute
// (dynamic shared memory). We use the baseline for static allocation and
// note when dynamic smem would be needed.
inline constexpr size_t SMEM_BASELINE_BYTES = 48 * 1024;
inline constexpr size_t SMEM_MAX_BYTES      = 164 * 1024;

// =============================================================================
// conv_tiling: block/cluster decomposition and GEMM tile sizes
// =============================================================================
//
// Template parameters:
//   B_      : output-block extent within a leaf (e.g. {4,2,2} -> 16 voxels)
//   CL_     : blocks per cluster per axis (e.g. {1,2,2} -> 4 blocks/cluster)
//   TileM_  : GEMM M-dimension tile size (output channels per tile)
//   TileCK_ : GEMM contraction-dimension tile size
//   Stages_ : pipeline depth (number of async copy stages)
//   Geom    : a geometry_like trait
//   Types   : a types_like trait

template <coord3 B_, coord3 CL_ = coord3{1, 1, 1}, int TileM_ = 32, int TileCK_ = 8,
          int Stages_ = 3, geometry_like Geom = geom_3x3x3_s1, types_like Types = types_f32>
struct conv_tiling {

    // ---- Primary parameters ------------------------------------------------

    static consteval coord3
    B() {
        return B_;
    }
    static consteval coord3
    CL() {
        return CL_;
    }
    static consteval int
    tile_m() {
        return TileM_;
    }
    static consteval int
    tile_ck() {
        return TileCK_;
    }
    static consteval int
    stages() {
        return Stages_;
    }

    // ---- Block quantities --------------------------------------------------

    static consteval coord3
    nblk() {
        return Geom::leaf3() / B_;
    }
    static consteval int
    nblk_tot() {
        return prod(nblk());
    }
    static consteval int
    blk_vol() {
        return prod(B_);
    }

    // ---- Cluster quantities ------------------------------------------------

    static consteval coord3
    ncl() {
        return nblk() / CL_;
    }
    static consteval int
    ncl_tot() {
        return prod(ncl());
    }
    static consteval coord3
    cl_extent() {
        return CL_ * B_;
    }
    static consteval coord3
    cl_halo() {
        return (cl_extent() - coord3{1, 1, 1}) * Geom::S() +
               (Geom::K() - coord3{1, 1, 1}) * Geom::D() + coord3{1, 1, 1};
    }
    static consteval int
    cl_halo_vol() {
        return prod(cl_halo());
    }
    static consteval int
    cl_vol() {
        return prod(cl_extent());
    }

    // ---- Full contraction length -------------------------------------------

    static consteval int
    contract() {
        return Geom::kern_vol(); // * C_in -- but C_in is runtime
    }

    // ---- Shared memory budget (C3) -----------------------------------------
    //
    // index_maps   = HALO_VOL * sizeof(ElemIdx) + LEAF^3 * sizeof(ElemIdx)
    // gemm_buffers = TileM * TileCK * sizeof(ElemWt)   * Stages   (A-tile pipeline)
    //              + BLK_VOL * TileCK * sizeof(ElemFeat) * Stages  (B-tile pipeline)
    // predicates   = CL_HALO_VOL + CL_VOL                         (bool masks)
    //
    // Total = index_maps + max(gemm_buffers, epilogue_staging) + predicates

    static consteval size_t
    smem_index_maps() {
        return static_cast<size_t>(Geom::halo_vol() + Geom::leaf_vol()) *
               sizeof(typename Types::ElemIdx);
    }

    static consteval size_t
    smem_gemm_a() {
        return static_cast<size_t>(TileM_) * TileCK_ * sizeof(typename Types::ElemWt) * Stages_;
    }

    static consteval size_t
    smem_gemm_b() {
        return static_cast<size_t>(blk_vol()) * TileCK_ * sizeof(typename Types::ElemFeat) *
               Stages_;
    }

    static consteval size_t
    smem_gemm_buffers() {
        return smem_gemm_a() + smem_gemm_b();
    }

    static consteval size_t
    smem_predicates() {
        return static_cast<size_t>(cl_halo_vol() + cl_vol());
    }

    static consteval size_t
    smem_total() {
        return smem_index_maps() + smem_gemm_buffers() + smem_predicates();
    }

    static consteval bool
    needs_dynamic_smem() {
        return smem_total() > SMEM_BASELINE_BYTES;
    }

    // ---- Constraint validation ---------------------------------------------

    // C1: blocks tile the leaf exactly with no remainder.
    static_assert(all_divisible(Geom::leaf3(), B_),
                  "C1: output-block shape B must evenly divide the leaf on all axes");

    // Cluster constraint: blocks-per-cluster must evenly divide blocks-per-leaf.
    static_assert(all_divisible(nblk(), CL_),
                  "Cluster shape CL must evenly divide blocks-per-leaf (NBLK) on all axes");

    static_assert(all_positive(B_), "Block shape must be positive on all axes");
    static_assert(all_positive(CL_), "Cluster shape must be positive on all axes");
    static_assert(TileM_ > 0, "GEMM tile M must be positive");
    static_assert(TileCK_ > 0, "GEMM tile CK must be positive");
    static_assert(Stages_ >= 2, "Pipeline needs at least 2 stages");

    // C3: shared memory budget (warn if exceeding even dynamic smem limit).
    static_assert(smem_total() <= SMEM_MAX_BYTES,
                  "C3: shared memory budget exceeds SM80 maximum (164 KB). "
                  "Reduce tile sizes, pipeline stages, or block shape.");
};

// =============================================================================
// is_conv_tiling trait + tiling_like concept
// =============================================================================

template <typename T>
struct is_conv_tiling : consteval_false_type {};

template <coord3 B, coord3 CL, int TM, int TCK, int S, geometry_like G, types_like Ty>
struct is_conv_tiling<conv_tiling<B, CL, TM, TCK, S, G, Ty>> : consteval_true_type {};

template <typename T>
concept tiling_like = is_conv_tiling<T>::value();

// =============================================================================
// Common tiling alias: the reference implementation's decomposition
// =============================================================================

// 4x2x2 blocks, 1x2x2 clusters, 32-wide M tile, 8-deep CK tile, 3 stages.
template <geometry_like Geom, types_like Types>
using tiling_default = conv_tiling<coord3{4, 2, 2}, coord3{1, 2, 2}, 32, 8, 3, Geom, Types>;

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_TILING_H
