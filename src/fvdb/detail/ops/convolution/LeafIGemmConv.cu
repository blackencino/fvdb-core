// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// LeafIGemmConv.cu -- Leaf-level implicit GEMM sparse convolution.
//
// Single-file implementation of the algorithm described in
// docs/wip/leaf_level_igemm_sparse_conv.md.
//
// One threadblock per output leaf. The kernel fuses topology densification
// (building gather/scatter index maps in shared memory) with an implicit
// GEMM convolution. The im2col matrix is never materialised; instead,
// each B-matrix element is resolved at runtime through a shared-memory
// lookup table via CuTe's ComposedLayout.
//
// Compile-time parameterised over kernel size (K), stride (S), dilation (D),
// scalar types, and block/cluster decomposition.
//
// Target architecture: SM80 (Ampere) and newer.

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/LeafIGemmConv.h>

#include <nanovdb/NanoVDB.h>

// NOTE: torch / ATen / c10 headers MUST precede CuTe / CUTLASS headers.
// See CutlassGroupedGemm.cu for the full explanation (CCCL version mismatch
// between local nvcc 13.1 and conda CUDA 12.9 toolkit headers).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/arch.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_mma_decl.hpp>
#include <cutlass/gemm/collective/sm80_mma_multistage.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/half.h>

#include <cstddef>
#include <cstdint>

namespace fvdb {
namespace detail {
namespace leaf_igemm {

using namespace cute;

// ============================================================================
// Vec3i / Coord utilities
// ============================================================================

__device__ __forceinline__ int
encode3(nanovdb::Vec3i pos, nanovdb::Vec3i dims) {
    return pos[0] * dims[1] * dims[2] + pos[1] * dims[2] + pos[2];
}

__device__ __forceinline__ nanovdb::Vec3i
decode3(int i, nanovdb::Vec3i dims) {
    return nanovdb::Vec3i(i / (dims[1] * dims[2]), (i / dims[2]) % dims[1], i % dims[2]);
}

__device__ __forceinline__ int
prod(nanovdb::Vec3i v) {
    return v[0] * v[1] * v[2];
}

__device__ __forceinline__ nanovdb::Coord
toCoord(nanovdb::Vec3i v) {
    return nanovdb::Coord(v[0], v[1], v[2]);
}

__device__ __forceinline__ nanovdb::Vec3i
toVec3i(nanovdb::Coord c) {
    return nanovdb::Vec3i(c);
}

// ============================================================================
// ConvGeometry -- compile-time spatial parameters
// ============================================================================
//
// All derived quantities available as static constexpr int members for use
// in array sizes, static_asserts, and template arguments. Vec3i accessors
// provided for runtime (device) arithmetic.

template <int K0_   = 3,
          int K1_   = 3,
          int K2_   = 3,
          int S0_   = 1,
          int S1_   = 1,
          int S2_   = 1,
          int D0_   = 1,
          int D1_   = 1,
          int D2_   = 1,
          int LEAF_ = 8>
struct ConvGeometry {
    static constexpr int K0        = K0_;
    static constexpr int K1        = K1_;
    static constexpr int K2        = K2_;
    static constexpr int S0        = S0_;
    static constexpr int S1        = S1_;
    static constexpr int S2        = S2_;
    static constexpr int D0        = D0_;
    static constexpr int D1        = D1_;
    static constexpr int D2        = D2_;
    static constexpr int leaf_side = LEAF_;

    static constexpr int kern_vol = K0 * K1 * K2;
    static constexpr int leaf_vol = LEAF_ * LEAF_ * LEAF_;

    static constexpr int halo0    = (LEAF_ - 1) * S0_ + (K0_ - 1) * D0_ + 1;
    static constexpr int halo1    = (LEAF_ - 1) * S1_ + (K1_ - 1) * D1_ + 1;
    static constexpr int halo2    = (LEAF_ - 1) * S2_ + (K2_ - 1) * D2_ + 1;
    static constexpr int halo_vol = halo0 * halo1 * halo2;

    static constexpr int off0 = -((K0_ - 1) * D0_) / 2;
    static constexpr int off1 = -((K1_ - 1) * D1_) / 2;
    static constexpr int off2 = -((K2_ - 1) * D2_) / 2;

    static __host__ __device__ nanovdb::Vec3i
    K() {
        return {K0, K1, K2};
    }
    static __host__ __device__ nanovdb::Vec3i
    S() {
        return {S0, S1, S2};
    }
    static __host__ __device__ nanovdb::Vec3i
    D() {
        return {D0, D1, D2};
    }
    static __host__ __device__ nanovdb::Vec3i
    halo() {
        return {halo0, halo1, halo2};
    }
    static __host__ __device__ nanovdb::Vec3i
    leaf3() {
        return {leaf_side, leaf_side, leaf_side};
    }
    static __host__ __device__ nanovdb::Vec3i
    offset() {
        return {off0, off1, off2};
    }

    static_assert(K0_ > 0 && K1_ > 0 && K2_ > 0, "Kernel size must be positive on all axes");
    static_assert(S0_ > 0 && S1_ > 0 && S2_ > 0, "Stride must be positive on all axes");
    static_assert(D0_ > 0 && D1_ > 0 && D2_ > 0, "Dilation must be positive on all axes");
    static_assert(LEAF_ > 0, "Leaf side length must be positive");
    static_assert(halo0 <= 2 * LEAF_ + 1 && halo1 <= 2 * LEAF_ + 1 && halo2 <= 2 * LEAF_ + 1,
                  "C2 violated: halo exceeds 3-leaf neighborhood. "
                  "Reduce kernel size, stride, or dilation.");
};

using Geom3x3x3_S1 = ConvGeometry<3, 3, 3>;
using Geom3x3x3_S2 = ConvGeometry<3, 3, 3, 2, 2, 2>;
using Geom5x5x5_S1 = ConvGeometry<5, 5, 5>;

// ============================================================================
// MMA atom selection -- maps (ElemWt, ElemFeat, ElemAcc) to SM80 MMA op
// ============================================================================

template <typename ElemWt, typename ElemFeat, typename ElemAcc> struct mma_atom_selector {
    static_assert(sizeof(ElemWt) == 0,
                  "C6 violated: no SM80 MMA atom for this (ElemWt, ElemFeat, ElemAcc) triple.");
};

template <> struct mma_atom_selector<float, float, float> {
    using mma_op               = SM80_16x8x8_F32TF32TF32F32_TN;
    using mma_traits           = MMA_Traits<mma_op>;
    static constexpr int mma_k = 8;
};

template <> struct mma_atom_selector<cutlass::half_t, cutlass::half_t, float> {
    using mma_op               = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits           = MMA_Traits<mma_op>;
    static constexpr int mma_k = 16;
};

template <> struct mma_atom_selector<cutlass::bfloat16_t, cutlass::bfloat16_t, float> {
    using mma_op               = SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits           = MMA_Traits<mma_op>;
    static constexpr int mma_k = 16;
};

// ============================================================================
// ConvTypes -- scalar type parameter bundle
// ============================================================================

template <typename ElemWt_,
          typename ElemFeat_,
          typename ElemAcc_,
          typename ElemOut_,
          typename ElemIdx_>
struct ConvTypes {
    using ElemWt   = ElemWt_;
    using ElemFeat = ElemFeat_;
    using ElemAcc  = ElemAcc_;
    using ElemOut  = ElemOut_;
    using ElemIdx  = ElemIdx_;

    using mma_selector = mma_atom_selector<ElemWt_, ElemFeat_, ElemAcc_>;
    using mma_op       = typename mma_selector::mma_op;
    using mma_traits   = typename mma_selector::mma_traits;

    static_assert(sizeof(ElemOut_) <= sizeof(ElemAcc_),
                  "C7: ElemOut must not be wider than ElemAcc");
};

using types_f32 = ConvTypes<float, float, float, float, uint32_t>;
using types_f16 = ConvTypes<cutlass::half_t, cutlass::half_t, float, cutlass::half_t, uint32_t>;
using types_bf16 =
    ConvTypes<cutlass::bfloat16_t, cutlass::bfloat16_t, float, cutlass::bfloat16_t, uint32_t>;

// ============================================================================
// ConvTiling -- block/cluster decomposition, GEMM tile sizes, smem budget
// ============================================================================

static constexpr size_t SMEM_BASELINE_BYTES = 48 * 1024;
static constexpr size_t SMEM_MAX_BYTES      = 164 * 1024;

template <int B0_         = 4,
          int B1_         = 2,
          int B2_         = 2,
          int CL0_        = 1,
          int CL1_        = 1,
          int CL2_        = 1,
          int TileM_      = 32,
          int TileCK_     = 8,
          int Stages_     = 3,
          typename Geom_  = Geom3x3x3_S1,
          typename Types_ = types_f32>
struct ConvTiling {
    using Geom  = Geom_;
    using Types = Types_;

    static constexpr int tile_m  = TileM_;
    static constexpr int tile_ck = TileCK_;
    static constexpr int stages  = Stages_;

    static constexpr int blk_vol = B0_ * B1_ * B2_;

    static constexpr int nblk0    = Geom_::leaf_side / B0_;
    static constexpr int nblk1    = Geom_::leaf_side / B1_;
    static constexpr int nblk2    = Geom_::leaf_side / B2_;
    static constexpr int nblk_tot = nblk0 * nblk1 * nblk2;

    static constexpr int ncl0    = nblk0 / CL0_;
    static constexpr int ncl1    = nblk1 / CL1_;
    static constexpr int ncl2    = nblk2 / CL2_;
    static constexpr int ncl_tot = ncl0 * ncl1 * ncl2;

    static constexpr int cl_ext0 = CL0_ * B0_;
    static constexpr int cl_ext1 = CL1_ * B1_;
    static constexpr int cl_ext2 = CL2_ * B2_;
    static constexpr int cl_vol  = cl_ext0 * cl_ext1 * cl_ext2;

    static constexpr int cl_halo0    = (cl_ext0 - 1) * Geom_::S0 + (Geom_::K0 - 1) * Geom_::D0 + 1;
    static constexpr int cl_halo1    = (cl_ext1 - 1) * Geom_::S1 + (Geom_::K1 - 1) * Geom_::D1 + 1;
    static constexpr int cl_halo2    = (cl_ext2 - 1) * Geom_::S2 + (Geom_::K2 - 1) * Geom_::D2 + 1;
    static constexpr int cl_halo_vol = cl_halo0 * cl_halo1 * cl_halo2;

    static constexpr int blocks_per_cl = CL0_ * CL1_ * CL2_;

    static __host__ __device__ nanovdb::Vec3i
    B() {
        return {B0_, B1_, B2_};
    }
    static __host__ __device__ nanovdb::Vec3i
    CL() {
        return {CL0_, CL1_, CL2_};
    }
    static __host__ __device__ nanovdb::Vec3i
    nblk() {
        return {nblk0, nblk1, nblk2};
    }
    static __host__ __device__ nanovdb::Vec3i
    ncl() {
        return {ncl0, ncl1, ncl2};
    }
    static __host__ __device__ nanovdb::Vec3i
    cl_extent() {
        return {cl_ext0, cl_ext1, cl_ext2};
    }
    static __host__ __device__ nanovdb::Vec3i
    cl_halo() {
        return {cl_halo0, cl_halo1, cl_halo2};
    }

    // Shared memory budget (C3)
    static constexpr size_t smem_index_maps =
        static_cast<size_t>(Geom_::halo_vol + Geom_::leaf_vol) * sizeof(typename Types_::ElemIdx);
    static constexpr size_t smem_gemm_a =
        static_cast<size_t>(TileM_) * TileCK_ * sizeof(typename Types_::ElemWt) * Stages_;
    static constexpr size_t smem_gemm_b =
        static_cast<size_t>(blk_vol) * TileCK_ * sizeof(typename Types_::ElemFeat) * Stages_;
    static constexpr size_t smem_predicates = static_cast<size_t>(cl_halo_vol + cl_vol);
    static constexpr size_t smem_total =
        smem_index_maps + smem_gemm_a + smem_gemm_b + smem_predicates;

    // C1: blocks tile the leaf exactly.
    static_assert(Geom_::leaf_side % B0_ == 0 && Geom_::leaf_side % B1_ == 0 &&
                      Geom_::leaf_side % B2_ == 0,
                  "C1: output-block shape B must evenly divide the leaf on all axes");
    static_assert(nblk0 % CL0_ == 0 && nblk1 % CL1_ == 0 && nblk2 % CL2_ == 0,
                  "Cluster shape CL must evenly divide blocks-per-leaf on all axes");
    static_assert(B0_ > 0 && B1_ > 0 && B2_ > 0, "Block shape must be positive");
    static_assert(CL0_ > 0 && CL1_ > 0 && CL2_ > 0, "Cluster shape must be positive");
    static_assert(TileM_ > 0, "GEMM tile M must be positive");
    static_assert(TileCK_ > 0, "GEMM tile CK must be positive");
    static_assert(Stages_ >= 2, "Pipeline needs at least 2 stages");
    static_assert(smem_total <= SMEM_MAX_BYTES,
                  "C3: shared memory budget exceeds SM80 maximum (164 KB)");
};

template <typename Geom, typename Types>
using DefaultTiling = ConvTiling<4, 2, 2, 1, 2, 2, 32, 8, 3, Geom, Types>;

// ============================================================================
// conv_variant -- operation variant enum and traits
// ============================================================================

enum class conv_variant { forward, input_grad, weight_grad, transposed_conv };

template <conv_variant V> struct variant_traits;

template <> struct variant_traits<conv_variant::forward> {
    static constexpr bool flip         = false;
    static constexpr bool atomic_accum = false;
};

template <> struct variant_traits<conv_variant::input_grad> {
    static constexpr bool flip         = true;
    static constexpr bool atomic_accum = false;
};

template <> struct variant_traits<conv_variant::weight_grad> {
    static constexpr bool flip         = false;
    static constexpr bool atomic_accum = true;
};

template <> struct variant_traits<conv_variant::transposed_conv> {
    static constexpr bool flip         = true;
    static constexpr bool atomic_accum = false;
};

// ============================================================================
// NanoVDB sparse grid adapter
// ============================================================================
//
// The ONLY code that touches NanoVDB tree structure. Provides the four
// operations from the formal specification:
//   leaf_count()         -> int
//   leaf_origin(leaf_id) -> nanovdb::Coord
//   leaf_value(leaf_id, p) -> ElemIdx   (0 = inactive)
//   coord_to_idx(coord)    -> ElemIdx   (0 = inactive)

template <typename ElemIdx> struct nanovdb_sparse_grid {
    using grid_type = nanovdb::OnIndexGrid;
    using tree_type = typename grid_type::TreeType;
    using acc_type  = nanovdb::ReadAccessor<nanovdb::ValueOnIndex>;

    grid_type const *grid_ptr;
    int64_t voxel_offset;
    int64_t cum_leaf_offset;

    __device__ int
    leaf_count() const {
        return static_cast<int>(grid_ptr->tree().nodeCount(0));
    }

    __device__ nanovdb::Coord
    leaf_origin(int leaf_id) const {
        return grid_ptr->tree().template getFirstNode<0>()[leaf_id].origin();
    }

    __device__ ElemIdx
    leaf_value(int leaf_id, int p) const {
        auto const &leaf = grid_ptr->tree().template getFirstNode<0>()[leaf_id];
        if (!leaf.isActive(p)) {
            return static_cast<ElemIdx>(0);
        }
        return static_cast<ElemIdx>(voxel_offset + leaf.getValue(p));
    }

    __device__ ElemIdx
    coord_to_idx(nanovdb::Coord coord) const {
        auto acc = grid_ptr->getAccessor();
        if (!acc.isActive(coord)) {
            return static_cast<ElemIdx>(0);
        }
        return static_cast<ElemIdx>(voxel_offset + acc.getValue(coord));
    }
};

template <typename ElemIdx>
__device__ nanovdb_sparse_grid<ElemIdx>
make_sparse_grid(GridBatchImpl::Accessor const &acc, int64_t batch_idx) {
    auto const *grid = acc.grid(batch_idx);
    return {grid, acc.voxelOffset(batch_idx), acc.leafOffset(batch_idx)};
}

// ============================================================================
// im2col map -- virtual im2col address computation
// ============================================================================
//
// Maps a logical GEMM coordinate (contraction_idx, spatial_idx) to a
// physical feature-memory address via the shared-memory gather_map.
// This is the heart of the implicit GEMM.

template <typename ElemIdx> struct im2col_result {
    ElemIdx voxel_idx;
    int channel;
};

template <typename Geom, typename Tiling, conv_variant Variant> struct im2col_map {
    static constexpr bool flip = variant_traits<Variant>::flip;

    template <typename ElemIdx>
    __device__ __forceinline__ static im2col_result<ElemIdx>
    apply(int ck, int n, nanovdb::Vec3i block_orig, ElemIdx const *gather_map) {
        int const c     = ck / Geom::kern_vol;
        int const k_lin = ck % Geom::kern_vol;

        nanovdb::Vec3i const Kv = Geom::K();
        nanovdb::Vec3i delta    = decode3(k_lin, Kv);

        if constexpr (flip) {
            delta = nanovdb::Vec3i(Kv[0] - 1, Kv[1] - 1, Kv[2] - 1) - delta;
        }

        nanovdb::Vec3i const v = decode3(n, Tiling::B());
        nanovdb::Vec3i const h = (block_orig + v) * Geom::S() + delta * Geom::D();

        int const h_lin         = encode3(h, Geom::halo());
        ElemIdx const voxel_idx = gather_map[h_lin];

        return {voxel_idx, c};
    }

    __device__ __forceinline__ static int
    halo_position(int ck, int n, nanovdb::Vec3i block_orig) {
        int const k_lin = ck % Geom::kern_vol;

        nanovdb::Vec3i const Kv = Geom::K();
        nanovdb::Vec3i delta    = decode3(k_lin, Kv);

        if constexpr (flip) {
            delta = nanovdb::Vec3i(Kv[0] - 1, Kv[1] - 1, Kv[2] - 1) - delta;
        }

        nanovdb::Vec3i const v = decode3(n, Tiling::B());
        nanovdb::Vec3i const h = (block_orig + v) * Geom::S() + delta * Geom::D();

        return encode3(h, Geom::halo());
    }
};

// ============================================================================
// Index map construction (Phase 1)
// ============================================================================
//
// The ONLY phase that touches the SparseGrid. After completion (plus a
// __syncthreads), the algorithm operates on flat index buffers in shared
// memory and never touches the tree again.

template <typename Geom, typename ElemIdx, typename SparseGrid>
__device__ void
build_scatter_map(SparseGrid const &iter_grid, int leaf_id, ElemIdx *scatter_map) {
    constexpr int LEAF_VOL = Geom::leaf_vol;
    int const tid          = threadIdx.x;
    int const nthreads     = blockDim.x;

    for (int p = tid; p < LEAF_VOL; p += nthreads) {
        scatter_map[p] = iter_grid.leaf_value(leaf_id, p);
    }
}

template <typename Geom, typename ElemIdx, typename SparseGrid>
__device__ void
build_gather_map(SparseGrid const &gather_grid, nanovdb::Vec3i halo_origin, ElemIdx *gather_map) {
    constexpr int HALO_VOL = Geom::halo_vol;
    int const tid          = threadIdx.x;
    int const nthreads     = blockDim.x;

    nanovdb::Vec3i const halo_dims = Geom::halo();

    for (int h = tid; h < HALO_VOL; h += nthreads) {
        nanovdb::Vec3i const h3    = decode3(h, halo_dims);
        nanovdb::Vec3i const coord = halo_origin + h3;
        gather_map[h]              = gather_grid.coord_to_idx(toCoord(coord));
    }
}

template <typename Geom, conv_variant Variant, typename ElemIdx, typename SparseGrid>
__device__ void
build_index_maps(SparseGrid const &iter_grid,
                 SparseGrid const &gather_grid,
                 int leaf_id,
                 ElemIdx *scatter_map,
                 ElemIdx *gather_map) {
    build_scatter_map<Geom>(iter_grid, leaf_id, scatter_map);

    nanovdb::Vec3i const origin      = toVec3i(iter_grid.leaf_origin(leaf_id));
    nanovdb::Vec3i const halo_origin = origin * Geom::S() + Geom::offset();

    build_gather_map<Geom>(gather_grid, halo_origin, gather_map);
}

template <typename Geom, typename Tiling, typename ElemIdx>
__device__ void
build_cluster_predicates(ElemIdx const *gather_map,
                         ElemIdx const *scatter_map,
                         nanovdb::Vec3i cluster_orig,
                         bool *gather_pred,
                         bool *scatter_pred) {
    constexpr int CL_HALO_VOL = Tiling::cl_halo_vol;
    constexpr int CL_VOL      = Tiling::cl_vol;
    int const tid             = threadIdx.x;
    int const nthreads        = blockDim.x;

    nanovdb::Vec3i const Sv         = Geom::S();
    nanovdb::Vec3i const cl_halo_d  = Tiling::cl_halo();
    nanovdb::Vec3i const cl_ext     = Tiling::cl_extent();
    nanovdb::Vec3i const halo_dims  = Geom::halo();
    nanovdb::Vec3i const leaf3_dims = Geom::leaf3();

    for (int h = tid; h < CL_HALO_VOL; h += nthreads) {
        nanovdb::Vec3i const h3       = decode3(h, cl_halo_d);
        nanovdb::Vec3i const full_pos = cluster_orig * Sv + h3;
        int const full_idx            = encode3(full_pos, halo_dims);
        gather_pred[h]                = (gather_map[full_idx] != static_cast<ElemIdx>(0));
    }

    for (int p = tid; p < CL_VOL; p += nthreads) {
        nanovdb::Vec3i const p3       = decode3(p, cl_ext);
        nanovdb::Vec3i const full_pos = cluster_orig + p3;
        int const full_idx            = encode3(full_pos, leaf3_dims);
        scatter_pred[p]               = (scatter_map[full_idx] != static_cast<ElemIdx>(0));
    }
}

// ============================================================================
// CuTe gather tensor -- IndirectTensor via ComposedLayout
// ============================================================================
//
// Makes the shared-memory indirection through gather_map invisible to
// the GEMM template: the GEMM sees a standard tensor whose "stride"
// happens to perform an indirect lookup.

template <typename Index> struct IndexedGather {
    CUTE_HOST_DEVICE constexpr IndexedGather(Index const *indices = {}) : indices_(indices) {}

    template <typename I>
    CUTE_HOST_DEVICE constexpr Index
    operator()(I i) const {
        return indices_[i];
    }

    CUTE_HOST_DEVICE friend void
    print(IndexedGather const &) {
        cute::print("Indexed");
    }

    Index const *indices_;
};

template <typename Func, typename Stride> struct CustomStride {
    CUTE_HOST_DEVICE constexpr CustomStride(Func const &func, Stride const &stride)
        : func_(func), stride_(stride) {}

    template <typename I>
    CUTE_HOST_DEVICE constexpr friend auto
    operator*(I i, CustomStride const &s) {
        return s.func_(i) * s.stride_;
    }

    template <typename I>
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

    template <typename Div>
    CUTE_HOST_DEVICE constexpr friend auto
    safe_div(CustomStride const &s, Div const &div) {
        return CustomStride<Func, decltype(safe_div(s.stride_, div))>(s.func_,
                                                                      safe_div(s.stride_, div));
    }

    template <typename Shape>
    CUTE_HOST_DEVICE constexpr friend auto
    make_layout(Shape const &shape, CustomStride const &stride) {
        return Layout<Shape, CustomStride>(shape, stride);
    }

    Func func_;
    Stride stride_;
};

template <typename Stride, typename Func>
CUTLASS_HOST_DEVICE auto
make_custom_stride_layout(Stride const &stride, Func &&func) {
    auto idx        = find_if(stride, [](auto x) { return not is_constant<1, decltype(x)>{}; });
    constexpr int I = decltype(idx)::value;
    return make_layout(
        repeat_like(stride, _1{}),
        replace<I>(stride, CustomStride{static_cast<Func &&>(func), get<I>(stride)}));
}

template <typename Iterator, typename Shape, typename Stride, typename Func>
CUTLASS_HOST_DEVICE auto
make_gather_tensor(Iterator iter, Shape const &shape, Stride const &stride, Func &&func) {
    Layout<Shape, Stride> matrix_layout = make_identity_layout(shape);
    auto offset                         = as_arithmetic_tuple(repeat_like(shape, _0{}));
    auto gather_layout = make_custom_stride_layout(stride, static_cast<Func &&>(func));
    return make_tensor(iter, ComposedLayout{gather_layout, offset, matrix_layout});
}

// ============================================================================
// Scatter epilogue -- predicated scatter-write to global memory
// ============================================================================

template <conv_variant Variant, typename ElemAcc, typename ElemOut, typename ElemIdx>
__device__ void
scatter_write(ElemAcc const *accum_buf,
              int tile_m,
              int blk_vol,
              ElemIdx const *scatter_map,
              nanovdb::Vec3i block_orig,
              nanovdb::Vec3i block_shape,
              nanovdb::Vec3i leaf3,
              ElemOut *output,
              int C_out,
              int m_offset,
              bool const *scatter_pred) {
    int const tid      = threadIdx.x;
    int const nthreads = blockDim.x;
    int const total    = tile_m * blk_vol;

    for (int idx = tid; idx < total; idx += nthreads) {
        int const m = idx / blk_vol;
        int const n = idx % blk_vol;

        if (!scatter_pred[n]) {
            continue;
        }

        nanovdb::Vec3i const v       = decode3(n, block_shape);
        nanovdb::Vec3i const out_pos = block_orig + v;
        int const out_lin            = encode3(out_pos, leaf3);
        ElemIdx const global_idx     = scatter_map[out_lin];

        if (global_idx == static_cast<ElemIdx>(0)) {
            continue;
        }

        int64_t const addr = static_cast<int64_t>(global_idx) * C_out + m_offset + m;
        ElemAcc const val  = accum_buf[m * blk_vol + n];

        if constexpr (variant_traits<Variant>::atomic_accum) {
            atomicAdd(&output[addr], static_cast<ElemOut>(val));
        } else {
            output[addr] = static_cast<ElemOut>(val);
        }
    }
}

// ============================================================================
// Predicated mainloop dispatch policy
// ============================================================================

template <int Stages_> struct MainloopSm80Predicated {
    static constexpr int Stages = Stages_;
    using ArchTag               = cutlass::arch::Sm80;
    using Schedule              = cutlass::gemm::KernelMultistage;
    using ClusterShape          = Shape<_1, _1, _1>;
};

// ============================================================================
// Kernel parameters and shared storage
// ============================================================================

template <typename Types> struct KernelParams {
    using ElemWt   = typename Types::ElemWt;
    using ElemFeat = typename Types::ElemFeat;
    using ElemOut  = typename Types::ElemOut;
    using ElemIdx  = typename Types::ElemIdx;

    GridBatchImpl::Accessor iter_acc;
    GridBatchImpl::Accessor gather_acc;
    int64_t iter_batch_idx;
    int64_t gather_batch_idx;

    ElemFeat const *features;
    ElemWt const *weights;
    ElemOut *output;

    int C_in;
    int C_out;
};

template <typename Geom, typename Tiling, typename Types> struct SharedStorage {
    using ElemIdx = typename Types::ElemIdx;

    ElemIdx gather_map[Geom::halo_vol];
    ElemIdx scatter_map[Geom::leaf_vol];

    bool gather_pred[Tiling::cl_halo_vol];
    bool scatter_pred[Tiling::cl_vol];
};

// ============================================================================
// Kernel
// ============================================================================

template <typename Geom, typename Types, typename Tiling, conv_variant Variant>
__global__ void __launch_bounds__(256)
leaf_igemm_kernel(KernelParams<Types> params) {
    using ElemIdx  = typename Types::ElemIdx;
    using ElemFeat = typename Types::ElemFeat;
    using ElemWt   = typename Types::ElemWt;
    using ElemAcc  = typename Types::ElemAcc;
    using ElemOut  = typename Types::ElemOut;

    int const leaf_id = blockIdx.x;

    auto iter_grid   = make_sparse_grid<ElemIdx>(params.iter_acc, params.iter_batch_idx);
    auto gather_grid = make_sparse_grid<ElemIdx>(params.gather_acc, params.gather_batch_idx);

    extern __shared__ char smem_raw[];
    auto &smem = *reinterpret_cast<SharedStorage<Geom, Tiling, Types> *>(smem_raw);

    // ==== PHASE 1: Build index maps ====

    build_index_maps<Geom, Variant>(
        iter_grid, gather_grid, leaf_id, smem.scatter_map, smem.gather_map);
    __syncthreads();

    // ==== PHASE 2-3: Tiled GEMM with predicated gather/scatter ====

    int const C_in     = params.C_in;
    int const C_out    = params.C_out;
    int const contract = C_in * Geom::kern_vol;
    int const M_total  = C_out;

    constexpr int TILE_M  = Tiling::tile_m;
    constexpr int BLK_VOL = Tiling::blk_vol;

    ElemAcc accum[TILE_M * BLK_VOL];

    nanovdb::Vec3i const B_shape    = Tiling::B();
    nanovdb::Vec3i const leaf3_dims = Geom::leaf3();

    for (int m_tile = 0; m_tile < M_total; m_tile += TILE_M) {
        for (int cl_id = 0; cl_id < Tiling::ncl_tot; ++cl_id) {
            nanovdb::Vec3i const cl_coord = decode3(cl_id, Tiling::ncl());
            nanovdb::Vec3i const cl_orig  = cl_coord * Tiling::cl_extent();

            build_cluster_predicates<Geom, Tiling>(
                smem.gather_map, smem.scatter_map, cl_orig, smem.gather_pred, smem.scatter_pred);
            __syncthreads();

            for (int local_blk = 0; local_blk < Tiling::blocks_per_cl; ++local_blk) {
                nanovdb::Vec3i const blk_in_cl  = decode3(local_blk, Tiling::CL());
                nanovdb::Vec3i const block_orig = cl_orig + blk_in_cl * B_shape;

                for (int i = 0; i < TILE_M * BLK_VOL; ++i) {
                    accum[i] = ElemAcc(0);
                }

                // Inner GEMM loop over contraction tiles.
                // TODO: Replace with CuTe ComposedLayout GEMM pipeline
                // (MainloopSm80Predicated + CollectiveMma) once the full
                // integration is in place.

                int const tid      = threadIdx.x;
                int const nthreads = blockDim.x;

                for (int ck_tile = 0; ck_tile < contract; ck_tile += Tiling::tile_ck) {
                    for (int idx = tid; idx < TILE_M * BLK_VOL; idx += nthreads) {
                        int const m = idx / BLK_VOL;
                        int const n = idx % BLK_VOL;

                        ElemAcc partial = ElemAcc(0);

                        for (int ck_local = 0;
                             ck_local < Tiling::tile_ck && (ck_tile + ck_local) < contract;
                             ++ck_local) {
                            int const ck = ck_tile + ck_local;

                            auto [voxel_idx, c] = im2col_map<Geom, Tiling, Variant>::apply(
                                ck, n, block_orig, smem.gather_map);

                            ElemFeat b_val = ElemFeat(0);
                            if (voxel_idx != ElemIdx(0)) {
                                int64_t const feat_addr =
                                    static_cast<int64_t>(voxel_idx) * C_in + c;
                                b_val = params.features[feat_addr];
                            }

                            int64_t const wt_addr =
                                static_cast<int64_t>(m_tile + m) * contract + ck;
                            ElemWt const a_val = params.weights[wt_addr];

                            partial += static_cast<ElemAcc>(a_val) * static_cast<ElemAcc>(b_val);
                        }

                        accum[idx] += partial;
                    }
                    __syncthreads();
                }

                // Scatter epilogue
                bool block_scatter_pred[BLK_VOL];
                for (int n = 0; n < BLK_VOL; ++n) {
                    nanovdb::Vec3i const v       = decode3(n, B_shape);
                    nanovdb::Vec3i const out_pos = block_orig + v;
                    int const out_lin            = encode3(out_pos, leaf3_dims);
                    block_scatter_pred[n]        = (smem.scatter_map[out_lin] != ElemIdx(0));
                }

                scatter_write<Variant>(accum,
                                       TILE_M,
                                       BLK_VOL,
                                       smem.scatter_map,
                                       block_orig,
                                       B_shape,
                                       leaf3_dims,
                                       params.output,
                                       C_out,
                                       m_tile,
                                       block_scatter_pred);
                __syncthreads();
            }
        }
    }
}

// ============================================================================
// Launch wrapper
// ============================================================================

template <typename Geom, typename Types, typename Tiling, conv_variant Variant>
void
launch_leaf_igemm(KernelParams<Types> const &params, int num_leaves, cudaStream_t stream) {
    if (num_leaves == 0) {
        return;
    }

    constexpr int THREADS = 256;
    constexpr size_t SMEM = sizeof(SharedStorage<Geom, Tiling, Types>);

    if constexpr (SMEM > SMEM_BASELINE_BYTES) {
        cudaFuncSetAttribute(leaf_igemm_kernel<Geom, Types, Tiling, Variant>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(SMEM));
    }

    leaf_igemm_kernel<Geom, Types, Tiling, Variant><<<num_leaves, THREADS, SMEM, stream>>>(params);
}

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

// ============================================================================
// CollectiveMma specialization for predicated B-matrix loads
// ============================================================================
//
// The single modification to a stock CUTLASS SM80 mainloop that makes
// the GEMM sparse-aware: B-matrix global-to-shared copies use copy_if
// gated by a predicate tensor in shared memory.
//
// Must be in cutlass::gemm::collective for CUTLASS template dispatch.
//
// Not yet integrated into the kernel above (the kernel uses an explicit
// loop for now). Included here so the full algorithm is visible in one
// place.

namespace cutlass::gemm::collective {

using namespace cute;

template <int Stages,
          class TileShape_,
          class ElementA_,
          class StrideA_,
          class ElementB_,
          class StrideB_,
          class TiledMma_,
          class GmemTiledCopyA_,
          class SmemLayoutAtomA_,
          class SmemCopyAtomA_,
          class TransformA_,
          class GmemTiledCopyB_,
          class SmemLayoutAtomB_,
          class SmemCopyAtomB_,
          class TransformB_>
struct CollectiveMma<fvdb::detail::leaf_igemm::MainloopSm80Predicated<Stages>,
                     TileShape_,
                     ElementA_,
                     StrideA_,
                     ElementB_,
                     StrideB_,
                     TiledMma_,
                     GmemTiledCopyA_,
                     SmemLayoutAtomA_,
                     SmemCopyAtomA_,
                     TransformA_,
                     GmemTiledCopyB_,
                     SmemLayoutAtomB_,
                     SmemCopyAtomB_,
                     TransformB_> {
    using DispatchPolicy     = MainloopSm80CpAsyncUnpredicated<Stages>;
    using TileShape          = TileShape_;
    using ElementA           = ElementA_;
    using StrideA            = StrideA_;
    using ElementB           = ElementB_;
    using StrideB            = StrideB_;
    using TiledMma           = TiledMma_;
    using ElementAccumulator = typename TiledMma::ValTypeC;
    using GmemTiledCopyA     = GmemTiledCopyA_;
    using GmemTiledCopyB     = GmemTiledCopyB_;
    using SmemLayoutAtomA    = SmemLayoutAtomA_;
    using SmemLayoutAtomB    = SmemLayoutAtomB_;
    using SmemCopyAtomA      = SmemCopyAtomA_;
    using SmemCopyAtomB      = SmemCopyAtomB_;
    using TransformA         = TransformA_;
    using TransformB         = TransformB_;
    using ArchTag            = typename DispatchPolicy::ArchTag;
    using CtaShape_MNK       = TileShape;

    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
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
        array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_a;
        array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_b;
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

    // Pipelined mainloop with predicated B-loads.
    //   A-loads: unpredicated copy (filter weights always present)
    //   B-loads: copy_if (zero-fill for inactive input voxels)
    template <class FrgTensorD,
              class TensorA,
              class TensorB,
              class TensorP,
              class FrgTensorC,
              class KTileIterator,
              class ResidueMNK>
    CUTLASS_DEVICE void
    operator()(FrgTensorD &accum,
               TensorA gA,
               TensorB gB,
               TensorP sP,
               FrgTensorC const &src_accum,
               KTileIterator k_tile_iter,
               int k_tile_count,
               ResidueMNK residue_mnk,
               int thread_idx,
               char *smem_buf) {
        static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
        static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
        static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");
        static_assert(is_smem<TensorP>::value, "Predicate tensor must be smem resident.");
        static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

        SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
        Tensor sA              = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
        Tensor sB              = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

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

        // Prefetch: fill all pipeline stages except the last
        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < Stages - 1; ++k_pipe) {
            copy(gmem_tiled_copy_A, tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, k_pipe));
            copy_if(gmem_tiled_copy_B,
                    tBsP(_, _, _, *k_tile_iter),
                    tBgB(_, _, _, *k_tile_iter),
                    tBsB(_, _, _, k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) {
                ++k_tile_iter;
            }
        }

        // MMA partitioning
        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA  = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB  = thr_mma.partition_fragment_B(sB(_, _, 0));

        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
        Tensor tCsA            = smem_thr_copy_A.partition_S(sA);
        Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);

        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
        Tensor tCsB            = smem_thr_copy_B.partition_S(sB);
        Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);

        // Pipelined main loop
        int smem_pipe_read  = 0;
        int smem_pipe_write = Stages - 1;

        Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
        Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

        auto K_BLOCK_MAX = size<2>(tCrA);

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
                copy(smem_tiled_copy_A,
                     tCsA_p(_, _, k_block_next),
                     tCrA_copy_view(_, _, k_block_next));
                copy(smem_tiled_copy_B,
                     tCsB_p(_, _, k_block_next),
                     tCrB_copy_view(_, _, k_block_next));

                if (k_block == 0) {
                    copy(gmem_tiled_copy_A,
                         tAgA(_, _, _, *k_tile_iter),
                         tAsA(_, _, _, smem_pipe_write));
                    copy_if(gmem_tiled_copy_B,
                            tBsP(_, _, _, *k_tile_iter),
                            tBgB(_, _, _, *k_tile_iter),
                            tBsB(_, _, _, smem_pipe_write));
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

// ============================================================================
// Host entry point
// ============================================================================

namespace fvdb {
namespace detail {
namespace ops {

static bool
deviceSupportsSm80(torch::Device device) {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device.index());
    return major >= 8;
}

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

    using Geom   = Geom3x3x3_S1;
    using Tiling = DefaultTiling<Geom, Types>;

    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const C_in  = features.size(1);
    int64_t const C_out = weights.size(0);

    auto iter_acc   = output_grid.deviceAccessor();
    auto gather_acc = input_grid.deviceAccessor();

    int64_t const total_output_voxels = output_grid.totalVoxels();
    int64_t const num_leaves          = output_grid.totalLeaves();

    auto accum_opts = torch::dtype(c10::CppTypeToScalarType<ElemAcc>::value).device(device);
    auto output     = torch::zeros({total_output_voxels, C_out}, accum_opts);

    if (total_output_voxels == 0 || num_leaves == 0) {
        return output.to(output_dtype);
    }

    auto W = weights.reshape({C_out, -1}).contiguous();

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

torch::Tensor
leafIGemmConv(torch::Tensor features,
              torch::Tensor weights,
              GridBatchImpl const &input_grid,
              GridBatchImpl const &output_grid) {
    TORCH_CHECK(features.is_cuda(), "leafIGemmConv: features must be on CUDA");
    TORCH_CHECK(deviceSupportsSm80(features.device()),
                "leafIGemmConv: requires SM80+ (Ampere or newer)");

    TORCH_CHECK(features.dim() == 2, "leafIGemmConv: features must be 2D [N, C_in]");
    TORCH_CHECK(features.is_contiguous(), "leafIGemmConv: features must be contiguous");

    TORCH_CHECK(weights.dim() == 5, "leafIGemmConv: weights must be 5D [C_out, C_in, k0, k1, k2]");
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
                    "leafIGemmConv: unsupported dtype ",
                    features.scalar_type(),
                    ". Supported: fp32, fp16.");
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
