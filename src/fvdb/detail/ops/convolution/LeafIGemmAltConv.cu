// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// LeafIGemmAltConv.cu -- Leaf-level implicit GEMM sparse convolution (Alt).
//
// Single-file implementation of the algorithm described in
// docs/wip/leaf_level_igemm_sparse_conv.md, following the Sifakis reference
// (sifakis/openvdb ex_sparse_convolution_igemm_nanovdb_cuda).
//
// Abstractions are derived from the C++20 concept/trait layer previously in
// the leaf_igemm/ header directory, unified here for readability.
//
// One threadblock per output leaf. The kernel fuses:
//   Phase 1: Topology densification (gather/scatter map construction)
//   Phase 2-3: Scalar GEMM with predicated gather + scatter epilogue
//
// Target architecture: SM80 (Ampere) and newer.

#include "LeafIGemmAltConv.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <nanovdb/NanoVDB.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_mma_decl.hpp>
#include <cutlass/gemm/collective/sm80_mma_multistage.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

#include <cstddef>
#include <cstdint>

// ============================================================================
// Section 1: Compile-time vocabulary types (core_types)
// ============================================================================

namespace fvdb {
namespace detail {
namespace leaf_igemm_alt {

struct coord3 {
    int x = 0;
    int y = 0;
    int z = 0;
    consteval bool operator==(coord3 const &) const = default;
};

consteval coord3
operator+(coord3 a, coord3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
consteval coord3
operator-(coord3 a, coord3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
consteval coord3
operator*(coord3 a, coord3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
consteval coord3
operator/(coord3 a, coord3 b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}
consteval coord3
operator*(coord3 v, int s) {
    return {v.x * s, v.y * s, v.z * s};
}
consteval coord3
operator/(coord3 v, int s) {
    return {v.x / s, v.y / s, v.z / s};
}
consteval coord3
operator-(coord3 a) {
    return {-a.x, -a.y, -a.z};
}
consteval int
prod(coord3 v) {
    return v.x * v.y * v.z;
}
consteval bool
all_positive(coord3 v) {
    return v.x > 0 && v.y > 0 && v.z > 0;
}
consteval bool
all_divisible(coord3 a, coord3 b) {
    return a.x % b.x == 0 && a.y % b.y == 0 && a.z % b.z == 0;
}
consteval bool
all_leq(coord3 a, coord3 b) {
    return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

template <bool B>
struct consteval_bool_type {
    static consteval bool value() { return B; }
};
using consteval_true_type  = consteval_bool_type<true>;
using consteval_false_type = consteval_bool_type<false>;

// ============================================================================
// Section 2: Spatial geometry trait (conv_geometry)
// ============================================================================

template <coord3 K_, coord3 S_ = coord3{1, 1, 1}, coord3 D_ = coord3{1, 1, 1}, int LEAF_ = 8>
struct conv_geometry {
    static consteval coord3 K() { return K_; }
    static consteval coord3 S() { return S_; }
    static consteval coord3 D() { return D_; }
    static consteval int leaf() { return LEAF_; }

    static consteval coord3 offset() { return -((K_ - coord3{1, 1, 1}) * D_) / 2; }
    static consteval coord3 halo() {
        return coord3{LEAF_ - 1, LEAF_ - 1, LEAF_ - 1} * S_ + (K_ - coord3{1, 1, 1}) * D_ +
               coord3{1, 1, 1};
    }
    static consteval int halo_vol() { return prod(halo()); }
    static consteval int kern_vol() { return prod(K_); }
    static consteval int leaf_vol() { return LEAF_ * LEAF_ * LEAF_; }
    static consteval coord3 leaf3() { return {LEAF_, LEAF_, LEAF_}; }

    static_assert(all_positive(K_), "Kernel size must be positive on all axes");
    static_assert(all_positive(S_), "Stride must be positive on all axes");
    static_assert(all_positive(D_), "Dilation must be positive on all axes");
    static_assert(LEAF_ > 0, "Leaf side length must be positive");
    static_assert(
        all_leq(halo(), coord3{2 * LEAF_ + 1, 2 * LEAF_ + 1, 2 * LEAF_ + 1}),
        "C2 violated: halo exceeds 3-leaf neighborhood");
};

template <typename T>
struct is_conv_geometry : consteval_false_type {};
template <coord3 K, coord3 S, coord3 D, int L>
struct is_conv_geometry<conv_geometry<K, S, D, L>> : consteval_true_type {};
template <typename T>
concept geometry_like = is_conv_geometry<T>::value();

using geom_3x3x3_s1 = conv_geometry<coord3{3, 3, 3}>;
using geom_5x5x5_s1 = conv_geometry<coord3{5, 5, 5}>;
using geom_3x3x3_s2 = conv_geometry<coord3{3, 3, 3}, coord3{2, 2, 2}>;
using geom_1x1x1_s1 = conv_geometry<coord3{1, 1, 1}>;

// ============================================================================
// Section 3: Scalar type traits (conv_types) -- no CuTe dependency
// ============================================================================

template <typename ElemWt_, typename ElemFeat_, typename ElemAcc_, typename ElemOut_, typename ElemIdx_>
struct conv_types {
    using ElemWt   = ElemWt_;
    using ElemFeat = ElemFeat_;
    using ElemAcc  = ElemAcc_;
    using ElemOut  = ElemOut_;
    using ElemIdx  = ElemIdx_;
    static_assert(sizeof(ElemOut_) <= sizeof(ElemAcc_), "C7: ElemOut must not be wider than ElemAcc");
};

template <typename T>
struct is_conv_types : consteval_false_type {};
template <typename Wt, typename Feat, typename Acc, typename Out, typename Idx>
struct is_conv_types<conv_types<Wt, Feat, Acc, Out, Idx>> : consteval_true_type {};
template <typename T>
concept types_like = is_conv_types<T>::value();

using types_f32 = conv_types<float, float, float, float, uint32_t>;

// ============================================================================
// Section 4: Block/cluster tiling trait (conv_tiling)
// ============================================================================

inline constexpr size_t SMEM_BASELINE_BYTES = 48 * 1024;
inline constexpr size_t SMEM_MAX_BYTES      = 164 * 1024;

template <coord3 B_, coord3 CL_ = coord3{1, 1, 1}, int TileM_ = 32, int TileCK_ = 8,
          int Stages_ = 3, geometry_like Geom = geom_3x3x3_s1, types_like Types = types_f32>
struct conv_tiling {
    static consteval coord3 B() { return B_; }
    static consteval coord3 CL() { return CL_; }
    static consteval int tile_m() { return TileM_; }
    static consteval int tile_ck() { return TileCK_; }
    static consteval int stages() { return Stages_; }

    static consteval coord3 nblk() { return Geom::leaf3() / B_; }
    static consteval int nblk_tot() { return prod(nblk()); }
    static consteval int blk_vol() { return prod(B_); }

    static consteval coord3 ncl() { return nblk() / CL_; }
    static consteval int ncl_tot() { return prod(ncl()); }
    static consteval coord3 cl_extent() { return CL_ * B_; }
    static consteval coord3 cl_halo() {
        return (cl_extent() - coord3{1, 1, 1}) * Geom::S() +
               (Geom::K() - coord3{1, 1, 1}) * Geom::D() + coord3{1, 1, 1};
    }
    static consteval int cl_halo_vol() { return prod(cl_halo()); }
    static consteval int cl_vol() { return prod(cl_extent()); }

    static consteval size_t smem_index_maps() {
        return static_cast<size_t>(Geom::halo_vol() + Geom::leaf_vol()) *
               sizeof(typename Types::ElemIdx);
    }
    static consteval size_t smem_predicates() {
        return static_cast<size_t>(cl_halo_vol() + cl_vol());
    }
    static consteval size_t smem_total() {
        return smem_index_maps() + smem_predicates();
    }

    static_assert(all_divisible(Geom::leaf3(), B_), "C1: B must divide leaf on all axes");
    static_assert(all_divisible(nblk(), CL_), "CL must divide nblk on all axes");
    static_assert(all_positive(B_));
    static_assert(all_positive(CL_));
    static_assert(TileM_ > 0);
    static_assert(TileCK_ > 0);
    static_assert(Stages_ >= 2);
    static_assert(smem_total() <= SMEM_MAX_BYTES, "C3: shared memory budget exceeded");
};

template <typename T>
struct is_conv_tiling : consteval_false_type {};
template <coord3 B, coord3 CL, int TM, int TCK, int S, geometry_like G, types_like Ty>
struct is_conv_tiling<conv_tiling<B, CL, TM, TCK, S, G, Ty>> : consteval_true_type {};
template <typename T>
concept tiling_like = is_conv_tiling<T>::value();

template <geometry_like Geom, types_like Types>
using tiling_default = conv_tiling<coord3{4, 2, 2}, coord3{1, 2, 2}, 32, 8, 3, Geom, Types>;

// ============================================================================
// Section 5: Operation variant enum and traits (conv_variant)
// ============================================================================

enum class conv_variant { forward, input_grad, weight_grad, transposed_conv };

template <conv_variant V>
struct variant_traits;

template <>
struct variant_traits<conv_variant::forward> {
    static consteval bool flip() { return false; }
    static consteval bool atomic_accum() { return false; }
};

template <>
struct variant_traits<conv_variant::input_grad> {
    static consteval bool flip() { return true; }
    static consteval bool atomic_accum() { return false; }
};

template <>
struct variant_traits<conv_variant::weight_grad> {
    static consteval bool flip() { return false; }
    static consteval bool atomic_accum() { return true; }
};

template <>
struct variant_traits<conv_variant::transposed_conv> {
    static consteval bool flip() { return true; }
    static consteval bool atomic_accum() { return false; }
};

// ============================================================================
// Section 6: Device-side coordinate arithmetic (from im2col_map.h)
// ============================================================================

struct dcoord3 {
    int x, y, z;
};

__device__ __forceinline__ dcoord3 operator+(dcoord3 a, dcoord3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ __forceinline__ dcoord3 operator-(dcoord3 a, dcoord3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ __forceinline__ dcoord3 operator*(dcoord3 a, dcoord3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }

__device__ __forceinline__ int dencode3(dcoord3 p, dcoord3 d) { return p.x*d.y*d.z + p.y*d.z + p.z; }
__device__ __forceinline__ dcoord3 ddecode3(int i, dcoord3 d) { return {i/(d.y*d.z), (i/d.z)%d.y, i%d.z}; }
__device__ __forceinline__ dcoord3 to_dcoord3(coord3 c) { return {c.x, c.y, c.z}; }

// ============================================================================
// Section 7: im2col map -- virtual im2col address computation
// ============================================================================

template <typename ElemIdx>
struct im2col_result {
    ElemIdx voxel_idx;
    int channel;
};

template <geometry_like Geom, tiling_like Tiling, conv_variant Variant>
struct im2col_map {
    static constexpr bool flip = variant_traits<Variant>::flip();

    template <typename ElemIdx>
    __device__ __forceinline__ static im2col_result<ElemIdx>
    apply(int ck, int n, dcoord3 block_orig, ElemIdx const *gather_map, int kern_vol) {
        int const c     = ck / kern_vol;
        int const k_lin = ck % kern_vol;

        dcoord3 const K = to_dcoord3(Geom::K());
        dcoord3 delta   = ddecode3(k_lin, K);

        if constexpr (flip) {
            dcoord3 const Km1 = {K.x - 1, K.y - 1, K.z - 1};
            delta             = Km1 - delta;
        }

        dcoord3 const B = to_dcoord3(Tiling::B());
        dcoord3 const v = ddecode3(n, B);
        dcoord3 const S = to_dcoord3(Geom::S());
        dcoord3 const D = to_dcoord3(Geom::D());
        dcoord3 const h = (block_orig + v) * S + delta * D;

        int const h_lin         = dencode3(h, to_dcoord3(Geom::halo()));
        ElemIdx const voxel_idx = gather_map[h_lin];

        return {voxel_idx, c};
    }
};

// ============================================================================
// Section 7b: CuTe gather tensor -- ComposedLayout infrastructure
// ============================================================================
//
// Adapted from Sifakis gather_tensor.hpp. Makes the shared-memory
// indirection through gather_map invisible to the GEMM template.

using namespace cute;

template <typename Index>
struct IndexedGather {
    CUTE_HOST_DEVICE constexpr IndexedGather(Index const *indices = {}) : indices_(indices) {}
    template <typename I>
    CUTE_HOST_DEVICE constexpr Index operator()(I i) const { return indices_[i]; }
    CUTE_HOST_DEVICE friend void print(IndexedGather const &) { cute::print("Indexed"); }
    Index const *indices_;
};

template <typename Func, typename Stride>
struct CustomStride {
    CUTE_HOST_DEVICE constexpr CustomStride(Func const &func, Stride const &stride)
        : func_(func), stride_(stride) {}
    template <typename I>
    CUTE_HOST_DEVICE constexpr friend auto operator*(I i, CustomStride const &s) {
        return s.func_(i) * s.stride_;
    }
    template <typename I>
    CUTE_HOST_DEVICE constexpr friend auto operator*(CustomStride const &s, I i) {
        return s.func_(i) * s.stride_;
    }
    CUTE_HOST_DEVICE friend void print(CustomStride const &s) {
        cute::print("Custom{"); print(s.func_); cute::print(","); print(s.stride_); cute::print("}");
    }
    template <typename Div>
    CUTE_HOST_DEVICE constexpr friend auto safe_div(CustomStride const &s, Div const &div) {
        return CustomStride<Func, decltype(safe_div(s.stride_, div))>(
            s.func_, safe_div(s.stride_, div));
    }
    template <typename Shape>
    CUTE_HOST_DEVICE constexpr friend auto make_layout(Shape const &shape, CustomStride const &stride) {
        return Layout<Shape, CustomStride>(shape, stride);
    }
    Func func_;
    Stride stride_;
};

template <typename Stride, typename Func>
CUTLASS_HOST_DEVICE auto
make_custom_stride_layout(Stride const &stride, Func &&func) {
    auto idx = find_if(stride, [](auto x) { return not is_constant<1, decltype(x)>{}; });
    constexpr int I = decltype(idx)::value;
    return make_layout(
        repeat_like(stride, _1{}),
        replace<I>(stride, CustomStride{static_cast<Func &&>(func), get<I>(stride)}));
}

template <typename Iterator, typename Shape, typename Stride, typename Func>
CUTLASS_HOST_DEVICE auto
make_gather_tensor(Iterator iter, Shape const &shape, Stride const &stride, Func &&func) {
    Layout<Shape, Stride> matrix_layout = make_identity_layout(shape);
    auto offset        = as_arithmetic_tuple(repeat_like(shape, _0{}));
    auto gather_layout = make_custom_stride_layout(stride, static_cast<Func &&>(func));
    return make_tensor(iter, ComposedLayout{gather_layout, offset, matrix_layout});
}

} // namespace leaf_igemm_alt
} // namespace detail
} // namespace fvdb

// ComposedLayout upcast overload (must be in namespace cute)
namespace cute {

template <int N, typename Shape_A, typename Stride_A, typename Offset, typename Shape, typename Stride>
CUTE_HOST_DEVICE constexpr auto
upcast(ComposedLayout<Layout<Shape_A, Stride_A>, Offset, Layout<Shape, Stride>> const &layout) {
    auto idx = find_if(layout.layout_a().stride(), [](auto x) {
        return is_constant<1, decltype(x)>{};
    });
    constexpr int I = decltype(idx)::value;
    auto outer  = upcast<N>(layout.layout_a());
    auto offset = as_arithmetic_tuple(
        replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));
    auto inner = upcast<N>(layout.layout_b().shape(), layout.layout_b().stride());
    return composition(outer, offset, inner);
}

} // namespace cute

// ============================================================================
// Section 7c: Predicated mainloop dispatch policy + CollectiveMma
// ============================================================================

namespace fvdb {
namespace detail {
namespace leaf_igemm_alt {

template <int Stages_>
struct MainloopSm80Predicated {
    static constexpr int Stages = Stages_;
    using ArchTag    = cutlass::arch::Sm80;
    using Schedule   = cutlass::gemm::KernelMultistage;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
};

} // namespace leaf_igemm_alt
} // namespace detail
} // namespace fvdb

namespace cutlass::gemm::collective {
using namespace cute;

template <int Stages, class TileShape_, class ElementA_, class StrideA_, class ElementB_,
          class StrideB_, class TiledMma_, class GmemTiledCopyA_, class SmemLayoutAtomA_,
          class SmemCopyAtomA_, class TransformA_, class GmemTiledCopyB_, class SmemLayoutAtomB_,
          class SmemCopyAtomB_, class TransformB_>
struct CollectiveMma<fvdb::detail::leaf_igemm_alt::MainloopSm80Predicated<Stages>, TileShape_,
                     ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_, GmemTiledCopyA_,
                     SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_,
                     SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_> {
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

    static_assert(rank(SmemLayoutAtomA{}) == 2);
    static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0);
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0);
    static_assert(rank(SmemLayoutAtomB{}) == 2);
    static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0);
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0);

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<Stages>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<Stages>{})));

    static_assert(DispatchPolicy::Stages >= 2);

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

    template <class FrgTensorD, class TensorA, class TensorB, class TensorP, class FrgTensorC,
              class KTileIterator, class ResidueMNK>
    CUTLASS_DEVICE void
    operator()(FrgTensorD &accum, TensorA gA, TensorB gB, TensorP sP,
               FrgTensorC const &src_accum, KTileIterator k_tile_iter, int k_tile_count,
               ResidueMNK residue_mnk, int thread_idx, char *smem_buf) {
        static_assert(is_rmem<FrgTensorD>::value);
        static_assert(is_gmem<TensorA>::value);
        static_assert(is_gmem<TensorB>::value);
        static_assert(is_smem<TensorP>::value);
        static_assert(is_rmem<FrgTensorC>::value);

        SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
        Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

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

        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < Stages - 1; ++k_pipe) {
            copy(gmem_tiled_copy_A, tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, k_pipe));
            copy_if(gmem_tiled_copy_B, tBsP(_, _, _, *k_tile_iter),
                    tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_iter; }
        }

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA  = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB  = thr_mma.partition_fragment_B(sB(_, _, 0));

        auto smem_tiled_copy_A  = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A    = smem_tiled_copy_A.get_thread_slice(thread_idx);
        Tensor tCsA             = smem_thr_copy_A.partition_S(sA);
        Tensor tCrA_copy_view   = smem_thr_copy_A.retile_D(tCrA);

        auto smem_tiled_copy_B  = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B    = smem_tiled_copy_B.get_thread_slice(thread_idx);
        Tensor tCsB             = smem_thr_copy_B.partition_S(sB);
        Tensor tCrB_copy_view   = smem_thr_copy_B.retile_D(tCrB);

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
                    if (k_tile_count > 0) { ++k_tile_iter; }
                    smem_pipe_write = smem_pipe_read;
                    ++smem_pipe_read;
                    smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
                }
                transform(tCrA(_, _, k_block), TransformA{});
                transform(tCrB(_, _, k_block), TransformB{});
                cute::gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrB(_, _, k_block), src_accum);
            });
        }
        cp_async_wait<0>();
        __syncthreads();
    }
};

} // namespace cutlass::gemm::collective

// ============================================================================
// Resume leaf_igemm_alt namespace
// ============================================================================

namespace fvdb {
namespace detail {
namespace leaf_igemm_alt {

// ============================================================================
// Section 8: NanoVDB sparse grid adapter (from sparse_grid.h)
// ============================================================================

template <typename ElemIdx>
struct nanovdb_sparse_grid {
    using grid_type = nanovdb::OnIndexGrid;

    grid_type const *grid_ptr;
    int64_t voxel_offset;

    __device__ int
    leaf_count() const {
        return static_cast<int>(grid_ptr->tree().nodeCount(0));
    }

    __device__ dcoord3
    leaf_origin(int leaf_id) const {
        auto const orig = grid_ptr->tree().template getFirstNode<0>()[leaf_id].origin();
        return {orig[0], orig[1], orig[2]};
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
    coord_to_idx(dcoord3 coord) const {
        auto acc = grid_ptr->getAccessor();
        nanovdb::Coord const c(coord.x, coord.y, coord.z);
        if (!acc.isActive(c)) {
            return static_cast<ElemIdx>(0);
        }
        return static_cast<ElemIdx>(voxel_offset + acc.getValue(c));
    }
};

template <typename ElemIdx>
__device__ nanovdb_sparse_grid<ElemIdx>
make_sparse_grid(GridBatchImpl::Accessor const &acc, int64_t batch_idx) {
    auto const *grid = acc.grid(batch_idx);
    return {grid, acc.voxelOffset(batch_idx)};
}

// ============================================================================
// Section 9: Phase 1 -- cooperative index map construction
// ============================================================================

template <geometry_like Geom, typename ElemIdx, typename SparseGrid>
__device__ void
build_scatter_map(SparseGrid const &iter_grid, int leaf_id, ElemIdx *scatter_map) {
    constexpr int LEAF_VOL = Geom::leaf_vol();
    for (int p = threadIdx.x; p < LEAF_VOL; p += blockDim.x) {
        scatter_map[p] = iter_grid.leaf_value(leaf_id, p);
    }
}

template <geometry_like Geom, typename ElemIdx, typename SparseGrid>
__device__ void
build_gather_map(SparseGrid const &gather_grid, dcoord3 halo_origin, ElemIdx *gather_map) {
    constexpr int HALO_VOL = Geom::halo_vol();
    dcoord3 const halo_dims = to_dcoord3(Geom::halo());
    for (int h = threadIdx.x; h < HALO_VOL; h += blockDim.x) {
        dcoord3 const h3    = ddecode3(h, halo_dims);
        dcoord3 const coord = halo_origin + h3;
        gather_map[h]       = gather_grid.coord_to_idx(coord);
    }
}

template <geometry_like Geom, conv_variant Variant, typename ElemIdx, typename SparseGrid>
__device__ void
build_index_maps(SparseGrid const &iter_grid,
                 SparseGrid const &gather_grid,
                 int leaf_id,
                 ElemIdx *scatter_map,
                 ElemIdx *gather_map) {
    build_scatter_map<Geom>(iter_grid, leaf_id, scatter_map);

    dcoord3 const origin      = iter_grid.leaf_origin(leaf_id);
    dcoord3 const S           = to_dcoord3(Geom::S());
    dcoord3 const off         = to_dcoord3(Geom::offset());
    dcoord3 const halo_origin = origin * S + off;

    build_gather_map<Geom>(gather_grid, halo_origin, gather_map);
}

// ============================================================================
// Section 10: Cluster predicate construction
// ============================================================================

template <geometry_like Geom, tiling_like Tiling, typename ElemIdx>
__device__ void
build_cluster_predicates(ElemIdx const *gather_map,
                         ElemIdx const *scatter_map,
                         dcoord3 cluster_orig,
                         bool *gather_pred,
                         bool *scatter_pred) {
    constexpr int CL_HALO_VOL = Tiling::cl_halo_vol();
    constexpr int CL_VOL      = Tiling::cl_vol();

    dcoord3 const S         = to_dcoord3(Geom::S());
    dcoord3 const cl_halo   = to_dcoord3(Tiling::cl_halo());
    dcoord3 const cl_extent = to_dcoord3(Tiling::cl_extent());
    dcoord3 const halo_dims = to_dcoord3(Geom::halo());
    dcoord3 const leaf3     = to_dcoord3(Geom::leaf3());

    for (int h = threadIdx.x; h < CL_HALO_VOL; h += blockDim.x) {
        dcoord3 const h3       = ddecode3(h, cl_halo);
        dcoord3 const full_pos = cluster_orig * S + h3;
        int const full_idx     = dencode3(full_pos, halo_dims);
        gather_pred[h]         = (gather_map[full_idx] != static_cast<ElemIdx>(0));
    }

    for (int p = threadIdx.x; p < CL_VOL; p += blockDim.x) {
        dcoord3 const p3       = ddecode3(p, cl_extent);
        dcoord3 const full_pos = cluster_orig + p3;
        int const full_idx     = dencode3(full_pos, leaf3);
        scatter_pred[p]        = (scatter_map[full_idx] != static_cast<ElemIdx>(0));
    }
}

// ============================================================================
// Section 11: Scatter epilogue
// ============================================================================

template <conv_variant Variant, typename ElemAcc, typename ElemOut, typename ElemIdx>
__device__ void
scatter_write(ElemAcc const *accum_buf,
              int tile_m,
              int blk_vol,
              ElemIdx const *scatter_map,
              dcoord3 block_orig,
              dcoord3 block_shape,
              dcoord3 leaf3,
              ElemOut *output,
              int C_out,
              int m_offset,
              bool const *scatter_pred) {
    int const total = tile_m * blk_vol;

    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int const m = idx / blk_vol;
        int const n = idx % blk_vol;

        if (!scatter_pred[n])
            continue;

        dcoord3 const v          = ddecode3(n, block_shape);
        dcoord3 const out_pos    = block_orig + v;
        int const out_lin        = dencode3(out_pos, leaf3);
        ElemIdx const global_idx = scatter_map[out_lin];

        if (global_idx == static_cast<ElemIdx>(0))
            continue;

        int64_t const addr = static_cast<int64_t>(global_idx - 1) * C_out + m_offset + m;
        ElemAcc const val  = accum_buf[m * blk_vol + n];

        if constexpr (variant_traits<Variant>::atomic_accum()) {
            atomicAdd(&output[addr], static_cast<ElemOut>(val));
        } else {
            output[addr] = static_cast<ElemOut>(val);
        }
    }
}

// ============================================================================
// Section 11b: igemm_layouts -- CuTe ComposedLayout tensor construction
// ============================================================================
//
// Mirrors the Sifakis IGEMM_Layouts, parameterized through conv_geometry
// and conv_tiling traits. Constructs the CuTe ComposedLayout for the
// B-matrix (activation gather) using IndexedGather + CustomStride.
//
// The inner layout maps logical coordinates ((block, voxel), (channel, kernel))
// to a 2D space: (halo_index, channel_offset). The outer layout applies
// IndexedGather to resolve halo_index through the shared-memory gather_map.

template <geometry_like Geom, tiling_like Tiling, int Cin>
struct igemm_layouts {
    static constexpr auto T  = Int<Geom::K().x>{};
    static constexpr auto R  = Int<Geom::K().y>{};
    static constexpr auto S  = Int<Geom::K().z>{};
    static constexpr auto Z  = Int<Tiling::B().x>{};
    static constexpr auto P  = Int<Tiling::B().y>{};
    static constexpr auto Q  = Int<Tiling::B().z>{};
    static constexpr auto C  = Int<Cin>{};
    static constexpr auto Bx = Int<Geom::leaf() / Tiling::B().x>{};
    static constexpr auto By = Int<Geom::leaf() / Tiling::B().y>{};
    static constexpr auto Bz = Int<Geom::leaf() / Tiling::B().z>{};
    static constexpr auto Hx = Int<Geom::halo().x>{};
    static constexpr auto Hy = Int<Geom::halo().y>{};
    static constexpr auto Hz = Int<Geom::halo().z>{};

    // Activation (B-matrix) composed gather layout.
    // Returns a ComposedLayout that maps:
    //   ((block_x, block_y, block_z, voxel_z, voxel_p, voxel_q),
    //    (channel, kernel_t, kernel_r, kernel_s))
    //   -> features[gather_map[halo_idx] * C + channel]
    template <typename ElemIdx>
    __hostdev__ static auto
    activationComposedGatherLayout(ElemIdx const *gather_map) {
        auto EG = E<0>{};
        auto EC = E<1>{};

        auto inner = make_layout(
            make_shape(
                make_shape(make_shape(Bx, By, Bz), Z, P, Q),
                make_shape(C, T, R, S)),
            make_stride(
                make_stride(
                    make_stride(Hy * Hz * Z * EG, Hz * P * EG, Q * EG),
                    Hy * Hz * EG, Hz * EG, EG),
                make_stride(EC, Hy * Hz * EG, Hz * EG, EG)));

        auto outer = make_layout(
            make_shape(_1{}, _1{}),
            make_stride(CustomStride{IndexedGather<ElemIdx>{gather_map}, C}, _1{}));

        return composition(outer, make_arithmetic_tuple(_0{}, _0{}), inner);
    }

    // Activation index layout (same shape as above, but returns halo indices
    // directly -- used for predicate construction).
    __hostdev__ static auto
    activationIndexLayout() {
        return make_layout(
            make_shape(
                make_shape(make_shape(Bx, By, Bz), Z, P, Q),
                make_shape(C, T, R, S)),
            make_stride(
                make_stride(
                    make_stride(Hy * Hz * Z, Hz * P, Q),
                    Hy * Hz, Hz, _1{}),
                make_stride(_0{}, Hy * Hz, Hz, _1{})));
    }

    // Filter (A-matrix) layout: ordered so that C is contiguous.
    template <int Cout>
    __hostdev__ static auto
    filterLayout() {
        return make_ordered_layout(
            make_shape(Int<Cout>{}, make_shape(C, T, R, S)),
            cute::tuple<_1, cute::tuple<_0, _4, _3, _2>>{});
    }

    // Cluster activation predicate stride (channels collapsed)
    __hostdev__ static auto
    clusterActivationPredicateStride() {
        static constexpr auto CHy = Int<Tiling::cl_halo().y>{};
        static constexpr auto CHz = Int<Tiling::cl_halo().z>{};
        return make_stride(
            make_stride(
                make_stride(CHy * CHz * Z, CHz * P, Q),
                CHy * CHz, CHz, _1{}),
            make_stride(_0{}, _0{}, _0{}, _0{}));
    }

    // Output composed scatter layout
    template <typename ElemIdx>
    __hostdev__ static auto
    outputComposedScatterLayout(ElemIdx const *scatter_map) {
        auto ES = E<0>{};
        auto EC = E<1>{};
        auto K_ = Int<64>{}; // placeholder, actual K comes from template
        return make_layout(
            make_shape(_1{}, _1{}),
            make_stride(_1{}, _1{}));
    }

    // Output index layout
    __hostdev__ static auto
    outputIndexLayout() {
        return make_layout(
            make_shape(Int<1>{}, make_shape(make_shape(Bx, By, Bz), Z, P, Q)),
            make_stride(_0{}, make_stride(
                make_stride(Int<64>{} * Z, Int<8>{} * P, Q),
                Int<64>{}, Int<8>{}, _1{})));
    }

    // Cluster output predicate stride
    __hostdev__ static auto
    clusterOutputPredicateStride() {
        static constexpr auto CVy = Int<Tiling::cl_extent().y>{};
        static constexpr auto CVz = Int<Tiling::cl_extent().z>{};
        return make_stride(
            _0{},
            make_stride(
                make_stride(CVy * CVz * Z, CVz * P, Q),
                CVy * CVz, CVz, _1{}));
    }
};

// ============================================================================
// Section 12: Kernel parameters
// ============================================================================

template <typename Types>
struct KernelParams {
    using ElemFeat = typename Types::ElemFeat;
    using ElemWt   = typename Types::ElemWt;
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

// ============================================================================
// Section 12b: Shared storage
// ============================================================================

template <geometry_like Geom, tiling_like Tiling, types_like Types>
struct SharedStorage {
    using ElemIdx = typename Types::ElemIdx;
    ElemIdx gather_map[Geom::halo_vol()];
    ElemIdx scatter_map[Geom::leaf_vol()];
    bool gather_pred[Tiling::cl_halo_vol()];
    bool scatter_pred[Tiling::cl_vol()];
};

// ============================================================================
// Section 13: The kernel (scalar GEMM -- correct reference implementation)
// ============================================================================

template <geometry_like Geom, types_like Types, tiling_like Tiling, conv_variant Variant>
__global__ void __launch_bounds__(256)
    leaf_igemm_alt_kernel(KernelParams<Types> params) {
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

    build_index_maps<Geom, Variant>(iter_grid, gather_grid, leaf_id,
                                    smem.scatter_map, smem.gather_map);
    __syncthreads();

    int const C_in     = params.C_in;
    int const C_out    = params.C_out;
    int const contract = C_in * Geom::kern_vol();

    constexpr int TILE_M  = Tiling::tile_m();
    constexpr int BLK_VOL = Tiling::blk_vol();

    ElemAcc accum[TILE_M * BLK_VOL];

    dcoord3 const B_shape = to_dcoord3(Tiling::B());
    dcoord3 const leaf3d  = to_dcoord3(Geom::leaf3());

    for (int m_tile = 0; m_tile < C_out; m_tile += TILE_M) {
        for (int cl_id = 0; cl_id < Tiling::ncl_tot(); ++cl_id) {
            dcoord3 const cl_coord = ddecode3(cl_id, to_dcoord3(Tiling::ncl()));
            dcoord3 const cl_orig  = cl_coord * to_dcoord3(Tiling::cl_extent());

            build_cluster_predicates<Geom, Tiling>(smem.gather_map, smem.scatter_map, cl_orig,
                                                   smem.gather_pred, smem.scatter_pred);
            __syncthreads();

            constexpr int BLOCKS_PER_CL = prod(Tiling::CL());
            for (int local_blk = 0; local_blk < BLOCKS_PER_CL; ++local_blk) {
                dcoord3 const blk_in_cl  = ddecode3(local_blk, to_dcoord3(Tiling::CL()));
                dcoord3 const block_orig = cl_orig + blk_in_cl * B_shape;

                int const actual_m = (m_tile + TILE_M <= C_out) ? TILE_M : (C_out - m_tile);

                for (int i = 0; i < actual_m * BLK_VOL; ++i)
                    accum[i] = ElemAcc(0);

                int const tid      = threadIdx.x;
                int const nthreads = blockDim.x;

                for (int ck_tile = 0; ck_tile < contract; ck_tile += Tiling::tile_ck()) {
                    for (int idx = tid; idx < actual_m * BLK_VOL; idx += nthreads) {
                        int const m = idx / BLK_VOL;
                        int const n = idx % BLK_VOL;
                        ElemAcc partial = ElemAcc(0);
                        for (int ck_local = 0;
                             ck_local < Tiling::tile_ck() && (ck_tile + ck_local) < contract;
                             ++ck_local) {
                            int const ck = ck_tile + ck_local;
                            auto [voxel_idx, c] =
                                im2col_map<Geom, Tiling, Variant>::apply(
                                    ck, n, block_orig, smem.gather_map, Geom::kern_vol());
                            ElemFeat b_val = ElemFeat(0);
                            if (voxel_idx != ElemIdx(0)) {
                                b_val = params.features[static_cast<int64_t>(voxel_idx - 1) * C_in + c];
                            }
                            ElemWt const a_val = params.weights[static_cast<int64_t>(m_tile + m) * contract + ck];
                            partial += static_cast<ElemAcc>(a_val) * static_cast<ElemAcc>(b_val);
                        }
                        accum[idx] += partial;
                    }
                    __syncthreads();
                }

                bool block_scatter_pred[BLK_VOL];
                for (int n = 0; n < BLK_VOL; ++n) {
                    dcoord3 const v       = ddecode3(n, B_shape);
                    dcoord3 const out_pos = block_orig + v;
                    block_scatter_pred[n] = (smem.scatter_map[dencode3(out_pos, leaf3d)] != ElemIdx(0));
                }
                scatter_write<Variant>(accum, actual_m, BLK_VOL, smem.scatter_map, block_orig,
                                       B_shape, leaf3d, params.output, C_out, m_tile,
                                       block_scatter_pred);
                __syncthreads();
            }
        }
    }
}

// ============================================================================
// Section 13b: CuTe GEMM kernel -- Sifakis AmperePredicatedFprop pipeline
// ============================================================================
//
// Uses SM80 tensor core MMA, cp.async 3-stage pipeline, ComposedLayout
// virtual im2col, and predicated B-loads via CollectiveMma.
// Requires compile-time Cin, Cout (multiples of 32).

template <geometry_like Geom, types_like Types, tiling_like Tiling,
          conv_variant Variant, int Cin, int Cout>
struct igemm_cute_op {
    using ElemIdx = typename Types::ElemIdx;
    using Layouts = igemm_layouts<Geom, Tiling, Cin>;

    using ElementFlt = cute::tfloat32_t;
    using ElementAct = cute::tfloat32_t;
    using ElementOut = float;

    static constexpr auto T = Int<Geom::K().x>{};
    static constexpr auto R = Int<Geom::K().y>{};
    static constexpr auto S = Int<Geom::K().z>{};
    static constexpr auto Z = Int<Tiling::B().x>{};
    static constexpr auto P = Int<Tiling::B().y>{};
    static constexpr auto Q = Int<Tiling::B().z>{};
    static constexpr auto C = Int<Cin>{};
    static constexpr auto K = Int<Cout>{};

    static constexpr int TilerCv = (Cin < 32) ? Cin : 32;
    static constexpr int TilerKv = (Cout < 32) ? Cout : 32;

    static constexpr auto Tiler_C_val = Int<TilerCv>{};
    static constexpr auto Tiler_K_val = Int<TilerKv>{};

    using TileM = Int<TilerKv>;
    using TileN = Shape<Int<Tiling::B().x>, Int<Tiling::B().y>, Int<Tiling::B().z>>;
    using TileK = Shape<Int<TilerCv>, Int<Geom::K().x>, Int<Geom::K().y>, Int<Geom::K().z>>;
    using PIPE  = _3;

    using TilerFlt = Shape<TileM, TileK>;
    using TilerAct = Shape<TileN, TileK>;
    using TilerOut = Shape<TileM, TileN>;

    static constexpr int TileSizeMv = TilerKv;
    static constexpr int TileSizeNv = Tiling::blk_vol();
    static constexpr int TileSizeKv = TilerCv * Geom::kern_vol();

    using TileSizeM = Int<TileSizeMv>;
    using TileSizeN = Int<TileSizeNv>;
    using TileSizeK = Int<TileSizeKv>;

    using ClusterShape = Shape<
        Int<Geom::leaf() / (Tiling::CL().x * Tiling::B().x)>,
        Int<Geom::leaf() / (Tiling::CL().y * Tiling::B().y)>,
        Int<Geom::leaf() / (Tiling::CL().z * Tiling::B().z)>>;

    using HaloLayout = decltype(make_layout(
        Shape<Int<Geom::halo().x>, Int<Geom::halo().y>, Int<Geom::halo().z>>{},
        GenRowMajor{}));

    using ClusterHaloLayout = decltype(make_layout(
        Shape<Int<Tiling::cl_halo().x>, Int<Tiling::cl_halo().y>, Int<Tiling::cl_halo().z>>{},
        GenRowMajor{}));

    using ClusterVoxelLayout = decltype(make_layout(
        Shape<Int<Tiling::cl_extent().x>, Int<Tiling::cl_extent().y>, Int<Tiling::cl_extent().z>>{},
        GenRowMajor{}));

    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
        Layout<Shape<_2, _2, _1>>,
        Tile<_32, _32, _8>>;

    static constexpr int MaxThreadsPerBlock = size(TiledMma{});
    static constexpr int MinBlocksPerMultiprocessor = 1;

    using GmemTiledCopyFlt = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementFlt>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _4>>{}));

    using SmemLayoutAtomFlt = decltype(
        composition(Swizzle<1, 2, 3>{},
                    Layout<Shape<_8, Shape<_4, _8>>, Stride<_4, Stride<_1, _32>>>{}));

    using SmemCopyAtomFlt = Copy_Atom<UniversalCopy<float>, float>;

    using GmemTiledCopyAct = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementAct>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _4>>{}));

    using SmemLayoutAtomAct = decltype(
        composition(Swizzle<1, 2, 3>{},
                    Layout<Shape<_8, Shape<_4, _8>>, Stride<_4, Stride<_1, _32>>>{}));

    using SmemCopyAtomAct = Copy_Atom<UniversalCopy<float>, float>;

    using GmemTiledCopyOut = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint32_t>, ElementAct>{},
        Layout<Shape<_16, _8>, Stride<_1, _16>>{},
        Layout<Shape<_1, _1>>{}));

    using SmemCopyAtomOut = Copy_Atom<UniversalCopy<float>, ElementOut>;

    using SmemLayoutOut = Layout<Shape<TileSizeM, TileSizeN>>;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
        MainloopSm80Predicated<PIPE::value>,
        Shape<TileSizeM, TileSizeN, TileSizeK>,
        ElementFlt, Underscore,
        ElementAct, Underscore,
        TiledMma,
        GmemTiledCopyFlt, SmemLayoutAtomFlt, SmemCopyAtomFlt, cute::identity,
        GmemTiledCopyAct, SmemLayoutAtomAct, SmemCopyAtomAct, cute::identity>;

    struct KernelSharedStorage {
        union {
            struct {
                ElementFlt sAMatrix[size(TileM{}) * size(TileK{}) * size(PIPE{})];
                ElementAct sBMatrix[size(TileN{}) * size(TileK{}) * size(PIPE{})];
            } mainloop;
            struct {
                ElementOut sCMatrix[size(TileM{}) * size(TileN{})];
            } epilogue;
        };
        ElemIdx sBIdxMatrix[Geom::halo_vol()];
        ElemIdx sCIdxMatrix[Geom::leaf_vol()];
        bool sBPredMatrix[Tiling::cl_halo_vol()];
        bool sCPredMatrix[Tiling::cl_vol()];
    };

    __device__ void
    operator()(Tensor<ElementFlt const *, decltype(Layouts::template filterLayout<Cout>())> mFlt,
               GridBatchImpl::Accessor const &iter_acc,
               GridBatchImpl::Accessor const &gather_acc,
               int64_t batch_idx,
               float const *actData,
               float *outData,
               char *smem_buf) const {
        int leafID = blockIdx.x;

        auto iter_grid   = make_sparse_grid<ElemIdx>(iter_acc, batch_idx);
        auto gather_grid = make_sparse_grid<ElemIdx>(gather_acc, batch_idx);

        auto sCIdx_ptr = &reinterpret_cast<KernelSharedStorage *>(smem_buf)->sCIdxMatrix[0];
        for (int v = threadIdx.x; v < Geom::leaf_vol(); v += MaxThreadsPerBlock)
            sCIdx_ptr[v] = iter_grid.leaf_value(leafID, v);

        auto sBIdx_ptr = &reinterpret_cast<KernelSharedStorage *>(smem_buf)->sBIdxMatrix[0];
        dcoord3 const origin = iter_grid.leaf_origin(leafID);
        dcoord3 const off    = to_dcoord3(Geom::offset());
        dcoord3 const filterOrigin = {origin.x + off.x, origin.y + off.y, origin.z + off.z};

        for (int v = threadIdx.x; v < Geom::halo_vol(); v += MaxThreadsPerBlock) {
            auto [i, j, k] = idx2crd(v, shape(HaloLayout{}), stride(HaloLayout{}));
            dcoord3 coord   = {filterOrigin.x + i, filterOrigin.y + j, filterOrigin.z + k};
            sBIdx_ptr[v]    = gather_grid.coord_to_idx(coord);
        }

        __syncthreads();

        // Adjust feature pointer: gather_map stores 1-based NanoVDB values.
        // Offset by -Cin so that gather_map[h] * Cin accesses the correct 0-based row.
        Tensor gAct    = make_tensor(make_gmem_ptr(actData - Cin),
                                     Layouts::activationComposedGatherLayout(sBIdx_ptr));
        Tensor sActIdx = make_tensor(make_smem_ptr(sBIdx_ptr),
                                     Layouts::activationIndexLayout());

        // Output uses 1-based scatter indices; offset by -Cout.
        Tensor gOut    = make_tensor(make_gmem_ptr(outData - Cout),
                                     Layouts::template outputComposedScatterLayout(sCIdx_ptr));
        Tensor sOutIdx = make_tensor(make_smem_ptr(sCIdx_ptr),
                                     Layouts::outputIndexLayout());

        TiledMma tiled_mma;
        Tensor accum = partition_fragment_C(tiled_mma, TilerOut{});

        Tensor gA_mk    = local_tile(mFlt, TilerFlt{}, make_coord(_, _));
        Tensor gB_nk    = local_tile(gAct, TilerAct{}, make_coord(_, _));
        Tensor sBIdx_nk = local_tile(sActIdx, TilerAct{}, make_coord(_, _));
        Tensor gC_mn    = local_tile(gOut, TilerOut{}, make_coord(_, _));
        Tensor sCIdx_mn = local_tile(sOutIdx, TilerOut{}, make_coord(_, _));

        for (int m_coord = 0; m_coord < size<2>(gA_mk); ++m_coord) {
            for (int clusterID = 0; clusterID < size(ClusterShape{}); ++clusterID) {
                clear(accum);

                auto clusterCoord = idx2crd(clusterID, ClusterShape{});
                auto n_coord      = make_tuple(clusterCoord, _0{}, _0{}, _0{});

                Tensor gA    = gA_mk(_, _, m_coord, _);
                Tensor gB    = gB_nk(_, _, n_coord, _);
                Tensor sBIdx = sBIdx_nk(_, _, n_coord, _);
                Tensor gC    = gC_mn(_, _, m_coord, n_coord);
                Tensor sCIdx = sCIdx_mn(_, _, m_coord, n_coord);

                // Build gather predicate
                auto sBPred_ptr = &reinterpret_cast<KernelSharedStorage *>(smem_buf)->sBPredMatrix[0];
                Tensor sBPred   = make_tensor(make_smem_ptr(sBPred_ptr), shape(sBIdx),
                                              Layouts::clusterActivationPredicateStride());
                for (int v = threadIdx.x; v < Tiling::cl_halo_vol(); v += MaxThreadsPerBlock) {
                    if (v < Tiling::cl_halo_vol()) {
                        auto [i, j, k] = idx2crd(v, shape(ClusterHaloLayout{}),
                                                  stride(ClusterHaloLayout{}));
                        auto coord     = make_tuple(
                            make_tuple(make_tuple(0, 0, 0), i, j, k),
                            make_tuple(0, 0, 0, 0));
                        sBPred(coord) = sBIdx(coord);
                    }
                }

                // Build scatter predicate
                auto sCPred_ptr = &reinterpret_cast<KernelSharedStorage *>(smem_buf)->sCPredMatrix[0];
                Tensor sCPred   = make_tensor(make_smem_ptr(sCPred_ptr), shape(sCIdx),
                                              Layouts::clusterOutputPredicateStride());
                for (int v = threadIdx.x; v < Tiling::cl_vol(); v += MaxThreadsPerBlock) {
                    if (v < Tiling::cl_vol()) {
                        auto [i, j, k] = idx2crd(v, shape(ClusterVoxelLayout{}),
                                                  stride(ClusterVoxelLayout{}));
                        auto coord     = make_tuple(0, make_tuple(make_tuple(0, 0, 0), i, j, k));
                        sCPred(coord) = sCIdx(coord);
                    }
                }

                __syncthreads();

                auto k_tile_iter  = cute::make_coord_iterator(size<2>(gA));
                int  k_tile_count = size<2>(gA);

                CollectiveMainloop collective_mma;
                collective_mma(accum, gA, gB, sBPred, accum,
                               k_tile_iter, k_tile_count,
                               Underscore{}, threadIdx.x, smem_buf);

                // Staged epilogue: accum -> smem -> global (predicated scatter)
                KernelSharedStorage &storage = *reinterpret_cast<KernelSharedStorage *>(smem_buf);
                Tensor sC = make_tensor(make_smem_ptr(&storage.epilogue.sCMatrix[0]), SmemLayoutOut{});

                auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomOut{}, tiled_mma);
                auto smem_thr_copy_C   = smem_tiled_copy_C.get_slice(threadIdx.x);
                auto tCrC              = smem_thr_copy_C.retile_S(accum);
                auto tCsC              = smem_thr_copy_C.partition_D(sC);
                copy(smem_tiled_copy_C, tCrC, tCsC);

                __syncthreads();

                GmemTiledCopyOut gmem_tiled_copy_C;
                auto gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(threadIdx.x);
                auto tDsC            = gmem_thr_copy_C.partition_S(sC);
                auto tDgC            = gmem_thr_copy_C.partition_D(gC);
                auto tDsCPred        = gmem_thr_copy_C.partition_D(sCPred);
                copy_if(gmem_tiled_copy_C, tDsCPred, tDsC, tDgC);

                __syncthreads();
            }
        }
    }
};

template <geometry_like Geom, types_like Types, tiling_like Tiling,
          conv_variant Variant, int Cin, int Cout>
__global__ void
__launch_bounds__(igemm_cute_op<Geom, Types, Tiling, Variant, Cin, Cout>::MaxThreadsPerBlock,
                  igemm_cute_op<Geom, Types, Tiling, Variant, Cin, Cout>::MinBlocksPerMultiprocessor)
leaf_igemm_alt_cute_kernel(
    Tensor<cute::tfloat32_t const *,
           decltype(igemm_layouts<Geom, tiling_default<Geom, Types>, Cin>::template filterLayout<Cout>())> mFlt,
    KernelParams<Types> params) {
    extern __shared__ char smem_buf[];
    igemm_cute_op<Geom, Types, Tiling, Variant, Cin, Cout> op;
    op(mFlt, params.iter_acc, params.gather_acc, params.iter_batch_idx,
       params.features, params.output, smem_buf);
}

// ============================================================================
// Section 14: Launch wrappers
// ============================================================================

template <geometry_like Geom, types_like Types, tiling_like Tiling, conv_variant Variant>
void
launch_leaf_igemm_alt(KernelParams<Types> const &params, int num_leaves, cudaStream_t stream) {
    if (num_leaves == 0)
        return;

    constexpr int THREADS = 256;
    constexpr size_t SMEM = sizeof(SharedStorage<Geom, Tiling, Types>);

    if constexpr (SMEM > SMEM_BASELINE_BYTES) {
        cudaFuncSetAttribute(leaf_igemm_alt_kernel<Geom, Types, Tiling, Variant>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(SMEM));
    }

    leaf_igemm_alt_kernel<Geom, Types, Tiling, Variant>
        <<<num_leaves, THREADS, SMEM, stream>>>(params);
}

} // namespace leaf_igemm_alt
} // namespace detail
} // namespace fvdb

// ============================================================================
// Section 15: Host entry point
// ============================================================================

namespace fvdb {
namespace detail {
namespace ops {

static bool
deviceSupportsSm80Alt(torch::Device device) {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device.index());
    return major >= 8;
}

template <typename Types, leaf_igemm_alt::geometry_like Geom>
static torch::Tensor
leafIGemmAltConvTyped(torch::Tensor features,
                      torch::Tensor weights,
                      GridBatchImpl const &input_grid,
                      GridBatchImpl const &output_grid,
                      torch::ScalarType output_dtype) {
    using namespace leaf_igemm_alt;

    using ElemWt   = typename Types::ElemWt;
    using ElemFeat = typename Types::ElemFeat;
    using ElemAcc  = typename Types::ElemAcc;
    using ElemOut  = typename Types::ElemOut;
    using ElemIdx  = typename Types::ElemIdx;

    using Tiling = tiling_default<Geom, Types>;

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

    launch_leaf_igemm_alt<Geom, Types, Tiling, conv_variant::forward>(
        params, static_cast<int>(num_leaves), stream);

    return output.to(output_dtype);
}

torch::Tensor
leafIGemmAltConv(torch::Tensor features,
                 torch::Tensor weights,
                 GridBatchImpl const &input_grid,
                 GridBatchImpl const &output_grid) {
    TORCH_CHECK(features.is_cuda(), "leafIGemmAltConv: features must be on CUDA");
    TORCH_CHECK(
        deviceSupportsSm80Alt(features.device()),
        "leafIGemmAltConv: requires SM80+ (Ampere or newer)");

    TORCH_CHECK(features.dim() == 2, "leafIGemmAltConv: features must be 2D [N, C_in]");
    TORCH_CHECK(features.is_contiguous(), "leafIGemmAltConv: features must be contiguous");

    TORCH_CHECK(weights.dim() == 5,
                "leafIGemmAltConv: weights must be 5D [C_out, C_in, k0, k1, k2]");
    TORCH_CHECK(features.size(1) == weights.size(1),
                "leafIGemmAltConv: C_in mismatch between features and weights");
    TORCH_CHECK(features.device() == weights.device(),
                "leafIGemmAltConv: features and weights must be on same device");
    TORCH_CHECK(features.scalar_type() == weights.scalar_type(),
                "leafIGemmAltConv: features and weights must have same dtype");

    int64_t const Cin  = features.size(1);
    int64_t const Cout = weights.size(0);
    TORCH_CHECK(Cin > 0 && Cin % 32 == 0,
                "leafIGemmAltConv: C_in must be a positive multiple of 32, got ", Cin);
    TORCH_CHECK(Cout > 0 && Cout % 32 == 0,
                "leafIGemmAltConv: C_out must be a positive multiple of 32, got ", Cout);

    int64_t const k0 = weights.size(2);
    int64_t const k1 = weights.size(3);
    int64_t const k2 = weights.size(4);

    if (features.scalar_type() == torch::kFloat32) {
        using Types = leaf_igemm_alt::types_f32;
        if (k0 == 3 && k1 == 3 && k2 == 3) {
            return leafIGemmAltConvTyped<Types, leaf_igemm_alt::geom_3x3x3_s1>(
                features, weights, input_grid, output_grid, torch::kFloat32);
        } else if (k0 == 5 && k1 == 5 && k2 == 5) {
            return leafIGemmAltConvTyped<Types, leaf_igemm_alt::geom_5x5x5_s1>(
                features, weights, input_grid, output_grid, torch::kFloat32);
        } else if (k0 == 1 && k1 == 1 && k2 == 1) {
            return leafIGemmAltConvTyped<Types, leaf_igemm_alt::geom_1x1x1_s1>(
                features, weights, input_grid, output_grid, torch::kFloat32);
        } else {
            TORCH_CHECK(false,
                        "leafIGemmAltConv: unsupported kernel size ",
                        k0, "x", k1, "x", k2,
                        ". Supported: 1x1x1, 3x3x3, 5x5x5.");
        }
    } else {
        TORCH_CHECK(false,
                    "leafIGemmAltConv: unsupported dtype ",
                    features.scalar_type(),
                    ". Supported: fp32.");
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
