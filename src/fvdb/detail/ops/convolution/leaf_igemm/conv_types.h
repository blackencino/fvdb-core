// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// conv_types.h -- Scalar type parameters and MMA atom selection for SM80.
//
// Encodes the five scalar type parameters from the formal specification
// (ElemWt, ElemFeat, ElemAcc, ElemOut, ElemIdx) and selects the appropriate
// SM80 tensor-core MMA atom via a specialized trait.
//
// Target architecture: SM80 (Ampere) only.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_TYPES_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_TYPES_H

#include "core_types.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

// CUTLASS / CuTe types for MMA atom definitions.
// This is the only Layer-2 header that touches CUTLASS.
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>

#include <cutlass/half.h>
#include <cutlass/bfloat16.h>

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// mma_atom_selector: maps (ElemWt, ElemFeat, ElemAcc) -> SM80 MMA atom
// =============================================================================
//
// Specialized for each valid MMA triple from the spec's C6 table.
// An unspecialized instantiation will trigger a static_assert.

template <typename ElemWt, typename ElemFeat, typename ElemAcc>
struct mma_atom_selector {
    static_assert(sizeof(ElemWt) == 0,
                  "C6 violated: no SM80 MMA atom for this (ElemWt, ElemFeat, ElemAcc) triple. "
                  "See conv_types.h for supported triples.");
};

// ---- f32 / tf32 -> f32 accumulator, SM80_16x8x8_F32TF32TF32F32_TN ----

template <>
struct mma_atom_selector<float, float, float> {
    using mma_op  = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using mma_traits = cute::MMA_Traits<mma_op>;

    static consteval int
    alignment_a() {
        return 4;
    }
    static consteval int
    alignment_b() {
        return 4;
    }
    static consteval int
    mma_k() {
        return 8;
    }
};

// ---- f16 x f16 -> f32 accumulator, SM80_16x8x16_F32F16F16F32_TN ----

template <>
struct mma_atom_selector<cutlass::half_t, cutlass::half_t, float> {
    using mma_op  = cute::SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = cute::MMA_Traits<mma_op>;

    static consteval int
    alignment_a() {
        return 8;
    }
    static consteval int
    alignment_b() {
        return 8;
    }
    static consteval int
    mma_k() {
        return 16;
    }
};

// ---- bf16 x bf16 -> f32 accumulator, SM80_16x8x16_F32BF16BF16F32_TN ----

template <>
struct mma_atom_selector<cutlass::bfloat16_t, cutlass::bfloat16_t, float> {
    using mma_op  = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits = cute::MMA_Traits<mma_op>;

    static consteval int
    alignment_a() {
        return 8;
    }
    static consteval int
    alignment_b() {
        return 8;
    }
    static consteval int
    mma_k() {
        return 16;
    }
};

// =============================================================================
// mma_alignment: per-element alignment requirement
// =============================================================================
//
// Returns the minimum channel alignment (number of elements) required by
// the MMA atom for this element type.

template <typename Elem>
struct mma_alignment;

template <>
struct mma_alignment<float> {
    static consteval int
    value() {
        return 4;
    }
};

template <>
struct mma_alignment<cutlass::half_t> {
    static consteval int
    value() {
        return 8;
    }
};

template <>
struct mma_alignment<cutlass::bfloat16_t> {
    static consteval int
    value() {
        return 8;
    }
};

// =============================================================================
// conv_types: scalar type parameter bundle
// =============================================================================

template <typename ElemWt_, typename ElemFeat_, typename ElemAcc_,
          typename ElemOut_, typename ElemIdx_>
struct conv_types {
    using ElemWt   = ElemWt_;
    using ElemFeat = ElemFeat_;
    using ElemAcc  = ElemAcc_;
    using ElemOut  = ElemOut_;
    using ElemIdx  = ElemIdx_;

    // MMA atom selected from the (Wt, Feat, Acc) triple.
    using mma_selector = mma_atom_selector<ElemWt_, ElemFeat_, ElemAcc_>;
    using mma_op       = typename mma_selector::mma_op;
    using mma_traits   = typename mma_selector::mma_traits;

    static consteval int
    alignment_a() {
        return mma_selector::alignment_a();
    }
    static consteval int
    alignment_b() {
        return mma_selector::alignment_b();
    }
    static consteval int
    mma_k() {
        return mma_selector::mma_k();
    }

    // C7: ElemOut must be assignable from ElemAcc (possibly with narrowing).
    // For identical types this is trivially true. For mixed precision
    // (e.g. f32 acc -> f16 out), the epilogue handles conversion.
    static_assert(sizeof(ElemOut_) <= sizeof(ElemAcc_),
                  "C7: ElemOut must not be wider than ElemAcc");
};

// =============================================================================
// is_conv_types trait + types_like concept
// =============================================================================

template <typename T>
struct is_conv_types : consteval_false_type {};

template <typename Wt, typename Feat, typename Acc, typename Out, typename Idx>
struct is_conv_types<conv_types<Wt, Feat, Acc, Out, Idx>> : consteval_true_type {};

template <typename T>
concept types_like = is_conv_types<T>::value();

// =============================================================================
// Common type aliases
// =============================================================================

// f32 features, f32 weights, f32 accumulator, f32 output, u32 indices
using types_f32 = conv_types<float, float, float, float, uint32_t>;

// f16 features, f16 weights, f32 accumulator, f16 output, u32 indices
using types_f16 = conv_types<cutlass::half_t, cutlass::half_t, float, cutlass::half_t, uint32_t>;

// bf16 features, bf16 weights, f32 accumulator, bf16 output, u32 indices
using types_bf16 =
    conv_types<cutlass::bfloat16_t, cutlass::bfloat16_t, float, cutlass::bfloat16_t, uint32_t>;

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_TYPES_H
