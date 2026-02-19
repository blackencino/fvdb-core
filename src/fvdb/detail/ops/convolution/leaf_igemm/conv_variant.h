// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// conv_variant.h -- Operation variant enum and specialized traits.
//
// Encodes the four operation variants (forward, input_grad, weight_grad,
// transposed_conv) from the formal specification's parameter table.
// Each variant specialization provides compile-time flags that control
// gather/scatter grid roles, kernel flip, GEMM layout, and accumulation
// semantics.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_VARIANT_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_VARIANT_H

#include "core_types.h"

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// conv_variant enum
// =============================================================================

enum class conv_variant { forward, input_grad, weight_grad, transposed_conv };

// =============================================================================
// grid_role: which grid a variant iterates / gathers from
// =============================================================================

enum class grid_role { output, input };

// =============================================================================
// gemm_layout: contraction arrangement
// =============================================================================
//
// nn: standard   A @ B        (forward, input_grad, transposed_conv)
// nt: transposed A @ B^T      (weight_grad)

enum class gemm_layout { nn, nt };

// =============================================================================
// variant_traits: specialized per variant
// =============================================================================
//
// Each specialization is a pure trait struct with consteval accessors,
// following the dispatch framework's pattern.

template <conv_variant V>
struct variant_traits;

// ---- Forward ----------------------------------------------------------------
// WorkUnit per: output leaf
// Gather from:  input grid
// im2col flip:  false
// GEMM:         A @ B,   (M,N,K) = (C_out, BLK_VOL, C_in * KERN_VOL)
// Accumulation: scatter

template <>
struct variant_traits<conv_variant::forward> {
    static consteval bool
    flip() {
        return false;
    }
    static consteval grid_role
    iter_grid() {
        return grid_role::output;
    }
    static consteval grid_role
    gather_grid() {
        return grid_role::input;
    }
    static consteval gemm_layout
    layout() {
        return gemm_layout::nn;
    }
    static consteval bool
    atomic_accum() {
        return false;
    }
};

// ---- Input Gradient ---------------------------------------------------------
// WorkUnit per: input leaf
// Gather from:  output grid
// im2col flip:  true   (kernel spatially reversed)
// GEMM:         A @ B,   (M,N,K) = (C_in, BLK_VOL, C_out * KERN_VOL)
// Accumulation: scatter

template <>
struct variant_traits<conv_variant::input_grad> {
    static consteval bool
    flip() {
        return true;
    }
    static consteval grid_role
    iter_grid() {
        return grid_role::input;
    }
    static consteval grid_role
    gather_grid() {
        return grid_role::output;
    }
    static consteval gemm_layout
    layout() {
        return gemm_layout::nn;
    }
    static consteval bool
    atomic_accum() {
        return false;
    }
};

// ---- Weight Gradient --------------------------------------------------------
// WorkUnit per: output leaf
// Gather from:  input grid
// im2col flip:  false
// GEMM:         A @ B^T,  (M,N,K) = (C_out, C_in * KERN_VOL, BLK_VOL)
// Accumulation: AtomicAccumulate (all leaves contribute partials)

template <>
struct variant_traits<conv_variant::weight_grad> {
    static consteval bool
    flip() {
        return false;
    }
    static consteval grid_role
    iter_grid() {
        return grid_role::output;
    }
    static consteval grid_role
    gather_grid() {
        return grid_role::input;
    }
    static consteval gemm_layout
    layout() {
        return gemm_layout::nt;
    }
    static consteval bool
    atomic_accum() {
        return true;
    }
};

// ---- Transposed Convolution ------------------------------------------------
// Structurally identical to InputGrad.
// WorkUnit per: output leaf (the upsampled grid)
// Gather from:  input grid  (the smaller grid)
// im2col flip:  true
// GEMM:         A @ B,   (M,N,K) = (C_out, BLK_VOL, C_in * KERN_VOL)
// Accumulation: scatter

template <>
struct variant_traits<conv_variant::transposed_conv> {
    static consteval bool
    flip() {
        return true;
    }
    static consteval grid_role
    iter_grid() {
        return grid_role::output;
    }
    static consteval grid_role
    gather_grid() {
        return grid_role::input;
    }
    static consteval gemm_layout
    layout() {
        return gemm_layout::nn;
    }
    static consteval bool
    atomic_accum() {
        return false;
    }
};

// =============================================================================
// Concept duals
// =============================================================================

template <conv_variant V>
concept flip_variant = variant_traits<V>::flip();

template <conv_variant V>
concept non_flip_variant = !variant_traits<V>::flip();

template <conv_variant V>
concept atomic_variant = variant_traits<V>::atomic_accum();

template <conv_variant V>
concept scatter_variant = !variant_traits<V>::atomic_accum();

template <conv_variant V>
concept nn_variant = (variant_traits<V>::layout() == gemm_layout::nn);

template <conv_variant V>
concept nt_variant = (variant_traits<V>::layout() == gemm_layout::nt);

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_VARIANT_H
