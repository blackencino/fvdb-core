// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// conv_geometry.h -- Spatial geometry trait for leaf-level iGEMM convolution.
//
// Encodes the compile-time spatial parameters (kernel size, stride, dilation,
// leaf side length) and computes all derived quantities from the formal
// specification's "Derived Quantities" section.
//
// Constraint C2 (reachability limit) is enforced via static_assert.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_GEOMETRY_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_GEOMETRY_H

#include "core_types.h"

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// conv_geometry: compile-time spatial parameter trait
// =============================================================================
//
// All quantities exposed as consteval member functions (not constexpr
// variables) to avoid addressable-object instantiation in deeply nested
// CUTLASS templates.

template <coord3 K_, coord3 S_ = coord3{1, 1, 1}, coord3 D_ = coord3{1, 1, 1}, int LEAF_ = 8>
struct conv_geometry {

    // ---- Primary parameters ------------------------------------------------

    static consteval coord3
    K() {
        return K_;
    }
    static consteval coord3
    S() {
        return S_;
    }
    static consteval coord3
    D() {
        return D_;
    }
    static consteval int
    leaf() {
        return LEAF_;
    }

    // ---- Derived quantities ------------------------------------------------

    // Kernel center offset (exact when K is odd per axis).
    static consteval coord3
    offset() {
        return -((K_ - coord3{1, 1, 1}) * D_) / 2;
    }

    // Halo extent in input-space: the full region of input voxels needed
    // to compute all outputs in one leaf.
    static consteval coord3
    halo() {
        return coord3{LEAF_ - 1, LEAF_ - 1, LEAF_ - 1} * S_ + (K_ - coord3{1, 1, 1}) * D_ +
               coord3{1, 1, 1};
    }

    static consteval int
    halo_vol() {
        return prod(halo());
    }

    // Filter volume (number of taps).
    static consteval int
    kern_vol() {
        return prod(K_);
    }

    // Leaf volume.
    static consteval int
    leaf_vol() {
        return LEAF_ * LEAF_ * LEAF_;
    }

    // Leaf extent as a coord3 (convenience).
    static consteval coord3
    leaf3() {
        return {LEAF_, LEAF_, LEAF_};
    }

    // ---- Constraint validation ---------------------------------------------

    static_assert(all_positive(K_), "Kernel size must be positive on all axes");
    static_assert(all_positive(S_), "Stride must be positive on all axes");
    static_assert(all_positive(D_), "Dilation must be positive on all axes");
    static_assert(LEAF_ > 0, "Leaf side length must be positive");

    // C2: Reachability limit.
    // The halo must fit within a 3-leaf neighborhood on each axis so that
    // coord_to_idx lookups stay within the immediate leaf neighbors.
    //
    //   halo[d] <= 2 * LEAF + 1
    //
    // Expanded:
    //   (LEAF - 1) * S[d] + (K[d] - 1) * D[d] <= 2 * LEAF
    //
    static_assert(
        all_leq(halo(), coord3{2 * LEAF_ + 1, 2 * LEAF_ + 1, 2 * LEAF_ + 1}),
        "C2 violated: halo exceeds 3-leaf neighborhood. "
        "Reduce kernel size, stride, or dilation.");
};

// =============================================================================
// is_conv_geometry trait + geometry_like concept
// =============================================================================

template <typename T>
struct is_conv_geometry : consteval_false_type {};

template <coord3 K, coord3 S, coord3 D, int L>
struct is_conv_geometry<conv_geometry<K, S, D, L>> : consteval_true_type {};

template <typename T>
concept geometry_like = is_conv_geometry<T>::value();

// =============================================================================
// Common geometry aliases
// =============================================================================

using geom_3x3x3_s1 = conv_geometry<coord3{3, 3, 3}>;
using geom_3x3x3_s2 = conv_geometry<coord3{3, 3, 3}, coord3{2, 2, 2}>;
using geom_5x5x5_s1 = conv_geometry<coord3{5, 5, 5}>;

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CONV_GEOMETRY_H
