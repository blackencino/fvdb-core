// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// core_types.h -- Vocabulary types for leaf-level iGEMM sparse convolution.
//
// Provides a constexpr-friendly coord3 (structural type, usable as NTTP),
// arithmetic operators, and the encode3/decode3 linearization helpers from
// the formal specification.
//
// Zero external dependencies. All functions are consteval so that derived
// quantities never produce addressable symbols in deeply nested templates
// (same rationale as dispatch::consteval_bool_type).
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CORE_TYPES_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CORE_TYPES_H

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// coord3: constexpr 3-component integer vector
// =============================================================================
//
// All members are public and the equality operator is defaulted, making
// coord3 a structural type eligible for use as a non-type template
// parameter (C++20 NTTP).

struct coord3 {
    int x = 0;
    int y = 0;
    int z = 0;

    consteval bool operator==(coord3 const &) const = default;
};

// -----------------------------------------------------------------------------
// Arithmetic operators (all consteval, element-wise)
// -----------------------------------------------------------------------------

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
operator-(coord3 a) {
    return {-a.x, -a.y, -a.z};
}

// Scalar-coord3 mixed arithmetic

consteval coord3
operator*(int s, coord3 v) {
    return {s * v.x, s * v.y, s * v.z};
}

consteval coord3
operator*(coord3 v, int s) {
    return {v.x * s, v.y * s, v.z * s};
}

consteval coord3
operator/(coord3 v, int s) {
    return {v.x / s, v.y / s, v.z / s};
}

// =============================================================================
// Reduction
// =============================================================================

consteval int
prod(coord3 v) {
    return v.x * v.y * v.z;
}

// =============================================================================
// Linearization / de-linearization (row-major)
// =============================================================================
//
// encode3((x,y,z), (X,Y,Z)) = x*Y*Z + y*Z + z
// decode3(i,        (X,Y,Z)) = (i/(Y*Z), (i/Z)%Y, i%Z)

consteval int
encode3(coord3 pos, coord3 dims) {
    return pos.x * dims.y * dims.z + pos.y * dims.z + pos.z;
}

consteval coord3
decode3(int i, coord3 dims) {
    return {i / (dims.y * dims.z), (i / dims.z) % dims.y, i % dims.z};
}

// =============================================================================
// Per-axis predicates
// =============================================================================

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

// =============================================================================
// consteval bool wrapper (matching dispatch framework idiom)
// =============================================================================

template <bool B>
struct consteval_bool_type {
    static consteval bool
    value() {
        return B;
    }
};

using consteval_true_type  = consteval_bool_type<true>;
using consteval_false_type = consteval_bool_type<false>;

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_CORE_TYPES_H
