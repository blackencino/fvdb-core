// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// DeviceTagDispatch.h - Tag-based device dispatch without macros
// =============================================================================
//
// This header provides a C++20 concepts-based mechanism for dispatching
// operations to device-specific implementations using tag types instead of
// macros. The key idea is that each device type (CPU, CUDA, PrivateUse1) has
// a corresponding tag type that carries the device type as a compile-time
// constant. Functions can be overloaded on these tags, and generic code can
// be written that is constrained to only accept valid device tags.
//
// -----------------------------------------------------------------------------
// BASIC USAGE: Defining device-specific overloads
// -----------------------------------------------------------------------------
//
// Instead of template specializations:
//
//     // OLD (template specialization approach):
//     template <torch::DeviceType> void gatherScatter(...);
//     template <> void gatherScatter<torch::kCPU>(...) { /* CPU impl */ }
//     template <> void gatherScatter<torch::kCUDA>(...) { /* CUDA impl */ }
//
// Use tag-based overloads:
//
//     // NEW (tag dispatch approach):
//     void gatherScatter(CpuTag, torch::Tensor in, torch::Tensor out, ...);
//     void gatherScatter(CudaTag, torch::Tensor in, torch::Tensor out, ...);
//
// -----------------------------------------------------------------------------
// DISPATCHING: Converting runtime device to compile-time tag
// -----------------------------------------------------------------------------
//
// Use the dispatch() function to convert a runtime torch::Device into the
// appropriate compile-time tag and invoke a callable:
//
//     // Dispatch to the correct gatherScatter overload based on device:
//     dispatch(device, [&](auto tag) {
//         gatherScatter(tag, inFeat, outFeat, kernel, nbMap, nbSizes, accel);
//     });
//
// By default, dispatch() only handles CPU and CUDA devices (the common case).
// The lambda receives either cpu_tag or cuda_tag.
// Inside the lambda, decltype(tag)::value gives the torch::DeviceType constant.
//
// For operations that also support PrivateUse1, the lambda may also receive
// privateuse1_tag (see "DEVICE SETS" below).
//
// -----------------------------------------------------------------------------
// DEVICE SETS: Controlling which devices are supported
// -----------------------------------------------------------------------------
//
// Many operations only have CPU and CUDA implementations. By default, dispatch()
// only handles these two devices:
//
//     // CPU/CUDA only (default) - errors on PrivateUse1 devices:
//     dispatch(device, gatherScatter, inFeat, outFeat, ...);
//
// For operations that also support PrivateUse1, use the WithPrivateUse1 selector:
//
//     // CPU, CUDA, and PrivateUse1:
//     dispatch(WithPrivateUse1, device, someOp, args...);
//
// This makes the device support explicit and readable at each call site.
//
// -----------------------------------------------------------------------------
// GENERIC FUNCTIONS: Writing code that works across all device tags
// -----------------------------------------------------------------------------
//
// Use the DeviceTag concept to constrain templates that should only accept
// valid device tag types:
//
//     template <DeviceTag Tag>
//     torch::Tensor allocateScratchBuffer(Tag, int64_t size) {
//         // Tag::value is the compile-time torch::DeviceType
//         auto opts = torch::TensorOptions().device(Tag::value);
//         return torch::empty({size}, opts);
//     }
//
// This function can be called with any device tag:
//
//     auto buf = allocateScratchBuffer(cuda_tag, 1024);
//     auto buf = allocateScratchBuffer(cpu_tag, 512);
//
// But will produce a clear compile error if called with an invalid type:
//
//     allocateScratchBuffer(42, 1024);  // ERROR: int does not satisfy DeviceTag
//
// -----------------------------------------------------------------------------
// COMPOSING OPERATIONS: Building higher-level device-generic algorithms
// -----------------------------------------------------------------------------
//
// You can write device-generic algorithms that compose lower-level operations:
//
//     template <DeviceTag Tag>
//     void convolutionForward(Tag tag,
//                             torch::Tensor input,
//                             torch::Tensor output,
//                             torch::Tensor kernel,
//                             torch::Tensor neighborMap) {
//         // Allocate scratch space on the correct device
//         auto gathered = allocateScratchBuffer(tag, input.size(0) * kernel.size(0));
//
//         // These call the correct overloads automatically via tag dispatch
//         gatherScatter(tag, input, gathered, ...);
//         matmulAccumulate(tag, gathered, kernel, output);
//     }
//
// Then expose a runtime-dispatched entry point:
//
//     void convolutionForward(torch::Device device, ...) {
//         dispatch(device, [&](auto tag) {
//             convolutionForward(tag, input, output, kernel, neighborMap);
//         });
//     }
//
// -----------------------------------------------------------------------------
// SUMMARY
// -----------------------------------------------------------------------------
//
// Device Tags:
//   - CpuTag, CudaTag, PrivateUse1Tag: Tag types with ::value = torch::DeviceType
//   - cpu_tag, cuda_tag, privateuse1_tag: Global instances for convenient use
//
// Concepts and Traits:
//   - DeviceTag: C++20 concept constraining templates to valid device tags
//   - is_device_tag_v<T>: Type trait for SFINAE/enable_if (C++17 fallback)
//
// Dispatch Functions:
//   - dispatch(device, callable)           : CPU/CUDA only (common case)
//   - dispatch(device, func, args...)      : CPU/CUDA only with args forwarding
//   - dispatch(WithPrivateUse1, device, ...)   : CPU, CUDA, and PrivateUse1
//
// =============================================================================

#ifndef FVDB_DETAIL_UTILS_DEVICETAGDISPATCH_H
#define FVDB_DETAIL_UTILS_DEVICETAGDISPATCH_H

#include <torch/types.h>

#include <concepts>
#include <type_traits>
#include <utility>

namespace fvdb {
namespace detail {

// =============================================================================
// Device Tag Types
// =============================================================================
//
// Each tag carries the corresponding torch::DeviceType as a static constexpr
// member, allowing compile-time access to the device type within generic code.

struct CpuTag {
    static constexpr torch::DeviceType value = torch::kCPU;
};

struct CudaTag {
    static constexpr torch::DeviceType value = torch::kCUDA;
};

struct PrivateUse1Tag {
    static constexpr torch::DeviceType value = torch::kPrivateUse1;
};

// Global tag instances for convenient use at call sites
inline constexpr CpuTag cpu_tag{};
inline constexpr CudaTag cuda_tag{};
inline constexpr PrivateUse1Tag privateuse1_tag{};

// =============================================================================
// Device Set Selectors
// =============================================================================
//
// Used as the first argument to dispatch() to specify which devices to support.
// By default, dispatch() only handles CPU and CUDA. Use WithPrivateUse1 to include
// PrivateUse1 as well.

struct WithPrivateUse1Selector {};

inline constexpr WithPrivateUse1Selector WithPrivateUse1{};

// =============================================================================
// Type Trait: is_device_tag
// =============================================================================
//
// A type trait to identify valid device tag types. Useful for SFINAE in C++17
// or as the foundation for the C++20 concept.

template <typename T> struct is_device_tag : std::false_type {};

template <> struct is_device_tag<CpuTag> : std::true_type {};

template <> struct is_device_tag<CudaTag> : std::true_type {};

template <> struct is_device_tag<PrivateUse1Tag> : std::true_type {};

template <typename T> inline constexpr bool is_device_tag_v = is_device_tag<T>::value;

// =============================================================================
// C++20 Concept: DeviceTag
// =============================================================================
//
// Constrains template parameters to valid device tag types. Provides clear
// compile-time error messages when an invalid type is used.

template <typename T>
concept DeviceTag = is_device_tag_v<T>;

// =============================================================================
// Runtime Dispatch Functions (CPU/CUDA only - default)
// =============================================================================
//
// Converts a runtime torch::Device value into the appropriate compile-time
// tag and invokes the provided callable with that tag.
//
// By default, only CPU and CUDA are supported. This is the common case for
// most operations. PrivateUse1 devices will produce a runtime error.
//
// Example:
//     dispatch(device, [&](auto tag) {
//         gatherScatter(tag, inFeat, outFeat, kernel, nbMap, nbSizes, accel);
//     });
//

template <typename F>
auto
dispatch(torch::Device device, F &&func) -> decltype(func(cpu_tag)) {
    if (device.is_cpu()) {
        return func(cpu_tag);
    } else if (device.is_cuda()) {
        return func(cuda_tag);
    }
    TORCH_CHECK(false, "Only CPU and CUDA devices are supported for this operation");
}

// Overload that forwards additional arguments to the callable:
//
//     dispatch(device, gatherScatter, inFeat, outFeat, kernel, nbMap, nbSizes, accel);
//
template <typename F, typename... Args>
auto
dispatch(torch::Device device,
         F &&func,
         Args &&...args) -> decltype(func(cpu_tag, std::forward<Args>(args)...)) {
    if (device.is_cpu()) {
        return func(cpu_tag, std::forward<Args>(args)...);
    } else if (device.is_cuda()) {
        return func(cuda_tag, std::forward<Args>(args)...);
    }
    TORCH_CHECK(false, "Only CPU and CUDA devices are supported for this operation");
}

// =============================================================================
// Runtime Dispatch Functions (All devices including PrivateUse1)
// =============================================================================
//
// Use the WithPrivateUse1 selector as the first argument to dispatch to operations
// that also support PrivateUse1:
//
//     dispatch(WithPrivateUse1, device, [&](auto tag) {
//         someOpWithPrivateUse1Support(tag, args...);
//     });
//
// Or with argument forwarding:
//
//     dispatch(WithPrivateUse1, device, someOp, arg1, arg2, ...);
//

template <typename F>
auto
dispatch(WithPrivateUse1Selector, torch::Device device, F &&func) -> decltype(func(cpu_tag)) {
    if (device.is_cpu()) {
        return func(cpu_tag);
    } else if (device.is_cuda()) {
        return func(cuda_tag);
    } else if (device.is_privateuseone()) {
        return func(privateuse1_tag);
    }
    TORCH_CHECK(false, "Only CPU, CUDA, and PrivateUse1 devices are supported");
}

template <typename F, typename... Args>
auto
dispatch(WithPrivateUse1Selector, torch::Device device, F &&func, Args &&...args)
    -> decltype(func(cpu_tag, std::forward<Args>(args)...)) {
    if (device.is_cpu()) {
        return func(cpu_tag, std::forward<Args>(args)...);
    } else if (device.is_cuda()) {
        return func(cuda_tag, std::forward<Args>(args)...);
    } else if (device.is_privateuseone()) {
        return func(privateuse1_tag, std::forward<Args>(args)...);
    }
    TORCH_CHECK(false, "Only CPU, CUDA, and PrivateUse1 devices are supported");
}

// =============================================================================
// Device Op Base Class
// =============================================================================
//
// A base class for device-specific operations.
// Derived classes must implement an execute() method that takes a tag and arguments.

template <typename Derived, typename DeviceSelector = void> struct DeviceDispatchOp {
    template <typename... Args>
    static auto
    apply(torch::Device device, Args &&...args) {
        // Create the lambda shim automatically
        auto shim = [](auto tag, auto &&...forwarded_args) {
            // Static Dispatch: Calls Derived::execute(tag, ...)
            return Derived::execute(tag, std::forward<decltype(forwarded_args)>(forwarded_args)...);
        };

        if constexpr (std::is_same_v<DeviceSelector, WithPrivateUse1Selector>) {
            return dispatch(WithPrivateUse1, device, shim, std::forward<Args>(args)...);
        } else {
            return dispatch(device, shim, std::forward<Args>(args)...);
        }
    }
};
// =============================================================================
// Concept: HasDeviceFunction
// =============================================================================
//
// 1. Use std::remove_cvref_t<T> to handle T being 'Tensor&', 'const Tensor&', etc.
// 2. We check if a CONST instance has .device(), ensuring we don't accidentally
//    accept types that only allow device queries on mutable objects.

template <typename T>
concept HasDeviceFunction = requires(std::remove_cvref_t<T> const &t) {
    { t.device() } -> std::convertible_to<torch::Device>;
};

// =============================================================================
// Function: checkDevicesAndGetFirst
// =============================================================================
//
// Uses Universal References (First&&) to preserve exact types.
// Static asserts verify that all arguments have a .device() method.
// Note: We use static_assert instead of requires clause because the latter
// can interfere with template argument deduction in some compilers.

template <typename First, typename... Rest>
inline torch::Device
checkDevicesAndGetFirst(First &&first, Rest &&...rest) {
    static_assert(HasDeviceFunction<First>,
                  "First argument must have a .device() method returning torch::Device");
    static_assert((HasDeviceFunction<Rest> && ...),
                  "All arguments must have a .device() method returning torch::Device");

    // We don't need std::forward here because we are just inspecting the property.
    // Accessing .device() is safe on l-values and r-values alike.
    torch::Device const target_device = first.device();

    auto check = [&](auto &&item) {
        TORCH_CHECK(item.device() == target_device,
                    "All arguments must be on the same device. Expected ",
                    target_device,
                    " but got ",
                    item.device());
    };

    // Fold expression expands to: check(rest_1), check(rest_2), ...
    (check(rest), ...);

    return target_device;
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_DEVICETAGDISPATCH_H
