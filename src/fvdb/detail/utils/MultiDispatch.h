// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_MULTIDISPATCH_H
#define FVDB_DETAIL_UTILS_MULTIDISPATCH_H

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <concepts>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace fvdb::detail::multi_dispatch {

template <auto... Vs> struct Values {};

template <typename... Ts> struct Types {};

template <auto V> using ValueTag = std::integral_constant<decltype(V), V>;

template <typename T> struct TypeTag {
    using type = T;
};

template <typename A, typename B> struct TypePair {
    using first  = A;
    using second = B;
};

template <typename... Ps> struct Pairs {};

template <typename T> struct ScalarOf {
    static constexpr auto value = c10::CppTypeToScalarType<T>::value;
};

namespace inner {

template <typename R = void, typename... Msg>
[[noreturn]] inline R
fail(Msg &&...msg) {
    TORCH_CHECK(false, std::forward<Msg>(msg)...);
    std::abort();
}

inline auto
deviceName(c10::DeviceType const d) {
    return c10::DeviceTypeName(d, /*lower_case=*/true);
}

inline auto
dtypeName(c10::ScalarType const s) {
    return c10::toString(s);
}

template <template <auto, typename...> class Impl, auto D, typename... Ts> struct ApplyCall {
  private:
    template <typename... Args>
    static auto has_call(int) -> decltype(Impl<D, Ts...>::call(std::declval<Args>()...),
                                          std::true_type{});

    template <typename...> static auto has_call(...) -> std::false_type;

  public:
    template <typename... Args>
    static decltype(auto)
    run(Args &&...args) {
        static_assert(
            decltype(has_call<Args...>(0))::value,
            "multi_dispatch::apply: missing Impl<D,Ts...>::call(...) matching this argument list");
        return Impl<D, Ts...>::call(std::forward<Args>(args)...);
    }
};

// CPO: spell it like a function template: inner::apply<Impl, D, Ts...>(args...)
template <template <auto, typename...> class Impl, auto D, typename... Ts>
inline constexpr auto apply = [](auto &&...args) -> decltype(auto) {
    return ApplyCall<Impl, D, Ts...>::run(std::forward<decltype(args)>(args)...);
};

// Dispatch runtime enum against values<V0,Vs...>
template <auto V0, auto... Vs, typename Enum, typename Func>
auto
dispatchValue(Values<V0, Vs...>,
              Enum const v,
              Func &&func) -> std::invoke_result_t<Func, ValueTag<V0>> {
    if (v == V0)
        return std::forward<Func>(func)(ValueTag<V0>{});
    if constexpr (sizeof...(Vs) > 0) {
        return dispatchValue(Values<Vs...>{}, v, std::forward<Func>(func));
    } else {
        return fail<std::invoke_result_t<Func, ValueTag<V0>>>(
            "fvdb::detail::multi_dispatch::dispatchValue: unsupported value");
    }
}

// Dispatch runtime dtype against types<T0,Ts...>
template <typename T0, typename... Ts, typename Func>
auto
dispatchType(Types<T0, Ts...>,
             c10::ScalarType const st,
             Func &&func) -> std::invoke_result_t<Func, TypeTag<T0>> {
    if (st == ScalarOf<T0>::value)
        return std::forward<Func>(func)(TypeTag<T0>{});
    if constexpr (sizeof...(Ts) > 0) {
        return dispatchType(Types<Ts...>{}, st, std::forward<Func>(func));
    } else {
        return fail<std::invoke_result_t<Func, TypeTag<T0>>>(
            "fvdb::detail::multi_dispatch::dispatchType: dtype ",
            dtypeName(st),
            " is not in the allowed set");
    }
}

// Dispatch two dtypes against explicit allowed pairs (avoids N×M)
template <typename P0, typename... Ps, typename Func>
auto
dispatchPair(Pairs<P0, Ps...>, c10::ScalarType const a, c10::ScalarType const b, Func &&func)
    -> std::invoke_result_t<Func, TypeTag<typename P0::first>, TypeTag<typename P0::second>> {
    using A0 = typename P0::first;
    using B0 = typename P0::second;

    if (a == ScalarOf<A0>::value && b == ScalarOf<B0>::value) {
        return std::forward<Func>(func)(TypeTag<A0>{}, TypeTag<B0>{});
    }
    if constexpr (sizeof...(Ps) > 0) {
        return dispatchPair(Pairs<Ps...>{}, a, b, std::forward<Func>(func));
    } else {
        return fail<std::invoke_result_t<Func, TypeTag<A0>, TypeTag<B0>>>(
            "fvdb::detail::multi_dispatch::dispatchPair: (",
            dtypeName(a),
            ", ",
            dtypeName(b),
            ") not in allowed pairs");
    }
}

} // namespace inner

// -------------------------
// Public dispatch API
// -------------------------

// 0 dtype
template <template <auto, typename...> class Impl, typename Devices, typename... Args>
decltype(auto)
dispatch(c10::Device const &dev, Args &&...args) {
    return inner::dispatchValue(Devices{}, dev.type(), [&](auto d) -> decltype(auto) {
        constexpr auto D = decltype(d)::value;
        return inner::apply<Impl, D>(std::forward<Args>(args)...);
    });
}

// 1 dtype
template <template <auto, typename...> class Impl,
          typename Devices,
          typename Types1,
          typename... Args>
decltype(auto)
dispatch(c10::Device const &dev, c10::ScalarType const t1, Args &&...args) {
    return inner::dispatchValue(Devices{}, dev.type(), [&](auto d) -> decltype(auto) {
        constexpr auto D = decltype(d)::value;
        return inner::dispatchType(Types1{}, t1, [&](auto a) -> decltype(auto) {
            using T1 = typename decltype(a)::type;
            return inner::apply<Impl, D, T1>(std::forward<Args>(args)...);
        });
    });
}

// 2 dtype — cartesian product
template <template <auto, typename...> class Impl,
          typename Devices,
          typename Types1,
          typename Types2,
          typename... Args>
decltype(auto)
dispatch(c10::Device const &dev,
         c10::ScalarType const t1,
         c10::ScalarType const t2,
         Args &&...args) {
    return inner::dispatchValue(Devices{}, dev.type(), [&](auto d) -> decltype(auto) {
        constexpr auto D = decltype(d)::value;
        return inner::dispatchType(Types1{}, t1, [&](auto a) -> decltype(auto) {
            using T1 = typename decltype(a)::type;
            return inner::dispatchType(Types2{}, t2, [&](auto b) -> decltype(auto) {
                using T2 = typename decltype(b)::type;
                return inner::apply<Impl, D, T1, T2>(std::forward<Args>(args)...);
            });
        });
    });
}

// 2 dtype — explicit allowed pairs (compile-time saver)
template <template <auto, typename...> class Impl,
          typename Devices,
          typename AllowedPairs,
          typename... Args>
decltype(auto)
dispatchPairs(c10::Device const &dev,
              c10::ScalarType const t1,
              c10::ScalarType const t2,
              Args &&...args) {
    return inner::dispatchValue(Devices{}, dev.type(), [&](auto d) -> decltype(auto) {
        constexpr auto D = decltype(d)::value;
        return inner::dispatchPair(AllowedPairs{}, t1, t2, [&](auto a, auto b) -> decltype(auto) {
            using T1 = typename decltype(a)::type;
            using T2 = typename decltype(b)::type;
            return inner::apply<Impl, D, T1, T2>(std::forward<Args>(args)...);
        });
    });
}

// Primary template (intentionally undefined)
template <template <auto, typename...> class Impl,
          typename Devices,
          typename Types1 = void,
          typename Types2 = void>
struct MultiDispatchOp {
    template <typename... Args>
    decltype(auto)
    operator()(c10::Device const &dev,
               c10::ScalarType const t1,
               c10::ScalarType const t2,
               Args &&...args) const {
        return dispatch<Impl, Devices, Types1, Types2>(dev, t1, t2, std::forward<Args>(args)...);
    }
};

// 0 dtype
template <template <auto, typename...> class Impl, typename Devices>
struct MultiDispatchOp<Impl, Devices, void, void> {
    template <typename... Args>
    decltype(auto)
    operator()(c10::Device const &dev, Args &&...args) const {
        return dispatch<Impl, Devices>(dev, std::forward<Args>(args)...);
    }
};

// 1 dtype
template <template <auto, typename...> class Impl, typename Devices, typename Types1>
struct MultiDispatchOp<Impl, Devices, Types1, void> {
    template <typename... Args>
    decltype(auto)
    operator()(c10::Device const &dev, c10::ScalarType const t1, Args &&...args) const {
        return dispatch<Impl, Devices, Types1>(dev, t1, std::forward<Args>(args)...);
    }
};

template <template <auto, typename...> class Impl, typename Devices, typename AllowedPairs>
struct MultiDispatchPairsOp {
    template <typename... Args>
    decltype(auto)
    operator()(c10::Device const &dev,
               c10::ScalarType const t1,
               c10::ScalarType const t2,
               Args &&...args) const {
        return dispatchPairs<Impl, Devices, AllowedPairs>(dev, t1, t2, std::forward<Args>(args)...);
    }
};

using CpuCudaDevices     = multi_dispatch::Values<c10::kCPU, c10::kCUDA>;
using CpuCudaPvt1Devices = multi_dispatch::Values<c10::kCPU, c10::kCUDA, c10::kPrivateUse1>;

using AllFloatTypes  = multi_dispatch::Types<float, double, c10::Half, c10::BFloat16>;
using StdFloatTypes  = multi_dispatch::Types<float, double>;
using HalfFloatTypes = multi_dispatch::Types<c10::Half, c10::BFloat16>;

template <auto Device, typename...> struct ExampleOpImpl;

template <typename T> struct ExampleOpImpl<c10::kCPU, T> {
    static void
    call(void const *x, void *y, int64_t n) {
        auto const *xp = static_cast<T const *>(x);
        auto *yp       = static_cast<T *>(y);
        printf("ExampleOpImpl<c10::kCPU, T>::call(xp, yp, n)\n");
    }
};

template <typename T> struct ExampleOpImpl<c10::kCUDA, T> {
    static void
    call(void const *x, void *y, int64_t n) {
        auto const *xp = static_cast<T const *>(x);
        auto *yp       = static_cast<T *>(y);
        printf("ExampleOpImpl<c10::kCUDA, T>::call(x, y, n)\n");
    }
};

using ExampleOp = MultiDispatchOp<ExampleOpImpl, CpuCudaDevices, AllFloatTypes>;

extern void *emptyBuffer(c10::Device const &dev, c10::ScalarType const t, int64_t n);

inline void *
exampleOp(c10::Device const &dev, c10::ScalarType const t, void const *x, int64_t n) {
    void *y = emptyBuffer(dev, t, n);
    ExampleOp op;
    op(dev, t, x, y, n);
    return y;
}

} // namespace fvdb::detail::multi_dispatch

#endif // FVDB_DETAIL_UTILS_MULTIDISPATCH_H
