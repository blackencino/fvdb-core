// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_COMMON_H
#define FVDB_DETAIL_AUTOGRAD_COMMON_H

#include <torch/autograd.h>
#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace autograd {

using variable_list   = torch::autograd::variable_list;
using AutogradContext = torch::autograd::AutogradContext;

template <typename... Args> using Function = torch::autograd::Function<Args...>;

inline torch::TensorOptions
tensorOptionsFrom(torch::Tensor t) {
    return torch::TensorOptions().dtype(t.dtype()).device(t.device());
}

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_COMMON_H
