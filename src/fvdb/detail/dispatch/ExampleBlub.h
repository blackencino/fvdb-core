// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_EXAMPLEBLUB_H
#define FVDB_DETAIL_DISPATCH_EXAMPLEBLUB_H

#include "fvdb/detail/dispatch/ConcreteTensor.h"
#include "fvdb/detail/dispatch/TorchTags.h"

#include <cstdint>

namespace fvdb {
namespace dispatch {
namespace example {

// -----------------------------------------------------------------------------
// Example: "blub" operation with sparse coverage
// -----------------------------------------------------------------------------

//---------------------------------------------------------------------------------
// FIRST WAY: Free Functions
//---------------------------------------------------------------------------------
void
blub_impl(TorchDeviceCpuTag, CpuTensor<torch::kFloat32, 1> in, CpuTensor<torch::kInt32, 1> out);

void
blub_impl(TorchDeviceCudaTag, CudaTensor<torch::kFloat32, 1> in, CudaTensor<torch::kInt32, 1> out);

template <torch::ScalarType Dtype>
void blub_impl(TorchDeviceCpuTag, CpuTensor<Dtype, 1> in, CpuTensor<Dtype, 1> out);

extern template void blub_impl<torch::kInt32>(TorchDeviceCpuTag,
                                              CpuTensor<torch::kInt32, 1>,
                                              CpuTensor<torch::kInt32, 1>);
extern template void blub_impl<torch::kInt64>(TorchDeviceCpuTag,
                                              CpuTensor<torch::kInt64, 1>,
                                              CpuTensor<torch::kInt64, 1>);
extern template void blub_impl<torch::kFloat16>(TorchDeviceCpuTag,
                                                CpuTensor<torch::kFloat16, 1>,
                                                CpuTensor<torch::kFloat16, 1>);
extern template void blub_impl<torch::kFloat32>(TorchDeviceCpuTag,
                                                CpuTensor<torch::kFloat32, 1>,
                                                CpuTensor<torch::kFloat32, 1>);
extern template void blub_impl<torch::kFloat64>(TorchDeviceCpuTag,
                                                CpuTensor<torch::kFloat64, 1>,
                                                CpuTensor<torch::kFloat64, 1>);

template <torch::DeviceType DeviceValue>
void blub_impl(Tag<DeviceValue>,
               ConcreteTensor<DeviceValue, torch::kFloat64, 1> in,
               ConcreteTensor<DeviceValue, torch::kInt32, 1> out);

extern template void blub_impl<torch::kCPU>(Tag<torch::kCPU>,
                                            ConcreteTensor<torch::kCPU, torch::kFloat64, 1>,
                                            ConcreteTensor<torch::kCPU, torch::kInt32, 1>);

extern template void blub_impl<torch::kCUDA>(Tag<torch::kCUDA>,
                                             ConcreteTensor<torch::kCUDA, torch::kFloat64, 1>,
                                             ConcreteTensor<torch::kCUDA, torch::kInt32, 1>);

torch::Tensor blub_eval(torch::Tensor in, torch::ScalarType out_dtype);

//---------------------------------------------------------------------------------
// SECOND WAY: Struct with statics
//---------------------------------------------------------------------------------

struct Blub {
static void
impl(TorchDeviceCpuTag, CpuTensor<torch::kFloat32, 1> in, CpuTensor<torch::kInt32, 1> out);

static void
impl(TorchDeviceCudaTag, CudaTensor<torch::kFloat32, 1> in, CudaTensor<torch::kInt32, 1> out);

template <torch::ScalarType Dtype>
static void impl(TorchDeviceCpuTag, CpuTensor<Dtype, 1> in, CpuTensor<Dtype, 1> out);

template <torch::DeviceType DeviceValue>
static void impl(Tag<DeviceValue>,
               ConcreteTensor<DeviceValue, torch::kFloat64, 1> in,
               ConcreteTensor<DeviceValue, torch::kInt32, 1> out);

static torch::Tensor eval(torch::Tensor in, torch::ScalarType out_dtype);
};

extern template void Blub::impl<torch::kInt32>(TorchDeviceCpuTag,
    CpuTensor<torch::kInt32, 1>,
    CpuTensor<torch::kInt32, 1>);
extern template void Blub::impl<torch::kInt64>(TorchDeviceCpuTag,
    CpuTensor<torch::kInt64, 1>,
    CpuTensor<torch::kInt64, 1>);
extern template void Blub::impl<torch::kFloat16>(TorchDeviceCpuTag,
      CpuTensor<torch::kFloat16, 1>,
      CpuTensor<torch::kFloat16, 1>);
extern template void Blub::impl<torch::kFloat32>(TorchDeviceCpuTag,
      CpuTensor<torch::kFloat32, 1>,
      CpuTensor<torch::kFloat32, 1>);
extern template void Blub::impl<torch::kFloat64>(TorchDeviceCpuTag,
      CpuTensor<torch::kFloat64, 1>,
      CpuTensor<torch::kFloat64, 1>);
extern template void Blub::impl<torch::kCPU>(Tag<torch::kCPU>,
        ConcreteTensor<torch::kCPU, torch::kFloat64, 1>,
        ConcreteTensor<torch::kCPU, torch::kInt32, 1>);
extern template void Blub::impl<torch::kCUDA>(Tag<torch::kCUDA>,
         ConcreteTensor<torch::kCUDA, torch::kFloat64, 1>,
         ConcreteTensor<torch::kCUDA, torch::kInt32, 1>);

//---------------------------------------------------------------------------------

inline torch::Tensor blub(torch::Tensor in, torch::ScalarType out_dtype) {
    return blub_eval(in, out_dtype);
}

inline torch::Tensor blub(torch::Tensor in) {
    return blub_eval(in, in.scalar_type());
}

} // namespace example
} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_EXAMPLEBLUB_H
