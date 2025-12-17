// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <pybind11/cast.h>
#include <pybind11/stl.h>

#include "TypeCasters.h"

#include <fvdb/ConvolutionBackends.h>

#include <torch/extension.h>

void
bind_convolution(py::module &m) {
    // Bind BackendConfig struct
    py::class_<fvdb::detail::autograd::convolution::BackendConfig>(m, "ConvBackendConfig")
        .def_readonly("source_grid",
                      &fvdb::detail::autograd::convolution::BackendConfig::sourceGrid)
        .def_readonly("target_grid",
                      &fvdb::detail::autograd::convolution::BackendConfig::targetGrid)
        .def_property_readonly(
            "kernel_size",
            [](const fvdb::detail::autograd::convolution::BackendConfig &self) {
                return std::array<int, 3>{self.kernelSize[0], self.kernelSize[1], self.kernelSize[2]};
            })
        .def_property_readonly(
            "stride",
            [](const fvdb::detail::autograd::convolution::BackendConfig &self) {
                return std::array<int, 3>{self.stride[0], self.stride[1], self.stride[2]};
            })
        .def("to",
             &fvdb::detail::autograd::convolution::BackendConfig::to,
             py::arg("device"));

    // Bind GatherScatterAutograd::Topology struct
    py::class_<fvdb::detail::autograd::convolution::GatherScatterAutograd::Topology>(
        m, "GatherScatterTopology")
        .def_readonly(
            "neighbor_map",
            &fvdb::detail::autograd::convolution::GatherScatterAutograd::Topology::neighborMap)
        .def_readonly(
            "neighbor_sizes",
            &fvdb::detail::autograd::convolution::GatherScatterAutograd::Topology::neighborSizes)
        .def_readonly(
            "source_total_voxel_count",
            &fvdb::detail::autograd::convolution::GatherScatterAutograd::Topology::sourceTotalVoxelCount)
        .def_readonly(
            "target_total_voxel_count",
            &fvdb::detail::autograd::convolution::GatherScatterAutograd::Topology::targetTotalVoxelCount)
        .def_property_readonly(
            "kernel_size",
            [](const fvdb::detail::autograd::convolution::GatherScatterAutograd::Topology &self) {
                return std::array<int, 3>{self.kernelSize[0], self.kernelSize[1], self.kernelSize[2]};
            })
        .def_property_readonly(
            "stride",
            [](const fvdb::detail::autograd::convolution::GatherScatterAutograd::Topology &self) {
                return std::array<int, 3>{self.stride[0], self.stride[1], self.stride[2]};
            })
        .def("to",
             &fvdb::detail::autograd::convolution::GatherScatterAutograd::Topology::to,
             py::arg("device"));

    // Bind GatherScatterBackend struct
    py::class_<fvdb::GatherScatterBackend>(m, "GatherScatterBackend")
        .def_readonly("config", &fvdb::GatherScatterBackend::config)
        .def_readonly("topology", &fvdb::GatherScatterBackend::topology)
        .def_static("create",
                    &fvdb::GatherScatterBackend::create,
                    py::arg("source_grid"),
                    py::arg("target_grid"),
                    py::arg("kernel_size"),
                    py::arg("stride"),
                    py::arg("expert_config") = std::map<std::string, std::string>{})
        .def("to", &fvdb::GatherScatterBackend::to, py::arg("device"))
        .def("execute",
             &fvdb::GatherScatterBackend::execute,
             py::arg("input"),
             py::arg("weights"));
}
