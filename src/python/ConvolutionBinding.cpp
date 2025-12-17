// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <pybind11/cast.h>
#include <pybind11/stl.h>

#include <fvdb/ConvolutionBackends.h>

#include <torch/extension.h>

void
bind_convolution(py::module &m) {
    using namespace fvdb::detail::autograd::convolution;
    // Bind BackendConfig struct
    py::class_<BackendConfig>(m, "ConvBackendConfig")
        .def_readonly("source_grid", &BackendConfig::sourceGrid)
        .def_readonly("target_grid", &BackendConfig::targetGrid)
        .def_property_readonly("kernel_size",
                               [](BackendConfig const &self) {
                                   return std::array<int, 3>{
                                       self.kernelSize[0], self.kernelSize[1], self.kernelSize[2]};
                               })
        .def_property_readonly("stride",
                               [](BackendConfig const &self) {
                                   return std::array<int, 3>{
                                       self.stride[0], self.stride[1], self.stride[2]};
                               })
        .def("to", &BackendConfig::to, py::arg("device"));

    // Bind GatherScatterAutograd::Topology struct
    py::class_<GatherScatterAutograd::Topology>(m, "GatherScatterAutogradTopology")
        .def_readonly("neighbor_map", &GatherScatterAutograd::Topology::neighborMap)
        .def_readonly("neighbor_sizes", &GatherScatterAutograd::Topology::neighborSizes)
        .def_readonly("source_total_voxel_count",
                      &GatherScatterAutograd::Topology::sourceTotalVoxelCount)
        .def_readonly("target_total_voxel_count",
                      &GatherScatterAutograd::Topology::targetTotalVoxelCount)
        .def_property_readonly("kernel_size",
                               [](GatherScatterAutograd::Topology const &self) {
                                   return std::array<int, 3>{
                                       self.kernelSize[0], self.kernelSize[1], self.kernelSize[2]};
                               })
        .def_property_readonly("stride",
                               [](GatherScatterAutograd::Topology const &self) {
                                   return std::array<int, 3>{
                                       self.stride[0], self.stride[1], self.stride[2]};
                               })
        .def("to", &GatherScatterAutograd::Topology::to, py::arg("device"));

    // Bind ConvBackendGatherScatter struct
    py::class_<fvdb::ConvBackendGatherScatter>(m, "ConvBackendGatherScatter")
        .def_readonly("config", &fvdb::ConvBackendGatherScatter::config)
        .def_readonly("topology", &fvdb::ConvBackendGatherScatter::topology)
        .def_static("create",
                    &fvdb::ConvBackendGatherScatter::create,
                    py::arg("source_grid"),
                    py::arg("target_grid"),
                    py::arg("kernel_size"),
                    py::arg("stride"),
                    py::arg("expert_config") = std::map<std::string, std::string>{})
        .def("to", &fvdb::ConvBackendGatherScatter::to, py::arg("device"))
        .def("execute",
             &fvdb::ConvBackendGatherScatter::execute,
             py::arg("input"),
             py::arg("weights"));
}
