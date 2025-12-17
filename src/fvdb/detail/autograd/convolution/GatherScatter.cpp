// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/autograd/convolution/GatherScatter.h>
#include <fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.h>
#include <fvdb/detail/ops/convolution/pack_info/ConvolutionKernelMap.h>
#include <fvdb/detail/utils/Utils.h>

#include <format>

namespace fvdb {
namespace detail {
namespace autograd {
namespace convolution {

namespace {

void
_checkFeatures(torch::Tensor features) {
    TORCH_CHECK_VALUE(features.is_contiguous(), "features must be contiguous");
    TORCH_CHECK_TYPE(features.is_floating_point(), "features must have a floating point type");
    TORCH_CHECK_VALUE(
        features.dim() == 2,
        std::format("Expected features to have 2 dimensions (shape (n, nF)) but got {} dimensions",
                    features.dim()));
}

void
_checkReshapedKernels(torch::Tensor kernels, int expectedInChannels, int expectedOutChannels = -1) {
    TORCH_CHECK_TYPE(kernels.is_floating_point(), "kernels must have a floating point type");

    TORCH_CHECK_VALUE(
        kernels.dim() == 5,
        std::format(
            "Expected reshaped kernels to have 5 dimensions (shape (k0, k1, k2, inC, outC)) but "
            "got {} dimensions",
            kernels.dim()));

    TORCH_CHECK_VALUE(kernels.size(3) == expectedInChannels,
                      std::format("Expected input channels of kernels ({}) to equal input channels "
                                  "of features: {}",
                                  kernels.size(3),
                                  expectedInChannels));
    if (expectedOutChannels != -1) {
        TORCH_CHECK_VALUE(
            kernels.size(4) == expectedOutChannels,
            std::format("Expected output channels of kernels ({}) to equal output channels "
                        "of features: {}",
                        kernels.size(4),
                        expectedOutChannels));
    }
    for (int i = 0; i < 5; ++i) {
        TORCH_CHECK_VALUE(kernels.size(i) != 0,
                          std::format("Reshaped kernels tensor has zero dimension (dim = {})", i));
    }
}

void
_checkTopology(torch::Tensor nbmaps, torch::Tensor nbsizes) {
    TORCH_CHECK(nbmaps.is_contiguous() && nbmaps.scalar_type() == torch::kInt32,
                "nbmaps must be contiguous and of type int32");
    TORCH_CHECK(nbsizes.is_contiguous() && nbsizes.scalar_type() == torch::kInt32,
                "nbsizes must be contiguous and of type int32");
    TORCH_CHECK(nbsizes.device().is_cpu(),
                "nbsizes tensor must be on CPU, but got device: ",
                nbsizes.device());
    //
}

} // namespace

// TODO(chorvath): This function is inefficient. It allocates a dense intermediate tensor of shape
// [outputVoxels, kernelVolume], fills it via computeConvolutionKernelMap, then post-processes with
// expensive PyTorch ops (transpose, mask, sum, nonzero, gather) to produce the sparse neighbor map.
//
// For sparse data with a 3x3x3 kernel, maybe only 10-30% of entries are valid, yet we allocate 100%
// of the dense tensor and scan all of it with torch::nonzero to find the valid entries.
//
// Better approach - modify computeConvolutionKernelMap (or create a new variant) to directly output
// the sparse neighbor map in one or two passes:
//   - Two-pass: (1) count valid pairs per kernel position, allocate exact-size output,
//               (2) fill sparse (input_idx, output_idx) pairs directly using prefix-sum offsets.
//   - Single-pass with atomics: each thread atomically appends to output buckets.
//
// This would reduce memory from O(outputVoxels * kernelVolume) to O(validPairs), and avoid
// multiple scan passes over the dense intermediate. The CUDA infrastructure (forEachVoxelCUDA,
// sparse grid accessors) already exists - only the output format needs changing.
auto
GatherScatterAutograd::topologyFromBackendConfig(BackendConfig config) -> Topology {
    auto const device = config.sourceGrid.device();
    TORCH_CHECK_VALUE(config.sourceGrid.device() == config.targetGrid.device(),
                      "Source and target grids must be on the same device");

    c10::DeviceGuard guard{device};

    auto const kernelVolume = config.kernelSize[0] * config.kernelSize[1] * config.kernelSize[2];
    auto const outputVoxelCount = config.targetGrid.total_voxels();

    // The kernel map is for each output voxel, kernel volume input indices, which may be -1.
    // It's a 2d tensor of shape [output_voxels, kernel_volume].
    auto kmap = torch::full({outputVoxelCount, kernelVolume},
                            -1,
                            torch::TensorOptions().dtype(torch::kInt32).device(device));
    FVDB_DISPATCH_KERNEL_DEVICE(device, [&]() {
        detail::ops::dispatchConvolutionKernelMap<DeviceTag>(*config.sourceGrid.implRawPtr(),
                                                             *config.targetGrid.implRawPtr(),
                                                             kmap,
                                                             config.kernelSize,
                                                             config.stride);
    });
    // SHAPE: [outputVoxelCount, kernelVolume]

    // Here we transpose it so now it is [kernelVolume, outputVoxelCount] in shape.
    // It represents the indices of the input voxels for each kernel element in the output voxels.
    // TODO: (chorvath): just build it transposed in the first place. This is the only place we
    // use it anyway.
    kmap = kmap.t(); // SHAPE: [kernelVolume, outputVoxelCount]

    // Get the valid indices.
    auto const valid_mask = (kmap != -1); // SHAPE: [kernelVolume, outputVoxelCount]

    // We start from `kmap`, which has shape [kernelVolume, outputVoxelCount] after transpose.
    // For each kernel element (i in kernelVolume), and each output voxel (j in outputVoxelCount),
    // kmap[i, j] is the index of the input voxel used for kernel position i at output voxel j, or
    // -1 if not present.

    // Compute the number of valid neighbors for each kernel element (not each output voxel!):
    // For each row (kernel position), sum the number of valid (non -1) positions across all
    // outputVoxels. This results in a tensor of shape [kernelVolume], where each entry is the count
    // for that kernel position, telling you how many active output voxels there are for each kernel
    // element.
    auto const nbsizes = torch::sum(valid_mask, -1); // [kernelVolume]

    // torch::nonzero returns the INDICES (at the rank of the input tensor) where the scalar value
    // of the output tensor is true (non false, non zero).
    auto nbmap = torch::nonzero(valid_mask).contiguous();
    // [N, 2], with N = total neighbor pairs [input idx, output idx]

    // These indices are into the flattened kmap tensor.
    //
    // The 'nbmap' tensor returned by torch::nonzero(valid_mask) is of shape [N, 2], where each row
    // contains:
    //   nbmap[i, 0] = kernel position index (k, from 0 to kernelVolume-1)
    //   nbmap[i, 1] = output voxel index (j, from 0 to outputVoxelCount-1)
    //
    // The 'kmap' tensor is of shape [kernelVolume, outputVoxelCount], where:
    //   kmap[k, j] = index of the input voxel for kernel position k and output voxel j (or -1 if
    //   not present)
    //
    // To gather the correct input voxel indices for each valid (k, j) pair in 'nbmap', we need to
    // compute the linear index for each [k, j] into the flattened kmap tensor (which was reshaped
    // with .reshape({-1})).
    //
    // For a 2D array of shape [rows, cols], the 1D (flattened) index of element (row, col) is:
    //   idx = row * cols + col
    // In our case:
    //   idx = nbmap[i, 0] * outputVoxelCount + nbmap[i, 1]
    //
    // This gives us the correct position of the [k, j] entry inside the 1D version of 'kmap', so we
    // can look up which input voxel index to put in the first column of 'nbmap'.
    auto const flatKmapIndices = nbmap.index({torch::indexing::Slice(), 0}) * outputVoxelCount +
                                 nbmap.index({torch::indexing::Slice(), 1}); // [N]

    // Replace the first column in nbmap with the input voxel index found via `flatKmapIndices`.
    // After this, nbmap[i, 0] contains the input voxel index used for this neighbor,
    // and nbmap[i, 1] is the output voxel index.
    // Each row is now: [input_voxel_idx, output_voxel_idx]
    nbmap.index_put_({torch::indexing::Slice(), 0}, kmap.reshape({-1}).index({flatKmapIndices}));

    // Save the results as int32 tensors.
    // neighborMap is [N,2] where each row is (input_voxel_idx, output_voxel_idx).
    // neighborSizes is [kernelVolume], where each entry is the number of valid output voxels for
    // that kernel element.
    return Topology{
        .neighborMap           = nbmap.to(torch::kInt32),
        .neighborSizes         = nbsizes.to(torch::kInt32),
        .sourceTotalVoxelCount = config.sourceGrid.total_voxels(),
        .targetTotalVoxelCount = config.targetGrid.total_voxels(),
        .kernelSize            = config.kernelSize,
        .stride                = config.stride,
    };
}

variable_list
GatherScatterAutograd::forward(AutogradContext *ctx,
                               torch::Tensor inFeatures,
                               torch::Tensor kernels,
                               Topology topology) {
    bool const middleAcceleration = topology.stride == nanovdb::Vec3i{1, 1, 1};

    _checkFeatures(inFeatures);
    auto const inC = inFeatures.size(1);

    auto const nbsizes_cpu_contiguous = topology.neighborSizes.cpu().contiguous();
    _checkTopology(topology.neighborMap, nbsizes_cpu_contiguous);

    // Reorder the memory, but don't erase dimensions
    auto const reshapedKernels = kernels.permute({2, 3, 4, 1, 0}).contiguous();
    _checkReshapedKernels(reshapedKernels, inC);
    auto const outC = reshapedKernels.size(4);

    // Save, which will preserve the various sizes, but also cache the work of making the
    // permutation contigous.
    ctx->save_for_backward({inFeatures, reshapedKernels});
    ctx->saved_data["nbmaps"]  = topology.neighborMap;
    ctx->saved_data["nbsizes"] = nbsizes_cpu_contiguous;

    torch::Tensor output;
    auto const opts = tensorOptionsFrom(inFeatures);
    if (topology.targetTotalVoxelCount > 0) {
        output = torch::zeros({topology.targetTotalVoxelCount, outC}, opts);

        FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
            ops::dispatchSparseConvolutionKernelMap<DeviceTag>(
                inFeatures,
                output,
                reshapedKernels.reshape({-1, inC, outC}),
                topology.neighborMap,
                nbsizes_cpu_contiguous,
                false,
                middleAcceleration);
        });
    } else {
        output = torch::empty({0, outC}, opts);
    }

    return {output};
}

variable_list
GatherScatterAutograd::backward(AutogradContext *ctx, variable_list grad_output) {
    // Use data saved in forward
    variable_list saved        = ctx->get_saved_variables();
    auto const inFeatures      = saved.at(0);
    auto const reshapedKernels = saved.at(1);
    auto const nbmaps          = ctx->saved_data["nbmaps"].toTensor();
    auto const nbsizes         = ctx->saved_data["nbsizes"].toTensor();
    auto const gradOut         = grad_output.at(0);

    auto const k0   = reshapedKernels.size(0);
    auto const k1   = reshapedKernels.size(1);
    auto const k2   = reshapedKernels.size(2);
    auto const inC  = inFeatures.size(1);
    auto const outC = gradOut.size(1);

    _checkFeatures(inFeatures);
    _checkReshapedKernels(reshapedKernels, inC, outC);
    _checkTopology(nbmaps, nbsizes);

    auto const flatk = reshapedKernels.reshape({-1, inC, outC});

    auto gradInput  = torch::zeros_like(inFeatures);
    auto gradWeight = torch::zeros_like(flatk);

    if (gradOut.size(0) != 0) {
        FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
            ops::dispatchSparseConvolutionKernelMapGrad<DeviceTag>(inFeatures,
                                                                   gradInput,
                                                                   gradOut.contiguous(),
                                                                   flatk,
                                                                   gradWeight,
                                                                   nbmaps,
                                                                   nbsizes,
                                                                   false);
        });
    }

    // The kernel map is built iterating k0 (slowest), k1, k2 (fastest) in ConvolutionKernelMap.cu,
    // so the flat kernel index corresponds to (k0, k1, k2) in row-major order.
    // We must reshape in the same order to correctly map gradients back to kernel positions.
    gradWeight = gradWeight.reshape({k0, k1, k2, inC, outC}).permute({4, 3, 0, 1, 2});

    return {gradInput, gradWeight, torch::Tensor()};
}

} // namespace convolution
} // namespace autograd
} // namespace detail
} // namespace fvdb
