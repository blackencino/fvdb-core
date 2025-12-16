// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/autograd/convolution/ConvGatherScatterBackendAG.h>
#include <fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.h>
#include <fvdb/detail/ops/convolution/pack_info/ConvolutionKernelMap.h>
#include <fvdb/detail/utils/Utils.h>

#include <format>

namespace fvdb {
namespace detail {
namespace autograd {

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
ConvGatherScatterBackendAG::topologyFromGridBatches(GridBatch sourceGrid,
                                                    GridBatch targetGrid,
                                                    nanovdb::Vec3i kernelSize,
                                                    nanovdb::Vec3i stride) -> Topology {
    auto const device = sourceGrid.device();
    TORCH_CHECK_VALUE(sourceGrid.device() == targetGrid.device(),
                      "Source and target grids must be on the same device");

    c10::DeviceGuard guard{device};

    auto const kernelVolume     = kernelSize[0] * kernelSize[1] * kernelSize[2];
    auto const outputVoxelCount = targetGrid.total_voxels();

    // The kernel map is for each output voxel, kernel volume input indices, which may be -1.
    // It's a 2d tensor of shape [output_voxels, kernel_volume].
    auto kmap = torch::full({outputVoxelCount, kernelVolume},
                            -1,
                            torch::TensorOptions().dtype(torch::kInt32).device(device));
    FVDB_DISPATCH_KERNEL_DEVICE(device, [&]() {
        detail::ops::dispatchConvolutionKernelMap<DeviceTag>(
            *sourceGrid.implRawPtr(), *targetGrid.implRawPtr(), kmap, kernelSize, stride);
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
        .sourceTotalVoxelCount = sourceGrid.total_voxels(),
        .targetTotalVoxelCount = targetGrid.total_voxels(),
        .kernelSize            = kernelSize,
        .stride                = stride,
    };
}

variable_list
ConvGatherScatterBackendAG::forward(AutogradContext *ctx,
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
ConvGatherScatterBackendAG::backward(AutogradContext *ctx, variable_list grad_output) {
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

#if 0



ConvolutionBackend::ConvolutionBackend(GridBatch sourceGrid,
    GridBatch targetGrid,
    nanovdb::Vec3i kernelSize,
    nanovdb::Vec3i stride,
    bool useTF32)
: mSourceGrid(sourceGrid)
, mTargetGrid(targetGrid)
, mKernelSize(kernelSize)
, mStride(stride)
, mUseTF32(useTF32) {
TORCH_CHECK(sourceGrid.device() == targetGrid.device(),
"Source and target grids must both be on the same device");
}

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
GatherScatterForwardMapping::GatherScatterForwardMapping(ConvolutionBackend const& backend) {
auto const device = backend.device();
auto const kernelVolume = backend.kernelVolume();
auto const outputVoxelCount = backend.targetGrid().total_voxels();

// The kernel map is for each output voxel, kernel volume input indices, which may be -1.
// It's a 2d tensor of shape [output_voxels, kernel_volume].
auto kmap =
torch::full({outputVoxelCount, kernelVolume},
-1,
torch::TensorOptions().dtype(torch::kInt32).device(device));
GridBatch::computeConvolutionKernelMap(backend.sourceGrid(), backend.targetGrid(), kmap,
backend.kernelSize(), backend.stride());
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
// kmap[i, j] is the index of the input voxel used for kernel position i at output voxel j, or -1 if not present.

// Compute the number of valid neighbors for each kernel element (not each output voxel!):
// For each row (kernel position), sum the number of valid (non -1) positions across all outputVoxels.
// This results in a tensor of shape [kernelVolume], where each entry is the count for that kernel position,
// telling you how many active output voxels there are for each kernel element.
auto const nbsizes = torch::sum(valid_mask, -1); // [kernelVolume]

// torch::nonzero returns the INDICES (at the rank of the input tensor) where the scalar value
// of the output tensor is true (non false, non zero).
auto nbmap = torch::nonzero(valid_mask).contiguous();
// [N, 2], with N = total neighbor pairs [input idx, output idx]

// These indices are into the flattened kmap tensor.
//
// The 'nbmap' tensor returned by torch::nonzero(valid_mask) is of shape [N, 2], where each row contains:
//   nbmap[i, 0] = kernel position index (k, from 0 to kernelVolume-1)
//   nbmap[i, 1] = output voxel index (j, from 0 to outputVoxelCount-1)
//
// The 'kmap' tensor is of shape [kernelVolume, outputVoxelCount], where:
//   kmap[k, j] = index of the input voxel for kernel position k and output voxel j (or -1 if not present)
//
// To gather the correct input voxel indices for each valid (k, j) pair in 'nbmap', we need to compute
// the linear index for each [k, j] into the flattened kmap tensor (which was reshaped with .reshape({-1})).
//
// For a 2D array of shape [rows, cols], the 1D (flattened) index of element (row, col) is:
//   idx = row * cols + col
// In our case:
//   idx = nbmap[i, 0] * outputVoxelCount + nbmap[i, 1]
//
// This gives us the correct position of the [k, j] entry inside the 1D version of 'kmap', so we can
// look up which input voxel index to put in the first column of 'nbmap'.
auto const flatKmapIndices = nbmap.index({torch::indexing::Slice(), 0}) * outputVoxelCount +
nbmap.index({torch::indexing::Slice(), 1}); // [N]

// Replace the first column in nbmap with the input voxel index found via `flatKmapIndices`.
// After this, nbmap[i, 0] contains the input voxel index used for this neighbor,
// and nbmap[i, 1] is the output voxel index.
// Each row is now: [input_voxel_idx, output_voxel_idx]
nbmap.index_put_({torch::indexing::Slice(), 0}, kmap.reshape({-1}).index({flatKmapIndices}));

// Save the results as int32 tensors.
// neighborMap is [N,2] where each row is (input_voxel_idx, output_voxel_idx).
// neighborSizes is [kernelVolume], where each entry is the number of valid output voxels for that kernel element.
neighborMap   = nbmap.to(torch::kInt32);
neighborSizes = nbsizes.to(torch::kInt32);
}

GatherScatterBackend::GatherScatterBackend(GridBatch sourceGrid,
GridBatch targetGrid,
nanovdb::Vec3i kernelSize,
nanovdb::Vec3i stride,
bool useTF32,
GatherScatterForwardMapping forwardMapping) : ConvolutionBackend(sourceGrid,
                                    targetGrid,
                                    kernelSize,
                                    stride,
                                    useTF32)
, mForwardMapping(forwardMapping) {
}

GatherScatterBackend::GatherScatterBackend(GridBatch sourceGrid,
        GridBatch targetGrid,
        nanovdb::Vec3i kernelSize,
        nanovdb::Vec3i stride,
        bool useTF32)
: ConvolutionBackend(sourceGrid,
targetGrid,
kernelSize, stride, useTF32)
, mForwardMapping(*this) {
}

std::unique_ptr<ConvolutionBackend> GatherScatterBackend::to(torch::Device device) const {
return std::make_unique<GatherScatterBackend>(mSourceGrid.to(device),
mTargetGrid.to(device),
mKernelSize,
mStride,
mUseTF32,
mForwardMapping.to(device));
}

JaggedTensor
GatherScatterBackend::forward(JaggedTensor const& input, torch::Tensor const& weights) const {
auto const _device = mSourceGrid.device();

TORCH_CHECK_VALUE(input.device() == _device, "Input must be on the same device as the backend");
TORCH_CHECK_VALUE(weights.device() == _device, "Weights must be on the same device as the backend");

// Check features
auto const input_jdata = input.jdata();
TORCH_CHECK_VALUE(input.jdata().is_contiguous(), "features must be contiguous");
TORCH_CHECK_TYPE(input_jdata.is_floating_point(), "features must have a floating point type");
TORCH_CHECK_VALUE(input_jdata.dim() == 2,
"Expected features to have 2 dimensions (shape (n, nF)) but got " +
std::to_string(input_jdata.dim()) + " dimensions");

// Check kernels
TORCH_CHECK_TYPE(weights.is_floating_point(), "kernels must have a floating point type");
for (int i = 0; i < weights.dim(); i += 1) {
TORCH_CHECK_VALUE(weights.size(i) != 0,
"kernels tensor has zero dimension (dim = " + std::to_string(i) + ")");
}

TORCH_CHECK_VALUE(input_jdata.size(0) == mSourceGrid.total_voxels(),
"The number of input features must match the number of voxels");
TORCH_CHECK_VALUE(weights.dim() == 5,
"Expected kernels to have 5 dimensions (shape (out_ch, in_ch, d, h, w)) "
"but got " +
std::to_string(weights.dim()) + " dimensions");
TORCH_CHECK_VALUE(weights.size(1) == input_jdata.size(1),
"Expected input channels of kernels (" + std::to_string(weights.size(1)) +
") to equal input channels of features: " +
std::to_string(input_jdata.size(1)));

auto const outC = weights.size(0);
auto const inC = weights.size(1);

torch::Tensor output;
if (mTargetGrid.total_voxels() > 0) {
auto const kernels = weights.permute({2, 3, 4, 1, 0}).reshape({-1, inC, outC}).contiguous();

auto opt = torch::TensorOptions().dtype(input_jdata.dtype()).device(_device);
output   = torch::zeros({mTargetGrid.total_voxels(), outC}, opt);

FVDB_DISPATCH_KERNEL_DEVICE(_device, [&]() {
detail::ops::dispatchSparseConvolutionKernelMap<DeviceTag>(input_jdata,
                                    output,
                                    kernels,
                                    mForwardMapping.neighborMap,
                                    mForwardMapping.neighborSizes.cpu().contiguous(),
                                    false /* not transposed */,
                                    middleAcceleration);
});
} else {
auto opt = torch::TensorOptions().dtype(inFeatures.dtype()).device(inFeatures.device());
output   = torch::empty({0, kernels.size(-1)}, opt);
}

return mTargetGrid.jagged_like(output);
}

// =============================================================================
// GatherScatterTransposeBackend Implementation
// =============================================================================

GatherScatterTransposeBackend::GatherScatterTransposeBackend(GridBatch sourceGrid,
                          GridBatch targetGrid,
                          Vec3i kernelSize,
                          Vec3i stride,
                          bool useTF32,
                          torch::Tensor neighborMap,
                          torch::Tensor neighborSizes)
: ConvolutionBackend(std::move(sourceGrid),
std::move(targetGrid),
kernelSize,
stride,
useTF32)
, mNeighborMap(std::move(neighborMap))
, mNeighborSizes(std::move(neighborSizes)) {}

std::unique_ptr<GatherScatterTransposeBackend>
GatherScatterTransposeBackend::create(GridBatch sourceGrid,
   GridBatch targetGrid,
   Vec3i kernelSize,
   Vec3i stride,
   bool useTF32) {
// Validate inputs
TORCH_CHECK(nanovdb::Coord(0, 0, 0) < kernelSize.value(),
"Expect kernel size to be larger than {0,0,0}, but got " + kernelSize.toString() +
".");
TORCH_CHECK(nanovdb::Coord(0, 0, 0) < stride.value(),
"Expect stride to be larger than 0, but got " + stride.toString() + ".");
TORCH_CHECK(sourceGrid.device() == targetGrid.device(),
"Source and target grids must both be on the same device");
TORCH_CHECK(!(kernelSize.value() == nanovdb::Coord(1, 1, 1) &&
stride.value() == nanovdb::Coord(1, 1, 1)),
"1x1 conv does not need kernel map to be built!");

// Build the kernel map (same as forward - the map structure is the same)
int kernelVolume = kernelSize.value().x() * kernelSize.value().y() * kernelSize.value().z();

torch::Tensor kmap =
torch::full({targetGrid.total_voxels(), kernelVolume},
-1,
torch::TensorOptions().dtype(torch::kInt32).device(targetGrid.device()));

GridBatch::computeConvolutionKernelMap(sourceGrid, targetGrid, kmap, kernelSize, stride);
kmap                  = kmap.t();
torch::Tensor kmask   = kmap != -1;
torch::Tensor nbsizes = torch::sum(kmask, -1);
torch::Tensor nbmap   = torch::nonzero(kmask).contiguous();

torch::Tensor indices = nbmap.index({torch::indexing::Slice(), 0}) * kmap.size(1) +
nbmap.index({torch::indexing::Slice(), 1});
nbmap.index_put_({torch::indexing::Slice(), 0}, kmap.reshape({-1}).index({indices}));
torch::Tensor neighborMap   = nbmap.to(torch::kInt32);
torch::Tensor neighborSizes = nbsizes.to(torch::kInt32);

return std::unique_ptr<GatherScatterTransposeBackend>(
new GatherScatterTransposeBackend(std::move(sourceGrid),
       std::move(targetGrid),
       kernelSize,
       stride,
       useTF32,
       std::move(neighborMap),
       std::move(neighborSizes)));
}

std::unique_ptr<GatherScatterTransposeBackend>
GatherScatterTransposeBackend::fromForward(const GatherScatterBackend &forwardBackend) {
// Share the same kernel map data (tensors are reference counted)
return std::unique_ptr<GatherScatterTransposeBackend>(
new GatherScatterTransposeBackend(forwardBackend.sourceGrid(),
       forwardBackend.targetGrid(),
       forwardBackend.kernelSize(),
       forwardBackend.stride(),
       forwardBackend.useTF32(),
       forwardBackend.neighborMap(),
       forwardBackend.neighborSizes()));
}

std::unique_ptr<GatherScatterTransposeBackend>
GatherScatterTransposeBackend::to(const torch::Device &device) const {
return std::unique_ptr<GatherScatterTransposeBackend>(
new GatherScatterTransposeBackend(mSourceGrid.to(device),
       mTargetGrid.to(device),
       mKernelSize,
       mStride,
       mUseTF32,
       mNeighborMap.to(device),
       mNeighborSizes.to(device)));
}

JaggedTensor
GatherScatterTransposeBackend::forward(const JaggedTensor &input,
    const torch::Tensor &weights) const {
// For transpose conv: input is on target grid, output is on source grid
TORCH_CHECK_VALUE(input.num_outer_lists() == mTargetGrid.grid_count(),
"Input batch size must match target grid batch size");
TORCH_CHECK_VALUE(input.element_count() == mTargetGrid.total_voxels(),
"Input element count must match target grid total voxels");

const torch::Tensor &inFeatures = input.jdata();
torch::Tensor kernels           = weights;

const std::vector<int> sizes    = {static_cast<int>(mSourceGrid.total_voxels()),
    static_cast<int>(mTargetGrid.total_voxels())};
const bool middleAcceleration   = mStride.value() == nanovdb::Coord(1, 1, 1);

// Check features
TORCH_CHECK_VALUE(inFeatures.is_contiguous(), "features must be contiguous");
TORCH_CHECK_TYPE(inFeatures.is_floating_point(), "features must have a floating point type");
TORCH_CHECK_VALUE(inFeatures.dim() == 2,
"Expected features to have 2 dimensions (shape (n, nF)) but got " +
std::to_string(inFeatures.dim()) + " dimensions");

// Check kernels
TORCH_CHECK_TYPE(kernels.is_floating_point(), "kernels must have a floating point type");
for (int i = 0; i < kernels.dim(); i += 1) {
TORCH_CHECK_VALUE(kernels.size(i) != 0,
"kernels tensor has zero dimension (dim = " + std::to_string(i) + ")");
}

// For transpose: input features are on target grid
TORCH_CHECK_VALUE(inFeatures.size(0) == sizes[1],
"The number of input features must match the number of voxels");
TORCH_CHECK_VALUE(kernels.dim() == 5,
"Expected kernels to have 5 dimensions (shape (in_ch, out_ch, d, h, w)) "
"but got " +
std::to_string(kernels.dim()) + " dimensions");
TORCH_CHECK_VALUE(kernels.size(0) == inFeatures.size(1),
"Expected input channels of kernels (" + std::to_string(kernels.size(0)) +
") to equal input channels of features: " +
std::to_string(inFeatures.size(1)));

const int inC = kernels.size(0), outC = kernels.size(1);
kernels = kernels.permute({2, 3, 4, 0, 1}).reshape({-1, inC, outC}).contiguous();

torch::Tensor output;
if (mSourceGrid.total_voxels() > 0) {
auto opt = torch::TensorOptions().dtype(inFeatures.dtype()).device(inFeatures.device());
output   = torch::zeros({sizes[0], kernels.size(-1)}, opt);

FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
detail::ops::dispatchSparseConvolutionKernelMap<DeviceTag>(inFeatures,
                                    output,
                                    kernels,
                                    mNeighborMap,
                                    mNeighborSizes.cpu().contiguous(),
                                    true /* transposed */,
                                    middleAcceleration);
});
} else {
auto opt = torch::TensorOptions().dtype(inFeatures.dtype()).device(inFeatures.device());
output   = torch::empty({0, kernels.size(-1)}, opt);
}

return mSourceGrid.jagged_like(output);
}

#endif

} // namespace autograd
} // namespace detail
} // namespace fvdb
