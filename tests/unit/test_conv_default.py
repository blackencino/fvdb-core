# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the default sparse convolution.

"""

import math
import unittest

import torch
from fvdb.types import DeviceIdentifier, resolve_device
from fvdb.utils.tests import (
    fourier_anti_symmetric_kernel,
    generate_hermit_impulses_dense,
    has_any_symmetry,
)
from fvdb.utils.tests.convolution_utils import (
    assert_coords_equal,
    compute_conv_grid_topology_ground_truth,
    conv_ground_truth_stride_1,
    disable_tf32,
    sort_coords_by_ijk,
)
from parameterized import parameterized

from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

# =============================================================================
# Test Configuration
# =============================================================================

ALL_DEVICE_DTYPE_COMBOS = [
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]

# Reduced coverage for tests where device/dtype doesn't affect the property being tested
REDUCED_DEVICE_DTYPE_COMBOS = [
    ["cuda", torch.float32],
]


# =============================================================================
# Test-Specific Helpers
# =============================================================================


def create_grid_from_coords(
    coords: torch.Tensor,
    device: torch.device,
) -> GridBatch:
    """Create a GridBatch from coordinate tensor."""
    ijks = JaggedTensor(coords.to(device=device, dtype=torch.int32))
    return GridBatch.from_ijk(ijks, device=device)


def get_cluster_near_origin(device: torch.device) -> torch.Tensor:
    """
    Get a sparse cluster of coordinates near the origin.

    This cluster includes coordinates at (0,0,0) which will produce negative
    output coordinates when convolved with stride=1.
    """
    return torch.tensor(
        [
            # Group 1: At/near origin - produces negative output coords with stride=1
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 2],
            # Group 2: Slightly separated - some kernel overlap with group 1
            [3, 4, 5],
            [4, 4, 5],
            [3, 5, 6],
            # Group 3: Further out - tests larger coordinate range
            [7, 8, 9],
        ],
        device=device,
        dtype=torch.int32,
    )


def get_cluster_edge_aligned(
    kernel_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Get a sparse cluster positioned so outputs stay non-negative.

    The minimum coordinate is offset by (half_kernel + 1) to ensure
    all output coordinates are non-negative.
    """
    kernel_half = tuple(k // 2 for k in kernel_size)
    base = tuple(k + 1 for k in kernel_half)  # Extra margin

    return torch.tensor(
        [
            [base[0] + 0, base[1] + 0, base[2] + 0],
            [base[0] + 2, base[1] + 0, base[2] + 1],
            [base[0] + 1, base[1] + 2, base[2] + 0],
            [base[0] + 3, base[1] + 3, base[2] + 3],
            [base[0] + 5, base[1] + 4, base[2] + 5],
        ],
        device=device,
        dtype=torch.int32,
    )


# =============================================================================
# Test Class
# =============================================================================


class TestConvDefault(unittest.TestCase):

    VOLUME_SHAPE = (71, 34, 58)
    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)
    NUM_CANDIDATES = 1000

    def setUp(self):
        torch.random.manual_seed(2024)

    # =========================================================================
    # Topology Tests (conv_grid)
    # =========================================================================

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_single_impulse_bounds(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Validate that conv_grid output bounds match expected kernel footprint
        for a single input coordinate.
        """
        device = resolve_device(device)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Check count
        kernel_volume = math.prod(self.KERNEL_SIZE)
        self.assertEqual(len(dst_ijks), kernel_volume)

        # Check bounds
        expected_start = tuple(self.SINGLE_COORD[i] - kernel_half[i] for i in range(3))
        expected_end = tuple(self.SINGLE_COORD[i] + kernel_half[i] + 1 for i in range(3))

        actual_start = tuple(dst_ijks[:, dim].min().item() for dim in range(3))
        actual_end = tuple(dst_ijks[:, dim].max().item() + 1 for dim in range(3))

        self.assertEqual(actual_start, expected_start)
        self.assertEqual(actual_end, expected_end)

    def _test_conv_grid_topology(
        self,
        device: DeviceIdentifier,
        dtype: torch.dtype,
        stride: tuple[int, int, int],
        cluster_coords: torch.Tensor,
        check_negative_outputs: bool = False,
        check_non_negative_outputs: bool = False,
    ):
        """
        Core topology test: verify conv_grid output matches dense ground truth.

        Args:
            device: Device identifier
            dtype: Data type
            stride: Stride tuple (s0, s1, s2)
            cluster_coords: Input coordinates
            check_negative_outputs: Assert that some outputs have negative coords
            check_non_negative_outputs: Assert that all outputs are non-negative
        """
        device = resolve_device(device)
        cluster_coords = cluster_coords.to(device=device)

        # Create grid and get conv_grid output
        grid_batch = create_grid_from_coords(cluster_coords, device)
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Compute ground truth
        expected_coords = compute_conv_grid_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
            dtype=dtype,
        )

        # Compare
        assert_coords_equal(
            dst_ijks,
            expected_coords,
            msg=f"stride={stride}",
        )

        # Additional checks
        if check_negative_outputs:
            has_negative = (dst_ijks < 0).any()
            self.assertTrue(has_negative, "Expected some negative output coordinates")

        if check_non_negative_outputs:
            all_non_negative = (dst_ijks >= 0).all()
            self.assertTrue(all_non_negative, "Expected all non-negative output coordinates")

    # --- Stride 1 tests (full device/dtype coverage) ---

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_topology_stride1_near_origin(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with coordinates near origin, stride=1."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_near_origin(device_resolved)
        self._test_conv_grid_topology(
            device,
            dtype,
            stride=(1, 1, 1),
            cluster_coords=cluster_coords,
            check_negative_outputs=True,
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_topology_stride1_edge_aligned(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with edge-aligned coordinates, stride=1."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_conv_grid_topology(
            device,
            dtype,
            stride=(1, 1, 1),
            cluster_coords=cluster_coords,
            check_non_negative_outputs=True,
        )

    # --- Strided tests (reduced device/dtype coverage) ---

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_topology_stride_uniform(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with uniform stride (2,2,2)."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_conv_grid_topology(
            device,
            dtype,
            stride=(2, 2, 2),
            cluster_coords=cluster_coords,
        )

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_topology_stride_nonuniform(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with non-uniform stride (1,2,3)."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_conv_grid_topology(
            device,
            dtype,
            stride=(1, 2, 3),
            cluster_coords=cluster_coords,
        )

    # =========================================================================
    # Convolution Value Tests (forward pass)
    # =========================================================================

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_activation_and_weights(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that each kernel weight position activates the correct output coordinate.

        For a single input impulse, iterates over each kernel position and verifies
        that the output impulse appears at the expected coordinate.
        """
        device = resolve_device(device)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)
        features = JaggedTensor(torch.ones((1, 1), device=device, dtype=dtype))

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )

        for k0 in range(self.KERNEL_SIZE[0]):
            for k1 in range(self.KERNEL_SIZE[1]):
                for k2 in range(self.KERNEL_SIZE[2]):
                    # Kernel with single impulse at (k0, k1, k2)
                    weights = torch.zeros((1, 1, *self.KERNEL_SIZE), device=device, dtype=dtype)
                    weights[0, 0, k0, k1, k2] = 1

                    output = conv_plan.execute(features, weights)
                    output_flat = output.jdata.flatten()

                    # Find the non-zero output location
                    nonzero_mask = output_flat != 0
                    self.assertEqual(nonzero_mask.sum().item(), 1)
                    got_coord = tuple(dst_ijks[nonzero_mask].flatten().tolist())

                    # Expected: input_coord - (kernel_idx - kernel_half)
                    expected_coord = (
                        self.SINGLE_COORD[0] - (k0 - kernel_half[0]),
                        self.SINGLE_COORD[1] - (k1 - kernel_half[1]),
                        self.SINGLE_COORD[2] - (k2 - kernel_half[2]),
                    )
                    self.assertEqual(got_coord, expected_coord)

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_forward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test forward convolution of a single impulse against dense ground truth.

        Creates a single-voxel input, convolves with an anti-symmetric kernel,
        and verifies the sparse output matches the dense convolution result.
        """
        device = resolve_device(device)
        half_kernel = tuple(k // 2 for k in self.KERNEL_SIZE)

        # Validate test configuration
        expected_volume = tuple(k + 2 for k in self.KERNEL_SIZE)
        expected_coord = tuple(1 + k for k in half_kernel)
        self.assertEqual(expected_volume, self.SINGLE_VOLUME_SHAPE)
        self.assertEqual(expected_coord, self.SINGLE_COORD)

        # Create input
        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)
        features = JaggedTensor(torch.ones((1, 1), device=device, dtype=dtype))

        # Anti-symmetric kernel (no symmetry that could hide bugs)
        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        kernel_5d = kernel.reshape(1, 1, *self.KERNEL_SIZE)
        kernel_sum = kernel_5d.sum().item()

        # Dense ground truth
        dense_input = torch.zeros((1, 1) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        dense_input[0, 0, coord[0], coord[1], coord[2]] = 1

        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=kernel_5d, padding="same")

        self.assertAlmostEqual(dense_output.sum().item(), kernel_sum, places=5)

        # Sparse convolution
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )
        self.assertEqual(str(conv_plan._backend), "ConvPackBackend.GATHER_SCATTER")

        sparse_output = conv_plan.execute(features, kernel_5d)
        sparse_output_flat = sparse_output.jdata.flatten()

        # Verify sparse matches dense at output locations
        dense_at_dst = dense_output[0, 0, dst_ijks[:, 0], dst_ijks[:, 1], dst_ijks[:, 2]]
        torch.testing.assert_close(sparse_output_flat, dense_at_dst, rtol=1e-5, atol=1e-6)

        # Verify sum matches kernel sum
        self.assertAlmostEqual(sparse_output_flat.sum().item(), kernel_sum, places=5)

        # Also test the utility function
        gt_activation, gt_convolved = conv_ground_truth_stride_1(
            grid_batch=grid_batch,
            activation=features,
            weights=kernel_5d,
            dense_dims=self.SINGLE_VOLUME_SHAPE,
            ijk_min=(0, 0, 0),
            allow_tf32=False,
        )
        torch.testing.assert_close(gt_convolved, dense_output, rtol=1e-5, atol=1e-6)

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_multiple_impulses_forward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test forward convolution of multiple isolated impulses.

        Uses hermit impulses (no kernel overlap) to ensure each impulse
        contributes independently to the output.
        """
        device = resolve_device(device)

        # Generate hermit impulses (no overlap)
        impulse_coords, impulse_field = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=dtype,
            device=device,
        )
        num_impulses = len(impulse_coords)

        # Anti-symmetric kernel
        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        self.assertTrue(torch.all(kernel != 0))
        kernel_5d = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        # Dense ground truth
        dense_input = impulse_field.reshape(1, 1, *self.VOLUME_SHAPE)
        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=kernel_5d, padding="same")

        # Sparse convolution
        grid_batch = create_grid_from_coords(impulse_coords, device)
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Verify topology matches dense non-zeros
        dense_nonzero = torch.nonzero(dense_output[0, 0]).to(torch.int32)
        assert_coords_equal(dst_ijks, dense_nonzero)

        # Execute convolution
        features = JaggedTensor(torch.ones((num_impulses, 1), device=device, dtype=dtype))
        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )

        sparse_output = conv_plan.execute(features, kernel_5d)
        sparse_flat = sparse_output.jdata.flatten()

        # Compare values (need to align ordering)
        dst_sorted, dst_perm = sort_coords_by_ijk(dst_ijks)
        dense_sorted, _ = sort_coords_by_ijk(dense_nonzero)

        dense_values_sorted = dense_output[0, 0, dense_sorted[:, 0], dense_sorted[:, 1], dense_sorted[:, 2]]
        sparse_values_sorted = sparse_flat[dst_perm]

        torch.testing.assert_close(sparse_values_sorted, dense_values_sorted, rtol=1e-5, atol=1e-6)

    # =========================================================================
    # Backward Pass Tests
    # =========================================================================

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_backward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test backward pass of sparse convolution against dense ground truth.

        Creates a single impulse, runs forward and backward passes on both
        sparse and dense implementations, and verifies gradients match.
        """
        device = resolve_device(device)
        half_kernel = tuple(k // 2 for k in self.KERNEL_SIZE)

        # Validate config
        expected_coord = tuple(1 + k for k in half_kernel)
        self.assertEqual(expected_coord, self.SINGLE_COORD)

        # Create input with gradient tracking
        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)

        features_data = torch.ones((1, 1), device=device, dtype=dtype, requires_grad=True)
        features = JaggedTensor(features_data)

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Kernel with gradient tracking
        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        kernel_5d = kernel.reshape(1, 1, *self.KERNEL_SIZE).clone().requires_grad_(True)

        # === Dense path ===
        dense_input = torch.zeros((1, 1) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        dense_input[0, 0, coord[0], coord[1], coord[2]] = 1
        dense_input = dense_input.clone().requires_grad_(True)
        dense_kernel = kernel_5d.detach().clone().requires_grad_(True)

        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=dense_kernel, padding="same")

        # === Sparse path ===
        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )
        sparse_output = conv_plan.execute(features, kernel_5d)

        # Verify forward match
        dense_at_dst = dense_output[0, 0, dst_ijks[:, 0], dst_ijks[:, 1], dst_ijks[:, 2]]
        torch.testing.assert_close(sparse_output.jdata.flatten(), dense_at_dst, rtol=1e-5, atol=1e-6)

        # === Backward ===
        # Create output gradient at center coordinate
        grad_coord = self.SINGLE_COORD

        # Dense gradient
        dense_grad = torch.zeros_like(dense_output)
        dense_grad[0, 0, grad_coord[0], grad_coord[1], grad_coord[2]] = 1
        dense_output.backward(dense_grad)

        # Sparse gradient
        grad_coord_tensor = torch.tensor(grad_coord, device=device, dtype=torch.int32)
        grad_idx = int(torch.nonzero((dst_ijks == grad_coord_tensor).all(dim=1)).squeeze().item())
        sparse_grad = torch.zeros_like(sparse_output.jdata)
        sparse_grad[grad_idx, 0] = 1
        sparse_output.jdata.backward(sparse_grad)

        # Compare input gradients
        dense_input_grad = dense_input.grad
        sparse_input_grad = features_data.grad
        assert dense_input_grad is not None and sparse_input_grad is not None

        dense_grad_at_coord = dense_input_grad[0, 0, coord[0], coord[1], coord[2]]
        torch.testing.assert_close(
            sparse_input_grad.flatten(),
            dense_grad_at_coord.unsqueeze(0),
            rtol=1e-5,
            atol=1e-6,
        )

        # Compare kernel gradients
        dense_kernel_grad = dense_kernel.grad
        sparse_kernel_grad = kernel_5d.grad
        assert dense_kernel_grad is not None and sparse_kernel_grad is not None

        torch.testing.assert_close(sparse_kernel_grad, dense_kernel_grad, rtol=1e-5, atol=1e-6)
