# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the PyTorch ground truth for convolutions in 3D.

The tests in this file will validate that the PyTorch dense versions of convolution
do what we expect them to do, and establish a set of baseline demonstrations of
convolution properties.
"""

import unittest

import torch
from fvdb.types import DeviceIdentifier, resolve_device
from fvdb.utils.tests import (
    fourier_anti_symmetric_kernel,
    generate_hermit_impulses_dense,
    has_any_symmetry,
)
from fvdb.utils.tests.convolution_utils import disable_tf32
from parameterized import parameterized

all_device_dtype_combos = [
    # ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]


def _validate_impulse_convolution(
    impulse_coord: torch.Tensor,
    kernel: torch.Tensor,
    convolved: torch.Tensor,
    kernel_size: tuple,
    test_case: unittest.TestCase,
    check_bounds: bool,
) -> None:
    """
    Validate that a convolution of an impulse at a specific coordinate produces the expected kernel.

    This function extracts the region around the impulse coordinate from the convolution result,
    flips it (since PyTorch conv3d performs cross-correlation), and compares it to the kernel.

    Args:
        impulse_coord: The coordinate of the impulse as a torch.Tensor
        kernel: The kernel used for convolution
        convolved: The result of the convolution
        kernel_size: The size of the kernel as a tuple
        test_case: The unittest.TestCase instance for assertions
        check_bounds: Whether to check that the bounds of the non-zero region exactly match the expected bounds
    """
    # Extract the region around the impulse coordinate where the kernel should appear
    # The kernel is centered at the impulse coordinate
    kernel_half = tuple(k // 2 for k in kernel_size)

    # Define the slice boundaries for extracting the kernel region
    start_coords = tuple(max(0, impulse_coord[i].item() - kernel_half[i]) for i in range(3))
    end_coords = tuple(min(convolved.shape[i + 1], impulse_coord[i].item() + kernel_half[i] + 1) for i in range(3))

    if check_bounds:
        # The non-zero region of convolved should be exactly inside the bounds we just computed.
        # Check that the bounds of the non-zero region exactly match the expected bounds
        non_zero_mask = convolved[0] != 0
        non_zero_coords = torch.nonzero(non_zero_mask)

        if non_zero_coords.shape[0] > 0:
            # Find the actual bounds of the non-zero region
            actual_start_coords = tuple(non_zero_coords[:, dim].min().item() for dim in range(3))
            actual_end_coords = tuple(non_zero_coords[:, dim].max().item() + 1 for dim in range(3))

            # The actual bounds should exactly match the expected bounds
            test_case.assertEqual(actual_start_coords, start_coords)
            test_case.assertEqual(actual_end_coords, end_coords)

    # Extract the convolved region around the impulse
    convolved_region = convolved[
        0, start_coords[0] : end_coords[0], start_coords[1] : end_coords[1], start_coords[2] : end_coords[2]
    ]

    test_case.assertEqual(convolved_region.shape, kernel.shape)

    # Since PyTorch conv3d performs cross-correlation, we need to flip to get the result to
    # match the kernel.
    convolved_region = torch.flip(convolved_region, dims=[0, 1, 2])

    # The flipped convolved region should match the kernel
    torch.testing.assert_close(convolved_region, kernel, rtol=1e-5, atol=1e-6)


class TestConvGroundTruth(unittest.TestCase):

    VOLUME_SHAPE = (71, 34, 58)
    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)
    NUM_CANDIDATES = 1000

    def setUp(self):
        torch.random.manual_seed(2024)

    @parameterized.expand(all_device_dtype_combos)
    def test_single_impulse(self, device: DeviceIdentifier, dtype: torch.dtype):
        device = resolve_device(device)

        # For single impulse, we just need to make sure it's far enough away from
        # the boundary of the volume.

        expected_volume_shape = tuple(a + 2 for a in self.KERNEL_SIZE)
        expected_impulse_coord = tuple(1 + a // 2 for a in self.KERNEL_SIZE)

        self.assertEqual(expected_volume_shape, self.SINGLE_VOLUME_SHAPE)
        self.assertEqual(expected_impulse_coord, self.SINGLE_COORD)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)

        impulse_field[0, coord[0], coord[1], coord[2]] = 1

        self.assertEqual(impulse_field.sum().item(), 1)

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))

        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        # Do a single convolution
        with disable_tf32():
            convolved = torch.nn.functional.conv3d(input=impulse_field, weight=kernel_with_channels, padding="same")
        self.assertEqual(impulse_field.shape, convolved.shape)

        # We know where the impulse coordinate is, so we should be able to test that the
        # convolution matches the kernel. Even though PyTorch calls it conv3d, it's actually a
        # cross-correlation, per the documentation. Therefore, convolving an impulse with a kernel
        # should produce exactly the kernel. We can test this by extracting the region around the impulse
        # coordinate and comparing it to the kernel.

        # Use the helper function to validate the impulse convolution
        _validate_impulse_convolution(
            impulse_coord=coord,
            kernel=kernel,
            convolved=convolved,
            kernel_size=self.KERNEL_SIZE,
            test_case=self,
            check_bounds=True,
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_single_impulse_activation_and_weights(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        This test iterates over each single weight location in the kernel space,
        creates a kernel that has just an impulse at that location, and convolves a single impulse
        conv grid with it. For each location within the kernel, we should expect the output to have
        a single impulse at the activation impulse coord minus the centered kernel impulse coord
        """
        device = resolve_device(device)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)

        impulse_field[0, coord[0], coord[1], coord[2]] = 1

        self.assertEqual(impulse_field.sum().item(), 1)

        kernel_half_width = tuple(k // 2 for k in self.KERNEL_SIZE)

        for k0 in range(self.KERNEL_SIZE[0]):
            for k1 in range(self.KERNEL_SIZE[1]):
                for k2 in range(self.KERNEL_SIZE[2]):
                    # Create a kernel that has just an impulse at the current location
                    weights = torch.zeros((1, 1, *self.KERNEL_SIZE), device=device, dtype=dtype)
                    weights[0, 0, k0, k1, k2] = 1
                    self.assertEqual(weights.sum().item(), 1)

                    convolved = torch.nn.functional.conv3d(
                        input=impulse_field, weight=weights, stride=1, padding="same"
                    )
                    self.assertEqual(impulse_field.shape, convolved.shape)
                    self.assertEqual(convolved.sum().item(), 1)

                    # Expected output coordinate
                    nonzero_coords = torch.nonzero(convolved[0])
                    self.assertEqual(nonzero_coords.shape[0], 1)
                    centered_kernel_coord = (
                        k0 - kernel_half_width[0],
                        k1 - kernel_half_width[1],
                        k2 - kernel_half_width[2],
                    )
                    expected_output_coord = (
                        self.SINGLE_COORD[0] - centered_kernel_coord[0],
                        self.SINGLE_COORD[1] - centered_kernel_coord[1],
                        self.SINGLE_COORD[2] - centered_kernel_coord[2],
                    )
                    got_output_coord = tuple(nonzero_coords[0].tolist())
                    self.assertEqual(got_output_coord, expected_output_coord)

    @parameterized.expand(all_device_dtype_combos)
    def test_multiple_impulses(self, device: DeviceIdentifier, dtype: torch.dtype):
        device = resolve_device(device)

        impulse_coords, impulse_field = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=dtype,
            device=device,
        )

        num_impulses = len(impulse_coords)
        print(f"Number of generated impulses: {num_impulses}")

        total_value = torch.sum(impulse_field).item()
        print(f"Total sum of impulse_field: {total_value}")

        # Test that the impulse field's shape matches the volume shape
        self.assertEqual(impulse_field.shape, self.VOLUME_SHAPE)

        # Test that the total value of the impulse field matches the total number of impulses
        self.assertEqual(round(total_value), num_impulses)

        # Get a kernel and convolve the impulse field with it
        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))

        impulse_field_with_channel = impulse_field.reshape(1, *self.VOLUME_SHAPE)
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        with disable_tf32():
            convolved = torch.nn.functional.conv3d(
                input=impulse_field_with_channel, weight=kernel_with_channels, padding="same"
            )
        self.assertEqual(impulse_field_with_channel.shape, convolved.shape)

        for i in range(num_impulses):
            impulse_coord = impulse_coords[i]
            _validate_impulse_convolution(
                impulse_coord=impulse_coord,
                kernel=kernel,
                convolved=convolved,
                kernel_size=self.KERNEL_SIZE,
                test_case=self,
                check_bounds=False,
            )

    # ==================================================================================
    # Backward convolution tests
    # ==================================================================================
    #
    # These tests verify that PyTorch's conv3d backward pass produces expected gradients.
    # The backward pass computes:
    #   - Gradient w.r.t. input: dy convolved with flipped weights (or conv_transpose(dy, w))
    #   - Gradient w.r.t. weights: correlation of input with output gradient
    #
    # For cross-correlation (which PyTorch conv3d implements), if an impulse in the output
    # gradient at coord C produces input gradients, those gradients should form the original
    # kernel centered at C (since backward effectively convolves dy with flip(flip(w)) = w).

    @parameterized.expand(all_device_dtype_combos)
    def test_single_impulse_backward_input_grad(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that backward pass w.r.t. input produces expected gradients.

        If we place an impulse in the output gradient at a specific coordinate,
        the input gradient should contain the kernel (not flipped) centered at
        that coordinate. This is because the backward of cross-correlation
        involves convolving with the flipped kernel, which reverses the flip
        that happened in forward.
        """
        device = resolve_device(device)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        impulse_field[0, coord[0], coord[1], coord[2]] = 1
        impulse_field.requires_grad_(True)

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        with disable_tf32():
            # Forward pass
            output = torch.nn.functional.conv3d(input=impulse_field, weight=kernel_with_channels, padding="same")

            # Create output gradient with impulse at the same coord
            output_grad = torch.zeros_like(output)
            output_grad[0, coord[0], coord[1], coord[2]] = 1

            # Backward pass
            output.backward(output_grad)

        input_grad = impulse_field.grad
        assert input_grad is not None

        # Extract the region around the impulse coordinate
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)
        start_coords = tuple(int(coord[i].item()) - kernel_half[i] for i in range(3))
        end_coords = tuple(int(coord[i].item()) + kernel_half[i] + 1 for i in range(3))

        input_grad_region = input_grad[
            0, start_coords[0] : end_coords[0], start_coords[1] : end_coords[1], start_coords[2] : end_coords[2]
        ]

        self.assertEqual(input_grad_region.shape, kernel.shape)

        # For backward of cross-correlation, the input gradient should be the
        # original kernel (not flipped). This is because backward involves
        # conv_transpose which flips the kernel, canceling the forward flip.
        torch.testing.assert_close(input_grad_region, kernel, rtol=1e-5, atol=1e-6)

    @parameterized.expand(all_device_dtype_combos)
    def test_single_impulse_backward_weight_grad(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that backward pass w.r.t. weights produces expected gradients.

        For an impulse input at coord C and an impulse output gradient at coord G,
        the weight gradient at kernel position (i,j,k) equals:
            input[G + centered_offset(i,j,k)] * output_grad[G]

        With both impulses at the same coord and value 1, only the center weight
        (where centered_offset = 0) should have gradient = 1.
        """
        device = resolve_device(device)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        impulse_field[0, coord[0], coord[1], coord[2]] = 1

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE).clone()
        kernel_with_channels.requires_grad_(True)

        with disable_tf32():
            # Forward pass
            output = torch.nn.functional.conv3d(input=impulse_field, weight=kernel_with_channels, padding="same")

            # Create output gradient with impulse at the same coord
            output_grad = torch.zeros_like(output)
            output_grad[0, coord[0], coord[1], coord[2]] = 1

            # Backward pass
            output.backward(output_grad)

        weight_grad = kernel_with_channels.grad
        assert weight_grad is not None

        # For cross-correlation backward w.r.t. weights:
        # dw[i,j,k] = sum over spatial of x[s + i - k//2] * dy[s]
        # With impulse input at coord and impulse output grad at coord:
        # Only the center weight (i,j,k) = (k0//2, k1//2, k2//2) should have grad = 1
        kernel_center = tuple(k // 2 for k in self.KERNEL_SIZE)
        expected_grad = torch.zeros_like(weight_grad)
        expected_grad[0, 0, kernel_center[0], kernel_center[1], kernel_center[2]] = 1

        torch.testing.assert_close(weight_grad, expected_grad, rtol=1e-5, atol=1e-6)

    @parameterized.expand(all_device_dtype_combos)
    def test_single_impulse_backward_weight_grad_offset(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test weight gradient when input and output gradient impulses are at different coords.

        When input impulse is at coord_in and output gradient impulse is at coord_out,
        the weight gradient should have a single non-zero entry at the kernel position
        corresponding to the offset between them.
        """
        device = resolve_device(device)

        # Input impulse at center
        coord_in = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        impulse_field[0, coord_in[0], coord_in[1], coord_in[2]] = 1

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE).clone()
        kernel_with_channels.requires_grad_(True)

        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        # Test a few different offset positions within the kernel
        test_offsets = [
            (0, 0, 0),  # Center
            (1, 0, 0),  # Offset in first axis
            (0, -1, 1),  # Mixed offset
        ]

        for offset in test_offsets:
            # Reset gradient
            if kernel_with_channels.grad is not None:
                kernel_with_channels.grad.zero_()

            # Output gradient impulse at offset from input
            coord_out = (
                int(coord_in[0].item()) + offset[0],
                int(coord_in[1].item()) + offset[1],
                int(coord_in[2].item()) + offset[2],
            )

            with disable_tf32():
                output = torch.nn.functional.conv3d(input=impulse_field, weight=kernel_with_channels, padding="same")

                output_grad = torch.zeros_like(output)
                output_grad[0, coord_out[0], coord_out[1], coord_out[2]] = 1

                output.backward(output_grad)

            weight_grad = kernel_with_channels.grad
            assert weight_grad is not None

            # The non-zero weight gradient should be at kernel position:
            # (k//2 - offset[0], k//2 - offset[1], k//2 - offset[2])
            expected_kernel_pos = tuple(kernel_half[i] - offset[i] for i in range(3))

            # Check that exactly one position has non-zero gradient
            nonzero_mask = weight_grad[0, 0] != 0
            nonzero_coords = torch.nonzero(nonzero_mask)
            self.assertEqual(nonzero_coords.shape[0], 1, f"Expected 1 non-zero grad for offset {offset}")

            actual_pos = tuple(nonzero_coords[0].tolist())
            self.assertEqual(actual_pos, expected_kernel_pos, f"Wrong grad position for offset {offset}")

    @parameterized.expand(all_device_dtype_combos)
    def test_multiple_impulses_backward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test backward pass with multiple non-overlapping impulses.

        Similar to test_multiple_impulses but for the backward pass. Each impulse
        in the output gradient should independently produce a kernel in the input
        gradient without interference.
        """
        device = resolve_device(device)

        impulse_coords, impulse_field = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=dtype,
            device=device,
        )

        num_impulses = len(impulse_coords)
        print(f"Number of generated impulses for backward test: {num_impulses}")

        impulse_field_with_channel = impulse_field.reshape(1, *self.VOLUME_SHAPE)
        impulse_field_with_channel = impulse_field_with_channel.clone().requires_grad_(True)

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        with disable_tf32():
            # Forward pass
            output = torch.nn.functional.conv3d(
                input=impulse_field_with_channel, weight=kernel_with_channels, padding="same"
            )

            # Create output gradient with impulses at the same coords as input
            output_grad = torch.zeros_like(output)
            for i in range(num_impulses):
                c = impulse_coords[i]
                output_grad[0, c[0], c[1], c[2]] = 1

            # Backward pass
            output.backward(output_grad)

        input_grad = impulse_field_with_channel.grad
        assert input_grad is not None

        # Verify each impulse produces the expected kernel in input gradient
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        for i in range(num_impulses):
            coord = impulse_coords[i]
            start_coords = tuple(max(0, int(coord[j].item()) - kernel_half[j]) for j in range(3))
            end_coords = tuple(
                min(input_grad.shape[j + 1], int(coord[j].item()) + kernel_half[j] + 1) for j in range(3)
            )

            # Extract region
            grad_region = input_grad[
                0, start_coords[0] : end_coords[0], start_coords[1] : end_coords[1], start_coords[2] : end_coords[2]
            ]

            # Only check if the region is fully within bounds (no clipping)
            expected_shape = tuple(end_coords[j] - start_coords[j] for j in range(3))
            if expected_shape == self.KERNEL_SIZE:
                torch.testing.assert_close(grad_region, kernel, rtol=1e-5, atol=1e-6)

    # ==================================================================================
    # Strided Convolution Tests
    # ==================================================================================
    #
    # Strided convolution changes the relationship between input and output coordinates.
    # Understanding this relationship is crucial for implementing sparse strided convolution.
    #
    # Key concepts for strided convolution (stride=S, kernel_size=K, padding=K//2):
    #
    # 1. OUTPUT COORDINATE MAPPING:
    #    For an input of size N with padding P and stride S:
    #      output_size = floor((N + 2*P - K) / S) + 1
    #
    #    With P = K//2 (half-padding), output[o] sees input centered at o*S.
    #
    # 2. WHICH INPUTS CONTRIBUTE TO WHICH OUTPUTS:
    #    Output at coordinate o receives contributions from inputs in the range:
    #      [o*S - K//2, o*S + K//2]
    #
    #    Conversely, an input at coordinate c contributes to outputs in the range:
    #      [ceil((c - K//2) / S), floor((c + K//2) / S)]
    #
    # 3. DOWNSAMPLING EFFECT:
    #    With stride S > 1, multiple input coordinates can map to the same output.
    #    For example, with stride=2 and kernel_size=3:
    #      - Input at c=0 contributes to output o=0 (since ceil((0-1)/2)=0, floor((0+1)/2)=0)
    #      - Input at c=1 contributes to output o=0 and o=1
    #      - Input at c=2 contributes to output o=1
    #
    # These tests validate these assumptions about PyTorch's strided conv3d.

    # Reduced coverage for strided tests (behavior is dtype/device independent)
    reduced_device_dtype_combos = [
        ["cuda", torch.float32],
    ]

    @parameterized.expand(reduced_device_dtype_combos)
    def test_strided_impulse_output_location(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that strided convolution produces output at the expected coordinates.

        With stride=S and half-padding, an impulse at input coordinate c produces
        non-zero output at coordinates o where:
            o*S - K//2 <= c <= o*S + K//2
        i.e., o in [ceil((c - K//2) / S), floor((c + K//2) / S)]

        This test validates this relationship for various input positions and strides.
        """
        device = resolve_device(device)

        # Use a simple 3x3x3 kernel for clarity
        kernel_size = (3, 3, 3)
        kernel_half = tuple(k // 2 for k in kernel_size)

        # Test different strides
        test_cases = [
            # (stride, input_coord, expected_output_coords)
            # stride=2: input at 0 -> output at 0 (ceil((0-1)/2)=0, floor((0+1)/2)=0)
            ((2, 2, 2), (4, 4, 4), [(2, 2, 2)]),
            # stride=2: input at 1 -> outputs at 0,1 (ceil((1-1)/2)=0, floor((1+1)/2)=1)
            (
                (2, 2, 2),
                (5, 5, 5),
                [(2, 2, 2), (2, 2, 3), (2, 3, 2), (2, 3, 3), (3, 2, 2), (3, 2, 3), (3, 3, 2), (3, 3, 3)],
            ),
            # stride=3: input at 0 -> output at 0
            ((3, 3, 3), (6, 6, 6), [(2, 2, 2)]),
        ]

        for stride, input_coord, expected_outputs in test_cases:
            # Create volume large enough for the test
            volume_size = tuple(max(c + k + 2 for c, k in zip(input_coord, kernel_size)) for _ in range(3))
            volume_size = (volume_size[0],) * 3  # Make it cubic for simplicity
            volume_size = (max(volume_size[0], 20),) * 3

            impulse_field = torch.zeros((1, 1) + volume_size, device=device, dtype=dtype)
            impulse_field[0, 0, input_coord[0], input_coord[1], input_coord[2]] = 1

            # All-ones kernel to see which outputs are activated
            kernel = torch.ones((1, 1) + kernel_size, device=device, dtype=dtype)

            with disable_tf32():
                output = torch.nn.functional.conv3d(
                    input=impulse_field, weight=kernel, stride=stride, padding=kernel_half
                )

            # Find non-zero output coordinates
            nonzero_coords = torch.nonzero(output[0, 0])
            actual_outputs = [tuple(c.tolist()) for c in nonzero_coords]

            # Verify the expected outputs are present
            for expected in expected_outputs:
                self.assertIn(
                    expected,
                    actual_outputs,
                    f"stride={stride}, input={input_coord}: expected output at {expected} not found. "
                    f"Actual outputs: {actual_outputs}",
                )

    @parameterized.expand(reduced_device_dtype_combos)
    def test_strided_activation_and_weights(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test the relationship between kernel position, stride, and output coordinate.

        For strided convolution, when we have:
        - An impulse at input coordinate c
        - A kernel with a single weight at position (k0, k1, k2)
        - Stride S and half-padding

        The output impulse appears at coordinate:
            o = floor((c + K//2 - k) / S)

        where k is the kernel index. This is because:
        - With half-padding, the kernel position k contributes to output o when
          the kernel is positioned such that position k aligns with input c.
        - The output coordinate o satisfies: c = o*S + k - K//2
        - Therefore: o = (c - k + K//2) / S

        This test validates this relationship.
        """
        device = resolve_device(device)

        kernel_size = (3, 3, 3)
        kernel_half = tuple(k // 2 for k in kernel_size)
        stride = (2, 2, 2)

        # Input impulse at a position that allows testing various kernel positions
        input_coord = (6, 6, 6)
        volume_size = (20, 20, 20)

        impulse_field = torch.zeros((1, 1) + volume_size, device=device, dtype=dtype)
        impulse_field[0, 0, input_coord[0], input_coord[1], input_coord[2]] = 1

        # Test each kernel position
        for k0 in range(kernel_size[0]):
            for k1 in range(kernel_size[1]):
                for k2 in range(kernel_size[2]):
                    # Create kernel with impulse at (k0, k1, k2)
                    kernel = torch.zeros((1, 1) + kernel_size, device=device, dtype=dtype)
                    kernel[0, 0, k0, k1, k2] = 1

                    with disable_tf32():
                        output = torch.nn.functional.conv3d(
                            input=impulse_field, weight=kernel, stride=stride, padding=kernel_half
                        )

                    # Find the output coordinate
                    nonzero_coords = torch.nonzero(output[0, 0])

                    # Compute expected output coordinate
                    # o = floor((c - k + K//2) / S) for each dimension
                    # But we need to be careful: the convolution shifts by (K//2 - k)
                    # So: output_idx = floor((input_idx + K//2 - k) / S)
                    # Wait, let me think more carefully...
                    #
                    # In PyTorch conv3d with padding P and stride S:
                    # output[o] = sum over k of input[o*S + k - P] * weight[k]
                    #
                    # With P = K//2, we have:
                    # output[o] = sum over k of input[o*S + k - K//2] * weight[k]
                    #
                    # For impulse at input_coord c and weight only at kernel position kp:
                    # output[o] = input[o*S + kp - K//2] * weight[kp]
                    #           = 1 if o*S + kp - K//2 == c, else 0
                    #           = 1 if o = (c - kp + K//2) / S
                    #
                    # This is only an integer when (c - kp + K//2) is divisible by S

                    expected_o = []
                    for dim in range(3):
                        c = input_coord[dim]
                        kp = [k0, k1, k2][dim]
                        kh = kernel_half[dim]
                        s = stride[dim]
                        # o*S = c - kp + kh
                        # o = (c - kp + kh) / S
                        numerator = c - kp + kh
                        if numerator % s == 0 and numerator >= 0:
                            expected_o.append(numerator // s)
                        else:
                            expected_o.append(None)

                    if all(o is not None for o in expected_o):
                        # Should have exactly one output
                        self.assertEqual(
                            len(nonzero_coords),
                            1,
                            f"kernel_pos=({k0},{k1},{k2}): expected 1 output, got {len(nonzero_coords)}",
                        )
                        actual_o = tuple(nonzero_coords[0].tolist())
                        expected_o_tuple = tuple(expected_o)
                        self.assertEqual(
                            actual_o,
                            expected_o_tuple,
                            f"kernel_pos=({k0},{k1},{k2}): expected output at {expected_o_tuple}, got {actual_o}",
                        )
                    else:
                        # Should have no output (the position doesn't align with stride grid)
                        self.assertEqual(
                            len(nonzero_coords),
                            0,
                            f"kernel_pos=({k0},{k1},{k2}): expected 0 outputs, got {len(nonzero_coords)}",
                        )

    @parameterized.expand(reduced_device_dtype_combos)
    def test_strided_backward_input_grad(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test backward pass w.r.t. input for strided convolution.

        For strided convolution backward:
        - The gradient w.r.t. input is computed via transposed convolution
        - An output gradient impulse at coord o produces input gradients at coords
          in the range [o*S - K//2, o*S + K//2]
        - The gradient values follow the kernel pattern (not flipped, due to
          the transpose operation canceling the forward flip)

        This test validates that the input gradient pattern and coordinates
        are as expected for strided convolution.
        """
        device = resolve_device(device)

        kernel_size = (3, 3, 3)
        kernel_half = tuple(k // 2 for k in kernel_size)
        stride = (2, 2, 2)
        volume_size = (20, 20, 20)

        # Create input (doesn't matter much, we care about grad location)
        input_coord = (6, 6, 6)
        impulse_field = torch.zeros((1, 1) + volume_size, device=device, dtype=dtype)
        impulse_field[0, 0, input_coord[0], input_coord[1], input_coord[2]] = 1
        impulse_field.requires_grad_(True)

        # Anti-symmetric kernel
        kernel = fourier_anti_symmetric_kernel(kernel_size, dtype=dtype, device=device)
        kernel_5d = kernel.reshape(1, 1, *kernel_size)

        with disable_tf32():
            output = torch.nn.functional.conv3d(
                input=impulse_field, weight=kernel_5d, stride=stride, padding=kernel_half
            )

            # Output gradient impulse at a specific coordinate
            output_coord = (3, 3, 3)
            output_grad = torch.zeros_like(output)
            output_grad[0, 0, output_coord[0], output_coord[1], output_coord[2]] = 1

            output.backward(output_grad)

        input_grad = impulse_field.grad
        assert input_grad is not None

        # The input gradient should be non-zero in the region
        # [o*S - K//2, o*S + K//2] for each dimension
        expected_center = tuple(output_coord[i] * stride[i] for i in range(3))
        expected_start = tuple(expected_center[i] - kernel_half[i] for i in range(3))
        expected_end = tuple(expected_center[i] + kernel_half[i] + 1 for i in range(3))

        # Extract the gradient region
        grad_region = input_grad[
            0,
            0,
            expected_start[0] : expected_end[0],
            expected_start[1] : expected_end[1],
            expected_start[2] : expected_end[2],
        ]

        self.assertEqual(grad_region.shape, kernel_size)

        # The gradient should match the kernel (not flipped)
        # This is because backward of conv is conv_transpose, which flips the kernel,
        # canceling the flip from the forward cross-correlation
        torch.testing.assert_close(grad_region, kernel, rtol=1e-5, atol=1e-6)

        # Also verify that the gradient is zero outside this region
        # (Create a mask for the expected region and check the rest is zero)
        mask = torch.zeros_like(input_grad, dtype=torch.bool)
        mask[
            0,
            0,
            expected_start[0] : expected_end[0],
            expected_start[1] : expected_end[1],
            expected_start[2] : expected_end[2],
        ] = True
        outside_region = input_grad[~mask]
        self.assertTrue(
            torch.all(outside_region == 0),
            f"Expected zeros outside grad region, but found non-zeros",
        )

    @parameterized.expand(reduced_device_dtype_combos)
    def test_strided_backward_weight_grad(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test backward pass w.r.t. weights for strided convolution.

        For strided convolution:
        - The weight gradient at kernel position k is the sum of products:
          sum over o of input[o*S + k - K//2] * output_grad[o]

        With an impulse input at coord c and impulse output grad at coord o:
        - Weight grad is non-zero at position k where c = o*S + k - K//2
        - i.e., k = c - o*S + K//2

        This test validates this relationship.
        """
        device = resolve_device(device)

        kernel_size = (3, 3, 3)
        kernel_half = tuple(k // 2 for k in kernel_size)
        stride = (2, 2, 2)
        volume_size = (20, 20, 20)

        # Input impulse at a specific coordinate
        input_coord = (6, 6, 6)
        impulse_field = torch.zeros((1, 1) + volume_size, device=device, dtype=dtype)
        impulse_field[0, 0, input_coord[0], input_coord[1], input_coord[2]] = 1

        kernel = fourier_anti_symmetric_kernel(kernel_size, dtype=dtype, device=device)
        kernel_5d = kernel.reshape(1, 1, *kernel_size).clone().requires_grad_(True)

        with disable_tf32():
            output = torch.nn.functional.conv3d(
                input=impulse_field, weight=kernel_5d, stride=stride, padding=kernel_half
            )

            # Output gradient impulse at coordinate that sees the input
            # For input at c=6 with stride=2 and K//2=1:
            # Output o sees input at o*2 + k - 1 for k in [0,1,2]
            # So output o=3 sees inputs at [5,6,7] (for k=0,1,2)
            output_coord = (3, 3, 3)
            output_grad = torch.zeros_like(output)
            output_grad[0, 0, output_coord[0], output_coord[1], output_coord[2]] = 1

            output.backward(output_grad)

        weight_grad = kernel_5d.grad
        assert weight_grad is not None

        # Expected kernel position: k = c - o*S + K//2
        expected_k = tuple(input_coord[i] - output_coord[i] * stride[i] + kernel_half[i] for i in range(3))

        # Check that it's within kernel bounds
        in_bounds = all(0 <= expected_k[i] < kernel_size[i] for i in range(3))
        self.assertTrue(in_bounds, f"Expected kernel position {expected_k} out of bounds")

        # Should have exactly one non-zero gradient
        nonzero_mask = weight_grad[0, 0] != 0
        nonzero_coords = torch.nonzero(nonzero_mask)
        self.assertEqual(len(nonzero_coords), 1, f"Expected 1 non-zero weight grad, got {len(nonzero_coords)}")

        actual_k = tuple(nonzero_coords[0].tolist())
        self.assertEqual(actual_k, expected_k, f"Expected weight grad at {expected_k}, got {actual_k}")
