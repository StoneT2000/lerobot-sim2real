"""Unit tests for camera.py utilities."""

import pytest
import numpy as np

from lerobot_sim2real.utils.camera import scale_intrinsics


def test_scale_intrinsics_no_scaling():
    """Test that scaling to the same dimensions returns the same intrinsics."""
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    original_width = 640
    original_height = 480
    new_width = 640
    new_height = 480

    result = scale_intrinsics(
        intrinsics, original_width, original_height, new_width, new_height
    )

    np.testing.assert_array_almost_equal(result, intrinsics)


def test_scale_intrinsics_uniform_downscale():
    """Test uniform downscaling (halving both dimensions)."""
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    original_width = 640
    original_height = 480
    new_width = 320
    new_height = 240

    result = scale_intrinsics(
        intrinsics, original_width, original_height, new_width, new_height
    )

    # With uniform scaling, factor = new_height / original_height = 0.5
    # fx, fy, cx, cy should all be scaled by 0.5
    expected = np.array([[250.0, 0, 160.0], [0, 250.0, 120.0], [0, 0, 1]])

    np.testing.assert_array_almost_equal(result, expected)


def test_scale_intrinsics_uniform_upscale():
    """Test uniform upscaling (doubling both dimensions)."""
    intrinsics = np.array([[250.0, 0, 160.0], [0, 250.0, 120.0], [0, 0, 1]])
    original_width = 320
    original_height = 240
    new_width = 640
    new_height = 480

    result = scale_intrinsics(
        intrinsics, original_width, original_height, new_width, new_height
    )

    # With uniform scaling, factor = new_height / original_height = 2.0
    expected = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])

    np.testing.assert_array_almost_equal(result, expected)


def test_scale_intrinsics_width_crop():
    """Test scaling where width needs to be cropped (original aspect wider)."""
    # Original: 1920x1080 (16:9), New: 1080x1080 (1:1)
    # This should crop the width
    intrinsics = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])
    original_width = 1920
    original_height = 1080
    new_width = 1080
    new_height = 1080

    result = scale_intrinsics(
        intrinsics, original_width, original_height, new_width, new_height
    )

    # crop_width = 1080 * (1080 / 1080) = 1080
    # cx adjustment: 960 - (1920 - 1080) / 2 = 960 - 420 = 540
    # factor = 1080 / 1080 = 1.0
    # Expected: fx=1000, fy=1000, cx=540, cy=540
    expected = np.array([[1000.0, 0, 540.0], [0, 1000.0, 540.0], [0, 0, 1]])

    np.testing.assert_array_almost_equal(result, expected)


def test_scale_intrinsics_height_crop():
    """Test scaling where height needs to be cropped (original aspect taller)."""
    # Original: 1080x1920 (9:16), New: 1080x1080 (1:1)
    # This should crop the height
    intrinsics = np.array([[1000.0, 0, 540.0], [0, 1000.0, 960.0], [0, 0, 1]])
    original_width = 1080
    original_height = 1920
    new_width = 1080
    new_height = 1080

    result = scale_intrinsics(
        intrinsics, original_width, original_height, new_width, new_height
    )

    # crop_height = 1080 * (1080 / 1080) = 1080
    # cy adjustment: 960 - (1920 - 1080) / 2 = 960 - 420 = 540
    # factor = 1080 / 1920 ≈ 0.5625
    # Expected: fx≈562.5, fy≈562.5, cx≈303.75, cy≈303.75
    factor = 1080 / 1920
    expected = np.array(
        [
            [1000.0 * factor, 0, 540.0 * factor],
            [0, 1000.0 * factor, 540.0 * factor],
            [0, 0, 1],
        ]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_scale_intrinsics_aspect_ratio_change():
    """Test scaling with aspect ratio change (16:9 to 4:3)."""
    # Original: 640x480 (4:3), New: 320x180 (16:9)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    original_width = 640
    original_height = 480
    new_width = 320
    new_height = 180

    result = scale_intrinsics(
        intrinsics, original_width, original_height, new_width, new_height
    )

    # original_width / new_width = 640/320 = 2.0
    # original_height / new_height = 480/180 = 2.667
    # Since 2.0 < 2.667, we take the else branch (crop height)
    # crop_height = 640 * (180 / 320) = 360
    # cy adjustment: 240 - (480 - 360) / 2 = 240 - 60 = 180
    # factor = 180 / 480 = 0.375
    factor = 180 / 480
    cx_adjusted = 320 * factor
    cy_adjusted = 180 * factor
    fx = 500 * factor
    fy = 500 * factor

    expected = np.array([[fx, 0, cx_adjusted], [0, fy, cy_adjusted], [0, 0, 1]])

    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_scale_intrinsics_preserves_bottom_row():
    """Test that the bottom row [0, 0, 1] is always preserved."""
    intrinsics = np.array([[600.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]])
    original_width = 800
    original_height = 600
    new_width = 400
    new_height = 300

    result = scale_intrinsics(
        intrinsics, original_width, original_height, new_width, new_height
    )

    # Check bottom row is [0, 0, 1]
    np.testing.assert_array_equal(result[2, :], np.array([0, 0, 1]))


def test_scale_intrinsics_off_diagonal_zeros():
    """Test that off-diagonal elements remain zero."""
    intrinsics = np.array([[700.0, 0, 512.0], [0, 700.0, 384.0], [0, 0, 1]])
    original_width = 1024
    original_height = 768
    new_width = 512
    new_height = 384

    result = scale_intrinsics(
        intrinsics, original_width, original_height, new_width, new_height
    )

    # Check off-diagonal elements are zero
    assert result[0, 1] == 0
    assert result[1, 0] == 0
    assert result[2, 0] == 0
    assert result[2, 1] == 0


@pytest.mark.parametrize(
    "original_dims,new_dims",
    [
        ((640, 480), (320, 240)),  # Half size
        ((1920, 1080), (960, 540)),  # Half HD to qHD
        ((800, 600), (400, 300)),  # SVGA to half
        ((1280, 720), (640, 360)),  # HD to half
    ],
)
def test_scale_intrinsics_various_resolutions(original_dims, new_dims):
    """Test scaling works correctly for various common resolutions."""
    original_width, original_height = original_dims
    new_width, new_height = new_dims

    intrinsics = np.array(
        [[500.0, 0, original_width / 2], [0, 500.0, original_height / 2], [0, 0, 1]]
    )

    result = scale_intrinsics(
        intrinsics, original_width, original_height, new_width, new_height
    )

    # Check result shape is 3x3
    assert result.shape == (3, 3)

    # Check that focal lengths are positive
    assert result[0, 0] > 0
    assert result[1, 1] > 0

    # Check that principal point is within image bounds
    assert 0 < result[0, 2] < new_width
    assert 0 < result[1, 2] < new_height
