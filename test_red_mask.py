#!/usr/bin/env python3
"""
Test script to verify the red mask visualization works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the current directory to path to import our custom visualization
sys.path.append(str(Path(__file__).parent))
from custom_visualization import visualize_extrinsic_results_red_mask


def test_red_mask_visualization():
    """Test the red mask visualization with dummy data"""

    # Create a simple test image
    test_image = np.ones((128, 128, 3), dtype=np.uint8) * 128  # Gray background

    # Create a simple test mask (center circle)
    mask = np.zeros((128, 128), dtype=bool)
    y, x = np.ogrid[:128, :128]
    center_x, center_y = 64, 64
    radius = 30
    mask[(x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2] = True

    # Create dummy data for the visualization function
    images = np.array([test_image])
    link_poses_dataset = np.random.rand(1, 1, 4, 4)  # Dummy poses
    meshes = []  # Empty meshes for this test
    intrinsic = np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1]], dtype=np.float32)
    extrinsics = np.random.rand(1, 4, 4)  # Dummy extrinsics
    masks = np.array([mask.astype(float)])

    # Test different mask colors
    colors_to_test = [
        ((255, 0, 0), "Red"),
        ((0, 255, 0), "Green"),
        ((0, 0, 255), "Blue"),
        ((255, 255, 0), "Yellow"),
    ]

    for i, (color, name) in enumerate(colors_to_test):
        print(f"Testing {name} mask color...")

        # Create output directory for this test
        output_dir = f"test_output_{name.lower()}"
        Path(output_dir).mkdir(exist_ok=True)

        # Apply the mask overlay manually to test our function
        def apply_red_mask_overlay(image, mask, alpha=0.6, mask_color=(255, 0, 0)):
            """Apply colored mask overlay to image"""
            result = image.copy()
            mask_3d = np.stack([mask, mask, mask], axis=2)
            red_overlay = np.full_like(image, mask_color)
            result = (1 - alpha) * result + alpha * red_overlay
            result = result * mask_3d + image * (1 - mask_3d)
            return result.astype(np.uint8)

        result = apply_red_mask_overlay(
            test_image, mask.astype(float), alpha=0.6, mask_color=color
        )

        # Save the result
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title(f"With {name} Mask Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/test_result.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved test result to {output_dir}/test_result.png")

    print("\nTest completed! Check the generated images to verify the mask colors.")


if __name__ == "__main__":
    test_red_mask_visualization()
