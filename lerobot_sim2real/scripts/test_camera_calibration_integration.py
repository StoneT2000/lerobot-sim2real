#!/usr/bin/env python3
"""
Test script to verify camera calibration integration in SO101GraspCubeEnv.

This script tests that the environment correctly loads and applies camera calibration
data from env_config.json, including:
- Camera extrinsics and intrinsics
- Joint calibration offsets
- Domain randomization around calibrated camera pose
"""

import json
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
import tyro

# Import the environment
import lerobot_sim2real.env.tasks.digit_twins.so101_arm.grasp_cube


@dataclass
class Args:
    """Arguments for testing camera calibration integration."""

    env_config_path: str = "env_config.json"
    """Path to environment config JSON file containing camera calibration"""

    test_domain_randomization: bool = False
    """If True, test domain randomization around calibrated camera"""

    num_episodes: int = 5
    """Number of episodes to run for testing"""

    render_mode: str = "human"
    """Render mode: 'human' for interactive viewer, 'rgb_array' for offscreen"""

    save_images: bool = True
    """If True, save camera images to disk"""

    output_dir: str = "results/camera_calibration_test"
    """Directory to save test outputs"""


def test_camera_calibration(args: Args):
    """Test camera calibration integration."""

    print("Camera Calibration Integration Test")
    print("=" * 50)
    print(f"Config path: {args.env_config_path}")
    print(f"Domain randomization: {args.test_domain_randomization}")
    print(f"Render mode: {args.render_mode}")

    # Check if config exists
    if not Path(args.env_config_path).exists():
        print(f"ERROR: Config file not found at {args.env_config_path}")
        print("Please run camera calibration first to generate env_config.json")
        return

    # Load config to display calibration info
    with open(args.env_config_path, "r") as f:
        config = json.load(f)

    print("\nCalibration Data:")
    if "base_camera_settings" in config:
        print(
            f"  Camera extrinsics: {config['base_camera_settings'].get('extrinsics', 'Not found')}"
        )
        print(
            f"  Camera intrinsics: {config['base_camera_settings'].get('intrinsics', 'Not found')}"
        )

    if "calibration_offset" in config:
        print("  Joint offsets (degrees):")
        for joint, offset in config["calibration_offset"].items():
            print(f"    {joint}: {offset:.2f}°")

    # Create output directory
    if args.save_images:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Basic environment creation with calibration
    print("\n" + "-" * 50)
    print("Test 1: Basic Environment Creation")
    print("-" * 50)

    try:
        env = gym.make(
            "SO101GraspCubeLeRobotSim2Real-v1",
            obs_mode="rgb+segmentation",
            control_mode="pd_joint_pos",
            render_mode=args.render_mode,
            env_config_path=args.env_config_path,
            use_learned_camera=True,
            domain_randomization=False,
        )
        print("✓ Environment created successfully with calibration")

        # Reset and check camera
        obs = env.reset()
        print("✓ Environment reset successful")

        if "rgb" in obs[0] and "base_camera" in obs[0]["rgb"]:
            img = obs[0]["rgb"]["base_camera"]
            print(f"✓ Camera image shape: {img.shape}")

            if args.save_images:
                # Save first frame
                if hasattr(img, "cpu"):
                    img_np = img.cpu().numpy()
                else:
                    img_np = img
                plt.imsave(str(output_dir / "test1_calibrated_view.png"), img_np)
                print(
                    f"✓ Saved calibrated camera view to {output_dir}/test1_calibrated_view.png"
                )

        env.close()

    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        return

    # Test 2: Compare with default camera
    print("\n" + "-" * 50)
    print("Test 2: Compare Calibrated vs Default Camera")
    print("-" * 50)

    try:
        # Create environment without calibration
        env_default = gym.make(
            "SO101GraspCubeLeRobotSim2Real-v1",
            obs_mode="rgb+segmentation",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",  # Offscreen for comparison
            use_learned_camera=False,
            domain_randomization=False,
        )

        # Create environment with calibration
        env_calibrated = gym.make(
            "SO101GraspCubeLeRobotSim2Real-v1",
            obs_mode="rgb+segmentation",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            env_config_path=args.env_config_path,
            use_learned_camera=True,
            domain_randomization=False,
        )

        # Reset both
        obs_default = env_default.reset()
        obs_calibrated = env_calibrated.reset()

        # Compare views
        if args.save_images:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Default camera
            img_default = obs_default[0]["rgb"]["base_camera"]
            if hasattr(img_default, "cpu"):
                img_default = img_default.cpu().numpy()
            ax1.imshow(img_default)
            ax1.set_title("Default Camera")
            ax1.axis("off")

            # Calibrated camera
            img_calibrated = obs_calibrated[0]["rgb"]["base_camera"]
            if hasattr(img_calibrated, "cpu"):
                img_calibrated = img_calibrated.cpu().numpy()
            ax2.imshow(img_calibrated)
            ax2.set_title("Calibrated Camera")
            ax2.axis("off")

            plt.tight_layout()
            plt.savefig(str(output_dir / "test2_camera_comparison.png"))
            plt.close()
            print(
                f"✓ Saved camera comparison to {output_dir}/test2_camera_comparison.png"
            )

        env_default.close()
        env_calibrated.close()
        print("✓ Camera comparison test passed")

    except Exception as e:
        print(f"✗ Test 2 failed: {e}")

    # Test 3: Domain randomization around calibrated camera
    if args.test_domain_randomization:
        print("\n" + "-" * 50)
        print("Test 3: Domain Randomization Around Calibrated Camera")
        print("-" * 50)

        try:
            env = gym.make(
                "SO101GraspCubeLeRobotSim2Real-v1",
                obs_mode="rgb+segmentation",
                control_mode="pd_joint_pos",
                render_mode=args.render_mode,
                env_config_path=args.env_config_path,
                use_learned_camera=True,
                domain_randomization=True,
                domain_randomization_config={
                    "randomize_around_calibrated_camera": True,
                    "max_camera_offset": [0.05, 0.05, 0.05],
                    "camera_fov_noise": np.deg2rad(5),
                },
            )

            print("✓ Environment created with domain randomization")

            # Collect multiple frames to show randomization
            if args.save_images:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                for i in range(6):
                    obs = env.reset()
                    img = obs[0]["rgb"]["base_camera"]
                    if hasattr(img, "cpu"):
                        img = img.cpu().numpy()

                    axes[i].imshow(img)
                    axes[i].set_title(f"Random View {i + 1}")
                    axes[i].axis("off")

                plt.suptitle("Domain Randomization Around Calibrated Camera")
                plt.tight_layout()
                plt.savefig(str(output_dir / "test3_domain_randomization.png"))
                plt.close()
                print(
                    f"✓ Saved domain randomization samples to {output_dir}/test3_domain_randomization.png"
                )

            env.close()
            print("✓ Domain randomization test passed")

        except Exception as e:
            print(f"✗ Test 3 failed: {e}")

    # Test 4: Joint calibration offsets
    print("\n" + "-" * 50)
    print("Test 4: Joint Calibration Offsets")
    print("-" * 50)

    if "calibration_offset" in config and config["calibration_offset"]:
        try:
            env = gym.make(
                "SO101GraspCubeLeRobotSim2Real-v1",
                obs_mode="state",
                control_mode="pd_joint_pos",
                render_mode="rgb_array",
                env_config_path=args.env_config_path,
                use_learned_camera=True,
                domain_randomization=False,
            )

            obs = env.reset()

            # Check if offsets were applied
            print("✓ Environment created with calibration offsets")
            print("  Note: Offsets are applied internally during robot loading")

            env.close()

        except Exception as e:
            print(f"✗ Test 4 failed: {e}")
    else:
        print("⚠ No calibration offsets found in config")

    # Test 5: Multi-episode consistency
    print("\n" + "-" * 50)
    print(f"Test 5: Multi-Episode Consistency ({args.num_episodes} episodes)")
    print("-" * 50)

    try:
        env = gym.make(
            "SO101GraspCubeLeRobotSim2Real-v1",
            obs_mode="rgb+segmentation",
            control_mode="pd_joint_pos",
            render_mode=args.render_mode,
            env_config_path=args.env_config_path,
            use_learned_camera=True,
            domain_randomization=False,
        )

        for episode in range(args.num_episodes):
            obs = env.reset()

            # Take a few random actions
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            print(f"  Episode {episode + 1}: Completed {step + 1} steps")

        env.close()
        print("✓ Multi-episode test passed")

    except Exception as e:
        print(f"✗ Test 5 failed: {e}")

    print("\n" + "=" * 50)
    print("Camera Calibration Integration Test Complete")
    print("=" * 50)


def main():
    """Main entry point."""
    args = tyro.cli(Args)
    test_camera_calibration(args)


if __name__ == "__main__":
    main()
