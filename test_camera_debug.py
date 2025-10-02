#!/usr/bin/env python3
"""Debug script to test camera calibration in SO101GraspCubeEnv"""

import gymnasium as gym
import numpy as np
from pathlib import Path

# Import the environment
import lerobot_sim2real.env.tasks.digit_twins.so101_arm.grasp_cube

# Create environment with calibration
print("Creating environment with calibration...")
env = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_pos",
    render_mode="human",
    env_config_path="env_config.json",
    use_learned_camera=True,
    domain_randomization=False,
)

# Check if calibration data was loaded
unwrapped = env.unwrapped
print(f"use_learned_camera: {unwrapped.use_learned_camera}")
print(f"domain_randomization: {unwrapped.domain_randomization}")
print(f"calibration_data loaded: {unwrapped.calibration_data is not None}")

if unwrapped.calibration_data:
    print(f"Calibration data keys: {unwrapped.calibration_data.keys()}")
    if 'pose' in unwrapped.calibration_data:
        pose = unwrapped.calibration_data['pose']
        print(f"Calibrated pose position: {pose.p}")
        print(f"Calibrated pose quaternion: {pose.q}")
    if 'fov' in unwrapped.calibration_data:
        print(f"Calibrated FOV: {np.rad2deg(unwrapped.calibration_data['fov']):.2f} degrees")

print(f"\nBase camera settings:")
print(f"  pos: {unwrapped.base_camera_settings['pos']}")
print(f"  target: {unwrapped.base_camera_settings['target']}")
print(f"  fov: {np.rad2deg(unwrapped.base_camera_settings['fov']):.2f} degrees")

# Reset environment
print("\nResetting environment...")
obs = env.reset()

# Check camera mount pose after reset
print(f"\nCamera mount pose after reset:")
camera_mount_pose = unwrapped.camera_mount.pose
print(f"  Position: {camera_mount_pose.p}")
print(f"  Quaternion: {camera_mount_pose.q}")

# Test sample_camera_poses
print("\nTesting sample_camera_poses(1):")
sampled_pose = unwrapped.sample_camera_poses(1)
print(f"  Type: {type(sampled_pose)}")
if hasattr(sampled_pose, 'p'):
    print(f"  Position: {sampled_pose.p}")
    print(f"  Quaternion: {sampled_pose.q}")

# Check camera observation
if "rgb" in obs[0] and "base_camera" in obs[0]["rgb"]:
    img = obs[0]["rgb"]["base_camera"]
    print(f"\nCamera image shape: {img.shape}")
    
    # Save the image
    import matplotlib.pyplot as plt
    if hasattr(img, 'cpu'):
        img_np = img.cpu().numpy()
    else:
        img_np = img
    plt.imsave("debug_camera_view.png", img_np)
    print("Saved camera view to debug_camera_view.png")

print("\nPress Enter to close...")
input()

env.close()
