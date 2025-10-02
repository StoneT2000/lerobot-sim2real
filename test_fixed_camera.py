#!/usr/bin/env python3
"""Test with a potential fix for the camera issue"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mani_skill.utils import sapien_utils

# Import the environment
import lerobot_sim2real.env.tasks.digit_twins.so101_arm.grasp_cube

# Load calibration data directly
import json
with open("env_config.json", 'r') as f:
    config = json.load(f)

# Get the calibrated position and compute a look_at pose
calibrated_pos = config["base_camera_settings"]["pos"]
calibrated_target = config["base_camera_settings"]["target"]

print(f"Calibrated position: {calibrated_pos}")
print(f"Calibrated target: {calibrated_target}")

# Create environment with override camera settings
env = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    use_learned_camera=False,  # Don't use the learned camera system
    domain_randomization=False,
    base_camera_settings=dict(
        fov=config["base_camera_settings"]["fov"],
        pos=calibrated_pos,
        target=calibrated_target,
    ),
)

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

# Get camera image
img = obs["sensor_data"]["base_camera"]["rgb"]
if hasattr(img, 'cpu'):
    img = img.cpu().numpy()
if img.ndim == 4:
    img = img[0]

# Save the image
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title("Camera View with Calibrated Position/Target")
plt.axis('off')
plt.tight_layout()
plt.savefig("test_fixed_camera_view.png", dpi=150)
print("\nSaved test view to test_fixed_camera_view.png")

# Also print camera parameters
sensor_params = obs["sensor_param"]["base_camera"]
print(f"\nCamera extrinsic_cv:\n{sensor_params['extrinsic_cv']}")
print(f"\nCamera mount pose:")
unwrapped = env.unwrapped
print(f"  Position: {unwrapped.camera_mount.pose.p}")
print(f"  Quaternion: {unwrapped.camera_mount.pose.q}")

env.close()

# Compare with the problematic learned camera approach
print("\n" + "="*50)
print("Now testing with learned camera system...")

env2 = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    env_config_path="env_config.json",
    use_learned_camera=True,
    domain_randomization=False,
)

obs2 = env2.reset()
if isinstance(obs2, tuple):
    obs2 = obs2[0]

# Get camera parameters
sensor_params2 = obs2["sensor_param"]["base_camera"]
print(f"\nLearned camera extrinsic_cv:\n{sensor_params2['extrinsic_cv']}")

unwrapped2 = env2.unwrapped
print(f"\nLearned camera mount pose:")
print(f"  Position: {unwrapped2.camera_mount.pose.p}")
print(f"  Quaternion: {unwrapped2.camera_mount.pose.q}")

env2.close()
