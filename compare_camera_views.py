#!/usr/bin/env python3
"""Compare camera views between default and calibrated cameras"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the environment
import lerobot_sim2real.env.tasks.digit_twins.so101_arm.grasp_cube

# Create figure for comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Test 1: Default camera
print("Creating environment with default camera...")
env_default = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    use_learned_camera=False,
    domain_randomization=False,
)
obs_default = env_default.reset()
# Handle different observation formats
if isinstance(obs_default, tuple):
    obs_default = obs_default[0]
img_default = obs_default["sensor_data"]["base_camera"]["rgb"]
if hasattr(img_default, "cpu"):
    img_default = img_default.cpu().numpy()
if img_default.ndim == 4:  # Remove batch dimension
    img_default = img_default[0]
axes[0].imshow(img_default)
axes[0].set_title("Default Camera")
axes[0].axis("off")
env_default.close()

# Test 2: Calibrated camera
print("Creating environment with calibrated camera...")
env_calibrated = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    env_config_path="env_config.json",
    use_learned_camera=True,
    domain_randomization=False,
)
obs_calibrated = env_calibrated.reset()
# Handle different observation formats
if isinstance(obs_calibrated, tuple):
    obs_calibrated = obs_calibrated[0]
img_calibrated = obs_calibrated["sensor_data"]["base_camera"]["rgb"]
if hasattr(img_calibrated, "cpu"):
    img_calibrated = img_calibrated.cpu().numpy()
if img_calibrated.ndim == 4:  # Remove batch dimension
    img_calibrated = img_calibrated[0]
axes[1].imshow(img_calibrated)
axes[1].set_title("Calibrated Camera (Environment)")
axes[1].axis("off")
env_calibrated.close()

# Test 3: Reference render
ref_path = Path("results/debug_so101_extrinsic/test_calibrated_view.png")
if ref_path.exists():
    ref_img = plt.imread(str(ref_path))
    axes[2].imshow(ref_img)
    axes[2].set_title("Calibrated Camera (Render Script)")
    axes[2].axis("off")
else:
    axes[2].text(0.5, 0.5, "Reference not found", ha="center", va="center")
    axes[2].axis("off")

plt.tight_layout()
plt.savefig("camera_comparison.png", dpi=150)
print("Saved comparison to camera_comparison.png")
plt.close()

# Check camera mount details
print("\nDebugging camera mount pose...")
env = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    env_config_path="env_config.json",
    use_learned_camera=True,
    domain_randomization=False,
)
env.reset()
unwrapped = env.unwrapped

print(f"Camera mount pose: {unwrapped.camera_mount.pose.p}")
print(f"Camera mount quat: {unwrapped.camera_mount.pose.q}")

if unwrapped.calibration_data:
    print(f"\nCalibration data:")
    print(f"  Extrinsic matrix:\n{unwrapped.calibration_data['extrinsic_opencv']}")

env.close()
