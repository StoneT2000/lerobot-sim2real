#!/usr/bin/env python3
"""Final test to verify camera calibration is working correctly"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the environment
import lerobot_sim2real.env.tasks.digit_twins.so101_arm.grasp_cube

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Test 1: Environment with calibrated camera
print("Creating environment with calibrated camera...")
env = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    env_config_path="env_config.json",
    use_learned_camera=True,
    domain_randomization=False,
)

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

# Get camera image
img = obs["sensor_data"]["base_camera"]["rgb"]
if hasattr(img, "cpu"):
    img = img.cpu().numpy()
if img.ndim == 4:
    img = img[0]

axes[0].imshow(img)
axes[0].set_title("Calibrated Camera (ManiSkill Environment)")
axes[0].axis("off")

# Get camera parameters for verification
sensor_params = obs["sensor_param"]["base_camera"]
print("\nCalibrated camera parameters:")
print(f"  Position: {env.unwrapped.camera_mount.pose.p.cpu().numpy()}")
if hasattr(env.unwrapped, "calibration_data") and env.unwrapped.calibration_data:
    print(f"  FOV: {np.rad2deg(env.unwrapped.calibration_data['fov']):.2f} degrees")

env.close()

# Test 2: Reference render
ref_path = Path("results/debug_so101_extrinsic/test_calibrated_view.png")
if ref_path.exists():
    ref_img = plt.imread(str(ref_path))
    axes[1].imshow(ref_img)
    axes[1].set_title("Calibrated Camera (Render Script)")
    axes[1].axis("off")
else:
    axes[1].text(0.5, 0.5, "Reference render not found", ha="center", va="center")
    axes[1].axis("off")

plt.suptitle("Camera Calibration Verification", fontsize=16)
plt.tight_layout()
plt.savefig("final_camera_verification.png", dpi=150)
print("\nSaved final verification to final_camera_verification.png")
plt.close()

print("\nCamera calibration integration is now using only extrinsics and intrinsics!")
print("The camera position and orientation come from the extrinsic matrix.")
print("The camera FOV is computed from the intrinsic matrix.")
