#!/usr/bin/env python3
"""Debug camera parameters in ManiSkill"""

import gymnasium as gym
import numpy as np
import lerobot_sim2real.env.tasks.digit_twins.so101_arm.grasp_cube

# Create environment with calibrated camera
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
unwrapped = env.unwrapped

# Get sensor parameters from observation
if isinstance(obs, tuple):
    obs = obs[0]

sensor_params = obs["sensor_param"]["base_camera"]
print("Sensor parameters from observation:")
print(f"  extrinsic_cv shape: {sensor_params['extrinsic_cv'].shape}")
print(f"  extrinsic_cv:\n{sensor_params['extrinsic_cv']}")
print(f"  cam2world_gl shape: {sensor_params['cam2world_gl'].shape}")
print(f"  cam2world_gl:\n{sensor_params['cam2world_gl']}")
print(f"  intrinsic_cv shape: {sensor_params['intrinsic_cv'].shape}")
print(f"  intrinsic_cv:\n{sensor_params['intrinsic_cv']}")

# Check camera mount
print(f"\nCamera mount pose:")
print(f"  Position: {unwrapped.camera_mount.pose.p}")
print(f"  Quaternion: {unwrapped.camera_mount.pose.q}")

# Get camera object
cameras = unwrapped._sensors["base_camera"]
if hasattr(cameras, "_sensor_poses"):
    print(f"\nCamera sensor poses: {cameras._sensor_poses}")
    
# Check the actual camera configuration
# print(f"\nCamera FOV: {np.rad2deg(cameras.fov)} degrees")

# Compare with default camera settings
print(f"\nDefault camera settings:")
default_env = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    use_learned_camera=False,
    domain_randomization=False,
)
default_obs = default_env.reset()
if isinstance(default_obs, tuple):
    default_obs = default_obs[0]
default_params = default_obs["sensor_param"]["base_camera"]
print(f"  Default extrinsic_cv:\n{default_params['extrinsic_cv']}")

# Check if they're the same
if np.allclose(sensor_params['extrinsic_cv'], default_params['extrinsic_cv']):
    print("\nWARNING: Camera extrinsics are the same as default!")
else:
    print("\nGood: Camera extrinsics are different from default")

env.close()
default_env.close()
