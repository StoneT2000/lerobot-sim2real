#!/usr/bin/env python3
"""Debug observation structure"""

import gymnasium as gym
import lerobot_sim2real.env.tasks.digit_twins.so101_arm.grasp_cube

# Create environment
env = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    use_learned_camera=False,
    domain_randomization=False,
)

obs = env.reset()
print(f"Type of obs: {type(obs)}")

if isinstance(obs, tuple):
    print(f"Tuple length: {len(obs)}")
    obs_data = obs[0]
else:
    obs_data = obs

print(f"Type of obs_data: {type(obs_data)}")

if isinstance(obs_data, dict):
    print(f"Keys: {obs_data.keys()}")
    for key in obs_data:
        print(f"  {key}: type={type(obs_data[key])}")
        if isinstance(obs_data[key], dict):
            print(f"    subkeys: {obs_data[key].keys()}")
            for subkey in obs_data[key]:
                if isinstance(obs_data[key][subkey], dict):
                    print(f"      {subkey}: {obs_data[key][subkey].keys()}")

env.close()
