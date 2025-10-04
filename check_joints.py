#!/usr/bin/env python3
"""Check joint names in SO101 robot"""

import gymnasium as gym
import lerobot_sim2real.env.tasks.digit_twins.so101_arm.grasp_cube

# Create environment without calibration to avoid the error
env = gym.make(
    "SO101GraspCubeLeRobotSim2Real-v1",
    obs_mode="none",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    use_learned_camera=False,
    domain_randomization=False,
)

# Get joint names
unwrapped = env.unwrapped
joints = unwrapped.agent.robot.get_joints()
print("Joint names in SO101 robot:")
for i, joint in enumerate(joints):
    print(f"  {i}: {joint.name}")

# Check qpos size
qpos = unwrapped.agent.robot.get_qpos()
print(f"\nQpos shape: {qpos.shape}")

env.close()
