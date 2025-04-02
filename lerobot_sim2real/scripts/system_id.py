"""
Code for system ID to tune joint stiffness and damping and test out controllers compared with real robots. You are recommended to copy this code
to your own project and modify it according to your own hardware

The strategy employed here is to sample a number of random joint positions and move the robot to each of those positions with the same actions and see
how the robot's qpos and qvel values evolve over time compared to the real world values.


To run we first save a motion profile from the real robot

python -m system_id.py \
    --robot_uid so100 --real
"""
from dataclasses import dataclass
import time

import gymnasium as gym
import numpy as np
import torch
import tyro
import matplotlib.pyplot as plt

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.sim2real_env import Sim2RealEnv
from lerobot.common.robot_devices.robots.configs import KochRobotConfig, So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
@dataclass
class Args:
    robot_uid: str = "so100"
    """the ID of the robot to use"""
    real: bool = True
    """whether to use the real robot"""
    seed: int = 0
    """the seed to use for the random number generator"""
    sim_freq: int = 120
    """the frequency of the simulation"""
    control_freq: int = 30
    """the frequency of the control"""
    
    sim: bool = False
    """Whether to run the same actions applied to the real robot in simulation and print tracking error"""
    vis: bool = False
    """Whether to open up the GUI, which lets you move around and see the simulated robot"""
    


def main(args: Args):
    ### Fill in the code for your real robot here ###
    real_agent = None  #
    robot_config = So100RobotConfig(
        leader_arms={},
        cameras={},
    )
    real_robot = ManipulatorRobot(robot_config)
    real_agent = LeRobotRealAgent(real_robot)
    real_agent.start()
    
    # pick a few goal joint positiions for the robot to move to, make sure they cover a wide space!
    goal_joint_positions = torch.tensor([
        [0, 2.7, 2.7, 1.0, -np.pi / 2, 0],
        [-np.pi / 2, np.pi / 2, np.pi / 2, -0.250, 0, 1.7],
        [np.pi / 4, 0.5, 0.0, -0.5, np.pi / 2, 0.3],
    ])
    ### end of real robot code ###

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.sim:
        sim_env = gym.make("Empty-v1", obs_mode="none", robot_uids="so100", control_mode="pd_joint_pos", sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq))
        sim_env.reset()
        base_env: BaseEnv = sim_env.unwrapped
        real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, obs_mode="none")
        if args.vis:
            base_env.render_human().paused=True
            base_env.render_human()
    
    # track and record some qpos values and the corresponding target joint positions
    qpos_history = []
    sim_qpos_history = []
    target_qpos_history = []
    # TODO (stao): randomize max radians per step?
    max_rad_per_step = 0.05
    freq = args.control_freq
    for goal_joint_position in goal_joint_positions:
        target_qpos = real_agent.get_qpos()
        for _ in range(int(20 * freq)): # give it max 20 seconds to reach the target
            start_loop_t = time.perf_counter()
            cur_qpos = real_agent.get_qpos()
            
            if args.sim:
                sim_qpos = base_env.agent.robot.get_qpos()
                sim_qpos_history.append(sim_qpos[0])
            delta_step = (goal_joint_position - target_qpos).clip(
                min=-max_rad_per_step, max=max_rad_per_step
            )
            target_qpos += delta_step
            real_agent.set_target_qpos(target_qpos)
            if args.sim:
                base_env.step(target_qpos)
                if args.vis:
                    base_env.render_human()
            dt_s = time.perf_counter() - start_loop_t
            if dt_s < 1 / freq:
                time.sleep(1 / freq - dt_s)
            qpos_history.append(cur_qpos[0].clone().numpy())
            target_qpos_history.append(target_qpos[0].clone().numpy())
            if np.linalg.norm(delta_step) <= 1e-4:
                break
            
    qpos_history = np.array(qpos_history)
    sim_qpos_history = np.array(sim_qpos_history)
    target_qpos_history = np.array(target_qpos_history)
    np.save(f"system_id_{args.robot_uid}.npy", {"qpos": qpos_history, "target_qpos": target_qpos_history})

    if args.sim:

        # Create one figure with subplots for all dimensions
        num_joints = qpos_history.shape[1]
        plt.figure(figsize=(15, 2 * num_joints))
        
        for i in range(num_joints):
            # Plot for real vs sim comparison
            plt.subplot(num_joints, 2, 2*i + 1)
            plt.plot(qpos_history[:, i], label="real")
            plt.plot(sim_qpos_history[:, i], label="sim")
            plt.title(f"Joint {i} - Real vs Sim")
            plt.xlabel("Time steps")
            plt.ylabel("Joint position (rad)")
            plt.legend()
            
            # Plot for target vs actual comparison
            plt.subplot(num_joints, 2, 2*i + 2)
            plt.plot(qpos_history[:, i], label="actual")
            plt.plot(target_qpos_history[:, i], label="target")
            plt.title(f"Joint {i} - Target vs Actual")
            plt.xlabel("Time steps")
            plt.ylabel("Joint position (rad)")
            plt.legend()
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
