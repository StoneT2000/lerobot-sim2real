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
    """Run this script to perform system ID on a robot. Code is currently hardcoded for the SO100 robot.
    
    Run 
    
    ```
    python lerobot_sim2real/scripts/system_id.py --real 
    ```
    
    to first gather qpos and qpos tracking data from the real robot. Then run
    
    ```
    python lerobot_sim2real/scripts/system_id.py --sim
    ```
    
    to run the same actions in simulation, print tracking error, and try to optimize for the best robot parameters (stiffness and damping)
    to use in simulation to replicate real world behavior.
    
    """
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
    if args.real:
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
        [0, 2.7, 0.2, 1.0, -np.pi / 2, 0],
        [0.1, 2.7, 2.7, 1.0, -np.pi / 2, 0],
        [-np.pi / 2, np.pi / 2, np.pi / 2, -0.250, 0, 1.7],
        [np.pi / 4, 0.5, 0.0, -0.5, np.pi / 2, 0.3],
    ])
    ### end of real robot code ###
    
    ### parameters for for system ID ###
    method = "random_sample"
    init_stiffness = np.array([1e3] * 6)
    init_damping = np.array([1e2] * 6)
    stiffness_low = np.array([1e3 * 0.5] * 6)
    stiffness_high = np.array([1e3 * 1.5] * 6)
    damping_low = np.array([1e2 * 0.5] * 6)
    damping_high = np.array([1e2 * 1.5] * 6)
    optimization_metric = "ee_pos_error"
    optimization_metric = "joint_pos_error"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.sim:
        sim_env = gym.make("Empty-v1", obs_mode="none", robot_uids="so100", control_mode="pd_joint_pos", sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq))
        sim_env.reset()
        base_env: BaseEnv = sim_env.unwrapped
        if args.real:
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
    if not args.real and args.sim:
        # optimize sim parameters to match
        kinematic_env = gym.make("Empty-v1", obs_mode="none", robot_uids="so100", control_mode="pd_joint_pos", sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq))
        data = np.load(f"system_id_{args.robot_uid}.npy", allow_pickle=True).reshape(-1)[0]
        prev_qpos_history = data["qpos"]
        prev_target_qpos_history = data["target_qpos"]
        print(prev_target_qpos_history.shape, prev_qpos_history.shape)
        
        trials = 1000
        best_stiffness = None
        best_damping = None
        best_error = np.inf
        
        for t in range(trials):
            
            if method == "random_sample":
                stiffness = np.random.uniform(stiffness_low, stiffness_high)
                damping = np.random.uniform(damping_low, damping_high)
                
                

            sim_env.reset()
            base_env.agent.controllers["pd_joint_pos"].config.stiffness = stiffness
            base_env.agent.controllers["pd_joint_pos"].config.damping = damping
            base_env.agent.controllers["pd_joint_pos"].set_drive_property()
            base_env.agent.robot.set_qpos(prev_qpos_history[0])
            if base_env.gpu_sim_enabled:
                base_env.scene._gpu_apply_all()
                base_env.scene.px.gpu_update_articulation_kinematics()
                base_env.scene._gpu_fetch_all()
        
            error = []
            for i in range(len(prev_qpos_history) - 1):
                # prev_real_qpos = prev_qpos_history[i]
                next_real_qpos = prev_qpos_history[i+1]
                target_qpos = prev_target_qpos_history[i]
                sim_env.step(target_qpos)
                kinematic_env.agent.robot.set_qpos(next_real_qpos)
                
                if optimization_metric == "ee_pos_error":
                    real_ee_pos = kinematic_env.agent.tcp_pos
                    sim_ee_pos = sim_env.agent.tcp_pos
                elif optimization_metric == "joint_pos_error":
                    real_joint_pos = kinematic_env.agent.robot.qpos[0]
                    sim_joint_pos = base_env.agent.robot.qpos[0]
                    error.append(real_joint_pos - sim_joint_pos)
                # print("EE Tracking error", np.linalg.norm(real_ee_pos - sim_ee_pos), real_ee_pos - sim_ee_pos)
                # error.append(real_ee_pos - sim_ee_pos)
                # sim_env.render_human()
            print(f"=== Trial {t} ===")
            print(f"Stiffness: {stiffness}")
            print(f"Damping: {damping}")
            print(f"Error over time: {np.linalg.norm(error, axis=0)}")
            print(f"Max error over time: {np.max(np.abs(error), axis=0)}")
            
            if np.linalg.norm(error) < best_error:
                best_error_results = {
                    "stiffness": stiffness,
                    "damping": damping,
                    "error": error,
                }
                best_error = np.linalg.norm(error)
                
        print(f"Best stiffness: {best_error_results['stiffness']}")
        print(f"Best damping: {best_error_results['damping']}")
        print(f"Best error over time: {np.linalg.norm(best_error_results['error'], axis=0)}")
        print(f"Best max error over time: {np.max(np.abs(best_error_results['error']), axis=0)}")
        print(f"Best median error over time: {np.median(np.abs(best_error_results['error']), axis=0)}")
        
    if args.real:
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
                qpos_history.append(cur_qpos[0].clone().numpy())
                target_qpos_history.append(target_qpos[0].clone().numpy())
                
                real_agent.set_target_qpos(target_qpos)
                if args.sim:
                    base_env.step(target_qpos)
                    if args.vis:
                        base_env.render_human()
                dt_s = time.perf_counter() - start_loop_t
                if dt_s < 1 / freq:
                    time.sleep(1 / freq - dt_s)
                if np.linalg.norm(delta_step) <= 1e-4:
                    break
        qpos_history.append(real_agent.get_qpos()[0])
        qpos_history = np.array(qpos_history)
        sim_qpos_history.append(base_env.agent.robot.get_qpos()[0])
        sim_qpos_history = np.array(sim_qpos_history)
        target_qpos_history = np.array(target_qpos_history)
        # prev_qpos_history = np.load(f"system_id_{args.robot_uid}.npy", allow_pickle=True).reshape(-1)[0]["qpos"]
        # np.save(f"system_id_{args.robot_uid}.npy", {"qpos": qpos_history, "target_qpos": target_qpos_history})

        if args.sim:
            print("Norm of joint differences over time", np.linalg.norm(qpos_history - sim_qpos_history, axis=0))
            print("Max error per joint", np.max(np.abs(qpos_history - sim_qpos_history), axis=0))
            
            print("===")
            # print("Per joint error", np.linalg.norm(qpos_history - prev_qpos_history, axis=0))
            # print("Max error per joint", np.max(np.abs(qpos_history - prev_qpos_history), axis=0))

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
                plt.plot(qpos_history[1:, i], label="actual")
                plt.plot(target_qpos_history[:, i], label="target")
                plt.title(f"Joint {i} - Target vs Actual")
                plt.xlabel("Time steps")
                plt.ylabel("Joint position (rad)")
                plt.legend()
            
            plt.tight_layout()
            plt.show()

    if args.real:
        real_agent.stop()

if __name__ == "__main__":
    main(tyro.cli(Args))
