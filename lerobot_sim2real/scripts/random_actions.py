import gymnasium as gym
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
import numpy as np
from lerobot.common.robot_devices.cameras.configs import (
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.robots.configs import KochRobotConfig, So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

import mani_skill.envs.tasks.digital_twins.koch_arm.grasp_cube
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm
from lerobot_sim2real.config.real_robot import create_real_robot


def main():
    real_robot = create_real_robot(uid="s100")
    real_robot.connect()
    # max control freq for lerobot really is just 60Hz
    real_agent = LeRobotRealAgent(real_robot)


    max_episode_steps = 30
    sim_env = gym.make(
        "SO100GraspCube-v1",
        obs_mode="rgb+segmentation",
        sim_config={"sim_freq": 120, "control_freq": 30},
        render_mode="sensors", # only sensors mode is supported right now for real envs, basically rendering the direct visual observations fed to policy
        max_episode_steps=max_episode_steps, # give our robot more time to try and re-try the task
    )
    # you can apply most wrappers freely to the sim_env and the real env will use them
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    sim_env = RecordEpisode(sim_env, output_dir="videos", save_trajectory=False, video_fps=sim_env.unwrapped.control_freq)
    
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, obs_mode="rgb")
    sim_env.print_sim_details()
    sim_obs, _ = sim_env.reset()
    real_obs, _ = real_env.reset()
    # while True:
    # while True:
    #     sim_env.render_human()
    #     print(real_env.agent.qpos)

    for k in sim_obs.keys():
        print(
            f"{k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}"
        )
    pbar = tqdm(range(max_episode_steps))
    done = False
    while not done:
        action = real_env.action_space.sample()
        # action[0] = 1
        real_obs, _, terminated, truncated, info = real_env.step(action)
        done = terminated or truncated
        pbar.update(1)
    sim_env.close()
    real_agent.stop()

    print("Saved video to videos/0.mp4")

if __name__ == "__main__":
    main()