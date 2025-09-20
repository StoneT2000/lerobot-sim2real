import json
import time
from typing import Optional
import gymnasium as gym
import torch
from lerobot_sim2real.utils.safety import setup_safe_exit
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.agents.robots.so101.lerobot_manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
import cv2
import numpy as np
import tyro
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils import sapien_utils
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import the environment to register it with gymnasium
from lerobot_sim2real.env.tasks.digit_twins.so101_arm.grasp_cube import (
    SO101GraspCubeEnv,
)


@dataclass
class Args:
    env_id: str = "SO101GraspCubeLeRobotSim2Real-v1"
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""


def main(args: Args):
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="sensors",
        reward_mode="none",
        # use larger camera resolution to make it easier to align. In training we won't use this however
        sensor_configs=dict(width=512, height=512),
    )
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            data = json.load(f)
            calibration_offset = data.pop("calibration_offset")
            env_kwargs.update(**data)

    real_robot = create_real_robot(uid="so101")
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot, calibration_offset=calibration_offset)

    sim_env = gym.make(
        args.env_id,
        **env_kwargs,
    )
    sim_env = FlattenRGBDObservationWrapper(sim_env)

    def real_reset_function(self: Sim2RealEnv, seed=None, options=None):
        self.sim_env.reset(seed=seed, options=options)
        for _ in range(2):
            # we move to a rest qpos twice since some robot arms don't reach the goal precisely the first time
            self.agent.reset(qpos=self.base_sim_env.agent.robot.qpos.cpu().flatten())

    real_env = Sim2RealEnv(
        sim_env=sim_env, agent=real_agent, real_reset_function=real_reset_function
    )
    real_obs, _ = real_env.reset()
    print(
        "Check the real robot lines up with the guide picture here: https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/assets/calibration_guide.png. If not, tune the calibration offset and try again."
    )
    input("Press enter to stop the program.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
