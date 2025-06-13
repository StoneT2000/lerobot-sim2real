from typing import Optional
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
import cv2
import numpy as np
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    env_id: str
    """The environment id to train on"""
    out: str
    """Path to save the greenscreen image to"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""


def main(args: Args):
    # TODO (can we avoid activating the robot?)
    real_robot = create_real_robot(uid="so100")
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)
    
    sim_env = gym.make(
        args.env_id,
        obs_mode="rgb+segmentation",
        render_mode="sensors",
    )
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    # we use our created simulation environment to determine how to process the real observations
    # e.g. if the sim env uses 128x128 images, the real_env will preprocess the real images down to 128x128 as well
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, obs_mode="rgb")
    real_obs, _ = real_env.reset()

    
    # Convert from RGB to BGR since OpenCV uses BGR
    rgb_img = real_obs["rgb"].cpu().numpy()[0].astype(np.uint8)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    
    # Save the image
    cv2.imwrite(args.out, bgr_img)
    print(f"Saved image to {args.out}")

    real_env.close()
    sim_env.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)