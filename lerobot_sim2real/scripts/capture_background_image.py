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
    real_robot = create_real_robot(uid="so100")
    real_robot.connect()
    # we don't need to move the robot. We only want to take a picture
    real_robot.bus.disable_torque()
    real_agent = LeRobotRealAgent(real_robot)
    
    sim_env = gym.make(
        args.env_id,
        obs_mode="rgb+segmentation",
        render_mode="sensors",
        reward_mode="none"
    )
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    # we use our created simulation environment to determine how to process the real observations
    # e.g. if the sim env uses 128x128 images, the real_env will preprocess the real images down to 128x128 as well
    # we skip data checks here since we don't want to actually move the robot. 
    # we also modify the default reset function to not move the robot
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, skip_data_checks=True, real_reset_function=lambda self, seed, options: None)
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