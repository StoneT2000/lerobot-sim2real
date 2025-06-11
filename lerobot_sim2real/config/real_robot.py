from pathlib import Path
import gymnasium as gym
from lerobot.common.robots.robot import Robot
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.robots.utils import make_robot_from_config
import numpy as np
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig

def create_real_robot(uid: str = "s100") -> Robot:
    if uid == "s100":
        robot_config = SO100FollowerConfig(
            port="/dev/ttyACM0",
            use_degrees=True,
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            cameras={
                "base_camera": RealSenseCameraConfig(serial_number_or_name="146322070293", fps=30, width=640, height=480)
            },
            id="stone_home",
        )
        real_robot = make_robot_from_config(robot_config)
        return real_robot