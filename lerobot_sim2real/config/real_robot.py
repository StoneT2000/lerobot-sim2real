from pathlib import Path
import gymnasium as gym
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower import SO100FollowerConfig
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config
import numpy as np
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


# todo(jackvial): Make the cameras configurable via the env_config.json file
def create_real_robot(uid: str = "so100") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file."""
    if uid == "so100":
        robot_config = SO100FollowerConfig(
            port="/dev/ttyACM0",
            id="so100_follower",
            use_degrees=True,
            cameras={
                "base_camera": OpenCVCameraConfig(
                    index_or_path=Path("/dev/video0"),
                    height=720,
                    width=1280,
                    fps=30,
                    warmup_s=2,
                )
                # "base_camera": RealSenseCameraConfig(serial_number_or_name="146322070293", fps=30, width=1280, height=720)
            },
        )
    elif uid == "so101":
        robot_config = SO101FollowerConfig(
            port="/dev/ttyACM0",
            id="so101_follower",
            use_degrees=True,
            cameras={
                # You will likely need to adjust this for your own camera setup
                # run `python -m lerobot.find_cameras opencv` to find your available camera indices
                "base_camera": OpenCVCameraConfig(
                    index_or_path=Path("/dev/video0"),
                    height=480,
                    width=640,
                    fps=30,
                    warmup_s=2,
                )
            },
        )
    else:
        raise ValueError(f"Unknown robot uid: {uid}. Supported: 'so100', 'so101'")

    robot = make_robot_from_config(robot_config)
    return robot
