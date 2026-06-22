from pathlib import Path
import gymnasium as gym
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower import SO100FollowerConfig
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config
import numpy as np
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

def create_real_robot(uid: str = "so100") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file."""
    if uid == "so100":
        robot_config = SO100FollowerConfig(
            port="/dev/ttyACM0",
            id="so100_follower",
            use_degrees=True, 
             cameras={
                "base_camera": OpenCVCameraConfig(
                    index_or_path=Path("/dev/video2"),
                    height=720,
                    width=1280,
                    fps=30,
                    warmup_s=2,
                )
            },
        )
    elif uid == "so101":
        robot_config = SO101FollowerConfig(
            port="/dev/ttyACM0",  
            id="so101_follower",  
            use_degrees=True,  
            cameras={
                "base_camera": OpenCVCameraConfig(
                    index_or_path=Path("/dev/video2"),
                    height=720,
                    width=1280,
                    fps=30,
                    warmup_s=2,
                )
            },
        )
    else:
        raise ValueError(f"Unknown robot uid: {uid}. Supported: 'so100', 'so101'")
    
    robot = make_robot_from_config(robot_config)
    return robot 