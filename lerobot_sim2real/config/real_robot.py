import gymnasium as gym
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
import numpy as np
from lerobot.common.robot_devices.cameras.configs import (
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

def create_real_robot(uid: str = "s100") -> ManipulatorRobot:
    if uid == "s100":
        robot_config = So100RobotConfig(
            leader_arms={},
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            cameras={
                "base_camera": IntelRealSenseCameraConfig(serial_number=146322070293, fps=30, width=640, height=480)
            }
        )
        real_robot = ManipulatorRobot(robot_config)
        return real_robot