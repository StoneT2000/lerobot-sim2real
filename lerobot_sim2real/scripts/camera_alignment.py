import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import tyro
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import REGISTERED_ENVS, register_env
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from transforms3d.euler import euler2quat


@dataclass
class Args:
    env_id: str = "KochGraspCube-v1"
    output_photo_path: Optional[str] = None
    """path to save photo from real_env"""


def overlay_envs(sim_env, real_env):
    """
    Overlays sim_env observtions onto real_env observations
    Requires matching ids between the two environments' sensors
    e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
    """
    real_obs = real_env.get_obs()["sensor_data"]
    sim_obs = sim_env.get_obs()["sensor_data"]
    assert sorted(real_obs.keys()) == sorted(
        sim_obs.keys()
    ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

    overlaid_dict = sim_env.get_obs()["sensor_data"]
    overlaid_imgs = []
    for name in overlaid_dict:
        real_imgs = real_obs[name]["rgb"][0] / 255
        sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)

    return tile_images(overlaid_imgs)


camera_offset = torch.zeros(3, dtype=torch.float32)
fov_offset = 0.0
active_keys = set()
last_frame_time = time.time()
MOVEMENT_SPEED = 0.1  # units per second
FOV_CHANGE_SPEED = 0.1  # radians per second
help_message_printed = False  # Flag to track if we've printed the help message


def on_key_press(event):
    global active_keys
    active_keys.add(event.key)


def on_key_release(event):
    global active_keys
    active_keys.discard(event.key)


def update_camera(sim_env):
    global camera_offset, fov_offset, last_frame_time, help_message_printed
    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time

    # Reset camera position and FOV on backspace
    if "backspace" in active_keys:
        camera_offset = torch.zeros(3, dtype=torch.float32)
        fov_offset = 0.0

    # Camera movement mapping based on active keys
    if "w" in active_keys:
        camera_offset[0] -= MOVEMENT_SPEED * delta_time  # Move forward
    if "s" in active_keys:
        camera_offset[0] += MOVEMENT_SPEED * delta_time  # Move back
    if "d" in active_keys:
        camera_offset[1] += MOVEMENT_SPEED * delta_time  # Move right
    if "a" in active_keys:
        camera_offset[1] -= MOVEMENT_SPEED * delta_time  # Move left
    if "up" in active_keys:
        camera_offset[2] += MOVEMENT_SPEED * delta_time  # Move up
    if "down" in active_keys:
        camera_offset[2] -= MOVEMENT_SPEED * delta_time  # Move down

    # FOV control
    if "left" in active_keys:
        fov_offset -= FOV_CHANGE_SPEED * delta_time
    if "right" in active_keys:
        fov_offset += FOV_CHANGE_SPEED * delta_time

    # update camera position and fov
    pos = sim_env.unwrapped.base_camera_settings["pos"] + camera_offset
    pose = sapien_utils.look_at(pos, sim_env.unwrapped.base_camera_settings["target"])
    sim_env.unwrapped.camera_mount.set_pose(pose)
    sim_env.unwrapped._sensors["base_camera"].camera.set_fovy(
        sim_env.unwrapped.base_camera_settings["fov"] + fov_offset
    )

    if len(active_keys) > 0:
        print("current_camera_position", pose.p)
        print(
            "current_camera_fov",
            sim_env.unwrapped.base_camera_settings["fov"] + fov_offset,
        )
        help_message_printed = False  # Reset the flag when there's movement
    elif (
        not help_message_printed
    ):  # Only print help message if it hasn't been printed yet
        print(
            "press: (w), (a) to move in x, (s), (d) to move in y, (up), (down) to move in z, (left), (right) to change fov"
        )
        print("press: (backspace) to reset, close figure to exit")
        help_message_printed = True


if __name__ == "__main__":
    args = args = tyro.cli(Args)

    # create duplicate environment with alignment dots
    @register_env("AlignmentEnv-v1")
    class Alignment_Env(REGISTERED_ENVS[args.env_id].cls):
        def _load_scene(self, options: dict):
            super()._load_scene(options)
            # overlay table dots for camera alignment
            def make_alignment_dot(name, color, init_pose):
                builder = self.scene.create_actor_builder()
                builder.add_cylinder_visual(
                    radius=0.005,
                    half_length=1e-4,
                    material=sapien.render.RenderMaterial(base_color=color),
                )
                builder.initial_pose = init_pose
                return builder.build_kinematic(name=name)

            alignment_dot_pos = [
                [0.2, 0.1, 0],  ## close to camera
                [0.2, -0.1, 0],  ## far from camera
                [0.35, 0, 0],  ## far infront of robot
                [0.35, 0.1, 0],  ## far infront of robot and close camera
            ]

            self.alignment_dots = []
            for i, pos in enumerate(alignment_dot_pos):
                dot = make_alignment_dot(
                    f"position{i}",
                    np.array([1, 1, 0, 1]),
                    sapien.Pose(p=pos, q=euler2quat(0, np.pi / 2, 0)),
                )
                self.alignment_dots.append(dot)
            cam_target_dot = make_alignment_dot(
                f"cam_target_dot",
                np.array([0, 1, 0, 1]),
                sapien.Pose(
                    p=self.base_camera_settings["target"], q=euler2quat(0, np.pi / 2, 0)
                ),
            )
            self.alignment_dots.append(cam_target_dot)

    # create robot from config
    # TODO: (xhin stao): make a separate config file to share among all scripts?
    robot_config = KochRobotConfig(
        leader_arms={},
        follower_arms={
            "main": DynamixelMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        },
        cameras={
            "base_camera": OpenCVCameraConfig(
                camera_index=0,  # <--- CHANGE HERE
                fps=60,
                width=640,
                height=480,
                rotation=90,  # <--- CHANGE If Necessary
            ),
        },
        calibration_dir="koch_calibration",  # <--- CHANGE HERE
    )
    real_robot = ManipulatorRobot(robot_config)

    # max control freq for lerobot really is just 60Hz
    real_agent = LeRobotRealAgent(real_robot)

    max_episode_steps = 200
    sim_env = gym.make(
        "AlignmentEnv-v1",
        obs_mode="rgb+segmentation",
        sim_config={"sim_freq": 120, "control_freq": 15},
        render_mode="sensors",  # only sensors mode is supported right now for real envs, basically rendering the direct visual observations fed to policy
        max_episode_steps=max_episode_steps,  # give our robot more time to try and re-try the task
        num_envs=1,
        # domain_randomization=False,
    )
    # you can apply most wrappers freely to the sim_env and the real env will use them
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    sim_env = RecordEpisode(
        sim_env,
        output_dir="videos",
        save_trajectory=False,
        video_fps=sim_env.unwrapped.control_freq,
    )
    sim_env.unwrapped.rgb_overlay_paths = None

    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, obs_mode="rgb")
    sim_env.print_sim_details()
    sim_obs, _ = sim_env.reset()
    real_obs, _ = real_env.reset()

    # for plotting robot camera reads
    fig = plt.figure()
    ax = fig.add_subplot()

    # Disable all default key bindings
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.manager.key_press_handler_id = None

    # initialize the plot
    im = ax.imshow(overlay_envs(sim_env, real_env))

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    if args.output_photo_path is not None:
        path, ext = args.output_photo_path.split(".")
        real_obs = real_env.get_obs()["sensor_data"]
        for i, name in enumerate(real_obs):
            plt.imsave(path + "_" + name + "." + ext, real_obs[name]["rgb"][0].numpy())
    else:
        obs, _ = real_env.reset()
        print("Camera alignment: Move real camera to align, close figure to exit")
        while True:
            overlaid_imgs = overlay_envs(sim_env, real_env)
            im.set_data(overlaid_imgs)
            # Update camera position based on active keys
            update_camera(sim_env)
            # Redraw the plot
            fig.canvas.draw()
            fig.show()
            fig.canvas.flush_events()
            if not plt.fignum_exists(fig.number):
                print("The figure has been closed.")
                break
