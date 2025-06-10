import time
import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
import cv2
import numpy as np
import tyro
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils import sapien_utils
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Args:
    env_id: str = "SO100GraspCube-v1"
    greenscreen_overlay_path: str = "greenscreen_background.png"

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

def main(args: Args):
    real_robot = create_real_robot(uid="s100")
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)

    sim_env = gym.make(
        args.env_id,
        obs_mode="rgb+segmentation",
        sim_config={"sim_freq": 120, "control_freq": 30},
        render_mode="sensors",
        # greenscreen_overlay_path=args.greenscreen_overlay_path,
        base_camera_settings=dict(
            pos=[0.6800, 0.3500, 0.2861],
            fov=0.61,
            target=[0.185, -0.135, 0.03]
        ),
        sensor_configs=dict(width=512, height=512)
    )
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, obs_mode="rgb")
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

    # if args.output_photo_path is not None:
    #     path, ext = args.output_photo_path.split(".")
    #     real_obs = real_env.get_obs()["sensor_data"]
    #     for i, name in enumerate(real_obs):
    #         plt.imsave(path + "_" + name + "." + ext, real_obs[name]["rgb"][0].numpy())
    # else:
        # obs, _ = real_env.reset()
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


    real_env.close()
    sim_env.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)