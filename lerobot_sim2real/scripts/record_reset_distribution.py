import json
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional

import numpy as np
from lerobot_sim2real.utils.camera import scale_intrinsics
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro


@dataclass
class Args:
    env_id: str = "SO100GraspCube-v1"
    """The environment id to use"""
    record_dir: str = "videos"
    """Directory to save recordings of the camera captured images. If none no recordings are saved"""
    env_kwargs_json_path: Optional[str] = None
    """path to a json file containing additional environment kwargs"""
    num_resets: int = 100
    """Number of resets to record"""
def main(args: Args):
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="sensors",
        domain_randomization=True,
        reward_mode="none",
        sensor_configs=dict(shader_pack="default")
    )
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            data = json.load(f)
            calibration_offset = data.pop("calibration_offset")
            env_kwargs.update(**data)
        env_kwargs["base_camera_settings"]["extrinsics"] = np.load(env_kwargs["base_camera_settings"]["extrinsics"])
        env_kwargs["base_camera_settings"]["intrinsics"] = np.load(env_kwargs["base_camera_settings"]["intrinsics"])
    env = gym.make(args.env_id, **env_kwargs)
    if args.record_dir is not None:
        env = RecordEpisode(env, output_dir=args.record_dir, save_video=False, save_trajectory=False, video_fps=15)
    env.reset()
    for _ in range(args.num_resets):
        env.reset()
        env.render_images.append(env.capture_image())
    name = f"{args.env_id}_reset_distribution"
    env.flush_video(name=name)
    print(f"Saved video to {env.output_dir}/{name}.mp4")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)