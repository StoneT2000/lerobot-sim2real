"""Simple script to train a RGB PPO policy in simulation"""

from dataclasses import dataclass, field
import json
from typing import Optional
import numpy as np
import tyro

from lerobot_sim2real.rl.ppo_rgb import PPOArgs, train


@dataclass
class Args:
    env_id: str
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    ppo: PPOArgs = field(default_factory=PPOArgs)
    """PPO training arguments"""


def main(args: Args):
    args.ppo.env_id = args.env_id
    if args.env_kwargs_json_path is not None:
        # Use the camera calibration integration system
        # The environment will load extrinsics/intrinsics from the file paths
        env_kwargs = {
            "env_config_path": args.env_kwargs_json_path,
            "use_learned_camera": True,
            "domain_randomization": True,  # Enable DR for training
            "domain_randomization_config": {
                "randomize_around_calibrated_camera": True,  # Randomize around calibrated pose
                "apply_calibration_offset_noise": False,  # Could enable for more robustness
            },
        }
        args.ppo.env_kwargs = env_kwargs
        print(f"Using camera calibration from {args.env_kwargs_json_path}")
        print("Camera position and FOV will be loaded from extrinsics/intrinsics files")
    else:
        print(
            "No env kwargs json path provided, using default env kwargs with default settings"
        )
    train(args=args.ppo)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
