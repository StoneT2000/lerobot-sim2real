"""Simple script to train a RGB PPO policy in simulation"""

from dataclasses import dataclass, field
import json
from typing import Optional
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
        with open(args.env_kwargs_json_path, "r") as f:
            data = json.load(f)
            calibration_offset = data.pop("calibration_offset")
            args.ppo.env_kwargs = data
    else:
        print("No env kwargs json path provided, using default env kwargs with default settings")
    train(args=args.ppo)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)