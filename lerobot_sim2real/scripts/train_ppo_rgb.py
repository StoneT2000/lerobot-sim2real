"""Simple script to train a RGB PPO policy in simulation"""

import json
from typing import Optional
import tyro

from lerobot_sim2real.rl.ppo_rgb import PPOArgs, train

class Args:
    env_id: str
    env_kwargs_json_path: Optional[str] = None
    ppo: PPOArgs = PPOArgs()

def main(args: Args):
    args.ppo.env_id = args.env_id
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs = json.load(f)
        args.ppo.env_kwargs = env_kwargs
    else:
        print("No env kwargs json path provided, using default env kwargs with default settings")
    train(args=args.ppo)

if __name__ == "__main__":

    args = tyro.cli(Args)
    main(args)
