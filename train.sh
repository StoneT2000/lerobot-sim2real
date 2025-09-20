#!/bin/bash

# Ensure this script runs from the repo root even if invoked elsewhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# To run this script in the background and save logs:
# nohup ./train.sh > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# 
# This will:
# - Run the training in the background (continues even if you close the terminal)
# - Save all output to a timestamped log file (e.g., training_20240115_143022.log)
# - Return control to your terminal immediately
# - You can monitor progress with: tail -f training_*.log
# - To stop training: ps aux | grep train_ppo_rgb.py then kill <PID>

seed=44
python -m lerobot_sim2real.scripts.train_ppo_rgb \
  --env-id="SO101GraspCubeLeRobotSim2Real-v1" \
  --env-kwargs-json-path=env_config.json \
  --ppo.seed=${seed} \
  --ppo.num_envs=1024 \
  --ppo.num-steps=16 \
  --ppo.update_epochs=8 \
  --ppo.num_minibatches=32 \
  --ppo.total_timesteps=100_000_000 \
  --ppo.gamma=0.9 \
  --ppo.num_eval_envs=16 \
  --ppo.num-eval-steps=64 \
  --ppo.no-partial-reset \
  --ppo.exp-name="gandalf-so101graspcube-${seed}" \
  --ppo.track \
  --ppo.wandb_project_name "SO101-ManiSkill"