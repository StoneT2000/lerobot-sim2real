export seed=3
cd ~/lerobot_ws/lerobot-sim2real
python lerobot_sim2real/scripts/train_ppo_rgb.py --env-id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json --ppo.seed=${seed} --ppo.num_envs=256 --ppo.num-steps=16 --ppo.update_epochs=8 --ppo.num_minibatches=16 --ppo.total_timesteps=100_000_000 --ppo.gamma=0.9 --ppo.num_eval_envs=4 --ppo.num-eval-steps=64 --ppo.no-partial-reset --ppo.exp-name="ppo-SO100GraspCube-v1-rgb-${seed}" --ppo.track --ppo.wandb_project_name "SO100-ManiSkill"
