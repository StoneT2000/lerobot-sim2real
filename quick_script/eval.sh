#python lerobot_sim2real/scripts/eval_ppo_rgb.py --env_id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json     --checkpoint=$1 --debug --record-dir=eval_records --no-continuous-eval 
cd ~/lerobot_ws/lerobot-sim2real
python lerobot_sim2real/scripts/eval_ppo_rgb.py \
	--env_id="SO100GraspCube-v1" \
	--env-kwargs-json-path=env_config.json \
	--checkpoint=$1 \
	--seed=3 \
	--debug \
	--record-dir=eval_records 
