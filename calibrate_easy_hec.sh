#!/bin/bash
source .venv/bin/activate
python lerobot_sim2real/scripts/easyhec_camera_calibration.py \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint ../sam2/checkpoints/sam2.1_hiera_large.pt \
  --env-kwargs-json-path env_config.json