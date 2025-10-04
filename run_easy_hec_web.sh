#!/bin/bash

source .venv/bin/activate
python lerobot_sim2real/scripts/web_easyhec_camera_calibration.py   --model-cfg sam2.1/sam2.1_hiera_l   --checkpoint ../sam2/checkpoints/sam2.1_hiera_large.pt   --server-name 0.0.0.0 --server-port 7860 --env-kwargs-json-path ./env_config.json