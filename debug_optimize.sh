#!/bin/bash

source .venv/bin/activate

for d in -0.03 0 0.03; do
  python -m lerobot_sim2real.scripts.debug_optimize_from_results \
    --init_tx -0.40 --init_ty 0.10 --init_tz 0.50 \
    --iterations 1500 --early_stopping_steps 400
done