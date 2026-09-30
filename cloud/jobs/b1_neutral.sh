#!/bin/bash
# B1 control: NERVA's style-conditioned environment fixed at neutral.
# Uses the same task, steps and upstream PPO configuration as B0.
set -euo pipefail
cd /work/Open_Duck_Playground
python -c "import jax; assert jax.devices()[0].platform == 'gpu', jax.devices()"
( while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader \
      >> /work/out/gpu.csv
    sleep 60
  done ) &
/usr/bin/time -v python -m nerva.training.train_style \
  --neutral \
  --task flat_terrain_backlash \
  --num_timesteps "${TRAIN_TIMESTEPS:-300000000}" \
  --feet_height_scale "${FEET_HEIGHT_SCALE:-0}" \
  --output_dir /work/out/checkpoints \
  2>&1 | tee /work/out/train.log
