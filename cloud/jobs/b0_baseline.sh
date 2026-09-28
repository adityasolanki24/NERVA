#!/bin/bash
# B0 (docs/style_policy_design.md): retrain the UPSTREAM baseline, unmodified, with the
# upstream README's "current win" command. Measures wall time for budgeting S1.
set -euo pipefail
cd /work/Open_Duck_Playground
python -c "import jax; assert jax.devices()[0].platform == 'gpu', jax.devices()"
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> /work/out/gpu.csv; sleep 60; done ) &
/usr/bin/time -v python playground/open_duck_mini_v2/runner.py \
  --task flat_terrain_backlash --num_timesteps 300000000 --output_dir /work/out/checkpoints \
  2>&1 | tee /work/out/train.log
ls -la /work/out/checkpoints
