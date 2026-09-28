#!/bin/bash
# Pipeline smoke test: the UPSTREAM baseline runner, unmodified, for only 2M env steps.
# Checks: GPU visible to JAX, MJX compiles, PPO trains, checkpoints + ONNX export work,
# results reach GCS, VM deletes itself. Not a scientific result.
set -euo pipefail
cd /work/Open_Duck_Playground
python -c "import jax; assert jax.devices()[0].platform == 'gpu', jax.devices()"
/usr/bin/time -v python playground/open_duck_mini_v2/runner.py \
  --task flat_terrain --num_timesteps 2000000 --output_dir /work/out/checkpoints \
  2>&1 | tee /work/out/train.log
ls -la /work/out/checkpoints
