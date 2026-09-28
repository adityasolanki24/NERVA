#!/bin/bash
# Pipeline smoke test: the upstream environment and PPO path for one 200k-step batch.
# Checks: GPU visible to JAX, MJX compiles, PPO trains, checkpoints + ONNX export work,
# results reach GCS, VM deletes itself. Not a scientific result.
set -euo pipefail
cd /work/Open_Duck_Playground
python -c "import jax; assert jax.devices()[0].platform == 'gpu', jax.devices()"
/usr/bin/time -v python /work/NERVA/cloud/smoke_train.py \
  2>&1 | tee /work/out/train.log
ls -la /work/out/checkpoints
