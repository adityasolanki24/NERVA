#!/bin/bash
# Short, non-scientific hardware comparison using the exact upstream PPO path.
# Three evaluations = two training chunks. The FIRST chunk includes compiling the
# training step, so the number to compare is steady_training_steps_per_second in
# benchmark.json: Brax's own training/walltime between the last two evaluations
# (no compilation, evaluation or export). 1.5 M requested steps -> 2 chunks of
# 819,200 steps with upstream's batch settings.
set -euo pipefail
cd /work/Open_Duck_Playground
python -c "import jax; assert jax.devices()[0].platform == 'gpu', jax.devices()"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  | tee /work/out/gpu-model.txt
( while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
      --format=csv,noheader >> /work/out/gpu.csv
    sleep 30
  done ) &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
/usr/bin/time -v python /work/NERVA/cloud/smoke_train.py \
  --timesteps "${BENCHMARK_TIMESTEPS:-1500000}" \
  --evals 3 \
  --task flat_terrain \
  --output-dir /work/out/checkpoints \
  --summary-json /work/out/benchmark.json \
  2>&1 | tee /work/out/train.log
cat /work/out/benchmark.json
