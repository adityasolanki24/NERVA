#!/bin/bash
# Short, non-scientific hardware comparison using the exact upstream PPO path.
# Two evaluations keep export coverage while checkpoint timestamps measure only
# the batch-rounded training interval. Defaults to about three L4 PPO batches.
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
  --timesteps "${BENCHMARK_TIMESTEPS:-1000000}" \
  --evals 2 \
  --task flat_terrain \
  --output-dir /work/out/checkpoints \
  --summary-json /work/out/benchmark.json \
  2>&1 | tee /work/out/train.log
cat /work/out/benchmark.json
