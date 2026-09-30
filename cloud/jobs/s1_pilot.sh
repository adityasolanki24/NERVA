#!/bin/bash
# S1: train one policy across the seven R1 pilot styles.
set -euo pipefail
: "${NERVA_INPUT_URI:?launch with --input-run or --input-dir}"
mkdir -p /work/refs
gsutil -m cp "$NERVA_INPUT_URI/*.pkl" /work/refs/
gsutil cp "$NERVA_INPUT_URI/styles.json" /work/refs/styles.json
cd /work/Open_Duck_Playground
python -c "import jax; assert jax.devices()[0].platform == 'gpu', jax.devices()"
( while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader \
      >> /work/out/gpu.csv
    sleep 60
  done ) &
/usr/bin/time -v python -m nerva.training.train_style \
  --refs /work/refs \
  --task flat_terrain_backlash \
  --num_timesteps "${TRAIN_TIMESTEPS:-300000000}" \
  --feet_height_scale "${FEET_HEIGHT_SCALE:-0}" \
  ${EXTRA_TRAIN_ARGS:-} \
  --output_dir /work/out/checkpoints \
  2>&1 | tee /work/out/train.log
