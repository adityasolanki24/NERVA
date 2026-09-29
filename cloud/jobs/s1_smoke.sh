#!/bin/bash
# Tiny end-to-end compile/train check for the seven-style environment. Not a result.
set -euo pipefail
: "${NERVA_INPUT_URI:?launch with --input-run or --input-dir}"
mkdir -p /work/refs
gsutil -m cp "$NERVA_INPUT_URI/*.pkl" /work/refs/
gsutil cp "$NERVA_INPUT_URI/styles.json" /work/refs/styles.json
cd /work/Open_Duck_Playground
python -c "import jax; assert jax.devices()[0].platform == 'gpu', jax.devices()"
/usr/bin/time -v python -m nerva.training.train_style \
  --refs /work/refs \
  --task flat_terrain_backlash \
  --output_dir /work/out/checkpoints \
  --smoke \
  2>&1 | tee /work/out/train.log
