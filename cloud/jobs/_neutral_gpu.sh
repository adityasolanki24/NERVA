#!/bin/bash
# Shared capped neutral GPU training job: `bash _neutral_gpu.sh EXPERIMENT` (an entry of
# experiments/locomotion_curriculum/gpu_pilot.py EXPERIMENTS). Same guards as neutral_gpu_pilot.sh:
# GPU assertion, checksum-verified input bundle, 180 s result sync, trainer deadline, hard `timeout`
# at uptime 2,580 s under the VM's 45-minute delete-on-expiry cap.
set -euo pipefail
EXPERIMENT="$1"
export OPEN_DUCK_ROOT=/work
export NERVA_CODE_SHA=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/nerva-code-sha)
OUT_GS="gs://$NERVA_BUCKET/runs/$NERVA_RUN_ID/out"
cd /work/NERVA
python -c "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', d"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee /work/out/gpu-model.txt
( while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader >> /work/out/gpu.csv
    sleep 30
  done ) &
( while true; do sleep 180; gsutil -q -m rsync -r /work/out "$OUT_GS" || echo "WARN: sync failed"; done ) &
# Inputs: only the declared bundle, checksum-verified before use.
gsutil -q cp "$NERVA_INPUT_URI/inputs.tar.gz" "$NERVA_INPUT_URI/MANIFEST.sha256" /work/
tar -xzf /work/inputs.tar.gz -C /work/NERVA
( cd /work/NERVA && sha256sum -c --strict /work/MANIFEST.sha256 ) > /work/out/input-verification.txt
cp /work/MANIFEST.sha256 /work/out/MANIFEST.sha256
UP=$(cut -d. -f1 /proc/uptime)
LEFT=$(( 2580 - UP ))
echo "uptime ${UP}s; trainer hard timeout ${LEFT}s"
[ "$LEFT" -gt 300 ] || { echo "ERROR: not enough time left before the cap"; exit 20; }
timeout --signal=INT --kill-after=60 "$LEFT" \
  python -m experiments.locomotion_curriculum.gpu_pilot --experiment "$EXPERIMENT" --out /work/out 2>&1 | tee /work/out/train.log
