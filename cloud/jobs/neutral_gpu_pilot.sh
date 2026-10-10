#!/bin/bash
# Capped GPU neutral continuation (docs/gpu_neutral_pilot.md). Launch with:
#   python cloud/launch.py launch --job neutral_gpu_pilot --hw l4 --max-minutes 45 \
#     --input-bundle cloud/inputs/neutral_gpu_pilot.txt --wait --cleanup-network
# Independent guards: the VM's 45-minute delete-on-expiry cap (Google), the trainer's own deadline at
# uptime 2,220 s, and a hard `timeout` here at uptime 2,580 s. Results sync every 180 s.
set -euo pipefail
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
  python -m experiments.locomotion_curriculum.gpu_pilot --out /work/out 2>&1 | tee /work/out/train.log
