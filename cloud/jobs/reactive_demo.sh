#!/bin/bash
# Render the reactive-behaviour demo (experiments/reactive/render.py) on a CPU VM with the S1 policy.
set -euo pipefail
: "${NERVA_BUCKET:?}"
S1=${S1_RUN:-s1_pilot-20260929-231102}
latest=$(gsutil ls "gs://$NERVA_BUCKET/runs/$S1/out/checkpoints/*.onnx" | sort | tail -1)
gsutil -q cp "$latest" /work/policy.onnx
echo "$S1 <- $latest" | tee /work/out/policy.txt
apt-get update -qq && apt-get install -y -qq libosmesa6 fonts-dejavu-core >/dev/null
source /work/venv/bin/activate
uv pip install matplotlib==3.11.2 imageio==2.38.0 imageio-ffmpeg==0.6.0
export OPEN_DUCK_ROOT=/work JAX_PLATFORMS=cpu MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
cd /work/NERVA
python experiments/reactive/render.py --policy /work/policy.onnx --out /work/out/reactive 2>&1   | grep -vE "^(joint|actuator|backlash) names" | tee /work/out/render.log
