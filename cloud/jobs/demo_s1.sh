#!/bin/bash
# Render the S1 affect demo video (experiments/demo_video/run_s1.py) on a CPU VM.
# Offscreen MuJoCo rendering via OSMesa; video packages pinned to the local dev versions.
set -euo pipefail
: "${NERVA_BUCKET:?}"
S1=${S1_RUN:-s1_pilot-20260929-231102}
latest=$(gsutil ls "gs://$NERVA_BUCKET/runs/$S1/out/checkpoints/*.onnx" | sort | tail -1)
gsutil -q cp "$latest" /work/s1.onnx
echo "$S1 <- $latest" | tee /work/out/policy.txt
apt-get update -qq && apt-get install -y -qq libosmesa6 fonts-dejavu-core >/dev/null
source /work/venv/bin/activate
uv pip install matplotlib==3.11.2 imageio==2.38.0 imageio-ffmpeg==0.6.0
export OPEN_DUCK_ROOT=/work JAX_PLATFORMS=cpu MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
cd /work/NERVA
python experiments/demo_video/run_s1.py --policy /work/s1.onnx --out /work/out/demo_s1 2>&1   | grep -vE "^(joint|actuator|backlash) names" | tee /work/out/demo.log
