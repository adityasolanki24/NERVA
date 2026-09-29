#!/bin/bash
# S1 evaluation (docs/style_policy_design.md §5) on CPU: MuJoCo rollouts of the final
# S1, B0 and B1 ONNX policies from their cloud runs (2026-09-29).
set -euo pipefail
: "${NERVA_BUCKET:?}"
S1=${S1_RUN:-s1_pilot-20260929-231102}
B0=${B0_RUN:-b0_baseline-20260929-230717}
B1=${B1_RUN:-b1_neutral-20260929-230837}
mkdir -p /work/policies
for run in "$S1" "$B0" "$B1"; do
  latest=$(gsutil ls "gs://$NERVA_BUCKET/runs/$run/out/checkpoints/*.onnx" | sort | tail -1)
  gsutil -q cp "$latest" "/work/policies/$run.onnx"
  echo "$run <- $latest" | tee -a /work/out/policies.txt
done
source /work/venv/bin/activate
export OPEN_DUCK_ROOT=/work JAX_PLATFORMS=cpu
cd /work/NERVA
python experiments/style_policy/evaluate.py \
  --s1 "/work/policies/$S1.onnx" --b0 "/work/policies/$B0.onnx" --b1 "/work/policies/$B1.onnx" \
  --out /work/out/s1_eval 2>&1 | grep -vE "^(joint|actuator|backlash) names" | tee /work/out/eval.log
