#!/bin/bash
# R1 pilot: generate neutral plus +/-1 on each preregistered style axis.
# This is CPU-only and produces styles.json plus seven fitted reference pickles.
set -euo pipefail
GEN_SHA=3d7bc6fd80e6c35587b2a5279314d946cf63456b
JOBS=${R1_JOBS:-$(nproc)}
mkdir -p /work/out/r1
command -v git-lfs >/dev/null || { apt-get update -qq && apt-get install -y -qq git-lfs; }
git lfs install --skip-repo
git clone -q https://github.com/apirrone/Open_Duck_reference_motion_generator.git /work/gen
cd /work/gen && git checkout -q "$GEN_SHA" && git lfs pull
uv sync
uv pip freeze > /work/out/r1/generator-freeze.txt
uv run python /work/NERVA/cloud/generate_styled_references.py \
  --generator-root /work/gen \
  --output-root /work/out/r1/references \
  --jobs "$JOBS"
