#!/bin/bash
# R0 (docs/style_policy_design.md): regenerate the NEUTRAL reference set with the upstream
# generator, unmodified, using its own README commands, to check the generator still works
# (its README has an open TODO about this) and how long generation takes.
# Output: /work/out/r0/{recordings/, polynomial_coefficients_neutral.pkl, generator-freeze.txt, timing}
set -euo pipefail
GEN_SHA=3d7bc6fd80e6c35587b2a5279314d946cf63456b
JOBS=${R0_JOBS:-$(nproc)}
mkdir -p /work/out/r0
command -v git-lfs >/dev/null || { apt-get update -qq && apt-get install -y -qq git-lfs; }
git lfs install --skip-repo
git clone -q https://github.com/apirrone/Open_Duck_reference_motion_generator.git /work/gen
cd /work/gen && git checkout -q "$GEN_SHA" && git lfs pull
uv sync                                   # resolves python 3.10.12 + placo 0.6.3 per its pyproject
uv pip freeze > /work/out/r0/generator-freeze.txt
start=$(date +%s)
uv run scripts/auto_waddle.py -j "$JOBS" --duck open_duck_mini_v2 --sweep \
  --output_dir /work/out/r0/recordings > /work/out/r0/generate.log 2>&1
gen_s=$(( $(date +%s) - start ))
uv run scripts/fit_poly.py --ref_motion /work/out/r0/recordings > /work/out/r0/fit.log 2>&1
mv polynomial_coefficients.pkl /work/out/r0/polynomial_coefficients_neutral.pkl
n=$(ls /work/out/r0/recordings/*.json | wc -l)
echo "gaits=$n generate_seconds=$gen_s jobs=$JOBS" | tee /work/out/r0/timing.txt
