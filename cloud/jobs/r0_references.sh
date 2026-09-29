#!/bin/bash
# R0 (docs/style_policy_design.md): regenerate the NEUTRAL reference set with the upstream
# generator, unmodified, using its own README commands, to check the generator still works
# (its README has an open TODO about this) and how long generation takes.
# Output: /work/out/r0/{recordings/, polynomial_coefficients_neutral.pkl, generator-freeze.txt, timing}
set -euo pipefail
GEN_SHA=3d7bc6fd80e6c35587b2a5279314d946cf63456b
JOBS=${R0_JOBS:-$(nproc)}
WORK_ROOT=${NERVA_WORK_ROOT:-/work}
OUT_ROOT=${NERVA_OUT_ROOT:-$WORK_ROOT/out}
GEN_ROOT=${R0_GENERATOR_ROOT:-$WORK_ROOT/gen}
mkdir -p "$OUT_ROOT/r0"
overall_start=$(date +%s)
if ! command -v git-lfs >/dev/null; then
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update -qq && apt-get install -y -qq git-lfs
  elif sudo -n true 2>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
  else
    echo "git-lfs is required (install it or provide passwordless sudo)" >&2
    exit 2
  fi
fi
git lfs install --skip-repo
git clone -q https://github.com/apirrone/Open_Duck_reference_motion_generator.git "$GEN_ROOT"
cd "$GEN_ROOT" && git checkout -q "$GEN_SHA" && git lfs pull
uv sync                                   # resolves python 3.10.12 + placo 0.6.3 per its pyproject
uv pip freeze > "$OUT_ROOT/r0/generator-freeze.txt"
start=$(date +%s)
uv run scripts/auto_waddle.py -j "$JOBS" --duck open_duck_mini_v2 --sweep \
  --output_dir "$OUT_ROOT/r0/recordings" > "$OUT_ROOT/r0/generate.log" 2>&1
gen_s=$(( $(date +%s) - start ))
fit_start=$(date +%s)
uv run scripts/fit_poly.py --ref_motion "$OUT_ROOT/r0/recordings" > "$OUT_ROOT/r0/fit.log" 2>&1
fit_s=$(( $(date +%s) - fit_start ))
mv polynomial_coefficients.pkl "$OUT_ROOT/r0/polynomial_coefficients_neutral.pkl"
n=$(find "$OUT_ROOT/r0/recordings" -maxdepth 1 -name '*.json' -type f | wc -l)
total_s=$(( $(date +%s) - overall_start ))
echo "gaits=$n generate_seconds=$gen_s fit_seconds=$fit_s total_seconds=$total_s jobs=$JOBS generator_sha=$GEN_SHA" \
  | tee "$OUT_ROOT/r0/timing.txt"
R0_SUMMARY="$OUT_ROOT/r0/summary.json" uv run python - "$n" "$gen_s" "$fit_s" "$total_s" "$JOBS" "$GEN_SHA" <<'PY'
import json
import os
import sys
from pathlib import Path

gaits, generate, fit, total, jobs, sha = sys.argv[1:]
Path(os.environ["R0_SUMMARY"]).write_text(json.dumps({
    "gaits": int(gaits),
    "generate_seconds": int(generate),
    "fit_seconds": int(fit),
    "total_seconds": int(total),
    "jobs": int(jobs),
    "generator_sha": sha,
}, indent=2, sort_keys=True) + "\n")
PY
