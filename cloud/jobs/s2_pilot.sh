#!/bin/bash
# S2_PILOT: as s1_pilot plus the per-style feet-height cost (docs/development_log.md, 2026-09-30).
# Weight -30 fixed before training: at the measured under-lift (~14 of 40 mm) it costs
# about 0.9/step, comparable to but below each tracking term and imitation (~1.4-1.6/step, B0).
set -euo pipefail
FEET_HEIGHT_SCALE=-30 exec bash /work/NERVA/cloud/jobs/s1_pilot.sh
