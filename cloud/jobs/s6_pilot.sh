#!/bin/bash
# S6: S5 (7 styles + feet-height -10) plus a feet air-time reward (2.0), so that never stepping can no
# longer avoid the swing cost (S5 learned to stand still, 2026-10-01).
set -euo pipefail
FEET_HEIGHT_SCALE=-10 EXTRA_TRAIN_ARGS="--feet_air_time_scale 2.0" exec bash /work/NERVA/cloud/jobs/s1_pilot.sh
