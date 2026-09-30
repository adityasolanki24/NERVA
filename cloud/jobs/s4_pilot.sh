#!/bin/bash
# S4: as S3 (head commands applied) plus backward-emphasis command sampling (30%), because every
# policy walks backward at only ~25% of the command (2026-10-01).
set -euo pipefail
EXTRA_TRAIN_ARGS="--apply_head_commands --backward_fraction 0.3" exec bash /work/NERVA/cloud/jobs/s1_pilot.sh
