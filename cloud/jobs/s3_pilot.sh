#!/bin/bash
# S3: as S1 (7 R1 styles, no feet-height cost) but trained with head commands applied to the head
# motors, as the hardware runtime applies them, so the policy can walk while gazing (2026-09-30).
set -euo pipefail
EXTRA_TRAIN_ARGS=--apply_head_commands exec bash /work/NERVA/cloud/jobs/s1_pilot.sh
