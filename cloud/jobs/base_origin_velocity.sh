#!/bin/bash
# Base-origin reward velocity continuation (docs/base_origin_velocity_pilot.md). Launch with:
#   python cloud/launch.py launch --job base_origin_velocity --hw l4 --max-minutes 45 \
#     --input-bundle cloud/inputs/base_origin_velocity.txt --wait --cleanup-network
set -euo pipefail
bash /work/NERVA/cloud/jobs/_neutral_gpu.sh base_origin_velocity
