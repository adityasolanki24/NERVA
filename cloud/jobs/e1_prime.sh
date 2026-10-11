#!/bin/bash
# E1' pitch x crouch conditioning (docs/expressive_posture_e1prime.md). Launch with the authorized cap, e.g.:
#   python cloud/launch.py launch --job e1_prime --hw l4 --max-minutes 90 \
#     --input-bundle cloud/inputs/e1_prime.txt --wait --cleanup-network
set -euo pipefail
bash /work/NERVA/cloud/jobs/_neutral_gpu.sh e1_prime
