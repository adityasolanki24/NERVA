#!/bin/bash
# Turn-translation tracking continuation (docs/turn_translation_pilot.md). Launch with:
#   python cloud/launch.py launch --job turn_translation --hw l4 --max-minutes 45 \
#     --input-bundle cloud/inputs/turn_translation.txt --wait --cleanup-network
set -euo pipefail
bash /work/NERVA/cloud/jobs/_neutral_gpu.sh turn_translation
