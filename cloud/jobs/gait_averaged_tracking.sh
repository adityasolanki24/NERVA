#!/bin/bash
# Gait-averaged tracking continuation (docs/gait_averaged_tracking_pilot.md). Launch with:
#   python cloud/launch.py launch --job gait_averaged_tracking --hw l4 --max-minutes 45 #     --input-bundle cloud/inputs/gait_averaged_tracking.txt --wait --cleanup-network
set -euo pipefail
bash /work/NERVA/cloud/jobs/_neutral_gpu.sh gait_averaged_tracking
