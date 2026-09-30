#!/bin/bash
# S5: S1 (7 styles) + a gentler per-style feet-height cost (-10; S2 used -30 and lost tempo expression).
# B2 showed the feet-height term is what makes backward walking work (2026-10-01).
set -euo pipefail
FEET_HEIGHT_SCALE=-10 exec bash /work/NERVA/cloud/jobs/s1_pilot.sh
