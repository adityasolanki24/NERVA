#!/bin/bash
# S2 evaluation: same protocol as S1 (experiments/style_policy/evaluate.py). In the report,
# "S1" is the style policy under test (here S2) and "B1" its neutral control (here B2).
set -euo pipefail
S1_RUN=s2_pilot-20260930-163509 B1_RUN=b2_neutral-20260930-163257 exec bash /work/NERVA/cloud/jobs/s1_eval.sh
