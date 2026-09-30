#!/bin/bash
# S3 style evaluation: same protocol as S1 (experiments/style_policy/evaluate.py); "S1" in the report is S3.
set -euo pipefail
S1_RUN=s3_pilot-20260930-235950 exec bash /work/NERVA/cloud/jobs/s1_eval.sh
