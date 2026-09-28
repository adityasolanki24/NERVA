#!/bin/bash
# First real session: R0 (reference regeneration, CPU, background) + B0 (baseline training, GPU).
set -uo pipefail
R0_JOBS=4 bash /work/NERVA/cloud/jobs/r0_references.sh > /work/out/r0_job.log 2>&1 &
r0=$!
bash /work/NERVA/cloud/jobs/b0_baseline.sh
b0_code=$?
wait $r0; r0_code=$?
echo "b0_exit=$b0_code r0_exit=$r0_code" | tee /work/out/session1_exit.txt
exit $(( b0_code || r0_code ))
