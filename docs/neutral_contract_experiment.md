# Shared neutral motor contract: preregistration

Completed: all fixed criteria pass across 2,180 NumPy/JAX ticks. Infrastructure
only; existing policies unchanged. [Results](../experiments/locomotion_curriculum/results_contract/README.md).

Reference subset failed 0/7; do not adopt it or train a candidate. This phase
implements independent contract infrastructure, not motor-performance claims.

## Exact contract

Rest iff planar command speed ≤0.005 m/s AND |yaw command| ≤0.02 rad/s.
Canonicalise the first three command slots to zero at rest; preserve head
slots. Freeze phase index at 0 with features `[1,0]` at rest. On movement onset
index is 0; each subsequent moving step increments by one modulo period steps.
Reset index=0, previous_rest=True, initial features `[1,0]` for every command,
so the first tick of an initially moving sequence also starts at index 0.
Initial motor baseline period is 27 steps (0.54 s at 50 Hz). No variable tempo.

One backend-parametric pure function supplies NumPy inference and JAX training
adapters; no duplicate threshold/phase definitions. Symmetric planar tracking
is `exp(-sum((canonical_xy-local_xy)^2)/0.01)`, no lateral tolerance. This helper
does not alter existing rewards, safety or any historical policy's clock.
No wiring to B2 or automatic inference/default adoption. Adapters are future
candidate infrastructure and require explicit future policy contract metadata.

## Fixed conformance evaluation

Seed 123, 1,000 commands: vx/vy uniform [-0.15,0.15], yaw uniform [-0.60,0.60],
head slots zero; every tenth command exactly zero. Then fixed 10-step blocks:
stop, exact rest boundary (vx=0.005,yaw=0.02), outside planar boundary vx=0.006,
outside yaw boundary yaw=0.025, forward 0.15, stop, head-only yaw 0.4,
left turn 0.60, stop. Evaluate all 1,090 ticks once for periods 27 and 20.
NumPy and JAX adapter phases agree within 1e-6, canonical commands within
1e-7; indices/modes agree exactly; phase norm within 1e-6; resting indices
zero, first moving index zero, all indices in range. No nonfinite values.
Symmetric tracking at zero command gives equal rewards for vx=0.05 and
vy=0.05 within 1e-6, both strictly less than rest reward.

All criteria required. Add meaningful tests for thresholds, head-only commands,
transitions, wraparound, batching/JIT and NumPy/JAX parity. Cap the standalone
local evaluation at 120 s wall time; no dynamics, PPO or paid work. Stop on
nonfinite/malformed state, mismatches or timeout, preserve failures. Commit
protocol then code before evaluation; record revisions and stream hash, small
JSON reports only. Full suite, Ruff, docs/log, public checks, read-only cloud
audit and commit/push after final verification.

Passing confirms only observation/clock conformance. Because the reference
subset failed, candidate environment/PPO smoke and restore continuation remain
blocked. B2's historical gate and original reference experiment criteria stay
unchanged. Next is a preregistered fit/derivative/contact investigation.
