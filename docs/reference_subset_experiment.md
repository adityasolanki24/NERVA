# Corrected neutral reference subset: preregistration

Completed: **0/7 pass**, no adoption. Known kinematics checks 12/12 pass;
all moving joint-velocity fits fail, static contact labels fail, and left turn
also fails knee/command checks. [Results](../experiments/locomotion_curriculum/results_reference_subset/README.md).

New local experiment, not a repeat of the completed 240-gait R0 or S1–S6.
Inspecting generator code found both angular double-angle differentiation and
an unconditional +0.00955 yaw-step bias. Preserve historical files unchanged.

## Fixed design

Generate exactly seven recordings: static stand `(0,0,0)`, forward/backward
vx ±0.074 m/s, left/right vy ±0.074 m/s, yaw ±0.60 rad/s. Other components
zero. Medium preset, period 0.54 s, step parameters = command × period / 2,
eight recorded seconds at nominal 50 Hz. Use `--stand` only for static stand.
One attempt per condition, two workers, each subprocess ≤180 s, experiment
≤900 s wall time. No repairs, neighbour substitutions, full-grid sweep, training
or cloud work. Preserve failed recordings and partial reports.

An isolated source wrapper removes the fixed yaw-step bias and records actual
Placo timestamps, keeping the original asset/import paths. Do not write to
upstream checkout/preset files. Reconstruct velocities from raw poses:
quaternions SciPy xyzw, world linear `(p[k]-p[k-1])/dt`, world angular
`rotvec(R[k] R[k-1]^-1)/dt`; express both in current body axes via R[k]^-1.
Joint velocity = joint-position difference / actual dt. First sample has no
derivative and is excluded. Verify known pure and mixed rotations at yaw 0/1,
dt 0.01/0.02/0.04; round-trip frame error ≤1e-8 and angular error ≤1e-6.

Fit a new explicitly labelled periodic Fourier schema (five harmonics), not
legacy degree-15 polynomials. Fit on timestamps [2,4), validate once on [4,6).
Phase = timestamp / recorded period modulo 1. Fit joint positions, contacts,
body linear/angular velocity. Derive fitted joint velocity analytically from
position coefficients. Static stand uses a constant pose and zero derivatives.
Do not relabel a moving gait as stationary or pretend this seven-point set is
a Cartesian training grid. Reject unsupported lookups; no implicit neighbours.

## All-required subset acceptance

For every condition: complete finite recording, increasing timestamps, period
0.54 ±1e-6, at least 80 samples in each fitting/validation interval; both knee
positions >0 throughout [2,6), including the fitted 100-phase cycle. No repair.
Held-out per-element joint-position RMSE ≤0.03 rad and joint-velocity RMSE
≤0.5 rad/s. Body linear RMSE ≤0.03 m/s and angular RMSE ≤0.10 rad/s. Evaluate
contacts by threshold >0.5; agreement ≥90% across held-out feet/samples.

Held-out signed commanded-axis mean ≥50% of requested magnitude; absolute
command-axis mean error ≤30% of magnitude. Noncommanded planar means each
≤0.02 m/s and noncommanded yaw mean ≤0.10 rad/s. Static stand linear/angular
maximum absolute velocities ≤1e-6. Joint-limit exceedances are reported,
not relaxed or silently repaired. Overall usable only if all seven pass.
No physical feasibility claim from fitting alone; dynamic motor gates remain
required and original B2 gate remains failed.

## Outputs and continuation

Commit protocol, then tested code before evaluation. Record source/preset/schema
and raw/output hashes, upstream revision, caps and fixed commands; small public
reports only, recordings/fits/logs remain ignored. Strip personal paths from
public reports. Stop on malformed input or timeout, preserve negatives.
Run full tests and Ruff before final results commit/push.

After evaluation, safely implement the shared rest/phase contract under its
own preregistration. A failed subset blocks training/reference adoption but
does not block unit-tested contract infrastructure. No paid work, deletion of
cloud artifacts, historical defaults or safety changes in this phase.
