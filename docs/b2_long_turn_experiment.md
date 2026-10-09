# B2 longer-turn support-region measurement

Completed: **aggregate inconclusive; all five zero-command controls migrate**.
Fixed criteria unchanged. [Results](../experiments/locomotion_curriculum/results_long_turn/README.md).

Preregistered before implementing or evaluating new rollouts. Follow-up to the
inconclusive short-record diagnostic; original B2 gate stays failed.

## Hypotheses and design

H1: over a one-minute steady interval, rotation-averaged robot and foot-region
positions stay within a small local region. H2: robot and foot region migrate
together across that interval. These are finite-duration descriptions, not
claims of asymptotic boundedness or readiness for expressive training.

Run exactly 15 primary trials: commands `(0,0,+0.60)`, `(0,0,-0.60)` and
`(0,0,0)`, seeds 0–4 each, 65 s each. Existing final B2 checkpoint, flat backlash
scene, 50 Hz, raw accelerometer, training observation noise, ±0.02 rad initial
joint noise, neutral style, zero head offsets. No affect, perception or memory.
Discard the first 5 s for scoring. No pushes, training, tuning or reruns.

Log base pose/velocity, whole-robot mass-weighted COM, both foot-site world
positions and foot contacts. Compute kinematics from a separate MuJoCo data
object at the recorded qpos/qvel; never call forward on the live physics state.
Select the free-joint robot subtree, excluding static scene objects. Contact
centroid is the mean of contacting foot-site positions; missing when airborne.
It is a site-based proxy, not a force centre or a measured support polygon.
Both-foot midpoint includes the swing foot and is a gait-region proxy.

For turns, unwrap base yaw, anchor at the first sample t ≥5, and divide by
signed displacement into consecutive complete 2π rotations. For each rotation,
average horizontal base, COM, both-foot midpoint and available contact-centroid
positions in time. Exclude only the unfinished final rotation. If yaw reverses
by >0.10 rad in any step or fewer than four full rotations occur, classify
inconclusive. For stop controls, use six fixed 10-second blocks [5,15), …,
[55,65], final block inclusive. Report contact coverage per block.

For every point series report maximum pairwise separation of block centres,
first-to-last centre displacement, and its magnitude divided by the difference
in block mean timestamps. Also report existing smoothed base-velocity gate
metrics over 5–65 s, falls, max tilt and shadow safety interventions.

Per trial, support **local-region** if COM and both-foot midpoint maximum
separation are each ≤0.05 m and endpoint drift speeds each ≤0.001 m/s.
Support **coherent migration** if both endpoint displacements are ≥0.10 m
and their vectors agree within 30 degrees. Otherwise inconclusive. Require
≥4 blocks, ≥90% contact-centroid coverage in every block, no fall, completed
duration and zero shadow interventions before supporting either description.
An aggregate description requires ≥4/5 supporting traces in each turn direction
and ≥4/5 stop controls supporting local-region. Report all failures separately.
Thresholds are prospective engineering choices, not modified motor-gate limits.

## Safety, stopping and provenance

SafetySupervisor remains shadow-only at 50 Hz. Stop the individual trial at
tilt >45 degrees or nonfinite state; mark it incomplete and unsupported. Stop
the entire experiment at 900 s wall time or missing/malformed inputs; preserve
partial outputs, never retry or overwrite. Maximum 975 simulated seconds.
No cloud work, payments, resource deletion or safety changes.

Commit protocol before code, and code before evaluation. Record revisions,
checkpoint/model hashes, upstream revision, seeds, duration and output hashes.
Small summary reports may be public; raw traces stay ignored. Test known orbit,
migration and incomplete/airborne signals, and verify kinematic COM against
MuJoCo subtree COM without altering the live state. Run full tests including
slow tests and Ruff; update docs/log; commit and push results. Retain negative
and inconclusive outcomes. Even local-region support cannot overturn the
original pure-turn base-velocity failure or authorise expressive training.
