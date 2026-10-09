# Neutral reference repair package

Preregistered 2026-10-09, before implementation or evaluation. Previous subset
and alignment results remain unchanged. This package comparison cannot identify
which individual change causes an improvement.

Hypothesis: a smooth, reachable gait with a deterministic positive knee branch,
consistent support labels and a richer position-derived velocity representation
can supply seven usable motor-only targets.

Fixed package: preserve the medium preset except COM height 0.215 m, foot lift
0.020 m and foot rise ratio 0.30 (replace the sharp 0.02 rise). Seed both knees
at 1.2 rad before initial IK. Enable URDF joint limits and set knee limits to
[0.01, pi/2]. Use the existing isolated recorder with zero yaw bias and actual
timestamps; set its solver substep duration to dt/10 on each tick. For stand,
require both frozen foot frames within 1 mm of the floor and label both as
geometric support; moving labels remain planned support, not measured force.
No pose clipping, command relabelling or nearest-gait substitution.

Fit a version-2 Fourier reference with exactly 12 harmonics. Joint positions
and their backward interval velocities share coefficients. Least squares stacks
position rows with 0.02 times interval-average derivative rows, using actual
timestamps (0.02 is seconds). Other channels use position-time least squares.
Runtime joint velocity is the analytic instantaneous derivative; held-out
comparison uses the exact interval average against recorded finite differences.
Contact threshold stays 0.5. Static joint/contact coefficients are constant;
static velocity is zero. Version-1/H5 results stay supported unchanged.

Use the same seven commands, one eight-second recording each, two workers,
180 s per recording, 900 s total, Linux timeout plus parent watchdog. Train
on [2,4), hold out [4,6), minimum 80 samples each, period 0.54 s. All seven
must pass: worst component RMSE positions <=0.03 rad, interval joint velocity
<=0.5 rad/s, body linear <=0.03 m/s, body angular <=0.10 rad/s; contact agreement
>=90%; positive knees and all recorded/fitted joints inside URDF limits (1e-5
numerical tolerance, fitted cycle 1,000 points). Commanded-axis mean has correct
sign, >=50% commanded magnitude and <=30% error; off-axis planar <=0.02 m/s,
yaw <=0.10 rad/s. Stand body velocities <=1e-6. Verify timestamps, joint order,
finite values, schema and known rotation/derivative cases before acceptance.

Stop for malformed data, source mismatch, timeout or generator errors. Collect
all seven results once; do not tune limits after observing them. A failed
package blocks training adoption. Raw artifacts remain ignored, public reports
contain hashes and measurements only. No paid work or upstream edits.

If all seven pass, implement an opt-in motor-only environment: seven discrete
commands with zero head offsets, shared rest/phase contract, body-frame imitation,
symmetric planar tracking, no foot-height/air-time objectives, rest-only pose
cost. Keep historical environments/defaults unchanged. Before any dynamic smoke,
record its exact protocol and cap. This does not establish learned robustness;
PPO training and long motor-readiness gates are separate phases.

## Result and follow-up registration

Protocol `1b86c37`, implementation `df51807`: 6/7 pass in 83.75 s. All fit,
contact, knee and joint-limit criteria pass. Moving interval-velocity RMSE
maxima are 0.217–0.353 rad/s. Left turn alone fails off-axis translation:
mean body vy +0.037961 m/s, limit 0.020. Right turn vy -0.006964; yaw means
±0.603689. No training adoption or candidate dynamics yet.

Before another evaluation, preregister a geometric pivot correction for the
two pure turns. The repetitive planner composes foot translation and then yaw
(see [Placo opposite_frame source](https://github.com/Rhoban/placo/blob/v0.6.4/src/placo/humanoid/humanoid_parameters.cpp)).
This suggests that a base forward offset from the foot midpoint contributes
body lateral velocity during yaw; this is a hypothesis, not a measured cause.
The installed 0.6.3 build has no matching Git tag; verify its actual opposite
frame operation against the formula before generation, rather than assuming
the adjacent 0.6.4 source matches.

After initial IK, define c as the planar base-position minus foot-midpoint
offset, expressed in the initial yaw frame. For a requested pure turn with
step angle theta, add internal planner translation `(I - Rz(theta)) c`.
Keep requested command labels zero translation and ±0.60 yaw; record the
offset and actual internal step in raw metadata. Do not translate recorded
poses or change fit/reward targets after generation. Analytic known-geometry
tests must show a fixed pivot after rotation; reject nonfinite/malformed
inputs. Zero yaw returns zero correction; both signs must be checked.

Generate only two new eight-second turns, one attempt each, two workers,
180 s each and 420 s total. Use the repaired preset, fit and every unchanged
criterion above. Retain the five passing nonturn references by verified hash;
this is explicitly a mixed-provenance subset, not seven newly rerun results.
Both turns must pass to admit the subset to candidate-only testing. Preserve
the 6/7 failure and stop adoption on any failure. No empirical parameter
tuning or paid work. An admitted subset is still not a trained motor policy.

The first pivot attempt (`3dca496`) aborted in its pre-generation verifier:
the 0.6.3 binding cannot return `Footstep.frame`'s Eigen transform. No recordings
were created and no turn outcomes were observed. Preserve the aborted reports.
Repair the verifier by comparing its exposed four-corner support polygon with
the same translation/yaw formula. Authorise one retry of the same two turns in
new output directories, under the same caps and criteria; this is an API
compatibility repair, not a parameter/criteria change.

The polygon-verifier retry passes 2/2 turns in 18.31 s (`bc176bb`, recorded in
`f92e19e`). Five passing repair targets plus these two turns are admitted only
to the candidate smoke. Left/right body lateral means +0.016680/+0.014414 m/s,
yaw ±0.603689 rad/s. The original 6/7 failure and preflight abort are preserved.
During final review, the joint-limit implementation was tightened from the
scored interval to every recorded frame, matching the original preregistration.
Independent confirmation checks all 400 raw frames plus 1,000 cycle samples
per admitted target: all pass, minimum joint-limit margin 0.137816 rad across
the subset. A regression test rejects out-of-bounds warmup frames. This adds
coverage; it does not change any criterion or fit. Evidence:
`results_neutral_smoke/full_recording_joint_limits.json`.
