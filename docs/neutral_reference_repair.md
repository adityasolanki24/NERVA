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
