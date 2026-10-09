# B2 turn-centre diagnostic: inconclusive

Protocol: [preregistration](../../../docs/b2_turn_diagnostic.md) at `69dc1e0`;
analysis at `661c4be`. Ten existing primary traces; no new physics or training.
Hashes and revisions are in provenance.json; all per-trace metrics are in report.json.

| Measurement | Left, five seeds | Right, five seeds |
|---|---:|---:|
| bounded-orbit classification | 0/5 | 1/5 |
| sustained-world-drift classification | 0/5 | 0/5 |
| inconclusive | 5/5 | 4/5 |
| held-out orbit position RMSE | 13.74–33.46 mm | 8.66–39.07 mm |
| fitted constant world drift speed | 1.88–4.46 mm/s | 1.11–9.97 mm/s |

All designs have full rank; drift-model condition numbers are 15.18–17.06.
Small fitted drift speeds do not prove bounded motion: nine orbit predictions
miss the preregistered 10 mm held-out tolerance. The short records and two fixed
models cannot distinguish moving turn centres or more complex drift. The
original base-velocity gate remains failed; neither aggregate hypothesis passes.

The shipped reference selects `(0,-0.037,+0.704)` for left and
`(0,-0.037,-0.593)` for right. Across the 50 fixed phases, mean reference
linear velocity is `(-0.00874,-0.01252,-0.00065)` and
`(-0.01242,-0.05852,-0.00075)` m/s respectively; mean reference yaw angular
velocity is +0.01425 / -0.00810 rad/s. These are the stored reward slices,
not evidence that the generator achieved the requested yaw velocities.
Reference lateral sign disagrees with all left traces and agrees with all
right traces. The nearest-reference asymmetry is a training-target concern,
not an established cause of the fixed policy's behaviour. Runtime does not
consult this reference.

Next recommended local phase: separately preregister longer pure-turn trials
with direct base, centre-of-mass and foot-position logging to measure whether
the support region drifts. Fix duration, seeds, windows and support-centre
metrics prospectively. Do not relax the original gate, regenerate references
as a purported runtime fix, or start expressive training from this result.
