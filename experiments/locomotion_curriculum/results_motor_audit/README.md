# Neutral motor audit: target issues confirmed; learned cause unproven

[Protocol](../../../docs/neutral_motor_audit.md) `fa22d19`; implementation
`a3d1220`. Seven nearest-reference targets sampled at the actual 27 float32
training phases; five synthetic reward-dispatch cases, 108 tracking probes,
18 touchdown cases and two generator yaw checks. No physics rollouts, training
or changes to historical rewards, references, policies, gates or safety.
Full numbers: report.json; source hashes and revisions: provenance.json.

## Stationary dispatch and sensitivity

| Fixed check | Result |
|---|---|
| imitation at zero motion, including head-only command | exactly zero |
| stationary joint pose/velocity cost | active; raw 2.8000004, scaled -0.5600001 in the declared synthetic state |
| motion norm exactly 0.010 | neither imitation nor stand-still active; strict `>`/`<` gates |
| motion norm 0.011 | imitation active; stand-still disabled |
| lateral velocities 0, 0.05, 0.10 at fixed vx | identical linear tracking reward; 0.11 reduces it |
| stop vx 0.013, other velocities zero | linear reward 1 → 0.983242; scaled loss 0.041895 |
| stop yaw 0.033, other velocities zero | angular reward 1 → 0.896820; scaled loss 0.619080 |
| no touchdown | height cost exactly zero at all declared lift values |
| 40 mm target touchdown | height cost zero |
| one 20 mm touchdown | raw height cost 0.25, scaled -7.5, same at stop and forward |

The height term is an ungated **touchdown cost**. Never stepping produces no
height cost, so it must not be claimed to reward starting to move. B2 has no
S6 air-time reward enabled. Its phase clock still advances at zero command.
These probes check formulas, not the total reward of a physically realised
state; the deliberately synthetic moving imitation state is not representative
of B2's real reward or proof of reward domination.

Linear tracking ignores up to 0.10 m/s of lateral error, at stop and during
pure turns as well as walking. It has no such tolerance for forward error.
This conflicts with the stricter 0.03 m/s horizontal pure-turn gate. It does
not prove that removing the tolerance alone will fix the trained controller.
No active reward anchors stationary world position; stand-still measures
actuator position/velocity, not accumulated COM displacement.

## References and velocity provenance

The grid has vx=0 but neither vy=0 nor yaw=0. Stop selects
`(0,-0.037,-0.074)`, with mean stored linear velocity
`(-0.000191,-0.039775,-0.000758)` m/s. This moving reference is looked up
even at rest, but the imitation gate disables its reward there. It is
therefore **not a demonstrated direct zero-command imitation incentive**.
Forward/backward also select nonzero lateral/yaw keys. Pure turns select
`(0,-0.037,+0.704)` / `(0,-0.037,-0.593)`, with mean stored yaw velocity
+0.014248 / -0.008100 rad/s. These results use the actual 27 training phases
and float32 evaluator; earlier diagnostics used 50 float64 phases.

The fitter's 40-signal ordering agrees with the reward's joint/contact/velocity
slices. Slice 3:7 is incorrectly treated as a reference quaternion, but its
orientation term is excluded from the returned reward. Do not attribute
the observed creep to that inactive term.

The inspected generator has a concrete derivative error: `as_rotvec()` already
contains the rotation angle, but `compute_angular_velocity` multiplies it by
`angle/dt` again. Pure +0.60 rad/s yaw at dt=0.02 produces **+0.0072 rad/s**
at both initial yaw 0 and 1. Correct differentiation is rotation-vector/dt.
Quaternions in this helper are consistently SciPy xyzw; ordering is not the
cause of this failed probe. Its relative rotation is expressed in the prior
body frame although the output is named world angular velocity.

Generator/fitter store world linear and nominal world angular velocities.
Imitation uses free-joint world translation and body angular components;
command tracking uses local linear velocity and gyro. The world linear target
is not rotated with the randomized reset yaw, so its axes need an explicit
frame contract. Angular reference/consumer conventions also need correction.
Source inspection establishes those contracts and the helper bug, **not**
which source produced the historical shipped pickle or causation of B2 drift.
Existing knee-only plausibility validation cannot validate these velocity
units/frame properties. No upstream code or pickle was modified.

Next: the [motor-only candidate design](../../../docs/neutral_motor_candidate.md)
requires verified reference derivatives/frames, an explicit rest mode and
two-axis velocity tracking before local smoke or any separately capped training.
The original B2 gate remains failed and expressive training remains blocked.

Validation: five new probe/dispatch tests; full suite including slow tests
258 passed. Ruff and public-content checks clean. The two existing JAX cast
warnings occurred in unchanged training tests. No cloud compute/network
resources; results bucket and runner identity retained.
