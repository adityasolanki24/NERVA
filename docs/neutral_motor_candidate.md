# Neutral motor baseline candidate (opt-in environment)

Status update 2026-10-09: the original subset stays 0/7 and alignment remains
insufficient. A preregistered smooth/limited repair passes 6/7; geometric pivot
correction passes both turns, yielding seven hash-verified candidate targets.
`NeutralJoystick` and explicit inference contract opt-in are implemented. The
80-step CPU smoke passes, with no terminations or zero-clipped rewards. A local
32-transition PPO check updates networks and restores actions; a separately
preregistered 32-transition retry passes strict live-byte checks in 203.19 s.
No validated learned motor-readiness result.

Grounding: [neutral motor audit](neutral_motor_audit.md) and longer-turn
measurements. This is an implemented candidate environment, not evidence of
improved learned locomotion. B2, S1–S6 and historical defaults remain reproducible. The audit
establishes target issues; it does not establish causes of the learned drift.

## Rest and movement contract

Infer a rest mode from planar command speed ≤0.005 m/s AND absolute yaw command
≤0.02 rad/s, avoiding the current mixed-unit norm and exact-boundary gap.
These thresholds are preregistered and implemented for the candidate only
(`neutral_motor_smoke.md`).
Keep head offsets zero for the initial motor-only baseline. At rest, use a
static neutral joint/contact target, no swing-height/air-time objectives, and
a fixed phase `[1,0]`; at movement onset reset phase predictably and resume the
reference clock. Training and inference must share this mode/phase rule.
Stopping/transitions must be evaluated, since a frozen phase can itself create
an observation discontinuity. The deterministic SafetySupervisor stays separate.

Rest priority is zero local planar/yaw velocity and stationary pose, verified
also by bounded accumulated COM/foot-region displacement over a fixed long
interval. Do not implement an unobserved absolute-world-position target in a
policy that has no position observation. If a later controller needs position
holding, expose the relevant state explicitly and treat it as a separate design.

## References before policy training

Declare quaternion convention, velocity units and frames in a schema. Correct
angular differentiation with rotation-vector/dt, transforming to the declared
frame; verify pure yaw and mixed rotations at multiple headings and time steps.
Align world/local reference velocity with the reward consumer and randomized
robot heading. Validate finite-difference velocities against sampled poses,
and joint velocities against joint trajectories; preserve knee checks.

Include exact zero lateral and yaw entries and neutral in-place turn targets.
Do not insert or relabel a nearest nonzero gait as zero. Generate and validate
a small fixed subset before considering a full grid. Report actual achieved
reference velocities; command-key labels alone are not validation. Replacing
references changes future training targets; it cannot fix a frozen ONNX policy.
Keep raw generated motions and old pickles recoverable and ignored.

## Motor-only reward and evaluation

The candidate implements symmetric planar tracking with existing sigma/scale,
angular tracking and unchanged effort/action-change costs. Height and air-time
objectives are absent; pose/velocity cost uses the static target only at rest.
Body-frame imitation is enabled only in movement. The CPU smoke records actual
state reward terms: weighted imitation -9.711 to +2.348, rest cost -3.533 to 0,
alive +20; total reward 0.2143–0.5582 per tick, no zero clipping. These zero-action
states do not establish that learning avoids a standing solution; future
training must measure it. Weights were not adjusted after these observations.

Treat this as a package comparison, not a single-variable causal experiment.
Before learning, preregister the exact configuration and an equal-step neutral
control, seed budget, reference validation tolerances, restoration method and
stopping conditions. Preserve the original B2 gate unchanged as a comparator.
A future candidate must pass its own fixed forward/backward/lateral/turn,
transition, push and head checks, plus prospectively declared long rest
displacement criteria. Passing one-minute rest descriptions alone is insufficient.

## Next phase and spending boundary

Reference fitting/contact/knee repair and geometric turns are complete
(`neutral_reference_repair.md`). The opt-in environment passes its fixed CPU
smoke (`neutral_motor_smoke.md`); the mixed subset has exactly seven discrete
commands, no full-grid or continuous coverage. Hashes protect raw recordings
and fitted targets; unsupported references fail closed. Shared rest/phase is
wired to the candidate and a matching-policy-hash inference opt-in; historical
policies retain their clock. Two local 16-transition PPO/warm-start stages now
complete (`neutral_ppo_restore_smoke.md`), with identical restored actions and
fresh optimizer/RNG/counters. The saved-checkpoint audit finds normalization
floor susceptibility; next preregister a normalization-control smoke and strict
live-byte checks before an equal-step motor-only comparison/readiness gate.
These tiny artifacts are not validated controllers. Do not rerun R0 or S1–S6. Any cloud pilot still
needs an explicit small deterministic cap, self-deletion, result fetching,
NAT cleanup and final resource audit. No paid run is authorised here.
