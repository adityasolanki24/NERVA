# Neutral motor baseline candidate (design only)

Status update 2026-10-09: seven-point corrected subset evaluated, 0/7 pass;
derivative alignment improves fit errors but is insufficient. Shared rest/phase
contract is implemented as isolated infrastructure and passes NumPy/JAX
conformance. No candidate environment, trained policy or reference adoption.

Grounding: [neutral motor audit](neutral_motor_audit.md) and longer-turn
measurements. This is a proposed package, not an implemented fix or training
authorisation. B2, S1–S6 and historical defaults remain reproducible. The audit
establishes target issues; it does not establish causes of the learned drift.

## Rest and movement contract

Infer a rest mode from planar command speed ≤0.005 m/s AND absolute yaw command
≤0.02 rad/s, avoiding the current mixed-unit norm and exact-boundary gap.
These are proposed thresholds requiring separate preregistration before use.
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

Propose symmetric two-axis planar tracking with the existing sigma and scale,
removing the 0.10 m/s lateral tolerance. Retain angular tracking and conservative
effort/action-change costs. Start with height and air-time scales both zero;
stationary pose cost only in rest. Validate reference imitation in movement
with correct units/frames. Check term magnitudes and reward clipping on actual
valid states before freezing a training protocol; synthetic audit values do
not justify reward-weight tuning.

Treat this as a package comparison, not a single-variable causal experiment.
Before learning, preregister the exact configuration and an equal-step neutral
control, seed budget, reference validation tolerances, restoration method and
stopping conditions. Preserve the original B2 gate unchanged as a comparator.
A future candidate must pass its own fixed forward/backward/lateral/turn,
transition, push and head checks, plus prospectively declared long rest
displacement criteria. Passing one-minute rest descriptions alone is insufficient.

## Next phase and spending boundary

The velocity/frame validator and seven-motion subset are complete; the subset
fails. Shared rest/phase infrastructure (`nerva.motor_contract`) passes
conformance but is not wired into historical policies. Next safe local work is
a preregistered derivative-consistent fitting and contact-label investigation,
including positive-knee initialization. Do not rerun R0 or S1–S6. Candidate
environment/PPO smoke waits for valid references and its own limits. Verify checkpoint
restore compatibility before planning continuation. Any cloud pilot still
needs an explicit small deterministic cap, self-deletion, result fetching,
NAT cleanup and final resource audit. No paid run is authorised here.
