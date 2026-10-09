# Neutral candidate environment smoke

Preregistered before candidate implementation/dynamic evaluation, 2026-10-09.
Reference repair is 6/7; the separately corrected turns pass 2/2 (18.31 s).
Admit a hash-verified mixed subset of five repair targets plus two pivot targets.
Preserve both prior failures. Subset scope is exactly seven discrete commands;
no claim of continuous command coverage or learned motor readiness.

Candidate `neutral_motor_v1`: upstream motor physics/control/random reset and
101 policy / 212 privileged observation slots, no style append, head commands
zero. Uniform sampling over the seven validated commands. Current-body linear
and angular imitation targets, analytic instantaneous joint derivatives,
planned contact threshold 0.5. Shared rest/canonical-command/phase clock from
`nerva.motor_contract`: rest index zero, phase [1,0], onset index zero, then
27-step clock. Reset features are [1,0]. A command resample is visible on the
next observation, preserving the existing step timing convention.

Reward scales fixed to upstream defaults: planar tracking 2.5, yaw tracking
6, torques -0.001, action rate -0.5, rest pose/velocity -0.2, alive 20,
imitation 1; sigma 0.01. Symmetric planar tracking, no deadband. Imitation
only outside rest, same existing active leg/velocity/contact terms and weights,
without the inactive incorrectly indexed quaternion expression. Rest pose
uses the validated static reference mapped to 14 actuators; no air-time or
height objectives. Affect and deterministic safety remain separate.

Before dynamics: verify all reference/report hashes, exact command coverage,
joint order/frames, NumPy/JAX target parity over 27 phases per command (position
<=1e-5, velocity <=1e-4), shared clock parity <=1e-6, unsupported lookup fails
closed. Candidate inference requires explicit contract metadata; historical
policies retain their existing clock. Do not transplant B2 semantics silently.

CPU smoke: one fixed seed 7, flat terrain, JIT reset/step. Seven separate resets
with one command each and eight zero-action steps; additional stand-forward-stop
sequence, eight steps per block. Exactly 80 steps, no optimiser, export or PPO.
Parent subprocess watchdog 600 s (includes compilation), no retries or cloud.
Require finite qpos/qvel, observations, raw reward components and scaled reward;
expected observation sizes, supported zero-head commands, exact reference
selection, clock mode/index agreement and phase error <=1e-6 at every tick.
Record every reward component, clipped-zero fraction and any termination;
termination is reported, not interpreted as learned-policy failure under zero
actions. Stop on nonfinite data or cap. Passing means environment plumbing only.

Run complete tests including legacy transition equality plus Ruff before final
adoption of this opt-in environment. A future PPO/restore smoke and neutral
policy gate require separate protocols and any paid work an explicit small cap.
