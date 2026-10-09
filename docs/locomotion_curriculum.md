# Robust locomotion before expressive objectives (design, 2026-10-09)

Status: design only; no new training or cloud run. S1–S6 remain closed results.

## Why the order changes

S5/S6 found a standing solution: a touchdown-gated height cost can be avoided
by not stepping. More reward terms from training onset did not resolve this.
B2 demonstrates usable backward locomotion, but its head-offset envelope and
command tracking are limited. A high episode reward is therefore insufficient
evidence of a robust motor skill. Affect success does not validate locomotion.

## Next local phase: establish the neutral gate

Evaluate the existing B2 checkpoint first, in its backlash training scene, with
neutral style and the measured capability envelope. Reuse the gait metrics,
training-level observation noise, paired seeds and recorded reference validation.
Write a separate preregistration before these new rollouts. It should cover
forward/backward/lateral/turn commands, starts and stops, transitions, and modest
controlled pushes. Report achieved velocity, tracking error, foot clearance,
falls, recovery time, and commanded-motion time spent standing. Split these
metrics by command direction; an average must not hide backward failure.
Treat the existing safety stop as a separate override and report its activation.

Proposed gate, requiring review and exact protocol before execution: no falls
in unperturbed commanded walking; correct direction of travel; achieved speed
at least half of nonzero commanded speed in each translational direction;
quantified push recovery and a declared head envelope. Freeze push magnitudes,
timing, stationary-speed threshold, seeds and tracking limits before observing
results. Existing B0/B2 reports are historical comparators, not new training jobs.

## Training stages after that gate

1. **Neutral motor skill.** No expressive height objective or variable style.
   Establish tracking, stepping and recovery with a verified neutral reference.
   B2 may be a starting checkpoint if the local gate supports it; otherwise
   preregister a motor-only candidate. Keep head commands fixed initially.
2. **Checkpoint continuation.** Prove that resuming the same neutral env retains
   its observations, normalisation and behaviour before adding objectives.
   The current train_style runner does not expose restore arguments; inspect
   the pinned Brax restore API and checkpoint schema before implementing this.
   ONNX alone is an inference artifact, not sufficient PPO training state.
3. **Expressive conditioning.** Add continuous joint sampling only over a
   reference space with measured feasibility. The current StyledReference picks
   discrete style indices and nearest command references; accepting a vector
   at inference is not continuous training. Preregister interpolation or a new
   validated reference set, phase handling, transition rate, and held-out
   combined styles before implementation. Start at neutral and widen only
   after locomotion gates pass.
4. **Expressive objectives.** Introduce them gradually from the established
   walking checkpoint. Compare an equal-step continuation control with the
   curriculum candidate. Explicitly measure the standing solution and stop
   when tracking, stepping or stability regresses; do not tune the gate after
   observing the run. Avoid starting another S5/S6 reward variant from scratch.

Keep intended-feature monotonicity, normalised cross-talk and command tracking
as distinct acceptance criteria. Step height remains an open research outcome;
do not assume that a schedule solves it. Human emotional interpretation remains
unmeasured. PAD does not enter motor targets or deterministic safety.

## Paid-work boundary

No training is authorised by this design. Before any paid pilot, provide an
explicit small hard duration/cost cap and expected throughput from a relevant
measurement. The early 95-hour projection was superseded by steady-state
measurements, but neither projection authorises a blind 300M-step launch.
Use the existing capped self-deleting launcher, attached result fetching,
network cleanup and final audit. Preserve recoverable checkpoint artifacts.
