# Robust locomotion before expressive objectives (design, 2026-10-09)

Status: local B2 motor gate evaluated; overall fail (2026-10-09). Local neutral
plumbing includes 32 PPO transitions and saved-checkpoint diagnosis, without a
learned readiness result or cloud run. S1–S6 remain closed results.

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

## Local gate result (2026-10-09)

Fixed protocol: `b2_robustness_gate.md`, preregistration `14a139c`, evaluator
`e8e3f34`. Reports: `experiments/locomotion_curriculum/results_b2/`.
B2 **fails** the all-required readiness decision because both in-place turn
commands exceed the 0.03 m/s horizontal translation RMS limit in all five
seeds each (left 0.0330–0.0379; right 0.0416–0.0439 m/s). Yaw tracking itself
passes. All other criteria pass: no falls in 115 trials, starts/stops and
reversals, no standing solution, all 40 modest pushes recovered (max 1.78 s),
and the standing/walking head checks. No shadow safety interventions occurred.

The short-record turn diagnostic and subsequent
[longer-turn measurement](b2_long_turn_experiment.md) are complete. Both
aggregate conclusions are inconclusive. In the longer measurement, all five
zero-command controls migrate: COM block-centre displacement 0.556–0.588 m,
with corresponding foot-region displacement. Local turn-region support is
3/5 left and 2/5 right, below the fixed 4/5 requirement. No falls or shadow
interventions occurred. The original motor gate remains failed.

The [neutral motor-target audit](neutral_motor_audit.md) is complete: zero
imitation is disabled; lateral tracking has a 0.10 m/s tolerance; the inspected
generator fails known-yaw velocity probes. Causes of learned behaviour remain
unproven. A [motor-only candidate design](neutral_motor_candidate.md) requires
verified reference derivatives/frames, symmetric tracking and explicit rest
semantics. The corrected seven-reference subset then failed 0/7; timing
alignment reduces velocity error but does not meet the fixed absolute limit.
Shared rest/phase conformance passes. The separately preregistered
[reference repair](neutral_reference_repair.md) fixes derivatives, static
contacts and knee limits (6/7); analytic base-pivot correction passes both turns.
Five preserved passing targets plus two corrected turns form a hash-verified
seven-command subset. The opt-in `NeutralJoystick` and matching-policy-hash
inference contract pass the [80-step CPU smoke](neutral_motor_smoke.md), no
terminations or zero-clipped rewards. A subsequent
[PPO/warm-start smoke](neutral_ppo_restore_smoke.md) completes two 16-transition
stages in 193.47 s: finite updates to both networks, normalization count 16->32,
exact action restoration and warm initialization. Adam/RNG/counters restart;
this is not full training-state continuation. The live comparison checked
numeric equality; subsequent literal-byte audit covers saved-tree reserialization,
so the original live bitwise criterion was not fully evidenced. The separately
[preregistered strict retry](neutral_ppo_byte_retry.md) closes that gap: all
criteria pass, 32 transitions in 203.19 s, every live-before-save/restored leaf
and warm initialization agree in literal bytes, 20 exact restored-action probes.
Fresh value loss
8.842e8/KL 1.552e11 are preserved, despite warm loss 0.0612/KL 0.1214.
[Normalization audit](neutral_normalization_audit.md) finds six policy and
seventeen privileged slots at the 1e-6 std floor; fixed synthetic probes amplify
to 449,070 and saturate actions. This does not prove rollout loss causation.
[Variance-floor control](neutral_normalization_control.md) is complete: 32
transitions in 225.75 s, epsilon 1e-4, frozen epsilon-zero comparator. Checkpoint
and amplification criteria pass, peak synthetic magnitude <=81.904, but overall
fails fresh KL 1489.402 >1; fresh value loss 10.4402 and warm loss/KL
0.05803/0.13245 pass. No criteria or defaults change. Pinned constant-LR PPO
collects using old statistics then updates them before SGD; its contribution
to KL is now demonstrated on one first batch. The original capture stopped
at old-control integrity; a first offline retry stopped at schema admission.
Both failures remain preserved. The separately preregistered
[offline replay](normalization_timing_offline_retry.md) passes every integrity
criterion: old KL 0.000140, updated KL 10.296526, deterministic action max shift
0.957049, no gradient or optimizer update. Eight retained transitions, zero new
transitions; 6.41 s under the 180 s cap. This isolates a preprocessing effect,
not the entire previous two-update KL or a learning fix. The subsequent
[offline schedule comparison](normalization_schedule_comparison.md) passes:
fixed preprocessing KL 0.023703, deferred deployment KL 10.288106; affine
first-layer inference rebase preserves policy/value outputs, but does not
transform Adam. The [identity on-policy/restore screen](identity_preprocessing_smoke.md)
passes 32 transitions (deployed fresh/warm KL 0.058854/0.243470), exact restoration
and zero statistics influence. The [carried-batch check](identity_multibatch_stability.md)
passes 128 new transitions/16 updates, max post-SGD KL 0.008009 and value loss
0.126727. It samples only lateral -0.074 m/s and yaw +0.6 rad/s. The subsequent
[balanced learned-controller pilot](neutral_learning_pilot.md) completes 28,672
local transitions/128 accepted updates from B2 walking weights, with frozen B2
statistics and all seven commands. All 63 paired 20 s native motor trials finish
without falls. The candidate passes rest and rightward motion across all three
seeds, but five commands fail and normalized tracking error improves only 2.82%
versus the converted untrained control (criterion: >=10%). Rest displacement
falls from historical B2's 0.257–0.261 m to 0.016–0.022 m; the untrained neutral
clock already passes rest, so this is not credited to PPO. Seven comparison
clips and a combined video are retained locally. Next preregister a longer,
explicitly capped neutral continuation for low-speed tracking and pure-turn
translation, then the long readiness gate; no expressive objectives yet.
There is no validated learned candidate. Subset command coverage is
discrete and limited; smoke success does not overturn B2's failed policy gate.
**GPU continuation (2026-10-10, `gpu_neutral_pilot.md`):**
- 60,318,720 transitions from the pilot's final checkpoint on one L4 (32.7 min VM, ≈ US$0.70).
- Backward and left pass, normalized RMSE ratio 0.730, no falls.
- Forward undershoots, right regresses from 3/3 to 0/3, and both turns still translate 0.039–0.044 m/s.
- The pilot fails; no checkpoint selection.
- Next: diagnose the asymmetry and turn translation before training further.

**Gait-averaged tracking continuation (2026-10-10, `gait_averaged_tracking_pilot.md`):**
- Only the tracking term changed (mean velocity over one 0.54 s gait period), from the GPU candidate;
  61,931,520 transitions on one L4 (≈ 33 min VM, ≈ US$0.70).
- Forward, backward, left and right pass 3/3 (79–122% of the request), rest 3/3, no falls; RMSE ratio
  0.352 vs untrained.
- Both turns fail: yaw rate ≈ 0.6 rad/s but translation 0.053–0.068 m/s. H supported; pilot fails.
- Next: diagnose turn drift and the MJX → native turn gap, then the long readiness gate.

**Turn pivot diagnostic (2026-10-10, `results_turn_pivot/`):** the rewards measure velocity at the IMU,
8 cm behind the base origin used by the references and the evaluation, and every policy turns about the
IMU. The base-origin reward continuation (`base_origin_velocity_pilot.md`) moved the pivot to the base:
6/7 commands pass (turn right included), no falls, RMSE ratio 0.317. It fails narrowly on turn left
(translation 0.033–0.035 m/s, a sideways pivot offset). Next: diagnose that asymmetry.

**Turn asymmetry diagnostic (2026-10-10, `results_turn_asymmetry/`):** in MJX the left turn is within the
limit (0.022–0.026 m/s); natively it is not. About half of that MJX → native gap comes from the training-only
0–2 step action delays; the reference turns are symmetric; contacts match between engines.

**Neutral long motor gate (2026-10-10, `neutral_motor_gate.md`, `results_neutral_gate/`):** every condition
runs without and with training-like latency; all required in both. The base-origin candidate fails only on
steady turn left (0/5 without latency, 2/5 with); no falls in 160 trials, all transitions pass, 80/80 pushes
recovered. Gate not passed; no expressive training.

**Turn-translation pilot (2026-10-11, `turn_translation_pilot.md`, `results_turn_translation/`):** one change,
a 4× tighter linear tracking width for the two pure turns, from the base-origin checkpoint (59 M transitions,
one L4, ≈ US$0.70). **The neutral long motor gate passes in both latency conditions:** every command 5/5
(turn left 0.021–0.025 m/s), no falls in 160 trials, all transitions, 80/80 pushes, no shadow interventions.
One training seed, simulation only; margins modest. This is the first validated neutral candidate; next is
preregistering expressive-objective work on top of it. No deployment or default change.

**E1 preregistered (2026-10-11, `expressive_posture_experiment.md`):** stage 3, continuous conditioning on
e = (torso pitch, body height) by imitating bilinearly interpolated verified styled references, fixed 0.54 s
period, no new reward objective; Phase A reference/interpolation gate (local), one capped GPU run
(needs authorization), then the unchanged neutral gate at e = 0 plus monotonicity, range, cross-talk and
style-space task criteria with a held-out combination. Tempo (E2) and stage-4 objectives come later.
**Outcome:** stopped at Phase A (`results_e1_references*/`). Pitch is feasible over −10°…+2°; body height
fails above neutral (COM 0.221 m: joint-velocity fit 0.53–0.61 > 0.5 rad/s) even after the single
preregistered halving. No training. A follow-up needs its own preregistration.

Do not relax the gate or start expressive training. B2 retains its
previously verified reactive role; the affect default remains Bv4.

## Training stages after that gate

1. **Neutral motor skill.** No expressive height objective or variable style.
   Establish tracking, stepping and recovery with a verified neutral reference.
   B2 may be a starting checkpoint if the local gate supports it; otherwise
   preregister a motor-only candidate. Keep head commands fixed initially.
2. **Checkpoint continuation.** Parameter warm start is implemented and checked
   in the candidate smoke: policy/value/normalizer restore with matching shapes,
   references and source hashes. Pinned Brax reinitializes Adam, RNG and counters;
   implement a separate full-state API if exact continuation is required.
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

The [short-record turn-centre diagnostic](b2_turn_diagnostic.md) and subsequent
longer-turn measurement do not overturn the failed gate. The latter reveals
zero-command COM and foot-region migration, which a future motor-only baseline
must address alongside pure-turn translation.

No paid training is authorised by this design. Before any paid pilot, provide an
explicit small hard duration/cost cap and expected throughput from a relevant
measurement. The early 95-hour projection was superseded by steady-state
measurements, but neither projection authorises a blind 300M-step launch.
Use the existing capped self-deleting launcher, attached result fetching,
network cleanup and final audit. Preserve recoverable checkpoint artifacts.
