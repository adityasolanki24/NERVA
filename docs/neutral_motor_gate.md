# Neutral long motor gate, with and without latency (preregistration, 2026-10-10)

Status: fixed before implementation and evaluation. No training, cloud or paid work. This is the
"long readiness gate" named in `locomotion_curriculum.md`. It adapts the B2 gate
(`b2_robustness_gate.md`) to the neutral motor contract and adds a latency condition. It does **not**
change or rescue any earlier result: every pilot keeps its recorded outcome.

## Why latency

Training delays every action by a random 0, 1 or 2 control steps. The native evaluation used so far has no
delay, and a real robot has some. `results_turn_asymmetry/` showed that removing the delays moves MJX
turning roughly half-way to native behaviour. A ready policy must therefore work **both** without latency
and with training-like latency: the gate is stricter than either condition alone.

Latency model (`OpenDuckSim(action_delay=True)`): the applied action is the policy output from 0, 1 or 2
control steps ago, uniformly resampled each control step from its own seeded stream (seed, 1); the history
starts at zero; the observation's last-action history stays undelayed, exactly as in training. Training's
IMU "delay" acts only on an unobserved gravity vector and has no counterpart.

## Policy

The base-origin candidate: `experiments/cloud_runs/base_origin_velocity-20261010-201858/candidate.onnx`,
SHA256 `9394fa5db7fc613f7ee311ec329e39a693f790ba219e26913db7a7464a11c5a2` (final accepted checkpoint, no
selection). It is evaluated knowing it failed the 3-seed pilot on the left turn without latency (0.033–0.035
m/s), so **an overall gate failure is expected**. The purpose is a complete robustness profile (more seeds,
transitions, pushes, latency) and to put the latency condition in place before any future candidate.

## Common setup

Native MuJoCo, backlash scene, raw accelerometer, training observation noise, initial joint noise ±0.02 rad,
neutral motor contract (zero head offsets), control dt 0.02 s, seeds 0–4. Every trial is run twice: latency
off and latency on, with the same seed (identical observation-noise stream). No safety override; the
unchanged `SafetySupervisor` runs in shadow and its interventions are reported, never substituted.

## Conditions (80 trials per latency condition, 160 in all)

- **Steady:** the seven neutral commands (rest; forward/backward ±0.074 m/s; left/right ±0.074 m/s;
  turns ±0.60 rad/s) × 5 seeds, 20 s, scored 5–20 s.
- **Transitions:** 5 seeds, eight consecutive 5 s phases (40 s): rest, forward, backward, left, right, turn
  left, turn right, rest. Each phase is scored over its last 3 s.
- **Pushes:** forward and backward × 4 heading-relative directions (0/90/180/270°) × 5 seeds, 15 s. At
  t = 8 s add 0.30 m/s of horizontal base velocity in the direction rotated by the robot's current yaw.

## Criteria (all required, in **each** latency condition)

1. **Stability:** every trial completes; no tilt > 45° in any steady or transition trial.
2. **Steady:** every command passes in **5/5** seeds under the pilot evaluation's per-trial rules
   (`nerva.analysis.motor_eval.motor_metrics`): signed axis mean ≥ 50% of the request; smoothed axis RMSE
   ≤ 60%; stationary fraction ≤ 10%; translation cross RMS ≤ 0.05 m/s and yaw RMS ≤ 0.20 rad/s; turns
   horizontal RMS ≤ 0.03 m/s; rest horizontal ≤ 0.02 m/s, yaw ≤ 0.15 rad/s, displacement ≤ 0.10 m.
3. **Transitions:** in every moving phase and seed, signed axis mean ≥ 50% and smoothed axis RMSE ≤ 60%
   of the request, and stationary fraction ≤ 10%; every rest phase has smoothed horizontal RMS ≤ 0.02 m/s
   and yaw RMS ≤ 0.15 rad/s. (As in the B2 gate, cross/turn-translation limits apply to steady trials only.)
4. **Pushes:** no falls in all 40 trials; recovery within 5 s in ≥ 4/5 seeds for each command/direction,
   with the B2 gate's recovery definition (first full 1 s window after the push with every tilt < 20°,
   commanded-axis window mean within max(0.03 m/s, 25% of the pre-push [6, 8) s mean) of that mean, and
   |window yaw rate| ≤ 0.20 rad/s).

**The gate passes only if 1–4 hold with latency off and with latency on.** Reported per condition and per
criterion, including every failing command and seed; shadow-safety counts are diagnostics. A pass would
still be one training seed and simulation only: it would justify preregistering expressive-objective work,
not hardware deployment.
