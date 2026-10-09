# B2 neutral motor gate: preregistration (2026-10-09)

Status: fixed at `14a139c` before implementation/evaluation; completed
2026-10-09, overall fail on pure-turn translation. Results:
`experiments/locomotion_curriculum/results_b2/`. No training or paid work.
This is a new motor-readiness gate, not a repeat of S1–S6 style experiments.

## Hypothesis and setup

The existing B2 final 300,482,560-step checkpoint is sufficiently robust to be
a neutral starting point for a later curriculum. Test it directly, without
affect, perception, memory or expressive objectives, in its backlash training
scene. Raw accelerometer, training observation noise, initial joint noise
uniform +/-0.02 rad; neutral style vector (0,0,0); control dt 0.02 s.
Pair seeds 0–4 across conditions. Record checkpoint SHA256 and committed code.

Primary motor rollouts apply requested commands without a safety override.
Run the unchanged deterministic SafetySupervisor in shadow at 50 Hz and report
its interventions/stop time separately; it cannot hide a motor failure.
Replay any primary trial with a shadow intervention with safety applied, using
the same seed/schedule, as a diagnostic only. Safety-on results never replace
primary results. No physical robot is involved.

## Fixed conditions (115 primary trials)

- Steady: six commands x five seeds, 20 s; metrics over [5,20) s:
  forward/backward vx = +/-0.15 m/s; left/right vy = +/-0.10 m/s;
  left/right yaw = +/-0.60 rad/s; other command components zero.
- Starts/stops/transitions: five seeds, eight consecutive 5 s phases, 40 s:
  stop, forward, backward, left, right, turn left, turn right, stop.
  Score each phase's final 3 s, excluding its first 2 s.
- Pushes: forward/backward x four heading-relative directions (0/90/180/270
  degrees) x five seeds, 15 s. At t = 8 s add 0.30 m/s horizontal base velocity
  in the direction rotated by the robot's current yaw. This is a training-style
  velocity kick, not a force impulse. Neutral head. Measure pre-push achieved
  commanded-axis velocity over [6,8) s and recovery afterward.
- Standing head envelope: four pitch/yaw corners (-0.2 or +0.6 rad, -0.4 or
  +0.4 rad), five seeds each, 15 s, zero velocity. Neck pitch/head roll zero.
- Walking head envelope: forward vx = +0.15, head pitch +/-0.15 with yaw zero,
  or head yaw +/-0.08 with pitch zero; five seeds each, 20 s. These are the
  registered B2 walking limits. Neck pitch/head roll zero.
  Ramp every nonzero head offset linearly from zero over 0.5 s; hold thereafter.

## Measurements and all-required acceptance criteria

Use existing heading-frame velocity, torso tilt and robust foot lift (95th–5th
percentile) metrics. Derive yaw rate from unwrapped yaw differences / dt.
For tracking/stationary metrics, average velocities in trailing 1 s windows;
score only windows fully inside the stated measurement interval. Report
unsmoothed mean achieved velocities and both foot lifts as well.

1. **Unperturbed stability:** no tilt >45 degrees in any steady, transition,
   standing-head or walking-head trial (whole trajectory). Nonfinite state or
   incomplete execution fails that condition; do not exclude failed seeds.
2. **Steady tracking**, every command and every seed: signed achieved
   commanded-axis mean >=50% of command magnitude; smoothed commanded-axis
   RMSE <=60% of command magnitude. Translational trials: smoothed orthogonal
   velocity RMS <=0.05 m/s and yaw RMS <=0.20 rad/s. Turn trials: smoothed
   horizontal-speed RMS <=0.03 m/s. These are engineering gate thresholds.
3. **No standing solution:** fraction of scored windows with |commanded-axis
   velocity| <0.01 m/s (translation) or <0.10 rad/s (turn) <=10%, in every
   steady and moving transition phase. The commanded axis, not total speed,
   prevents sideways drift from counting as forward progress.
4. **Transitions/stops:** every moving phase and seed meets the same >=50%
   signed mean and <=60% smoothed tracking RMSE as steady; every stop phase
   has smoothed horizontal-speed RMS <=0.02 m/s and yaw RMS <=0.15 rad/s.
5. **Pushes:** no falls in all 40 trials; recovery within 5 s in >=4/5 seeds
   for EACH command/direction. Recovery is the first full 1 s post-push window
   with every tilt sample <20 degrees, commanded-axis window mean within
   max(0.03 m/s, 25% of |pre-push achieved mean|) of its pre-push mean, and
   |window mean yaw rate| <=0.20 rad/s. Recovery time is window END minus 8 s;
   report null if absent; a later fall makes recovery unsuccessful.
6. **Head tolerance:** all standing corners remain upright; every walking-head
   condition meets steady tracking/stationary criteria, and signed forward
   mean is >=75% of its paired zero-head steady forward trial.

Report shadow-safety intervention counts and stop-time fractions in every
trial, plus foot lift and max tilt; these are diagnostics, not substitute
success criteria. Preserve all trial metrics, condition counts and negatives.

## Decision/stopping

All six criteria must pass to call B2 motor-ready under this protocol. A fail
blocks expressive curriculum training; it does not invalidate B2's existing
reactive-scenario results. No thresholds or pushes will be tuned in this round.
Finish all primary conditions even if early ones fail. Stop a trial on
nonfinite state and mark it failed; abort the experiment on a harness/environment
error instead of interpreting partial results. Safety diagnostics are capped at
one replay per affected primary trial. Keep raw traces local/git-ignored and
commit only small summaries. Then report the limiting motor capability and
recommend the next separately preregistered motor-only step; do not launch it.
