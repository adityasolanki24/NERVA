# Base-origin reward velocity continuation (preregistration, 2026-10-10)

Baseline: the gait-averaged tracking continuation (`gait_averaged_tracking_pilot.md`, results
`results_gait_averaged_tracking/`). It supported its hypothesis (forward, backward, left and right pass 3/3)
but failed overall because both pure turns translate 0.053–0.068 m/s at the base origin (limit 0.03).
Neutral motor skill only; no expressive objectives, affect, safety, default or deployment changes.

## What the diagnostic found (`results_turn_pivot/`, before this preregistration)

- Upstream's tracking and imitation rewards read body velocity from the IMU site, 8 cm behind the base
  origin. The evaluation and the verified references use the base origin.
- Every evaluated policy (including historical B2) turns about a point near the IMU. Turn translation is
  ≈ 0.02 m/s at the IMU but 0.034–0.067 m/s at the base.
- The reference's own turn scores 0.998 on the gait-averaged tracking term measured at the base, but 0.782
  measured at the IMU: the rewards penalize turning like the reference.
- For the gait-averaged candidate, turn translation is the only failing turn criterion.

## Hypothesis

**H:** with the tracking and imitation rewards measured at the base origin, and nothing else changed, a
continuation reduces pure-turn translation at the base origin to ≤ 0.03 m/s, without losing the four
translations, rest, or adding falls.

## Configuration (identical to the gait-averaged run except the two lines marked ★)

- ★ **Reward:** `BaseOriginGaitAveragedNeutralJoystick`.
  - Gait-averaged tracking, as before.
  - The linear velocity used by tracking and imitation is v_base = v_IMU − ω × r_IMU, with r_IMU =
    (−0.08, 0, 0.05) m read from the model. Tests check it against native MuJoCo's free-joint velocity
    to within 1e-5.
  - Observations, the critic's privileged inputs, costs, alive, noise, pushes and scales are unchanged.
- ★ **Starting checkpoint:** the gait-averaged run's final accepted checkpoint
  `experiments/cloud_runs/gait_averaged_tracking-20261010-170130/checkpoints/000061931520`, hash-verified
  against `results_gait_averaged_tracking/training_summary.json`. Parameters only; fresh
  Adam/RNG/environment (seed 41).
- **Unchanged:**
  - frozen B2-cropped statistics;
  - 8,064 balanced persistent-command environments, 1,000-step episodes;
  - unroll 20, 32 minibatches, 4 epochs;
  - LR 1e-4, clip 0.2, entropy 0.005, discount 0.97, GAE 0.95, value 0.5, gradient norm 1.0;
  - deadline at uptime 2,220 s; step ceiling by the same throughput rule (≤ 80 M);
  - checkpoints every 20 iterations with snapshots;
  - the same stops (KL > 0.1, replay error, coverage, nonfinite, statistics change).
- **Budget:** one L4 VM, 45 min hard cap, ≤ US$2. **Requires explicit authorization.**

## Evaluation (unchanged protocol; `motor_compare.py --protocol base_origin_velocity`)

- 84 paired native trials, seeds 0/1/2, seven commands, final accepted checkpoint only.
- Arms: historical B2, untrained neutral, gait-averaged candidate (this run's start), and the base-origin
  candidate.
- Videos: B2 | gait-averaged candidate | base-origin candidate.

## Criteria (fixed now)

1. **Hypothesis:** both pure turns have horizontal translation RMS ≤ 0.03 m/s at the base origin (the
   evaluation's `cross` criterion) in 3/3 seeds each.
2. **No regression:** rest, forward, backward, left and right each still pass 3/3, and there are no added
   falls compared with the untrained control and the start.
3. **Overall pilot:** all seven commands pass, and normalized RMSE ≤ 90% of the untrained control.

- **H is supported** if 1 and 2 hold. **The pilot passes** if 1–3 hold.
- Even then: one training seed, no robust-locomotion claim. A pass would justify preregistering the long
  robustness gate (more seeds, pushes, longer trials), not expressive training.
