# Gait-averaged tracking continuation (preregistration, 2026-10-10)

Baseline: the GPU neutral continuation (`gpu_neutral_pilot.md`, results `results_gpu_pilot/`), which
failed: forward undershoots, right regressed to 0/3, and both pure turns translate. Neutral motor skill
only; no expressive objectives, affect, safety, default or deployment changes.

## What the diagnostics found (2026-10-10, before this preregistration)

1. **The tracking reward prefers standing still.**
   - Upstream's form is 2.5·exp(−|c − v_instantaneous|²/0.01).
   - For the four 0.074 m/s translation commands, a robot that follows its own verified reference gait
     perfectly scores **0.77–0.80**. Standing still scores **1.45**; walking at half speed 0.58–0.74.
   - The gait's within-stride sway (≈ 0.15 m/s RMS) dominates the instantaneous comparison.
   - On velocity averaged over one gait period (0.54 s, 27 steps), the perfect gait scores **2.49**.
2. **The undershoot is in the policy, not a sim-to-sim gap** (`results_sim_gap/`).
   - The GPU candidate, deterministic in its own MJX training environment without pushes, moves at:
     forward +0.039, backward −0.047, left +0.047, right −0.021 m/s (28–64% of the request).
   - Native MuJoCo gives similar or larger speeds.
3. **The turn failures are largely a MJX → native MuJoCo gap.**
   - In MJX the turns run at 0.565 and −0.607 rad/s, with lateral drift −0.018 and +0.022 m/s.
   - Natively they run at 0.711 and −0.669 rad/s, with drift +0.043 and −0.025 m/s.
   - **This run does not address that gap.**

## Hypothesis

**H:** with tracking computed on the gait-period-averaged body velocity, and nothing else changed, a
continuation raises forward, backward, left and right speed to at least half the request in native
evaluation, without added falls and without losing rest.

## Configuration (identical to the GPU pilot except the two lines marked ★)

- ★ **Reward:** `GaitAveragedTrackingNeutralJoystick`.
  - `tracking_lin_vel` and `tracking_ang_vel` use the mean of the last ≤ 27 body-frame velocities (the
    window resets with the episode).
  - Imitation, costs, alive, noise, pushes and scales are unchanged.
- ★ **Starting checkpoint:** the GPU pilot's final accepted checkpoint
  `experiments/cloud_runs/neutral_gpu_pilot-20261010-152812/checkpoints/000060318720`, hash-verified
  against `results_gpu_pilot/training_summary.json`. Parameters only; fresh Adam/RNG/environment
  (seed 41).
- **Unchanged from the GPU pilot:**
  - frozen B2-cropped statistics;
  - 8,064 balanced persistent-command environments, 1,000-step episodes;
  - unroll 20, 32 minibatches, 4 epochs;
  - LR 1e-4, clip 0.2, entropy 0.005, discount 0.97, GAE 0.95, value 0.5, gradient norm 1.0;
  - deadline at uptime 2,220 s; step ceiling by the same throughput rule (≤ 80 M);
  - checkpoints every 20 iterations with snapshots;
  - the same stops (KL > 0.1, replay error, coverage, nonfinite, statistics change).
- **Budget:** one L4 VM, 45 min hard cap, ≤ US$2. **Requires explicit authorization.**

## Evaluation (unchanged protocol; `motor_compare.py`)

- 84 paired native trials, seeds 0/1/2, seven commands.
- Arms: historical B2, untrained neutral, GPU pilot candidate (this run's start), and the gait-averaged
  candidate (final accepted checkpoint, no selection).
- **The GPU pilot candidate replaces the local pilot as the "start" control.**

## Criteria (fixed now)

1. **Hypothesis:** forward, backward, left and right signed axis mean ≥ 50% of the request in 3/3 seeds
   each.
2. **No added falls**, and rest still passes 3/3.
3. **Overall pilot:** all seven commands pass, and normalized RMSE ≤ 90% of the untrained control.
   Turns are expected to fail because of the gap in finding 3. That is reported, not tuned.

- **H is supported** if 1 and 2 hold. **The pilot passes** if 1–3 hold.
- Even then: one seed, no robust-locomotion claim. The long gate and a turn-gap diagnostic come next.
