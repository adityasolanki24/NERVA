# Turn-translation tracking continuation (preregistration, 2026-10-11)

Authorized single capped pilot (handoff of 2026-10-11): one GPU VM, 45 min hard lifetime, ≤ US$2 total, no
replacement or second run. Neutral motor skill only; no expressive objectives, affect, safety, default or
deployment changes. If it fails, the failure is reported and no further run starts.

## Evidence and choice of intervention

The neutral long motor gate (`neutral_motor_gate.md`, `results_neutral_gate/`) passes everything for the
base-origin candidate except steady **turn left** translation: 0.033–0.035 m/s without latency (0/5) and
0.030–0.033 with latency (2/5); the limit is 0.03 m/s.

Strongest supported cause (existing measurements, no new diagnostics):
1. **The reward barely penalizes threshold-level turn drift.** Linear tracking is 2.5·exp(−|v|²/σ),
   σ = 0.01, on the gait-averaged base-origin velocity. A pure turn translating 0.03 m/s keeps 91% of the
   term (0.914); 0.02 m/s keeps 96%.
2. **So the trained turns sit near the threshold even in training:** in MJX, turn left 0.022–0.026 and turn
   right 0.025–0.030 m/s (`results_turn_asymmetry/`), both pivoting ≈ 2 cm to the robot's right.
3. **Native MuJoCo adds ≈ 0.01 m/s to the left turn** (0.023 → 0.034) and removes some from the right
   (0.025 → 0.020); about half of that gap is the training-only action delay.
4. **Not the cause:** the reference turns are mirror-symmetric (0.041 rad, same as the forward walk's
   0.040); commands are balanced (each turn 1/7 of environments, persistent); contacts match between
   engines.

**One intervention:** tighten the linear tracking width **for the two pure-turn commands only**, σ 0.01 →
0.0025 (tolerance 0.1 → 0.05 m/s). At 0.03 m/s the term then keeps 70% instead of 91%: a direct, ≈ 3×
stronger penalty on exactly the failing quantity, aimed at leaving margin for the native shift. It does not
touch the sim gap itself. No other change.

## Configuration (identical to the base-origin run except the lines marked ★)

- ★ **Environment/reward:** `TurnTranslationNeutralJoystick` = `BaseOriginGaitAveragedNeutralJoystick` with
  `tracking_lin_vel` for commands (0, 0, ±0.6) computed with σ = 0.0025 (`TURN_TRACKING_SIGMA`). All reward
  terms and scales otherwise unchanged: tracking_lin_vel 2.5 (σ 0.01 for all other commands), tracking_ang_vel
  6.0, torques −1e-3, action_rate, alive, imitation, stand_still as upstream/NERVA defaults; noise, action
  delay 0–2 steps, pushes (0.1–1.0 m/s every 5–10 s) unchanged.
- ★ **Starting checkpoint:** `experiments/cloud_runs/base_origin_velocity-20261010-201858/checkpoints/000060480000`
  (the base-origin run's final accepted checkpoint), hash-verified on the VM against
  `results_base_origin_velocity/training_summary.json` (11 files). Parameters only; fresh Adam/RNG/environments.
- **Command sampling:** balanced persistent commands, environment i → COMMANDS[i mod 7]; no resampling
  within an episode.
- **Environments:** 8,064 (1,152 per command); **episodes** 1,000 control steps (20 s).
- **Preprocessing:** frozen B2-cropped observation statistics (sha256 `9fcb8bc0…`), never updated; the run
  stops if they change.
- **PPO:** unroll 20, 32 minibatches, 4 epochs, LR 1e-4, clip 0.2, entropy 0.005, discount 0.97, GAE 0.95,
  value 0.5, gradient norm 1.0, seed 41; stops on KL > 0.1, replay error, coverage loss or nonfinite values.
- **Max training steps:** the step ceiling from throughput measured on iterations 2–4 after compilation,
  filling the time to the trainer deadline at VM uptime 2,220 s, capped at 80 M transitions (previous runs:
  ≈ 60 M).
- **Wall time:** VM hard lifetime 45 min (self-delete), trainer hard timeout from the shared job script.
- **Checkpoints:** every 20 iterations with optimizer/key snapshots (recoverable), synced to the bucket every
  180 s; **only the final accepted checkpoint is evaluated.**
- **Cost estimate:** previous identical-shape runs took ≈ 33 min of VM lifetime: L4 g2-standard-8 at
  $0.8536/h ≈ $0.48, Cloud NAT data processing ≈ $0.2, disk/storage < $0.03 → **≈ US$0.70**, within the
  US$2 cap; the 45 min hard lifetime bounds the VM at ≈ $0.64.

## Evaluation

1. **Decision: the full neutral long motor gate** (`neutral_motor_gate.md`, protocol unchanged; runner
   `neutral_gate.py --candidate turn_translation`): 7 steady commands × 5 seeds, 5 transition sequences, 40
   pushes, each without and with training-like latency; falls and shadow-safety interventions reported.
2. **Descriptive, not a criterion:** the 84-trial paired evaluation (`motor_compare.py --protocol
   turn_translation`; arms B2, untrained, base-origin candidate, turn-translation candidate) for comparison
   clips (B2 | current | trained), and a left-turn report (commanded and achieved yaw rate, vx, vy,
   horizontal translation, trajectory, foot contacts) for B2, the current and the trained candidate.

## Criteria (fixed now; thresholds unchanged)

- **H:** steady turn left horizontal translation RMS ≤ 0.03 m/s in 5/5 seeds, without and with latency.
- **The pilot passes** only if **the whole neutral gate passes** in both latency conditions (all four gate
  criteria, every command 5/5 including turn right, no falls, all transitions, push recovery).
- A pass is one training seed in simulation: it opens preregistration of expressive-objective work, not
  deployment. A fail is reported as a fail; no second run.
