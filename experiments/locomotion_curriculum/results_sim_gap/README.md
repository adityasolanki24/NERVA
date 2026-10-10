# Diagnostics after the GPU pilot: sim-to-sim gap and the tracking reward (2026-10-10)

Runner: `sim_gap_diagnostic.py` (its reading was fixed in the docstring before running). Diagnostic only:
no criteria, no training. CPU, local.

## 1. MJX training environment vs native MuJoCo

Deterministic actions, seeds 0/1/2, 1,000 steps, mean over 5–20 s. Each cell is vx, vy (m/s) and yaw
rate (rad/s), in the MJX body frame / gyro and the native heading frame.

| command | GPU candidate, MJX + pushes | GPU candidate, MJX no pushes | GPU candidate, native |
|---|---|---|---|
| forward +0.074 | +0.020, +0.021 | +0.039, +0.019 | +0.032, +0.027 |
| backward −0.074 | −0.053, −0.001 | −0.047, +0.008 | −0.065, +0.015 |
| left +0.074 | −0.004, +0.043 | −0.007, +0.047 | −0.018, +0.071 |
| right −0.074 | +0.003, −0.023 | −0.002, −0.021 | −0.017, −0.021 |
| turn left +0.60 | yaw 0.567, vy −0.018 | yaw 0.565, vy −0.018 | yaw **0.711**, vy +0.043 |
| turn right −0.60 | yaw −0.593, vy +0.030 | yaw −0.607, vy +0.022 | yaw **−0.669**, vy −0.025 |

Local pilot candidate (start of the GPU run), MJX without pushes: forward +0.038, backward −0.005, left
+0.029, right −0.024; turns 0.532 / −0.578 rad/s. With pushes on (as trained), episodes still end
occasionally (1–6 true terminations per command over three 20 s rollouts, across both arms); without pushes,
none do.

**Reading:**
- The translation undershoot exists in the policy's own training environment, so it is learned
  behaviour, not a sim-to-sim gap.
- Yaw rate is close to the command in MJX but 12–18% too high natively, with different lateral drift:
  the turn failures are largely a MJX → native MuJoCo gap.

## 2. The tracking reward on the verified reference gaits

Upstream's form is 2.5·exp(−|c − v|²/0.01), evaluated on the reference body velocity (54 s at 50 Hz).

| command | perfect reference gait, instantaneous | standing still | gait at 50% speed | perfect gait, averaged over one 0.54 s period |
|---|---|---|---|---|
| forward 0.074 | 0.772 | 1.446 | 0.591 | 2.490 |
| backward −0.074 | 0.766 | 1.446 | 0.580 | 2.490 |
| left 0.074 | 0.795 | 1.446 | 0.738 | 2.489 |
| right −0.074 | 0.789 | 1.446 | 0.730 | 2.490 |

At these speeds the gait's own within-stride sway dominates the instantaneous comparison, so **the
tracking term pays more for standing still than for walking correctly**. The reference means are correct:
translation 0.0738 m/s; turns ≤ 0.004 m/s mean translation and 0.5986 rad/s.

Files: `rollouts.json`, `summary.json`. Follow-up: `docs/gait_averaged_tracking_pilot.md`.
