# Diagnostics after the GPU pilot: sim-to-sim gap and the tracking reward (2026-10-10)

Runner: `sim_gap_diagnostic.py` (now `archive/`, byte-identical; original path at the recorded commits) (its reading was fixed in the docstring before running). Diagnostic only:
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

## 3. Turn gap: is it the solver iteration count? (2026-10-10, diagnostic only)

Upstream's XML uses one solver iteration. GPU candidate, same commands and seeds, deterministic, 5–20 s means.
Scratch scripts, not committed; the upstream checkout was left unchanged (verified clean).

| setting | turn left yaw (rad/s) | turn right yaw (rad/s) |
|---|---|---|
| MJX, 1 iteration (as trained) | +0.565 | −0.607 |
| MJX, 10 iterations | +0.567 | −0.594 |
| native, default | +0.711 | −0.669 |
| native, 10 iterations | +0.720 | — |
| native, warmstart disabled | +0.08 (barely moves) | — |

**Reading:** the iteration count explains neither side of the gap. Native turning depends strongly on
warmstart, which MJX does not use in the same way, so warmstart and contact-model differences remain the
open candidates. Not resolved here.
