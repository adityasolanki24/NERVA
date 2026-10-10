# Turn translation is a measurement-point mismatch: rewards measure the IMU, evaluation the base (2026-10-10)

Runner: `turn_pivot_diagnostic.py` (now `archive/`, byte-identical). Post-hoc diagnostic on the saved native rollouts (found while exploring
them after `results_gait_averaged_tracking/`): no criteria, no training, CPU only. Files: `summary.json`,
`trials.json`.

## Geometry (base frame, home keyframe)

| point | forward, left, up (m) |
|---|---|
| base origin (evaluation's `qpos[0:3]`/`qvel[0:3]`; the references' root) | 0, 0, 0 |
| IMU site (upstream's `velocimeter` and `gyro`, used by the tracking and imitation rewards) | −0.080, 0, +0.050 |
| feet (midpoint) | −0.030, 0, −0.161 |
| centre of mass | −0.026, 0, +0.044 |

During a 0.6 rad/s turn, two points 8 cm apart differ in speed by 0.6 × 0.08 = **0.048 m/s**. The turn
criterion allows 0.03 m/s at the base origin.

## Turn translation at each point (native rollouts, seeds 0/1/2, 5–20 s; horizontal RMS of the 1 s mean, m/s)

| arm | turn left: base / IMU | turn right: base / IMU | forward: base / IMU |
|---|---|---|---|
| historical B2 | 0.036 / 0.019 | 0.044 / 0.023 | 0.033 / 0.036 |
| untrained neutral | 0.036 / 0.019 | 0.045 / 0.021 | 0.034 / 0.036 |
| local pilot candidate | 0.034 / 0.021 | 0.044 / 0.020 | 0.034 / 0.037 |
| GPU pilot candidate | 0.043 / 0.022 | 0.040 / 0.043 | 0.042 / 0.044 |
| gait-averaged candidate | **0.067 / 0.024** | **0.054 / 0.022** | 0.060 / 0.060 |

- In translation the two points agree, as rigid-body kinematics requires.
- In turns, every arm translates about half as much at the IMU as at the base (one exception: the GPU pilot's
  right turn, whose pivot is also offset sideways). The policies turn roughly in place **about a point
  near the IMU**, 4–11 cm behind the base origin and behind their own feet.
- Only the turn translation criterion (`cross`) failed for the gait-averaged candidate. Direction,
  tracking and stationary passed in 3/3 for both turns.

## The rewards penalize the reference's own turn

The verified reference turns pivot about the base origin: mean body velocity ≤ 0.004 m/s there,
0.0495 m/s at the IMU. On the gait-averaged tracking term (normalized to 1), the reference turn scores
**0.998 measured at the base** and **0.782 measured at the IMU**. The imitation term's linear-velocity
target is the reference's base velocity too, but it is compared with the IMU reading.

So upstream's IMU-site rewards pay a policy to pivot about the IMU, not as the reference does. This
explains why making the tracking term sharper (gait-averaged) increased base-origin turn translation
from 0.040–0.043 to 0.054–0.067 m/s while the IMU stayed at ≈ 0.02 [inference from these measurements;
the causal test is a training run].

## Change (opt-in)

`BaseOriginGaitAveragedNeutralJoystick`: the tracking and imitation rewards use body velocity at the base
origin, from the exact rigid-body shift v_base = v_IMU − ω × r_IMU (r read from the model; checked against
native MuJoCo's free-joint velocity to 1e-5). Observations, the critic's inputs, costs, alive, noise and
pushes are unchanged. Next run: `docs/base_origin_velocity_pilot.md`.
