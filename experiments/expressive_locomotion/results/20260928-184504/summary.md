# RQ1 results — 20260928-184504

Style → phase factor 1 + 0.3·style. Command vx = 0.15 m/s. 10 seeded trials per style, metrics over t = 5–20 s.

Values: mean ± std across trials. `n/N`: seeds where the paired difference from Neutral has the same sign (N/N = consistent in every trial).

| metric | Style −1 | Neutral | Style +1 | −1 vs 0 | +1 vs 0 |
|---|---|---|---|---|---|
| v_fwd | 0.05492 ± 0.0026 | 0.107 ± 0.0012 | 0.1237 ± 0.00095 | −10/10 | +10/10 |
| v_lat | -0.001495 ± 0.00093 | 0.004776 ± 0.00078 | 0.005991 ± 0.0011 | −10/10 | +6/10 |
| gait_hz | 1.298 ± 0.0038 | 1.85 ± 2.2e-16 | 2.408 ± 4.4e-16 | −10/10 | +10/10 |
| cadence_steps_per_s | 2.595 ± 0.0076 | 3.7 ± 4.4e-16 | 4.817 ± 8.9e-16 | −10/10 | +10/10 |
| stride_m | 0.04233 ± 0.002 | 0.05782 ± 0.00064 | 0.05136 ± 0.00039 | −10/10 | −10/10 |
| lift_left_mm | 12.19 ± 0.61 | 14.14 ± 0.23 | 11.4 ± 0.12 | −10/10 | −10/10 |
| lift_right_mm | 10.5 ± 0.48 | 12.54 ± 0.13 | 9.916 ± 0.13 | −10/10 | −10/10 |
| base_height_m | 0.1612 ± 0.00014 | 0.1647 ± 9.2e-05 | 0.1646 ± 5.7e-05 | −10/10 | −9/10 |
| pitch_mean_deg | 4.458 ± 0.15 | 1.81 ± 0.068 | 1.054 ± 0.058 | +10/10 | −10/10 |
| pitch_std_deg | 0.9074 ± 0.056 | 0.6459 ± 0.019 | 0.5896 ± 0.02 | +10/10 | −10/10 |
| roll_std_deg | 3.038 ± 0.13 | 2.608 ± 0.024 | 1.883 ± 0.014 | +10/10 | −10/10 |
| sway_rate_rms | 0.4356 ± 0.011 | 0.4716 ± 0.0026 | 0.4453 ± 0.003 | −10/10 | −10/10 |
| action_rate_rms | 0.1099 ± 0.00077 | 0.1287 ± 0.00063 | 0.153 ± 0.00076 | −10/10 | +10/10 |
| joint_range_rms_rad | 0.2055 ± 0.0014 | 0.2068 ± 0.0012 | 0.172 ± 0.0011 | −8/10 | −10/10 |
| tau_sq | 7.251 ± 0.066 | 12.51 ± 0.079 | 13.39 ± 0.043 | −10/10 | +10/10 |
| power_w | 5.861 ± 0.08 | 10.46 ± 0.081 | 9.989 ± 0.039 | −10/10 | −10/10 |
| cost_of_transport | 5.172 ± 0.19 | 4.73 ± 0.037 | 3.907 ± 0.032 | +10/10 | −10/10 |
| max_tilt_deg | 8.383 ± 0.39 | 5.453 ± 0.19 | 3.806 ± 0.16 | +10/10 | −10/10 |
| fell | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 | 0 |

## Push robustness (falls / trials)

| magnitude m/s | Style −1 | Neutral | Style +1 |
|---|---|---|---|
| 0.3 | 0/8 | 0/8 | 0/8 |
| 0.6 | 1/8 | 2/8 | 2/8 |
| 0.9 | 4/8 | 6/8 | 6/8 |

## Run metadata

```json
{
  "timestamp": "20260928-184504",
  "wall_time_s": 101.6,
  "styles": {
    "-1.0": 0.7,
    "0.0": 1.0,
    "1.0": 1.3
  },
  "command_vx": 0.15,
  "raw_accel": true,
  "init_joint_noise_rad": 0.02,
  "obs_noise": "training noise_config (joystick.py), seeded",
  "gait": {
    "trials": 10,
    "seconds": 20.0,
    "window_start_s": 5.0
  },
  "push": {
    "at_s": 8.0,
    "seconds": 15.0,
    "magnitudes": [
      0.3,
      0.6,
      0.9
    ],
    "directions_deg": [
      0,
      45,
      90,
      135,
      180,
      225,
      270,
      315
    ]
  },
  "policy": "BEST_WALK_ONNX_2.onnx",
  "nerva_rev": "d41d8b5",
  "open_duck_playground_rev": "b9be205",
  "open_duck_mini_rev": "b23317a",
  "python": "3.12.4"
}
```
