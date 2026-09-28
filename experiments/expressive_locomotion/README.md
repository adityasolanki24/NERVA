# Experiment: expressive locomotion via the gait-phase clock (RQ1, method A)

**Question.** Can one variable `style ∈ [-1, 1]` make the *same* walking controller move measurably differently while staying stable?

**Method A.** `style` sets the rate of the policy's gait-phase clock: `phase_factor = 1 + 0.3·style` (`nerva/style.py`). So Style −1 is 0.7×, Neutral 1.0× and Style +1 1.3×. The policy is the unmodified upstream `BEST_WALK_ONNX_2.onnx`; nothing is retrained.

## Protocol (`run.py`)

- **Task, identical for all conditions:** walk forward, commanded 0.15 m/s, flat ground, 20 s.
- **Simulation:** `nerva/open_duck_sim.py`. The control loop is tested to reproduce upstream `mujoco_infer.py` bit for bit (`tests/test_open_duck_sim.py`). Two deliberate differences:
  - **Raw accelerometer.** The +1.3 offset that only upstream sim inference adds is removed, which matches training and hardware.
  - **Training-level observation noise.** The noise from `joystick.py` `noise_config`, resampled every step and seeded. This is the source of trial-to-trial variation: without it, every run converges to the same limit cycle within seconds.
- **Gait trials:** 10 seeds per style, with the same seeds across styles (paired). Initial joint noise is ±0.02 rad. Metrics are computed over t = 5–20 s (`nerva/gait_metrics.py`).
- **Push trials:** per style, one base-velocity kick at t = 8 s, at 3 magnitudes × 8 directions, 15 s total. Fall = tilt > 45°.

**Reproduce:**
```bash
<venv>/Scripts/python experiments/expressive_locomotion/run.py
```
It takes about 2 minutes on the development laptop and writes `results/<timestamp>/`.

## Results (run `results/20260928-184504`)

Full table: `results/20260928-184504/summary.md`. Mean ± std over 10 trials. "10/10" means every paired seed differed from Neutral in the same direction.

| metric | Style −1 | Neutral | Style +1 |
|---|---|---|---|
| measured forward speed (m/s) | 0.055 ± 0.003 | 0.107 ± 0.001 | 0.124 ± 0.001 |
| cadence (steps/s) | 2.60 | 3.70 | 4.82 |
| stride length per gait cycle (m) | 0.042 | **0.058** | 0.051 |
| foot lift, left/right (mm) | 12.2 / 10.5 | **14.1 / 12.5** | 11.4 / 9.9 |
| mean torso pitch (°, + = nose down) | **4.46** | 1.81 | 1.05 |
| torso roll variability, std (°) | **3.04** | 2.61 | 1.88 |
| max tilt (°) | 8.4 | 5.5 | 3.8 |
| policy action rate, RMS per step | 0.110 | 0.129 | 0.153 |
| mechanical power (W) | 5.9 | 10.5 | 10.0 |
| cost of transport | 5.17 | 4.73 | **3.91** |
| falls while walking | 0/10 | 0/10 | 0/10 |

Every metric above differed from Neutral in the same direction in 10/10 paired trials.

**Push robustness (falls / 8 directions):**

| push | Style −1 | Neutral | Style +1 |
|---|---|---|---|
| 0.3 m/s | 0 | 0 | 0 |
| 0.6 m/s | 1 | 2 | 2 |
| 0.9 m/s | 4 | 6 | 6 |

The differences are not statistically distinguishable (Fisher exact, pooled over magnitudes: p ≈ 0.5).

A rerun with the same seeds reproduced both CSVs byte for byte.

## What this shows

**Measured:**
- **RQ1, first answer: yes.** A single variable changes the gait measurably and consistently, with no falls in unperturbed walking. There is no detectable loss of push robustness at this sample size (8 pushes per cell, which is a small sample).
- **Cadence is directly controlled.** It follows the clock exactly: 1.30 / 1.85 / 2.41 Hz, against predicted 1.30 / 1.85 / 2.41.
- **The effects are not a single "more/less" axis.**
  - Stride length and foot lift are **largest at Neutral** and decrease in *both* directions. The policy's gait is "best formed" at the clock rate it was trained on.
  - Style −1 is not bigger, slower steps. It is smaller, slower steps with **more forward torso pitch and more roll sway**.
  - Style +1 is quicker, shorter, lower steps with a **more upright, steadier torso** and the lowest cost of transport.
- **Speed is coupled to style.** At the same command, Style −1 walks at about half Neutral's speed. Style +1 is only 16% faster: speed saturates above factor 1.0.

**Not shown (do not claim):**
- **That any style looks hesitant, confident, calm or anything else.** Only human evaluation (RQ4) can support such labels.
- **That the differences come from "style" rather than from speed.** Many of them (pitch, sway, power) could simply follow from walking slower or faster. A **speed-matched** comparison is needed to separate the two.
- **Behaviour on hardware, on rough ground, or under other pushes.** Clock factors ≠ 1.0 are outside the policy's training distribution.

## Limitations

- One policy, one command (forward 0.15 m/s), flat ground, simulation only.
- Trial-to-trial variation comes only from training-level sensor noise and small initial-pose noise. It is not an estimate of real-world variability.
- Cadence resolution is limited by the FFT bin width (about 0.008 Hz after padding). That is why its std can print as ~0.
- Push trials use one seed per direction and 8 directions; that is enough to spot large effects only.

---

# RQ1b: speed-matched comparison (`speed_matched.py`)

**Question.** Does the gait-clock style change *how* the robot walks once walking *speed* is matched? Or were the RQ1 differences just consequences of speed?

## Procedure

1. **Calibration.**
   - Per style, sweep the commanded vx over 0 to 0.15 in steps of 0.015. That's 11 commands × 3 calibration seeds (100–102), disjoint from the evaluation seeds.
   - Measure steady-state forward speed (t = 5–20 s).
2. **Command selection.** Linear interpolation on the first bracket where the mean speed crosses the target, then up to 4 secant refinements on the calibration seeds.
3. **Evaluation.**
   - The same paired seeds 0–9 and simulator settings as RQ1: raw accelerometer, training observation noise, ±0.02 rad initial joint noise.
   - A style counts as matched if the mean achieved speed is within **±0.005 m/s** of the target.
4. **Speed-mismatch control.** The local slope of each metric vs measured speed, taken from Neutral calibration trials near the target, predicts how much of a difference the leftover speed mismatch could explain. This appears as the "speed-explained" column in `summary.md`.
5. **Command-confound control** (`same_command_check.py`). Matching speed requires *different commands* per style, and the command is a policy input. The calibration sweep compares styles at *identical* commands.
6. **Push robustness** at the primary target, using the same protocol as RQ1.
7. **Target choice.** Only commands inside the trained range (≤ 0.15) are used, where Style −1 reaches at most about 0.056 m/s. So the targets are 0.045 m/s (primary) and 0.025 m/s (secondary).

**Reproduce:**
```bash
<venv>/Scripts/python experiments/expressive_locomotion/speed_matched.py
<venv>/Scripts/python experiments/expressive_locomotion/same_command_check.py experiments/expressive_locomotion/results/speed_matched-20260928-192454
```
The run took about 4 minutes. `results/speed_matched-20260928-192454` was generated from clean commit `3cf1fa1`, and an earlier run of the same code reproduced all three CSVs byte for byte.

## Results

### Calibration: the policy has a command dead zone [measured]

| command vx | 0.075 | 0.090 | 0.105 | 0.120 | 0.135 | 0.150 |
|---|---|---|---|---|---|---|
| Style −1 speed (m/s) | 0.000 | 0.009 | 0.017 | 0.030 | 0.045 | 0.056 |
| Neutral | 0.000 | 0.051 | 0.069 | 0.084 | 0.096 | 0.108 |
| Style +1 | 0.001 | 0.068 | 0.086 | 0.100 | 0.113 | 0.125 |

- Below about 0.08 commanded, no style walks forward.
- Neutral and Style +1 *jump* from 0 to 0.05–0.07 m/s between 0.075 and 0.09. Style −1 ramps up gradually.
- The speed range all three styles share is only about 0.045–0.056 m/s.

### Primary target 0.045 m/s: matched

| | Style −1 | Neutral | Style +1 |
|---|---|---|---|
| command vx | 0.1355 | 0.0873 | 0.0800 |
| achieved speed | 0.0424 ± 0.0030 | 0.0443 ± 0.0022 | 0.0461 ± 0.0023 |
| cadence (steps/s) | 2.60 | 3.70 | 4.82 |
| stride per gait cycle (m) | **0.033** | 0.024 | **0.019** |
| foot lift L / R (mm) | **9.7 / 9.2** | 7.5 / 7.9 | **6.3 / 5.8** |
| joint range, RMS (rad) | **0.198** | 0.168 | **0.134** |
| mean torso pitch (°, + = nose down) | **4.12** | 1.40 | **0.95** |
| roll sway, std (°) | **2.71** | 2.15 | **1.82** |
| base height (mm) | 160.9 | 163.2 | 163.2 |
| policy action rate, RMS | 0.108 | 0.119 | 0.133 |
| mechanical power (W) | 5.38 | **7.08** | 5.83 |
| cost of transport | 6.17 | **7.75** | 6.13 |
| max tilt (°) | 7.6 | 4.6 | 4.1 |
| falls while walking | 0/10 | 0/10 | 0/10 |

- Each row below achieved speed differs from Neutral in the same direction in 10/10 paired seeds, with two exceptions: Style +1's base height (no difference, 7/10) and Style +1's max tilt (9/10).
- The leftover speed mismatch (±0.002 m/s) predicts differences that are small compared with those observed, and often of the *opposite* sign. For example, for Style −1 it predicts −0.24 mm of lift against an observed +2.1 mm, and −0.009° of pitch against an observed +2.7°. The largest ratio is Σ τ² for Style +1: a predicted +0.10 against an observed −0.34, again of opposite sign. Full columns are in `summary.md`.
- **Push falls, per 8 directions:**

  | push | Style −1 | Neutral | Style +1 |
  |---|---|---|---|
  | 0.6 m/s | 2 | 2 | 1 |
  | 0.9 m/s | 4 | 6 | 6 |

  As in RQ1, there's no detectable difference.

### Command-confound check [measured, calibration seeds]

- At *identical* commands (0.105–0.15), Style −1 still leans forward 3.4–4.5°, Neutral 1.4–1.8° and Style +1 0.7–1.2°. Across that whole range, the command alone moves Neutral's pitch only from 1.43° to 1.81°. **The pitch difference is caused by the clock, not by the command or the speed.**
- Style +1's smaller joint range and lower roll sway also appear at identical commands.
- Style −1's *larger* lift, joint range and roll sway appear only at matched speed. At the same command it walks much slower, and those values are similar to or smaller than Neutral's. For Style −1, these three features therefore depend on how the comparison is made.

### Secondary target 0.025 m/s: not matched (negative result)

- Style +1 reached 0.010 ± 0.011 m/s. **It stalled completely in 4 of 10 seeds.**
- Neutral's per-seed speeds ranged from 0.007 to 0.033.
- Near the dead zone the policy walks stop-and-go, so no valid speed-matched comparison exists below about 0.045 m/s. The 0.025 data is kept in the results for transparency but is not interpreted.

## What this shows

**Measured:**
- **The style differences do not disappear when speed is matched.** At 0.045 m/s the gait-clock variable still changes cadence, stride, foot lift, joint amplitude, torso pitch, roll sway and effort, consistently across paired seeds, and without falls.
- **Speed matching removes the earlier non-monotonic pattern.** In RQ1, stride and lift peaked at Neutral. At matched speed they become **monotonic**: slower clock → longer, higher, larger-amplitude steps. The RQ1 peak was a speed confound.
- **The effects are coupled, not independent:**
  - At a fixed speed, stride = speed / gait frequency. So cadence and stride are *one* degree of freedom, not two.
  - Pitch, amplitude and sway move *together* with the clock. The variable cannot produce, for example, slow steps with an upright torso, or quick steps with large amplitude.
- **Neutral is the least efficient** at 0.045 m/s (highest power and cost of transport). This is non-monotonic, and we don't yet know why.

**Not shown:**
- That any of these patterns *looks* hesitant, confident or anything else. That still requires human evaluation.
- That the effects hold at other speeds, above about 0.056 m/s, where Style −1 cannot follow.

## Limitations

- **The usable speed-matched range is very narrow,** about 0.045–0.056 m/s, and only one speed was validly matched.
- The command dead zone and the steep command→speed curve make matching sensitive. Refinement needed up to 4 iterations.
- Same single policy, flat ground, simulation, and sensor-noise-only variability as RQ1.
