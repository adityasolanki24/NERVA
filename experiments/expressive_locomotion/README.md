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
