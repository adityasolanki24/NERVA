# Design: style-conditioned imitation policy π(s, c, e) — experiment S1

**Status:** design only, for review before any code or cloud spend. It follows from RQ1b/RQ1c (`experiments/expressive_locomotion/README.md`) and is roadmap stages 9–11.

**Claim labels:**
- **[fact]** verified in code or by running
- **[design]** a NERVA choice
- **[hypothesis]** what the experiment tests
- **[unknown]** to be measured

## 1. Question

> Can **one** locomotion policy, trained with imitation plus RL, take a small style vector `e` as input and produce **independently controllable** expressive features, while walking as stably and tracking commands as well as the upstream baseline?

**Why this is the next step [fact]:**
- **RQ1b:** runtime modulation of the gait clock gives one *coupled* axis (tempo↔amplitude↔posture) inside a narrow speed band.
- **RQ1c:** head posture applied at runtime costs up to 84% of walking speed. The loss comes from the physical head motion the policy never trained with.
- **So independent style dimensions have to be trained into the policy.**

## 2. Style vector e (3 dimensions, each in [−1, 1]) [design]

The dimensions are chosen because the literature on emotion perception from gait names them:
- speed/tempo, posture, and amplitude/heaviness: Roether et al. 2009; Montepare et al. 1987
- Laban Effort *Time* and *Weight*: Knight & Simmons 2014

They are also chosen because the upstream reference generator already exposes a parameter for each **[fact]** (`placo_presets/medium.json`, `gait_generator.py` arguments).

| e | name | Placo parameter | neutral (medium preset) [fact] | proposed range [design, provisional] |
|---|---|---|---|---|
| e₁ | tempo | `single_support_duration` | 0.18 s | 0.18 × (1 − 0.25·e₁) → 0.225 … 0.135 s |
| e₂ | step height | `walk_foot_height` | 0.04 m | 0.04 × (1 + 0.5·e₂) → 0.02 … 0.06 m |
| e₃ | torso pitch | `walk_trunk_pitch` | −4° | −4° + 6°·e₃ → −10° … +2° |

- **Not included yet:** crouch (`walk_com_height`), head posture, smoothness/flow. Add a dimension only if S1 works with three.
- **The ranges are provisional.** A gait is kept only if Placo generates it and it replays stably (§4, step R1).
- **[unknown]** the sign convention of `walk_trunk_pitch`, i.e. whether + means leaning forward. It will be checked by rendering one reference.
- **Labels stay neutral.** e₁…e₃ are "tempo, step height, torso pitch", never "confident" and the like.

## 3. How the policy uses e [design]

**Upstream baseline [fact, `docs/open_duck_baseline.md`]:**
- observation of 101 values; action of 14 joint targets at 50 Hz
- reward = alive + velocity tracking + imitation of Placo references + regularisers
- PPO (Brax) on MJX, 8192 environments, 150–300 M steps
- noise, delays, pushes and domain randomisation

**S1 changes only these things:**
1. **Observation:** e is appended, giving 104 values. The critic's privileged observation also gets e.
2. **Episodes:** e is sampled at episode start and resampled when the command is resampled (every 10 s) **[design]**. Sampling is uniform over the generated style grid.
3. **Reference:**
   - The imitation target is the reference for **(command, e)**, using nearest neighbour on a velocity grid per style, like upstream's lookup.
   - Each style has its **own gait period**, because tempo changes it, so the phase clock in the observation uses that style's period.
4. **Everything else is unchanged:** reward terms and weights, noise, randomisation, pushes and PPO hyperparameters. That way any difference is attributable to e.

**Upstream code stays unmodified.** S1 is a NERVA subclass of `playground.open_duck_mini_v2.joystick.Joystick`. Upstream's `step()` is monolithic, so the subclass has to re-implement it.

**Guard: an equivalence test, as for `OpenDuckSim`.** With a single neutral style and upstream's reference file, the NERVA environment must produce **identical** transitions to upstream's `Joystick` for the same RNG seeds.

## 4. Pipeline and controls

| step | what | where | purpose |
|---|---|---|---|
| **R0** | Regenerate upstream's *neutral* reference set with the current generator, and compare it with the shipped `polynomial_coefficients.pkl` | Linux VM (CPU) | The generator README lists an open TODO, *"Validate that we can train policies with these motions (might have broken something during the port...)"*, so its output must be checked before we rely on it |
| **B0** | Retrain the **upstream baseline** unchanged | GPU VM (L4) | reproduces the published result; **measures time and cost per run** |
| **R1** | Generate styled references: first a **pilot** of 7 styles (neutral plus ±1 on each axis), then 27 (3 levels per axis) if the pilot works | CPU VM | feasibility; drop styles Placo cannot make |
| **B1** | NERVA env with e fixed at neutral | GPU VM | should match B0; catches environment bugs |
| **S1** | NERVA env with e sampled | GPU VM | the experiment |

**Velocity grid [fact]:** reference keys are *computed* velocities, `steps_to_vel(dx, period)`, rounded to 3 decimals. Each style therefore gets its own regular grid. The upstream sweep config (`auto_gait.json`) generates 6 dx × 4 dy × 9 dθ ≈ 216 gaits per style.
- **Note [fact]:** that config does not reproduce the shipped pickle's grid (6 × 4 × 10 with different values). The settings that produced the shipped references are **[unknown]**, which is another reason for R0.
- **Generation cost [unknown]:** 7 styles ≈ 1,500 gaits for the pilot; 27 styles ≈ 5,800. Seconds per gait will be measured in R0.

## 5. Evaluation (reuses `nerva/open_duck_sim.py` and `nerva/gait_metrics.py`) [design]

- **For each dimension,** sweep e_k over {−1, −0.5, 0, 0.5, 1} with the others at 0, under the same command and paired noise seeds as RQ1, and measure:
  - the **intended feature:** e₁ → cadence; e₂ → foot lift; e₃ → torso pitch
  - **cross-talk:** the change in the *other* features and in speed
- **Controllability:** how closely the policy realises the reference's own feature values, e.g. commanded vs achieved torso pitch.
- **Stability and task:** falls, push robustness, and command-tracking error compared with B0.
- **Success criteria [design, fixed before running]:**
  - each intended feature changes **monotonically**, 10/10 paired seeds between e = ±1
  - the cross-talk on each other feature is smaller than the intended change, in normalised units
  - no walking falls
  - forward-tracking error no worse than B0's by more than 25%
  - otherwise the result is reported as negative
- **Not claimed:** any emotional reading. That needs human evaluation (roadmap stage 15).

## 6. Compute and cost controls [design, facts verified 2026-09-28]

- **Project `nerva-adityapersonal`,** us-central1:
  - NVIDIA L4 quota: 8 on-demand, 8 preemptible **[fact]**
  - L4 is offered in zones a, b and c **[fact]**
- **Image:** `common-cu129-ubuntu-2204-nvidia-580`, the Deep Learning VM (CUDA 12.9, driver 580) **[fact: image family exists]**. The training environment uses the **same pins as the local lockfile** (playground 0.0.4, mujoco 3.3.0, jax 0.5.3), with `jax[cuda12]` plus tensorflow/tf2onnx for ONNX export.
- **Hard cost caps, deterministic, not alerts:**
  - every VM is created with `--max-run-duration` and `--instance-termination-action=DELETE`. Both flags exist in gcloud 586 **[fact]**. Results are copied to a Cloud Storage bucket *before* the VM ends.
  - the training script shuts the VM down when it finishes or crashes
  - the budget alert "nerva" ($150) is a backstop only
- **Per-run cost [unknown]:** measured in B0. Nothing beyond B0 is launched until that number is known.

## 7. Risks

- **The ported generator may be broken** (its own TODO). Mitigated by R0.
- **Some style combinations may be physically infeasible,** such as fast tempo with high steps. Mitigated by the pilot and by pruning.
- **Upstream `step()` is monolithic,** so the re-implementation could drift. Mitigated by the B1 equivalence test.
- **More conditioning may lower walking quality.** It's measured against B0; if tracking or stability degrade beyond the criteria, that's a finding, not a failure to hide.
- **The policy may ignore e** and imitate an average gait. Detected directly by the per-dimension sweeps.

## 8. Order of work

1. **Local (free):**
   - write the cloud scripts: create VM with max-run-duration → bootstrap env → run → copy results → shut down
   - write the NERVA environment subclass and its equivalence test
   - CPU smoke tests with tiny settings
2. **Cloud R0 + B0.** Report timing and cost, then decide the S1 budget with the user.
3. **Cloud R1 pilot → B1 → S1 pilot (7 styles).** Report.
4. **Only then** consider 27 styles, and later PAD → e (roadmap stage 4).
