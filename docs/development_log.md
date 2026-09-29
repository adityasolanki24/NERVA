# NERVA development log

Newest entry first. Each entry records what was done, what was actually run, and what is still unverified.

---

## 2026-09-29 — First L4 cloud smoke completed; all resources removed

- Full report: `docs/cloud_smoke_report.md`.
- Private `g2-standard-8` VM: JAX 0.5.3 saw the L4; MJX/PPO trained one 327,680-step batch; Orbax and ONNX exports completed; results synced; exit 0; VM self-deleted.
- The successful job took 14 min 15 s (about 18 min including VM bootstrap). Reward changed from 13.961 ± 8.708 to 17.348 ± 12.064. This is a pipeline smoke, not a trained walking result.
- The final policy remained upright in a 10 s replay but did not move forward. Video and raw artifacts are in the git-ignored `experiments/cloud_runs/` directory.
- The initial 2 M-step smoke was stopped when its measured rate showed it would exceed the one-hour cap. The smoke-only wrapper now requests one 200k-step batch and two evaluations.
- Organisation policy required private VMs; the launcher now has temporary Cloud NAT lifecycle commands. GPU stockouts occurred in different zones.
- Public dependency alerts were addressed by updating the export-only stack to TensorFlow CPU 2.20, tf2onnx 1.17, ONNX 1.22 and protobuf 5.29.6. A clean local Python 3.12 conversion test passed.
- Throughput projects the unchanged 300 M-step B0 to roughly 95 hours on this L4, so B0 was **not** launched.
- Teardown audit: zero instances, disks, addresses, forwarding rules, routers/NAT, snapshots, buckets and runner service accounts.

---

## 2026-09-28 — S1 training env + cloud runner (local design stage)

- **`nerva/training/style_joystick.py`:** `StyledReference` stacks upstream reference pickles per style. `StyleJoystick` subclasses the upstream `Joystick`; its `reset`/`step` are copies of upstream with lines marked `# NERVA`. Style keys use `fold_in`, so they never consume upstream randomness.
- **Tests:**
  - 5 fast unit tests on synthetic references pass.
  - **Slow:** the single neutral style reproduces upstream **exactly** over 8 steps (qpos, observations, reward, reference).
  - **Resample test:** the new style appears in the observation one step later, like upstream's command. The first version of this test wrongly expected it immediately and was fixed.
- **`nerva/training/train_style.py`:** CPU smoke PPO completed with 1 style (284 s) and with 3 styles (310 s). A suspected scan dtype mismatch did **not** occur.
- **`cloud/`:**
  - launcher: plan printed, `--yes` required, committed code only, `max-run-duration` + DELETE, self-delete
  - job scripts: smoke, b0_baseline, r0_references, session1
  - a Linux lockfile compiled with uv: numpy 2.0.2, protobuf 3.20.3
- **Not yet run:**
  - the `setup` and `launch` commands
  - whether the GPU and CUDA stack work on the VM
  - the per-hour price of `g2-standard-8` (the lookup was interrupted)

---

## 2026-09-28 — GPU compute set up; style-conditioned policy designed

### Compute
- **Google Cloud project:** a dedicated research project was created and linked to an approved billing account. Project identifiers, account addresses, credit balances and organisation details are intentionally kept out of the public repository.
- **Access:** the research account was granted the roles required to operate the project.
- **Budget "nerva":** $150, alerts only. Google's spend caps do not support Compute Engine.
- **Quota:** GPUs (all regions) 360; L4 in us-central1 8 on-demand and 8 preemptible. No requests were needed.
- **`gcloud` 586.0.0** was installed locally via winget and the user logged in. Read-only checks confirmed:
  - billing is enabled
  - L4 is offered in us-central1 zones a, b and c
  - no instances are running
  - the `--max-run-duration` and `--instance-termination-action` flags exist
  - the DLVM image family `common-cu129-ubuntu-2204-nvidia-580` exists

### Design (`docs/style_policy_design.md`)
- **Style vector** e = (tempo, step height, torso pitch), mapped to Placo's `single_support_duration`, `walk_foot_height` and `walk_trunk_pitch`, around the `medium` preset. Observation 101 → 104.
- **Reference and periods:** per-style reference grids and gait periods.
- **Pipeline R0 → B0 → R1 → B1 → S1,** with success criteria fixed in advance.

### Facts found while designing
- The generator needs `placo==0.6.3` and Python 3.10.12, and has no Windows build. So references are generated on a Linux VM.
- Reference keys are computed velocities (`steps_to_vel(dx, period)`, rounded to 3 decimals), not measured ones.
- The current `auto_gait.json` sweep does not reproduce the shipped pickle's velocity grid, so how the shipped references were generated is unknown. **(Corrected 2026-09-29: wrong. R0 reproduces exactly the shipped 240 keys; see the 2026-09-29 entry.)**
- The generator README has an open TODO questioning whether its output still trains. That is why step R0 exists.

---

## 2026-09-28 — Demo video: closed loop events → appraisal → PAD → behaviour → robot (user request)

- **What it is.** `experiments/demo_video/`: MuJoCo render beside emotion, PAD and behaviour panels revealed over time, with an event caption and a status line. The output is `results/nerva_affect_demo.mp4` (66 s, 1280×650, 25 fps, about 10 MB).
- **Labelled as a demo, not a result.** The PAD → behaviour mapping is hand-designed:
  - arousal → tempo style
  - valence + dominance → head posture, limited to ±0.5 because of RQ1c
  - stop when blocked, pause while arousal > 0.2
  WHAT and HOW stay separate, and PAD never drives joints.
- **The near_fall event includes a real 0.6 m/s sideways push.** Tilt peaks at 13.4° and the robot recovers. With the same seed, 0.75 m/s makes it fall.
- **Problems hit and fixed:**
  1. The first renders were black: several renderers were open at once. They are now closed.
  2. The camera framed the feet; it now follows the trunk with a raised look-at point.
  3. The Windows console couldn't print "→" (terminal output switched to ASCII).
  4. The model's offscreen buffer was 480 px, raised at runtime as upstream `base.py` does.
  5. The chart's "future" was only faded; it is now a true progressive reveal (background-only plus full chart).
- **Verified:**
  - The MP4 reads back as 1650 frames at 25 fps, 66 s.
  - The frames change over time (walking, stumble, head posture).
  - The README's event table was checked against `timeline.csv` (resume at 35.8 s and 46.0 s; minimum V −0.39, D −0.25).

---

## 2026-09-28 — RQ1c: head posture as a second channel (user-requested test)

- **Added:** `OpenDuckSim.set_head_offset`, which follows the hardware-runtime convention. The offset goes into the observation's command slots 3:7 and is added to head targets 5:9 after the speed limit. It defaults to zero, and the equivalence test still passes.
- **Direction check:** rendering showed that negative head_pitch raises the head. The first render attempt showed black frames because several renderers were left open at once; they are now closed after use.
- **Experiment:** `head_posture.py`, 5 head values × 3 tempo styles × 10 seeds, plus pushes and an observation-vs-actuation diagnostic.
  - The results are from clean commit `24ec0bb`.
  - An earlier run was made while `pyproject.toml` was being edited, so it was marked dirty. It was deleted before its CSVs were compared, which was a slip. Its printed v_fwd and push tables match the clean run exactly.
- **Result:**
  - The head angle follows the offset, independent of tempo.
  - **Walking speed drops with any offset:** 0.107 → 0.017 m/s head-up, → 0.053 head-down. Stride and lift shrink, and the torso compensates. No falls.
- **Diagnostic:** an observed-only offset has no effect (0.107 m/s); an actuated-only offset reproduces the whole loss. So the policy ignores head commands, as trained, and the loss is caused by head dynamics it never experienced.
- **Implication:** independent posture channels require retraining (option C). Runtime head posture is not an independent style dimension for this policy.

---

## 2026-09-28 — Affect model v0.1 and documentation cleanup

### Affect v0.1 (decisions by the user)
- **Fix 1, surprise diluting fear.**
  - A strong surprise (valence 0) still pulls fear's −0.64 toward 0 in the intensity-weighted average. So setting surprise's valence to 0 was not enough.
  - Implemented instead:
    - surprise acts on **arousal only** (0.8, WASABI's value) and is excluded from the valence and dominance averages
    - the emotion centre is computed per dimension
    - surprise decays with τ = 1 s
  - Result: valence minimum after near_fall is −0.39, where v0 gave −0.09.
- **Fix 2, routine events.**
  - Intensity is now multiplied by relevance, a NERVA extension of EMA.
  - Routine `successful_walking` relevance is lowered from 0.5 to 0.2.
  - Result after two routine walks: valence +0.09, where v0 gave +0.27.
  - **Limitation:** repeated routine success still settles at valence +0.16 (+0.22 at relevance 0.5). The pull aims at the emotion's anchor however weak the emotion is. The principled fix, habituation in history-aware appraisal, belongs to roadmap stage 5. It was not implemented, and is documented in `affect_model.md` with the alternative fix.
- **Fix 3, arousal persistence.** τ_return is now per dimension: 20 s, 6 s, 20 s. Arousal 11 s after near_fall is 0.16, where v0 gave 0.50.
- **Issue 4, weak emotions:** left unchanged as decided. Relevance scaling makes them weaker still (hope 0.12 → 0.05).
- **Architecture.**
  - New `AffectSystem` protocol in `nerva/interfaces.py`: `AppraisalState` in, `PADState` out.
  - `AffectModel` renamed to `CategoricalAffectModel` (Model A). The discrete labels are explicitly one implementation, not a layer.
  - Appraisal v0 is documented as a context-free table, to be replaced by contextual appraisal.
- **Documentation honesty.** `affect_model.md` now says the intensity rule is a *simplified* subset of EMA. EMA's appraisal frames, mood-adjusted intensity, focus and coping are not implemented.
- **Results.** The v0 demo outputs were moved to `experiments/affect_prototype/results/v0.0/`. The new ones are in `results/v0.1/`.
- **Tests: 52 passed.** New tests cover:
  - relevance scaling
  - surprise being arousal-only
  - surprise not diluting fear's valence
  - arousal recovering faster than valence and dominance
  - protocol conformance
  - repeated routine success staying below 0.2 (a design target)

### Documentation cleanup
- `overview.md` rewritten: accurate and not oversold (the old text had typos and described capabilities that don't exist).
- `architecture.md`:
  - the canonical architecture is stated explicitly (perception → context-aware appraisal → PAD → behaviour → learned policy π(s, c, e) → deterministic safety/control)
  - Model A is described as v0, not as a layer
  - the timescales are listed
  - the safety/affect dual pathway is described
  - a "future research direction" claim label is added
- New `roadmap.md`: the 20-stage research trajectory, the planned comparisons and the invariant constraints.
- **Licence:** `LICENSE` is still empty. **Flagged for the author to decide; nothing was inserted.**

---

## 2026-09-28 — RQ1b: speed-matched expressive locomotion

**Question:** does the gait-clock style change *how* the robot walks once measured speed is equal?

### Changed
- `experiments/expressive_locomotion/run.py`: added an optional `vx` parameter. The defaults are unchanged. **Verified:** rerunning the original RQ1 reproduced both original CSVs byte for byte.
- New `speed_matched.py`:
  - calibration sweep: 11 commands × 3 seeds × 3 styles
  - interpolation plus secant refinement
  - paired evaluation on seeds 0–9
  - a speed-sensitivity control
  - pushes
- New `same_command_check.py` (the command-confound control).
- Full write-up in `experiments/expressive_locomotion/README.md`.

### Run
- `results/speed_matched-20260928-192454`, from clean commit `3cf1fa1`, 254 s.
- The first run of the same code reproduced all three CSVs byte for byte.
- It was generated with uncommitted edits in the working tree, so it was discarded in favour of the clean run.

### Measured
- **Command dead zone:** below about 0.08 commanded vx, no style walks forward. Neutral and Style +1 jump to 0.05–0.07 m/s just above it.
- **Shared speed band:** only about 0.045–0.056 m/s, because Style −1 is at most 0.056 m/s inside the trained command range.
- **At 0.045 m/s** the styles matched at 0.0424 / 0.0443 / 0.0461 m/s (tolerance ±0.005). Differences remain, 10/10 paired seeds for nearly all metrics:

  | | Style −1 | Neutral | Style +1 |
  |---|---|---|---|
  | cadence (steps/s) | 2.60 | 3.70 | 4.82 |
  | stride (m) | 0.033 | 0.024 | 0.019 |
  | foot lift L (mm) | 9.7 | 7.5 | 6.3 |
  | joint range RMS (rad) | 0.198 | 0.168 | 0.134 |
  | torso pitch (°) | 4.1 | 1.4 | 1.0 |
  | roll std (°) | 2.7 | 2.1 | 1.8 |
  | power (W) | 5.4 | 7.1 | 5.8 |
  | falls | 0 | 0 | 0 |

- The leftover speed mismatch predicts much smaller effects, often of the opposite sign.
- **Command confound:** at identical commands, pitch is 3.4–4.5° / 1.4–1.8° / 0.7–1.2° for the three styles, while the command alone moves Neutral's pitch only 1.43° → 1.81°. **So pitch is a clock effect.** Style −1's larger lift, range and sway appear only at matched speed.
- **0.025 m/s target: not matched** (negative result). Style +1 stalled in 4 of 10 seeds (0.010 ± 0.011 m/s), and Neutral ranged 0.007–0.033 m/s per seed. The policy walks stop-and-go near the dead zone.
- **Pushes at 0.045 m/s:** 2/2/1 and 4/6/6 falls of 8. No detectable difference.

### Interpretation
- **Observation:** the RQ1 differences are not just speed effects. Speed matching also turns RQ1's non-monotonic stride and lift into monotonic trends.
- **Interpretation:**
  - The variable is one *coupled* axis: tempo goes up while stride and amplitude go down and the torso becomes more upright.
  - At fixed speed, cadence and stride are one degree of freedom (stride = v / f).
  - Posture cannot be set independently of tempo.
- **No emotional labels are claimed.**

### Mistakes caught during write-up
The first draft of the README claimed "10/10 in every row" and "speed mismatch explains at most about 10%". Both were checked against `summary.md` and corrected: Style +1's max tilt is 9/10, and the largest speed-predicted ratio is about 28%, of opposite sign.

---

## 2026-09-28 — Affect model v0: synthetic events → appraisal → emotions → PAD

The user asked to start the emotional-state system after researching it properly. It is **simulation only and not connected to movement**.

### Research (primary sources read, not summaries)
- **EMA:**
  - Marsella & Gratch 2009, pp. 80–81: the appraisal variables, and Table 2's appraisal pattern → emotion label. EMA mood is per-label and discrete; **EMA does not use PAD**.
  - Gratch & Marsella 2004, Table 3: intensity = abs(desirability × likelihood). The text was extracted locally with `pypdf` in a scratch folder.
- **ALMA** (Gebhard 2005): Table 2 maps OCC emotions to PAD. Decaying emotions form an intensity-weighted "virtual emotion center" that pulls the mood (plus a push phase), and the mood returns to a default.
- **WASABI** (Becker-Asano & Wachsmuth 2010):
  - event valence acts as an impulse, and the state is driven back to balance
  - dominance comes from situational context in cognition
  - Table 1 gives the PAD point for "surprised"
- Paper titles were checked against the PDFs' first pages.

### Built
- `nerva/appraisal.py`: a table of EMA-variable appraisals for the 5 synthetic events. The values are NERVA design choices, with the reasoning in the doc.
- `nerva/affect.py`:
  - `categorise()` implements the EMA rules
  - `EMOTION_PAD` holds the ALMA values
  - a controllability → dominance blend (NERVA hypothesis)
  - `AffectModel`: exponential emotion decay, an ALMA-style pull toward the emotion centre, and a return to baseline, integrated exactly (independent of dt)
  - every parameter lives in `AffectConfig`
- `docs/affect_model.md`: the design, with every number labelled by source.
- `experiments/affect_prototype/run.py`: a 70 s scripted timeline producing a CSV and a plot.
- `pyproject.toml`:
  - **Fixed:** `numpy` is now declared. `gait_metrics.py` already needed it, and the empty dependency list was wrong.
  - Added the optional `experiments` extra (`matplotlib`, installed 3.11.2).

### Verified
- **pytest: 48 passed.** The tests cover:
  - the EMA labels and intensities
  - the ALMA table values
  - controllability affecting dominance only
  - staying at baseline without events
  - rise after an event, then an exact exp(−t/τ) return
  - update-rate independence
  - bounds under extreme input
- **Two mistakes of mine were caught during testing:**
  - A test demanded a return below 0.001 within 2 minutes. That's wrong: the model's τ values predict about 0.002, so the test now checks the exact exponential instead.
  - The demo printed decayed intensities (0.01) instead of the intensities at elicitation, because the objects mutate. It now records them at fire time.

### Observed in the demo (details in `docs/affect_model.md`)
- Across the rapid approach, valence and dominance go down and arousal goes up. That is by construction.
- Open design issues, not fixed and left for the user to decide:
  - surprise's +0.1 valence dilutes fear
  - relevance doesn't scale intensity, so routine success moves PAD a lot
  - arousal persists for many seconds
  - weak emotions are barely visible

---

## 2026-09-28 — Phase 5: first expressive-locomotion experiment (method A)

**Decision** (with the user): method A. `style` sets the gait-phase clock rate, `1 + 0.3·style`, with no retraining. B (style as a policy input) and C (style-conditioned reference motions) need GPU retraining and are deferred.

### Built
- `nerva/style.py`: the style → phase-factor mapping.
- `nerva/open_duck_sim.py`: a headless simulator. It reuses upstream `MjInfer` (model, obs, policy, action scaling, speed limit) and replaces only its viewer loop. Options:
  - raw accelerometer
  - seeded initial joint noise
  - training-level observation noise
  - pushes
  - `set_behaviour(BehaviourCommand)`, which clips velocities to the trained range
- `nerva/gait_metrics.py`: pure-NumPy metrics.
- `experiments/expressive_locomotion/run.py`: the protocol runner, and its `README.md`.

### Verified
- **pytest: 34 passed.** Key tests:
  - **Exact equivalence:** after 3 s of walking, our loop's state is bit-identical to upstream `run()` at style 0 and +1. A positive control asserts the robot walked more than 0.1 m, and a negative control asserts a 1% clock change is detected.
  - The metrics recover known values from synthetic signals.
- **Upstream quirk found and replicated deliberately:** `joystick.py` computes joint-noise indices on the 10-joint no-head list but applies them to the 14-actuator vector. Head joints get leg noise; right hip roll/pitch, knee and ankle get none. `training_obs_noise_scale()` reproduces this, and a test recomputes it from upstream `constants`.
- **Methodology correction during the run:** the first smoke test showed trials differing only in initial pose converge to the same limit cycle (std about 1e-4), which would make any "consistent in N/N trials" claim vacuous. Added training-level observation noise as the source of variation.

### Result (`experiments/expressive_locomotion/results/20260928-184504`, generated from clean commit d41d8b5; about 100 s wall time; two further reruns reproduced both CSVs byte for byte)
- 10 paired trials per style, forward command 0.15 m/s.
- **No falls while walking.**

| | Style −1 | Neutral | Style +1 |
|---|---|---|---|
| speed | 0.055 m/s | 0.107 | 0.124 |
| cadence | 2.60 steps/s | 3.70 | 4.82 |
| stride | 0.042 m | 0.058 | 0.051 |
| foot lift L | 12.2 mm | 14.1 | 11.4 |
| torso pitch | 4.5° | 1.8° | 1.1° |
| roll std | 3.0° | 2.6° | 1.9° |
| power | 5.9 W | 10.5 | 10.0 |
| cost of transport | 5.2 | 4.7 | 3.9 |

- All of these differed from Neutral in the same direction in 10/10 paired trials.
- **Pushes:** falls at 0.6 m/s are 1 / 2 / 2 of 8, and at 0.9 m/s 4 / 6 / 6 of 8. Not distinguishable (Fisher exact p ≈ 0.5).

### Interpretation (measured vs not)
- **Measured:**
  - One variable changes the gait consistently and stays stable.
  - Stride and lift peak at Neutral, the trained clock, and fall off in both directions, so style is not a single "bigger/smaller" axis.
  - Speed is strongly coupled to style.
- **Not established:**
  - Any emotional reading. That needs human evaluation.
  - That the differences are style rather than speed effects.

### Next smallest step (proposed)
Speed-matched comparison. For each style, adjust the commanded vx so the **measured** speed matches a common target, then compare posture, sway, lift and effort. This separates "how" from "how fast".

---

## 2026-09-28 — Phases 3–4: NERVA package and layer interfaces; Phase 5 probe

### Structure (Phase 3)
- Added `pyproject.toml` and the package `nerva/`, which has **no dependencies**, so the interfaces import and test without MuJoCo, JAX or Open Duck.
- Installed editable into the Open Duck venv along with `pytest` 9.1.1: `pip install -e ".[dev]"`.
- **Decision:** a flat package instead of the suggested `affect/`, `appraisal/`, `behaviour/`, `control/`, `evaluation/` subpackages. Each would currently hold one dataclass or nothing. A layer gets a subpackage once it has real code.
- Added `docs/architecture.md` (layers, what exists, Open Duck/NERVA boundary, safety rule, claim labels) and `docs/research_questions.md` (RQ1 active; RQ2–6 recorded, inactive).

### Interfaces (Phase 4)
- `nerva/interfaces.py` holds frozen dataclasses with range checks:
  - `Event` / `PerceptionState`
  - `AppraisalState` (relevance, desirability, likelihood, expectedness, controllability)
  - `PADState`
  - `ExpressiveStyle` (labels "Style -1" / "Neutral" / "Style +1" only)
  - `BehaviourCommand` (vx, vy, yaw_rate, skill ∈ {"walk"}, style)
- No appraisal, affect dynamics or behaviour logic. Ranges are NERVA conventions and are documented as such.
- `pytest`: **14 passed**.

### Exploratory probe for Phase 5: the gait-phase clock
This was run from a scratch script, not committed; Phase 5 will reproduce it properly.
- **Setup:** upstream `MjInfer.run()`, forward command 0.15 m/s, raw accelerometer, 20 s per setting, steady state over the last 10 s. Changed only `phase_frequency_factor`, the rate at which the policy's observed phase clock advances.

| factor | clock Hz | foot-height dominant freq (FFT) | fwd speed | falls |
|---|---|---|---|---|
| 0.6 | 1.11 | 1.10 Hz | 0.022 m/s | no |
| 0.8 | 1.48 | 1.50 | 0.077 | no |
| 1.0 | 1.85 | 1.90 | 0.110 | no |
| 1.2 | 2.22 | 2.20 | 0.117 | no |
| 1.4 | 2.59 | 2.60 | 0.126 | no |

- **Cadence follows the clock closely.** Speed rises with the factor and saturates above 1.0.
- An earlier 0.8/1.0/1.2 run also showed:
  - effort (mean Σ τ²) of 9.1 / 12.9 / 13.6
  - max tilt of 7.0° / 5.3° / 4.1°
  - left-foot height range of 16.3 / 16.2 / 13.9 mm
- **Measurement lesson:** counting foot-contact on/off transitions gave nearly the same "step rate" at 0.8 and 1.0 because contact chatters. Cadence must be measured from foot height (or joint angles), not raw contact switches.
- **Caveat:** training always advanced the clock at factor 1.0 (`joystick.py:326`), so other factors are outside the training distribution. The probe used flat ground with no pushes, and stability under disturbance at other factors is untested.
- The speed in this probe is world-x displacement, which is close to heading-frame speed because yaw drift is small. Phase 5 uses the heading frame.

---

## 2026-09-28 — Phase 2: understanding the baseline

Wrote `docs/open_duck_baseline.md` from the upstream source code, the compiled MuJoCo model and the ONNX file. Nothing upstream was modified.

### What was checked directly (not just read)
- **Effective actuator parameters come from the compiled model:** kp 13.37, ±3.23 N·m, damping 0.56. `xmls/joints_properties.xml` (kp 17.8) is not included by any model and is stale.
- **The ONNX graph** has `onnx` installed to a scratch folder only, not the project venv. The network is 101→512→256→128→28 with swish activation, normalisation built into the graph, and `tanh` on the means. That matches the Berkeley Humanoid PPO config the runner uses.
- **Reference-motion pickle:** 240 gaits on a 6×4×10 grid (vx, vy, ωz), 40 signals each, degree-15 polynomials, period 0.54 s (27 policy steps), nearest-neighbour lookup.
- **Reference-motion speeds:** the reference gaits' own forward speed matches their command (0.155 m/s for the 0.148 gait), so the reference is not why the policy under-tracks forward.

### Finding: accelerometer offset mismatch
- `joystick.py:502` (training) intends to add 1.3 to accelerometer x, but uses `.at[0].set()` without assigning the result. In JAX that's a no-op, so training used the raw value.
- The hardware runtime also uses the raw value (`tare_x()` is disabled).
- Only `mujoco_infer.py:74` adds +1.3.
- Added `--raw-accel` to `scripts/check_open_duck_baseline.py` to cancel it (upstream untouched). Result, 20 s per command:
  - forward 0.097 → **0.112 m/s**
  - lateral 0.093 → 0.102
  - backward and turn unchanged
  - no falls
- Conclusion: a real but minor effect. **NERVA experiments will use the raw accelerometer.**

### Command tracking, current understanding
- **Lateral:** explained. The reward's 0.1 m/s dead-band plus a reference grid limited to ±0.111 m/s means about 0.10 m/s earns full reward.
- **Forward vs turning:** a likely explanation is that the absolute `tracking_sigma = 0.01` enforces m/s errors much more weakly than rad/s errors. That's inferred, not tested.
- **Backward:** about 21%, unexplained.

### For Phase 5 (noted, not acted on)
The phase clock `[cos φ, sin φ]` is the only trace of the reference motion at runtime. Its rate is already adjustable (`P`/`;` in sim; LB and a per-robot offset on hardware). That's a zero-retraining candidate for option A (gait-parameter modulation), but its effect on the gait hasn't been measured yet.

---

## 2026-09-28 — Phase 1: Open Duck Mini v2 baseline running in MuJoCo

### Machine
- Windows 11 Home, Intel i7-1355U (10 cores), 15.6 GB RAM, **Intel Iris Xe only (no NVIDIA GPU)**.
- Python 3.12 and 3.13 installed (`py -0`). 37 GB free on C:.
- Windows long-path support enabled by the user (`LongPathsEnabled = 1`, verified).

**Consequence:** Open Duck *training* uses JAX + MJX on CUDA (`jax[cuda12]` in the Playground `pyproject.toml`). CUDA JAX does not exist for native Windows and this machine has no NVIDIA GPU, so training cannot run here. Inference (running a trained ONNX policy in CPU MuJoCo) works fine. Training will need a Linux + NVIDIA machine (cloud, lab PC or Colab). That decision is deferred until we actually need to train.

### Upstream code used (unmodified)
Cloned under `<OPEN_DUCK_ROOT>`, outside OneDrive (OneDrive sync interferes with git repos, LFS meshes and venvs).

| Repo | Branch (default) | Commit |
|---|---|---|
| apirrone/Open_Duck_Playground | main | b9be205 (2025-08-05) |
| apirrone/Open_Duck_Mini | v2 | b23317a (2026-01-31) |
| apirrone/Open_Duck_Mini_Runtime | v2 | 376de65 (2026-09-08) |
| apirrone/Open_Duck_reference_motion_generator | main | 3d7bc6f (2025-04-03) |

- Policy: `Open_Duck_Mini/BEST_WALK_ONNX_2.onnx` (uploaded 2025-04-02). This is the policy the hardware runtime README tells users to deploy.
- Model: `Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml` (Open Duck Mini v2 MJCF).
- `git status` in all upstream repos is clean.

### Environment
Venv: `<OPEN_DUCK_ROOT>/.venv` (Python 3.12). Exact versions: `env/open_duck_inference.lock.txt`.

Setup steps, reproducible:
```bash
py -3.12 -m venv <OPEN_DUCK_ROOT>/.venv
<OPEN_DUCK_ROOT>/.venv/Scripts/python -m pip install -r env/open_duck_inference.lock.txt
<OPEN_DUCK_ROOT>/.venv/Scripts/python -m pip install -e <OPEN_DUCK_ROOT>/Open_Duck_Playground --no-deps
```

Decisions and why:
1. **Did not use `uv sync`** (the upstream-recommended installer). The upstream deps include `jax[cuda12]`, `tensorflow` and `tf2onnx`, which are for training and ONNX export only, and the CUDA wheel cannot install on Windows. Installed only what the inference path imports (checked file-by-file): `playground` (DeepMind MuJoCo Playground), `mujoco`, `mujoco-mjx`, `jax` (CPU), `onnxruntime`.
2. **Pinned to the era of the Duck code**, not the newest packages. The upstream `pyproject.toml` says `playground>=0.0.3` with no lockfile, so pip resolves today's 0.2.0 (2026-03), a year newer than the Duck env code. Pinned instead to `playground==0.0.4` (2025-03-07, when `base.py` was last changed), `mujoco==3.3.0` (2025-02-27, before the policy was uploaded), `jax==0.5.3`, `onnxruntime==1.21.0`. **Not pinned to that era:** `numpy` (2.5.3), `brax`, `flax`, `orbax-checkpoint`. They are transitive deps, and the inference path works with them. Revisit if physics or behaviour discrepancies appear.
3. **Editable install of Open_Duck_Playground with `--no-deps`.** Equivalent to what `uv run` does. Without it, `mujoco_infer.py` fails with `ModuleNotFoundError: No module named 'playground'`. The two similarly named packages don't collide: the Duck repo provides import name `playground`, and DeepMind's distribution `playground` provides import name `mujoco_playground`.
4. On first import, `mujoco_playground` 0.0.4 git-clones the whole of `mujoco_menagerie` (about 1.7 GB, commit 14ceccf) into `site-packages/mujoco_playground/external_deps/`. It takes several minutes and happens once per venv. Open Duck doesn't use menagerie models; the clone is just a side effect of the import.

### Problems hit (and causes)
- `pip install playground` failed with `OSError: No such file or directory` on a very long path. Cause: `orbax-checkpoint` ships test fixtures deeper than Windows' 260-character `MAX_PATH`. Fixed properly by the user enabling long paths. It was not worked around in the final environment.
- `ModuleNotFoundError: playground` when launching the upstream script: see decision 3.
- `cp -r src dest/` put the copied menagerie contents one level too high because `dest/` did not exist yet. Corrected, and the commit was verified.

### What ran
1. **GUI (upstream, unmodified):**
   ```bash
   cd <OPEN_DUCK_ROOT>/Open_Duck_Playground
   ../.venv/Scripts/python playground/open_duck_mini_v2/mujoco_infer.py -o ../Open_Duck_Mini/BEST_WALK_ONNX_2.onnx
   ```
   The MuJoCo viewer opened and rendered the duck standing on flat ground (screenshot checked). Keys in `key_callback` (`mujoco_infer.py`): arrows for forward/back/lateral, `Q`/`E` to turn, `H` to toggle head-control mode, `P`/`;` to raise/lower the gait phase frequency. These are GLFW keycodes 81/69/72/80/59. The upstream comments say "a" and "m" because the author uses an AZERTY keyboard; on QWERTY the keys are Q and ;. A keypress latches the command; it doesn't reset on release.

2. **Headless check** `scripts/check_open_duck_baseline.py`. It calls the upstream `MjInfer.run()` loop unchanged, with the viewer replaced by a recorder. 20 s simulated per command, with steady-state velocity measured over the last 10 s in the robot's heading frame:

   | Command | Fell | Max tilt | Base height | Measured |
   |---|---|---|---|---|
   | zero | no | 6.3° | 0.150–0.167 m | 0.000 m/s |
   | fwd 0.15 m/s | no | 5.0° | 0.150–0.169 m | fwd +0.097, lat +0.017 m/s |
   | back 0.15 m/s | no | 5.1° | 0.150–0.169 m | fwd −0.032 m/s |
   | left 0.2 m/s | no | 4.2° | 0.150–0.168 m | lat +0.093 m/s |
   | turn 1.0 rad/s | no | 4.0° | 0.150–0.169 m | yaw +0.81 rad/s |

   Two runs gave identical numbers, so the simulation is deterministic.

### Observations (measured, not interpreted)
- Stable in every tested command: no falls, tilt ≤ 6.3°.
- **Command tracking is incomplete:** about 65 % of the forward command, about 21 % of the backward command, about 47 % of the lateral command and about 81 % of the yaw command. We don't yet know whether this is expected for this policy, a sim-version effect, or an observation mismatch. Worth understanding in Phase 2 before any style experiment uses velocity as a metric.
- Policy loop: `sim_dt = 0.002 s`, `decimation = 10`, so the policy runs at 50 Hz. Observation dimension 101, action dimension 14 (ONNX I/O `obs[1,101] → continuous_actions[1,14]`).

### Follow-up: "I didn't see it move"
On the first GUI launch the user saw the duck standing still. Cause: the upstream script starts with a **zero command** (`self.commands = [0]*7`), so the policy stands and sways slightly until a key press reaches the viewer window. The window must have focus: click it, then press ↑.

To rule out other explanations, I ran a diagnostic (the upstream `MjInfer.run()` with the real viewer and `commands[0] = 0.15` preset):
- The duck walked **1.44 m in 20 s wall-clock** in the GUI. The user confirmed visually that it walks.
- **The GUI runs at about 0.75× real time on this laptop.** The upstream loop calls `viewer.sync()` after every 2 ms physics step, and its sleep only compensates when a step is faster than real time. This affects viewing speed only, not the physics. Headless runs are unaffected.
- The GUI velocity matched the headless measurement (about 0.097 m/s simulated), so the viewer and headless runs behave the same.

### Not yet verified
- Whether results would differ under the newest `playground`/`mujoco` versions.
