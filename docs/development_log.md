# NERVA development log

Newest entry first. Each entry records what was done, what was actually run, and what is still unverified.

---

## 2026-10-01 — S4 (backward emphasis): hypothesis refuted; cloud torn down

**S4:** `s4_pilot-20261001-021916` = S3 + 30% backward-emphasis commands, 300 M steps, exit 0.

**Mean forward velocity, 2 seeds, 15 s, neutral style [measured]:**

| policy | vx = −0.15 | −0.10 | +0.15 |
|---|---|---|---|
| S1 | −0.042 | −0.016 | +0.113 |
| S3 | −0.001 | +0.000 | +0.095 |
| S4 | −0.007 | −0.001 | +0.075 |

**Reading:**
- More backward episodes did **not** improve backward walking. S3 and S4 both walk backward far worse than S1, which suggests that training with head commands applied cost the backward gait [hypothesis].
- The cause of weak backward walking in all policies stays unknown. Candidates not tested: the tracking reward's σ, the stand-still cost, and the imitation weight of backward gaits.
- The fear retreat therefore remains limited. S1 stays the policy used in the reactive stage.

**Teardown [measured, `launch.py audit`]:** no instances, disks, addresses, forwarding rules, routers or snapshots. The results bucket and the runner service account remain.

---

## 2026-10-01 — Memory M2–M4: episodic store, sleep consolidation, spatial memory

**M2/M3 — episodic memory** (`nerva/episodic.py`, `14b4ef1`):
- **Salience-gated encoding:** relevance × (arousal, |prediction error|, novelty).
- **ACT-R base-level activation:** power-law decay; retrieval strengthens [tested: slope −0.5 on log–log].
- **Cue retrieval:** the entity cue is a filter. A vivid memory of B used to answer a cue for A; fixed.
- **Sleep consolidation,** run when no one has been in view for 5 s, at most every 20 s:
  - prioritised replay of emotionally significant episodes (intensity ≥ 0.3), which nudges the entity association at 0.2× the normal learning rate
  - gist merging of repeats
  - pruning of low-activation consolidated episodes
- **Persistence:** SQLite.
- **Two-person run [measured]:** 16 episodes plus 1 routine moment; two sleeps (the 6 touches merged into one gist, count 6); A's lunge is A's strongest memory.
- **Replaying low-intensity episodes diluted B's warmth** (0.19 → 0.08), hence the intensity threshold.
- **Tests:** a long run of 20,000 events stays within a 500-episode capacity at under 5 ms per event.
- **Ambiguous identities:** one seed had created a phantom third person from a partial view. A new identity is now created only when cosine < 0.5 to all known people; in between counts as ambiguous, and the track keeps its identity.
- **Evaluation** (5 seeds, vision) with M1–M4: memory 5/5 on all four criteria; no memory 0/5 on "B not blamed" and "A remembered".

**M4 — spatial memory** (`nerva/spatial.py`):
- 0.75 m place grid: familiarity (fades with time), place threat/valence learned from events there, and an exploration heading toward the nearest novel, safe cell.
- **Exploration only, 90 s, 3 seeds [measured]:** 9/10/11 cells visited with spatial memory vs 7/9/8 without (about +25%), limited by walking speed.

---

## 2026-10-01 — Real vision and entity memory (M1): person-specific, evolving associations

**Real vision** (`nerva/vision.py`, `a23a078`):
- Colour segmentation of the robot_eye RGB frame (HSV) → connected components → depth render gives distance → bearing/elevation from the pixel rays. The same tracker as the simulated detector.
- **On renders [measured]:** distance within 0.04–0.09 m (it measures to the surface rather than the centre); bearing within about 0.02 rad.
- **Self-body false positive:** the robot's orange feet were detected as a ball when looking down. Small objects closer than 0.2 m are now rejected.
- **Reactive 5-seed evaluation with vision:** identical verdicts to simulated perception (curiosity 5/5, habituation 5/5, safety 5/5, fear 0/5).

**Memory M1** (`nerva/memory.py`, `MemoryAppraiser`, `IdentityBinder`; design: `docs/memory_design.md`):
- **Per-identity records:** familiarity, threat/warmth/trust learned by prediction error with arousal-scaled rate, slow drift in absence.
- **Identity** comes from a clothing-colour histogram, which is None when uninformative.
- **Measured on renders:** A vs B far away, cosine 0.33; up close (trousers only) 0.99, indistinguishable. So identity stays bound to the track, and events with an unknown identity are held back and learned once resolved (confidence 0.8).
- **Touch:** gentle petting is simulated as contact events and appraised as pleasant.

**Two-person scenario, 5 seeds, vision, memory ON vs OFF [measured]:**

| criterion (stated in advance) | memory | no memory |
|---|---|---|
| B not blamed for A's lunge (46–58 s) | 5/5 | 0/5 |
| A still feared on return 70 s later (94–106 s) | 5/5 | 0/5 |
| B welcomed back (120–130 s) | 5/5 | 5/5 |
| petting credited to B (warmth B > 0 > A) | 5/5 | — |

- Without memory, the global 45 s threat timer makes the robot wary of innocent B and then forget A.
- Final records (seed 0): A threat 0.18, warmth −0.17; B threat 0, warmth +0.19 to +0.32.
- Worst tilt 7.4°.
- The demo video was rendered locally (it's local only).

**Limits:**
- One person per class may be tracked at a time (the tracker is keyed by class).
- Appearance is clothing colour, not faces.
- A's fear partly re-generates itself: each sighting elicits fear from memory, and fear is learned again. Extinction only happens through positive encounters.
- Episodic store, consolidation and spatial memory (M2–M4) are not implemented.

---

## 2026-10-01 — Behaviour v2 (utility arbitration), S3 (head commands in training), backward-walking diagnosis

**Behaviour v2 (`nerva/action_selection.py`):**
- Emotion-modulated action selection replaces the if/else rules. Each action's utility comes from drives (interest, hope+joy, fear, surprise, distress), per-target novelty from the appraiser, and proximity, plus persistence and committed freeze/back-step.
- The weights are hand-set design choices. It is not learned.
- 5-seed reactive evaluation with S1: curiosity 5/5, habituation 5/5, safety 5/5, **fear 0/5**. The robot retreats, but gains no distance within 3 s.

**S3** (`s3_pilot-20260930-235950`, `--apply_head_commands`, 300 M steps, exit 0) [measured]:
- **Walking with head offsets (vx 0.15, 15 s):**
  - Yaw 0.2 → 0.089 m/s (S1: 0.012); yaw 0.3 → 0.078; yaw 0.4 → 0.015; yaw 0.5 → steps in place.
  - Pitch +0.4 → 0.084 (S1: 0.027); pitch −0.2 → 0.083; pitch −0.4 → 0.014.
  - About three times S1's head range, but not the full range sampled in training.
- **Style evaluation (`s3_eval`, same §5 protocol):**
  - At the trained values, gait frequency still follows tempo: 1.52 / 1.85 / 2.50 Hz. At the untrained e1 = −0.5 it reads 3.17 Hz (harmonic or shuffle, unchecked).
  - Torso pitch is only monotonic from 0 upward (0.42 / 0.22 / 0.79 / 1.65 / 3.31°).
  - Tracking is worse: mean 0.071, worst condition 0.141, against B0's 0.053.
  - **FAIL.** S3 trades style fidelity for head tolerance.
- Reactive scenario with S3 and a wider walking-head limit: same verdicts as S1 (fear 0/5).

**Backward walking [measured]:**
- At vx −0.15 m/s commanded: B0 −0.036, S1 −0.043. Forward 0.15: 0.096 / 0.114.
- The references aren't the cause. The shipped backward reference moves at −0.115 m/s at the −0.148 key; forward is +0.155.
- Hypothesis: too few backward episodes in training. Test prepared as **S4** (S3 + 30% backward-emphasis commands via a derived key; test added). Launch failed: no L4 capacity in us-central1-a/b/c at the time.

---

## 2026-10-01 — Reactive stage 1: scene, perception, contextual appraisal, interest, behaviour modes

**User decision:** the S1 demo's style differences were too small to see. Make the robot genuinely react to its environment through the emotion model, with complex behaviours (curiosity, fear/stepping back). Stay in MuJoCo for now; Isaac Sim later. Design: `docs/reactive_behaviour_design.md`.

**Built** (commits `d54b7fd` … `33e7475`; 112 tests pass):
- **Scene** (`nerva/world.py`):
  - Visual-only mocap person (1.7 m) and ball, added via MjSpec, plus a `robot_eye` camera on the head, 0.09 m in front of the head site (inside the shell the view was blocked).
  - The robot's dynamics are bit-identical to the plain scene. This needed the solver warm start copied over, because `opt.iterations = 1`.
- **Perception** (`nerva/perception.py`):
  - A simulated head-camera detector: field of view and range, a distance-dependent miss rate, bearing/distance noise.
  - Tracks with memory. Approach speed is a least-squares slope with **ego-motion compensation**; without it, walking toward someone read as them approaching.
  - Separate re-arm flags for slow and rapid approaches (a slow event had masked a lunge).
- **Contextual appraisal** (`ContextualAppraiser`): novelty habituation, proximity and speed of approaches, threat memory (45 s), relief when a threat leaves, habituation to repeated lunges.
- **Affect v0.2:** a new emotion, "interest" (novel and non-harmful). The rule and anchor are NERVA choices. Events without novelty elicit exactly what v0.1 did (existing tests unchanged).
- **Behaviour v1** (`nerva/reactive_behaviour.py`): explore / orient / approach / inspect / freeze / retreat / watch / withdraw, with hysteresis and head gaze.
- **Scenario and renderer:** `experiments/reactive/`, rendered in the cloud (`cloud/jobs/reactive_demo.sh`).

**Scenario result, S1 policy, seed 0 (simulation) [measured]:**
- Ball appears → surprise and interest → approach → inspect (about 16 s).
- Person approaches slowly → hope and interest → the robot walks up to them.
- Lunge → fear 0.57, surprise 0.77 → **freeze → retreat**.
- Person out of sight → relief (joy).
- The person's return within the threat memory → the same slow approach is appraised as threatening → **watch** (wary, keeps distance) → habituates → explore.
- Worst tilt 7.7°, no fall.

**5-seed evaluation** (`experiments/reactive/evaluate.py`, design §5 criteria, S1 policy) [measured]:
- **Curiosity 5/5:** ball distance 1.29 → 0.62 m.
- **Habituation 5/5:** peak interest at the person's 1st vs 2nd appearance, 1.07 vs 0.20–0.33.
- **Safety 5/5:** worst tilt 7.8°.
- **Fear 1/5 — fails the stated criterion.** A freeze and a retreat occur in every seed, but the distance 3 s after the lunge doesn't grow (0.45 → 0.42–0.45 m). The backward step is too slow (the policy walks backward at about 21% of command), and the turn-away only starts after 1.5 s. The robot gets away over about 10 s.
- The seeds differ very little, so perception/initial-state noise barely changes the outcome.

**Findings along the way [measured]:**
- **Head offsets stop S1 from walking.** At vx 0.12, 10 s: head yaw 0.2 rad → 0.012 m/s (0.05 without); pitch/roll 0.2 → about 0.05–0.074 m/s; beyond 0.3 rad it nearly stops.
  - Cause [fact, upstream `joystick.py`]: training puts random head commands in the observation but never applies them to the head motors (the line is commented out). The hardware runtime does apply them.
  - Interim: the behaviour limits head offsets while walking and gazes fully only when standing.
  - Fix in training: **S3** (`--apply_head_commands`), run `s3_pilot-20260930-235950`, in progress.
- **Head-pitch sign.** On the robot_eye camera, positive `head_pitch` tilts the face **up** (+0.4 → 35° up, −0.4 → 14° down), while the head position barely moves. `experiments/expressive_locomotion/head_posture.py` (RQ1c) states the opposite ("negative head_pitch raises the head; verified by rendering"), so **RQ1c's head-up/head-down labels may be inverted**. Not yet re-checked visually.
- **The backward references are fine.** For straight gaits, the R1 neutral recordings achieve −0.10 to −0.12 m/s at key −0.148. The earlier side note (key −0.148, achieved +0.014) came from turning gaits. So the policy's weak backward walking (21% of command) has a different, unknown cause.
- **Small velocity commands barely move the policy,** so approaches use full speed or nothing.

---

## 2026-09-30 — Affect connected to S1: behaviour layer v0 and S1 demo video

**Decision (user):** adopt S1 (tempo and torso pitch working), park step height as an open problem, and connect affect.

**Code (`190df97`):**
- `nerva.interfaces.StyleVector` (e1, e2, e3). `BehaviourCommand.style_vector` is exclusive with the method-A phase-clock style.
- `OpenDuckSim.set_behaviour` applies it. When e1 changes the period, the phase *fraction* is kept, a deployment choice so a continuously varying tempo doesn't jump the clock.
- `nerva/behaviour.py`:
  - HOW: e1 = clip(3·A), e2 = 0, e3 = clip(−1.5·(V + D)).
  - WHAT: walk at 0.13 m/s; stop while the goal is blocked.
  - Directions are informed by emotional-gait studies. The gains are hand-chosen so the demo's PAD range spans S1's trained range. **Not validated.**
- `experiments/demo_video/run_s1.py` reuses run.py's scenario and renderer, which is now parametrised.
- 93 tests pass.

**Demo run.**
- Simulation only on the laptop (about 20 s of CPU) to check the loop. The render ran in the cloud: `demo_s1-20260930-204728`, CPU VM, OSMesa, 1,650 frames, exit 0, VM deleted, NAT removed.
- Results (simulation) [measured]:
  - No fall; maximum tilt 7.6° (the push at 27 s was recovered).
  - Fear phase (28–38 s): mean e = (+0.76, 0, +0.91), torso pitch +3.3°, speed 0.063 m/s.
  - Calm walking: e3 ≈ −0.28, pitch −0.3°, 0.095 m/s.
  - Blocked (standing): e3 +0.72 but pitch −0.3°, so lean is barely expressed when standing.
- Combined and intermediate styles are outside S1's training distribution; this is one scenario, not a test.
- Video and keyframes are local, git-ignored: `experiments/cloud_runs/demo_s1-20260930-204728/demo_s1/`.

---

## 2026-09-30 — S2 result: the feet-height cost lifts feet in B2 but degrades the multi-style policy; S1 remains the best style policy

**Runs.**
- `b2_neutral-20260930-163257` and `s2_pilot-20260930-163509` (code `991fd6b`): 300 M steps each, both exited 0. The reward curves were slightly below B1/S1, as expected from an added cost.
- Evaluation `s2_eval-20260930-184340` (code `4d6417f`), same protocol and seeds as S1. In the report files, "S1" means S2 and "B1" means B2.
- Network cleanup: the first router deletion failed ("in use" right after the NAT was deleted). It succeeded on manual retry, and `network-down` now retries.

**Results [measured]** (means over 10 seeds; S1 values in brackets):

| e_k | −1 | −0.5 | 0 | +0.5 | +1 | monotonic |
|---|---|---|---|---|---|---|
| e1 → gait frequency, Hz | 1.95 [1.52] | 2.49 [1.67] | 1.85 | 2.17 | 2.50 | **0/10** [10/10] |
| e2 → lift, mm (target 20–60) | 14.1 [12.9] | 17.8 | 19.5 [14.1] | 19.6 | 20.1 [13.7] | **1/10** [0/10] |
| e3 → torso pitch, ° | −1.25 | −1.07 | −0.48 | 0.42 | 1.37 | **10/10** [10/10] |

**Neutral control.** B2 lifts **30.1 mm**, against 13.3 for B0 and 14.4 for B1, with tracking error 0.057 against B0's 0.053. **For a single style the term works.**

**The multi-style policy:**
- **Step height:** S2 lift now rises with e2 on average (14 → 20 mm), but not reliably per seed. It stays far below the targets and below B2's neutral lift.
- **Tempo broke.** Slow tempos no longer slow the gait. e1 = −0.5 reads 2.49 Hz: whether that is a harmonic picked by the FFT or a real fast shuffle is unchecked.
- **Torso pitch** still works, with a smaller range: 2.6° against S1's 3.9°.
- **Tracking** is worse: mean 0.070 against B0's 0.053, worst condition 0.090.
- **Falls:** 0/170.

**Verdict: FAIL (preregistered).** S2 is worse than S1 on e1 and on tracking. S1 (e1 and e3 working) remains the best style policy.

**Interpretation [hypothesis]:** with 7 styles, the feet-height cost at weight −30 competes with the per-style clock and tracking. B2 shows the term itself is learnable. A lower weight, a style-dependent schedule, or more steps might help, but each attempt costs about 2 × 2 h L4 and wasn't tried.

---

## 2026-09-30 — S2 preregistration: per-style feet-height cost; B2 + S2 launched

**Change [NERVA design choice]:** `StyleJoystick(feet_height_scale=...)` adds

`feet_height = Σ_feet ((swing_peak − 0.003 m) / walk_foot_height(e2) − 1)² · first_contact`

- It has the form of MuJoCo Playground's Berkeley Humanoid `_cost_feet_height`.
- The target is the style's R1 foot height (20 / 40 / 60 mm).
- 0.003 m is the median stance foot-site height, measured in MuJoCo with the B0 policy.
- With scale 0 the reward is unchanged, so S1/B1 are reproducible. Tests cover the targets, the cost and the config key (87 passed).

**Weight −30, fixed before training.** B0 per-step episode means are: alive 20, tracking_lin 1.57, tracking_ang 1.37, imitation 1.51, action_rate −0.56. At the measured under-lift (about 14 of 40 mm, ≈0.074 touchdowns per step), −30 costs about 0.9 per step. That is comparable to, but below, each tracking term.

**Runs:**
- **B2** (`b2_neutral`): B1 + the term, shipped neutral references.
- **S2** (`s2_pilot`): S1 + the term, the 7 R1 styles.
- Both: 300 M steps, L4, 3 h cap.

**Evaluation (unchanged §5 criteria):**
- The cross-talk normaliser is now the within-condition std pooled over all S-policy conditions, floored at the metric resolution (gait frequency: FFT bin 0.0083 Hz).
- Re-analysing S1 with it leaves every S1 verdict unchanged.
- Tracking is compared against B0 as before. B2 is also reported, because the reward change may alter tracking.
- **Additional pre-stated check for S2:** achieved lift vs the reference target (20 / 30 / 40 / 50 / 60 mm).

---

## 2026-09-30 — Why e2 (step height) failed: all Open Duck policies under-lift; reward has no foot-height term

**The metric is right [measured].** Replaying the R1 reference *joint* trajectories kinematically (gait `0.148_0.037_-0.074`, base fixed) and applying our `lift_height` to the MuJoCo foot site gives:
- 21.7 / 42.5 / 63.4 mm for e2 = −1 / 0 / +1
- Placo's own toe trajectory in the recordings gives 21.7 / 42.6 / 63.5 mm
- `walk_foot_height` is 20 / 40 / 60 mm

So a policy that tracked the reference joints would read about 42 mm at neutral. B0, B1, S1 and the shipped `BEST_WALK_ONNX_2` all read **13–14 mm**. Under-lifting is a property of upstream-style training, not of the NERVA env.

**The references differ enough [measured].** Between e2 = −1 and +1 the reference knees differ by up to 0.60 / 0.69 rad, and the mean Σ(Δq²) over leg joints is 0.19 rad². That is the same order as e3 (0.27 rad²), which *was* learned.

**Upstream reward structure [fact, `joystick.py`, `custom_rewards.py`]:**
- The top-level scales are alive 20, tracking_lin_vel 2.5, tracking_ang_vel 6, action_rate −0.5, torques −1e-3, stand_still −0.2, **imitation 1.0**.
- Inside imitation, leg joint positions carry `−15·Σ(Δq²)`, and there is an explicit torso-orientation term.
- **The toe-position terms are commented out.** Nothing rewards foot height directly.
- `swing_peak` (maximum foot-site z per swing) is tracked but not rewarded.
- MuJoCo Playground's Berkeley Humanoid env has exactly such a term, `_cost_feet_height = Σ (swing_peak / max_foot_height − 1)² · first_contact`, disabled (scale 0) in its default config [fact].

**Interpretation [hypothesis]:**
- Torso pitch (e3) has its own reward term plus a near-constant hip offset, and it was learned at about 1/3 magnitude.
- Swing height only shows up through small, transient knee errors that PPO trades against alive/tracking/action-rate. Under the motor speed limit and action-rate cost, a larger swing is costly.

**Possible fix (not run; a design change needing a new baseline) [NERVA design proposal]:**
- Add a per-style feet-height term using the existing `swing_peak`, with target = the style's `walk_foot_height` above the stance foot-site height.
- Retrain a baseline B2 (neutral, same term) and S2 (7 styles), with the same evaluation and a normaliser for cross-talk fixed in advance.
- Cost is about 2 × 2 h L4.

---

## 2026-09-30 — S1 evaluation: tempo and torso pitch work; step height does not; preregistered verdict is FAIL

**Run.** `s1_eval-20260930-013427` (code `70191ab`), e2-standard-8, 170 trials in 77 s of compute. Final ONNX of each run; design §5 protocol (vx = 0.15 m/s, 10 paired seeds, 20 s, metrics over 5–20 s).
- Two earlier attempts crashed before any trial ran:
  - parallel workers raced to clone mujoco_playground's menagerie
  - B1 expects the 104-value observation (it is a StyleJoystick policy with e = 0)
- Both are fixed, with an up-front input-size check and unit tests of the analysis.

**Results [measured]** (means over 10 seeds):

| e_k | −1 | −0.5 | 0 | +0.5 | +1 | monotonic (10 seeds) | reference target |
|---|---|---|---|---|---|---|---|
| e1 → gait frequency, Hz | 1.52 | 1.67 | 1.85 | 2.17 | 2.50 | **10/10** | 1/period = 1.48 / 1.85 / 2.47 |
| e2 → foot lift, mm | 12.9 | 13.7 | 14.1 | 14.2 | 13.7 | **0/10** | foot height 20 / 40 / 60 mm |
| e3 → mean torso pitch, ° | −0.71 | −0.38 | 0.20 | 1.26 | 3.14 | **10/10** | trunk pitch −10 / −4 / +2 ° |

**Tempo (e1)**
- It works, and it is controllable: the achieved gait frequency matches the reference's 1/period within 0.04 Hz at the trained values.
- **Cross-talk [measured]:** mean foot lift falls about 4 mm at non-neutral tempos (9.6–10.2 mm at ±0.5 and +1).

**Step height (e2)**
- It is not expressed. Lift barely changes (range 1.3 mm, non-monotonic), while the reference spans 40 mm.
- Even at neutral, every policy lifts only about 13–14 mm against the 40 mm reference foot height. B0 does the same, so this is inherited from upstream training. Possible causes are the imitation-reward weighting, or the lift metric not measuring the same thing as `walk_foot_height` [hypothesis; both unchecked].

**Torso pitch (e3)**
- It works in direction, with about 1/3 of the reference magnitude: 3.9° achieved against a 12° span.

**Criteria (design §5)**
- Monotonic: e1 ✓, e2 ✗, e3 ✓.
- Cross-talk: e1 ✓, e2 ✗, e3 ✓.
- Falls: 0/170 ✓.
- **Tracking ✗:** the worst S1 condition is e1 = −0.5, with |v − 0.15| = 0.075 against the limit 1.25 × B0 (0.053) = 0.066. That value is **not a trained style** (its period is interpolated). The worst trained style (e3 = −1) scores 0.065 and passes. Neutral S1 tracks better than B0 (0.038 vs 0.053).
- **Overall: FAIL** by the preregistered rule. Reported as a negative result for e2, with e1 and e3 positive.

**Caveats**
- **The cross-talk normalisation I defined breaks for gait frequency.** The metric is quantised by the FFT resolution and identical across seeds at neutral, so its std is 0 and its normalised values are meaningless (≈10¹⁵). The verdicts don't depend on it: e1's raw frequency change is large, and gait frequency doesn't change for e2 or e3. Next time: a different normaliser, stated in advance.
- **Neutral S1 leans differently from B1 [measured]:** 0.2° vs 3.4° pitch, with the same neutral references. This is possible blending across styles [hypothesis].

---

## 2026-09-29 — B0, B1 and S1 pilot trained (300 M steps each, L4)

**S1 smoke** (`s1_smoke-20260929-224944`, code `8443fd4`) passed:
- all 7 styles loaded
- observation 104 = 101 + e
- per-style periods 0.54 / 0.675 / 0.405 s
- 575 s on the GPU, mostly compile

**Runs.** One g2-standard-8 (L4) each, 3 h cap, `flat_terrain_backlash`, 300,482,560 effective steps. All three exited 0 and self-deleted; the waiter removed the NAT and `audit` is clean.
- `b0_baseline-20260929-230717` (upstream runner, shipped references)
- `b1_neutral-20260929-230837` (StyleJoystick, **shipped** neutral references via `--neutral`; corrected 2026-09-30, earlier this said R1 neutral)
- `s1_pilot-20260929-231102` (StyleJoystick, 7 R1 styles)
- Wall time was about 1 h 59 min each, with 7,040–7,050 s in training. That is about 43 k steps/s, including 15 evaluations and exports, on the backlash task.

**Evaluation reward (upstream eval, one batch; std ≈ 150 across envs) [measured]:**

| steps (M) | 21 | 64 | 107 | 150 | 193 | 236 | 258 | 279 | 300 |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 216 | 244 | 271 | 301 | 282 | 261 | 310 | 305 | 249 |
| B1 | 208 | 248 | 284 | 281 | 287 | 273 | 297 | 298 | 250 |
| S1 | 212 | 248 | 265 | 277 | 282 | 233 | 289 | 290 | 224 |

**Reading:**
- B1 tracks B0 within the evaluation noise at every checkpoint. So the NERVA env reproduces upstream training with the same references [measured, reward only]. (Corrected 2026-09-30: B1 uses the shipped pickle, not R1 neutral; R1 neutral is used only inside S1.)
- S1 is slightly lower at most checkpoints, which is expected for a 7-style task [hypothesis].
- All three drop at the last checkpoint, which suggests evaluation variance rather than collapse [hypothesis].
- **Reward says nothing about whether styles are expressed.** That is the S1 evaluation (design §5), next.

**S1 evaluation tooling:**
- `OpenDuckSim.set_style_vector(e)` appends e to the observation (noise-free, as in training) and uses `nerva.style.s1_nb_steps_in_period(e1)` for the phase clock. The period is linear in e1, measured on the 3 R1 periods; values between them are interpolated.
- `experiments/style_policy/evaluate.py` applies the §5 sweep and criteria. Cross-talk "normalised units" had not been defined; before any results, I defined them as that feature's across-seed std at neutral S1.
- The evaluation runs in the cloud (`cloud/jobs/s1_eval.sh`, CPU). One 6 s trial takes about 1.5 s locally.

---

## 2026-09-29 — R1 pilot complete: 7 validated style reference sets

Run `r1_references-20260929-210511` (code `084806b`), e2-standard-8, 4,516 s total (618–679 s per style). The VM self-deleted; I removed the NAT and `audit` is clean. The local `launch.py wait` process died without output (exit 4, cause unknown), so the NAT was left up for about 16 min after the job ended.

**Results [measured]:**
- All 7 styles generated 240/240 recordings and fitted 240 keys.
- Every pickle passes an independent re-validation.
- All 7 load into `StyledReference` (grid 6×4×10, 40 signals, degree 15). Gait periods: 0.54 s, 0.675 s (e1−) and 0.405 s (e1+).

| style | changed parameter | gaits substituted (invalid after 2 regenerations) |
|---|---|---|
| neutral | — | 2 |
| e1− / e1+ | single support 0.225 / 0.135 s | 2 / 1 |
| e2− / e2+ | foot height 0.02 / 0.06 m | 3 / 1 |
| e3− / e3+ | trunk pitch −10° / +2° | **4** / 2 |

**Observations:**
- **Where the failures are [measured]:** every substituted gait sits at the maximum forward step with the maximum lateral step, mostly with large turning. **No regeneration on the VM repaired any gait.** All flagged gaits ended up substituted, so on this machine the failures are deterministic.
- **The substitution cap was reached exactly [measured].** e3− (more forward lean) needed 4 substitutions, the `MAX_SUBSTITUTIONS` cap. A wider style range on e3 would exceed it.
- **Knee limit [measured]:** in e2− only 138/240 gaits exceed the ±π/2 knee range (240/240 in the other styles). This is informational only.
- **Imitation impact [hypothesis]:** the substituted corner commands will imitate a neighbour's gait.

---

## 2026-09-29 — L4 throughput measured; R1 attempt 1 failed on two unrepairable gaits

Both cloud VMs (code `4f7ab2c`) ran and self-deleted within their caps. The temporary Cloud NAT was then removed; `audit` shows no instances, disks, addresses or routers (only the bucket and runner service account remain). The NAT stayed up idle for about 4 h after the jobs ended, because I didn't tear it down promptly.

**Steady-state throughput on an L4 (g2-standard-8) [measured]:** `throughput_benchmark`, 1,638,400 effective steps, 3 evaluations.
- Brax `training/walltime`: 281.0 s at 819,200 steps, and 293.5 s at 1,638,400 steps.
- That gives **65,500 training steps/s** between the last two chunks.
- The first chunk is dominated by the one-time compile of `pmap_training_epoch` (about 4.5 min; XLA printed "Very slow compile?").
- Wall time was 810 s including evaluations and ONNX export.
- **Projection [arithmetic, not measured]:** about 25 min of training per 10⁸ steps, plus compile and evaluations. The earlier 876 steps/s and "95 h / $81" figures were compile-dominated and are wrong. L4 is sufficient for B0/B1/S1.

**R1 attempt 1 failed [measured]:**
- The first style (neutral) stopped after 5 repair rounds, with two gaits still invalid: `0.222_-0.111_-1.111` and `0.222_0.111_0.963`.
- These are **exactly the shipped file's 2 defective gaits.**
- The mechanism is different from R0's:
  - The initial placement is correct (knees +0.84 and +1.09).
  - Then, at frame 33 and frame 47 respectively, the knee passes through full extension (0 rad) and continues on the backward branch.
  - This happened on every regeneration on the VM.
- Both combine maximum forward step, maximum lateral step and fast turning, a reach the leg apparently can't provide.
- Their achieved average velocities are near zero (for example −0.008 m/s against a key of 0.222).
- In the laptop R0 repair these same keys regenerated correctly, which suggests that solver behaviour differs between machines [hypothesis].

**Response [NERVA design choice]:**
- Upstream `PolyReferenceMotion` needs the full grid, so invalid gaits can't be dropped.
- After `MAX_REPAIR_ATTEMPTS = 2` regenerations, `substitute_invalid` replaces a still-invalid gait with its nearest valid gait in the range-scaled (dx, dy, dθ) grid.
- More than `MAX_SUBSTITUTIONS = 4` substitutions aborts the run.
- The mapping is recorded in each style's validation report.
- Applied to the shipped file, it maps the two gaits to their neighbours one yaw step inward (`…_-0.852` and `…_1.222`), and the result validates. The shipped pickle itself stays unmodified for B0.

---

## 2026-09-29 — R0 result: generator reproduces upstream but intermittently produces backward knees

Continued from the parallel Codex session. Its work (benchmark job, R1 generator, B1/S1 jobs, audit and teardown) was reviewed and committed as `d8d160a`. Codex had run R0 locally in WSL at no cloud cost: 240 gaits in 1,405 s with 6 workers, plus a 13 s fit.

**R0 comparison with the shipped `polynomial_coefficients.pkl` [measured]:**
- The **same 240 keys and 0.54 s period.** My earlier claim that the sweep config doesn't reproduce the shipped grid was wrong and is corrected in the design doc.
- Raw polynomial coefficients differ by up to 3.4e5, but degree-15 coefficients are ill-conditioned, so the evaluated trajectories were compared instead.
- For the median gait, joint positions differ by at most 0.0008 rad.
- **In 41/240 gaits a knee has the opposite sign.**
  - 39 gaits were flagged as backward-bent by the validator, with left or right knee negative for the whole cycle.
  - Each flipped gait's generator log shows the negative knee already in the **initial IK placement** (`Initial position reached … right_knee: -0.83`).
  - Re-running a flipped gait with identical parameters gave a correct knee every time: 3 standalone runs and 12 concurrent ones.
  - So it's intermittent, not caused by parameters or concurrency.
- **The shipped file has the same defect, less often:** 1 fully mirrored right knee, and 1 gait whose left knee switches branch mid-cycle (−1.49 → +1.76 rad).

**Response:**
- `nerva/reference_validation.py`: flags backward knees and non-finite values.
- `fit_validated`: regenerates each flagged gait with its exact logged parameters until the set is valid.
- **Repair of a copy of R0** (raw R0 kept): all 39 fixed in 2 rounds (30 on the first retry, 9 on the second), 593 s, and an independent re-validation passes.
- The repaired set matches the shipped set to within 0.002 rad on every gait except the shipped file's 2 defective gaits.
- The validated neutral pickle is local and git-ignored: `experiments/cloud_runs/r0-local-20260929/r0_repaired/`.

**Side observation [measured, generator logs]:** a gait's velocity key (the planned value) can differ strongly from its achieved average velocity. For example, key vx = −0.148 with an achieved +0.014 m/s, which is consistent with the preset limit `walk_max_dx_backward = 0.03`. This may be related to the baseline's poor backward tracking (Phase 2) and hasn't been investigated.

**Throughput correction:** the smoke run's 876 steps/s (and Brax's 1,154 `training/sps`) came from a single training chunk that included compiling the training step. So the "95 h / $81 for B0 on L4" projection is an upper bound, not an estimate. The benchmark now measures steady-state throughput from Brax's `training/walltime` between the last two of 3 evaluations. That can't be measured locally (no GPU), so the corrected benchmark is the next paid step.

**Tests:** 80 passed, 2 slow skipped.

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
