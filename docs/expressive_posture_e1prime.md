# E1′: continuous pitch × crouch conditioning over the admitted reference space (preregistration, 2026-10-11)

A new preregistration, not a change to E1 (`expressive_posture_experiment.md`, stopped at Phase A). Its style
domain is chosen from **E1's measured reference feasibility** (not from any policy result), and it adds a
neutral-anchored curriculum. Stage 3 of the curriculum: imitation of verified styled references; no new reward
objective; no affect/PAD input; no emotion labels. One capped GPU training run, which needs an explicit compute
cap from the user (§4); everything else is local.

Labels: [measured], [design], [hypothesis], [future].

## 1. Question and hypothesis

> From the neutral policy that passes the long motor gate, does a continuous input e = (e_pitch, e_crouch)
> produce monotonic, low-cross-talk, interpolating control of torso pitch and body height across the admitted
> domain, including a held-out combination, without changing commanded speed and without losing the neutral
> gate at e = 0?

**H [hypothesis]:** yes, at ≥ 50% of the references' own feature change.

## 2. Style domain [measured → design]

- **e_pitch ∈ [−1, 1]:** `walk_trunk_pitch` = −4° + 6°·e_pitch (−10°…+2°). Every pitch passed E1 Phase A at
  COM 0.215 and 0.209 m.
- **e_crouch ∈ [0, 1]:** `walk_com_height` = 0.215 − 0.006·e_crouch m (0.215…0.209 m). One-sided, because above-neutral
  height failed and below 0.209 m failed joint limits in E1.
- **e = (0, 0) is exactly the admitted neutral reference set** (`verified_references`). Neutral is a corner of
  the crouch axis.
- [measured] Reference feature changes: pitch 12.0° across e_pitch; base height ≈ 7 mm across e_crouch (half of
  E1-halved's 14.3 mm for 0.209–0.221 m; Phase A′ records the exact value). **The crouch range is small**; it is
  what the generator admits.

## 3. References (Phase A′, local; no regeneration)

Grid e_pitch ∈ {−1, 0, 1} × e_crouch ∈ {0, 1}, seven commands each:
- (0, 0): the admitted neutral set;
- (±1, 0): `results_e1_references_p1.0_h0.5/` styles (±1, 0) (COM 0.215 m);
- (−1, 1), (0, 1), (+1, 1): that run's styles (−1, −1), (0, −1), (+1, −1) (COM 0.209 m).

Interpolation checkpoints (validation only): (±0.5, 0.5) = that run's (±0.5, −0.5) (COM 0.212 m), whose
interpolation from exactly these four corners was measured there.

**Phase A′ gate (all required):** every grid and checkpoint recording and fitted reference matches its recorded
SHA-256; every grid recording passes all eight admission criteria; each checkpoint's measured interpolation
error is ≤ 0.05 rad with contact agreement ≥ 90%; the regenerated neutral set matched the admitted one. If any
fails, E1′ stops before training.

## 4. Training (one capped run; requires the user's compute cap)

Identical to the turn-translation run (`turn_translation_pilot.md`) except the lines marked ★.

- ★ **Start:** `experiments/cloud_runs/turn_translation-20261011-000711/checkpoints/000059189760` (hash-verified
  against `results_turn_translation/training_summary.json`).
- ★ **Inputs:** e appended to the policy state (101 → 103) and the privileged state (212 → 214); normalizer
  entries for e fixed at mean 0, std 1; all other statistics frozen as before. New first-layer weights of actor
  and critic are zero, so the start is exactly the neutral policy and value for every e. A test must show
  identical actions before training.
- ★ **Targets:** imitation target and `stand_still` rest pose = bilinear interpolation of the six grid references'
  coefficients at e (fixed 0.54 s period).
- ★ **Style sampling with a neutral-anchored curriculum** [design]: e is drawn at every episode start (including
  autoresets) and held for the episode.
  - With probability 0.2, **e = (0, 0) exactly** (neutral anchor), throughout training.
  - Otherwise e = (s·U(−1, 1), s·U(0, 1)), rejecting the **held-out region e_pitch > 0.5 and e_crouch > 0.5**.
  - The scale is s = min(1, 0.25 + 1.5·u), where u is the fraction of the step-ceiling iterations completed:
    s rises from 0.25 to 1 over the first half of training, then the full admitted domain for the second half.
- **Unchanged:**
  - all reward terms and scales (gait-averaged base-origin tracking, pure-turn width 0.0025);
  - balanced persistent commands, 8,064 environments, 1,000-step episodes;
  - noise, 0–2 step action delay, pushes;
  - PPO hyperparameters, seed 41, and all stops (KL > 0.1, replay, coverage, nonfinite, statistics change);
  - checkpoints every 20 iterations with snapshots, synced every 180 s.
- **Budget (to be authorized):** one L4 VM with hard lifetime L minutes, L ∈ {45, 90} as authorized. Trainer
  deadline at uptime 60·L − 480 s; step ceiling from throughput measured on iterations 2–4; ≤ 160 M
  transitions. Proposed L = 90 (≈ 120–160 M transitions, ≈ US$1.5 expected, cap US$3).
- **Final accepted checkpoint only**; no selection.

## 5. Evaluation (local, native MuJoCo; same scene/noise/initial conditions as the neutral gate)

Native inference appends e after the 101 state values (103-input ONNX).

1. **Neutral retention:** the full neutral long motor gate at e = (0, 0), unchanged (160 trials, both latency
   conditions), compared with the validated neutral candidate's recorded gate.
2. **Sweeps** (seeds 0–2, no latency, seven commands, 20 s, scored over 5–20 s):
   - e_pitch ∈ {−1, −0.5, 0, 0.5, 1} at e_crouch = 0;
   - e_crouch ∈ {0, 0.25, 0.5, 0.75, 1} at e_pitch = 0.
   - 9 distinct points × 7 × 3 = 189 trials. Features: mean base pitch and mean base height (qpos z).
3. **Style space:** trained points (−1, 0), (1, 0), (−1, 1), (0, 1) and held-out points (0.75, 0.75), (1, 1); seven
   commands × seeds 0–2 × both latency conditions (252 trials).
4. **Continuous ramps:** forward command, 30 s, e ramped linearly from one end of an axis to the other over
   5–25 s, the other axis at 0, seeds 0–2. Features in non-overlapping 1 s windows against the mean e of each
   window.
5. **Descriptive:** clips (neutral candidate | E1′ at e = 0 | extremes | held-out | ramps) with the same camera,
   duration, command and seed; failures kept.

## 6. Criteria (all required; fixed now)

1. Phase A′ passes.
2. **Neutral retention:** the neutral gate passes at e = 0 in both latency conditions.
3. **Direction and range:** for each axis, every command and every seed, the end-to-end feature change has the
   reference's sign and magnitude ≥ 50% of the reference's change for that command.
4. **Monotonicity:** for each axis and command, the seed-mean feature is strictly monotonic over the five sweep
   points.
5. **Cross-talk:** for each command, the other feature's change during an axis sweep is ≤ 25% of that feature's
   own change in its own sweep.
6. **Task across the style space:** at all six style-space points, every command passes the per-trial motor
   rules (`motor_metrics`) in 3/3 seeds without latency; no falls in any trial with or without latency.
7. **Task interference** (the S1 confound): at every sweep and style-space point without latency, each
   translation command's seed-mean signed speed is within [0.75, 1.33] × the same policy's e = 0 speed, and each
   turn's yaw rate is within the same ratio. Command calibration for speed matching is not possible under the
   neutral contract (only the seven trained commands are accepted), so speed equivalence is required instead.
8. **Continuity:** in every ramp and seed, the windowed feature has Spearman correlation ≥ 0.9 with window e in
   the reference's direction, and no window-to-window change exceeds 25% of that ramp's total feature change.

- **H supported** if 3–5, 7 and 8 hold. **E1′ passes** only if 1–8 all hold.
- Every failure is reported per criterion, axis, command and seed. No threshold, domain, curriculum or
  checkpoint changes after results.
- **Not claimed:** emotional readability (needs human evaluation), deployment, default changes. PAD → e stays
  unconnected until E1′ passes.
