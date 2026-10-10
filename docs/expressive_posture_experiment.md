# E1: continuous posture conditioning on the validated neutral candidate (preregistration, 2026-10-11)

Status: fixed before any implementation, reference generation, training or evaluation. Curriculum stage 3
("expressive conditioning", `locomotion_curriculum.md`): a style input trained by **imitating verified styled
references**, with **no new reward objective**. Stage 4 (expressive reward objectives, the S5/S6 failure
mode) is not part of E1. Phases A and C are local and free. **Phase B (one capped GPU run) needs its own
explicit authorization**; this document does not authorize spending.

Claim labels as in `style_policy_design.md`: [fact], [design], [hypothesis], [unknown].

## 1. Question and hypothesis

> Starting from the first policy that passes the neutral long motor gate, can a two-dimensional continuous style
> input e = (e_pitch, e_height) ∈ [−1, 1]² control torso pitch and body height independently and
> monotonically, while the neutral gate still passes at e = 0 and walking stays valid across the style space,
> including a held-out combination?

**H [hypothesis]:** imitation of interpolated styled references, with e appended to the observation and the
neutral policy as the starting point, yields monotonic, low-cross-talk control of both features, at ≥ 50% of
the references' own feature change, without breaking the neutral gate.

## 2. Why these two axes [fact/design]

- **Torso pitch** was expressed by S1 (monotonic 10/10, but about 1/3 of the nominal 12° span) and the
  references are feasible over −10°…+2° (R1).
- **Body height (crouch)** is a postural cue in the gait-perception literature (body contraction) and
  Placo exposes it (`walk_com_height`). Its feasibility is **[unknown]**, so Phase A gates it.
- Both keep the **gait period fixed at 0.54 s (27 steps)**. The neutral contract, the phase clock and the
  gait-averaged tracking window all assume that period.
- **Not in E1:** step height (S1 0/10, and S5/S6 collapsed to standing); tempo (needs a variable-period
  contract, so it is a separate E2); head posture. Labels stay neutral ("torso pitch", "body height"), never
  emotion words.

## 3. Style space and references (Phase A, local WSL, free)

**Mapping [design]**, around the exact preset of the admitted neutral references (COM 0.215 m, foot height
0.020 m, rise ratio 0.30, single support 0.18 s, double-support ratio 0.5, trunk pitch −4°):
- e_pitch → `walk_trunk_pitch` = −4° + 6°·e_pitch (−10°…+2°, S1's range);
- e_height → `walk_com_height` = 0.215 + 0.012·e_height m (0.203…0.227 m).

**Generation:** the same recorder, repair settings and geometric-turn correction that produced the admitted
neutral set (`record_reference.py`; preregistrations `1b86c37`, `f8728dc`), with only these two preset
fields changed; seven commands per style; one 8 s recording each.
- Grid styles: e ∈ {−1, 0, +1}² minus (0, 0): **8 styles × 7 commands = 56 recordings**.
- Interpolation checkpoints: e ∈ {±0.5}²: **4 × 7 = 28 recordings** (validation only, never trained on).
- Reproducibility check: regenerate (0, 0): **7 recordings**, compared with the admitted neutral set.

**Phase A gate (all required; thresholds fixed now):**
1. Every grid recording passes all eight admission criteria of the neutral set (`REQUIRED_CRITERIA`:
   positive knees, joint position/velocity, linear/angular and contact fits, command tracking, joint
   limits), unchanged.
2. The regenerated (0, 0) set matches the admitted neutral references with moving-joint position RMSE
   ≤ 0.05 rad per command (pipeline reproducibility).
3. **Interpolation validity:** at each of the 28 checkpoints, bilinear interpolation of the four surrounding
   grid references' Fourier coefficients (same period and phase convention) matches the generated
   reference with leg-joint position RMSE ≤ 0.05 rad, and contact-label agreement ≥ 90% of frames.
4. **Range rule (fixed):** if criterion 1 or 3 fails only because of points at |e_height| = 1 or |e_pitch|
   = 1, that axis's range is **halved once** and Phase A is rerun for that axis. If it still fails, E1
   **stops** and is reported; no training. No other adjustment is allowed.

**Reference feature targets:** for each style and command, mean base pitch and mean base height from the
recording's root pose. The target changes are Δpitch_ref and Δheight_ref between e = −1 and +1 on each
axis, with the other axis at 0. They define the direction and range criteria; nothing assumes the sign of
`walk_trunk_pitch`.

## 4. Training (Phase B, one capped GPU run; requires authorization)

Identical to the turn-translation run (`turn_translation_pilot.md`) except the lines marked ★.

- ★ **Starting checkpoint:** `experiments/cloud_runs/turn_translation-20261011-000711/checkpoints/000059189760`,
  hash-verified against `results_turn_translation/training_summary.json`.
- ★ **Observation:** policy state 101 → 103 and privileged state 212 → 214, with (e_pitch, e_height)
  appended. Their normalizer entries are fixed at mean 0, std 1; the other entries keep the frozen B2-cropped
  statistics. The new first-layer weights of actor and critic are **initialized to zero**, so the starting
  policy and value are exactly the neutral candidate's for every e. A test must verify identical actions
  before training.
- ★ **Imitation target and rest pose:** the reference for (command, e) is the bilinear interpolation of the
  four surrounding grid references' coefficients (one period, 27 steps). The rest-pose target of
  `stand_still` is the interpolated rest reference's static pose, not the neutral pose. Unsupported commands
  still return nonfinite targets.
- ★ **Style sampling:** e drawn uniformly per episode at reset from [−1, 1]², rejecting the **held-out region
  e_pitch > 0.5 and e_height > 0.5**. It is constant within an episode (no within-episode style changes in
  training).
- **Unchanged:**
  - every reward term and scale, including gait-averaged base-origin tracking and the pure-turn width 0.0025;
  - balanced persistent commands (env i → COMMANDS[i mod 7]), 8,064 environments, 1,000-step episodes;
  - noise, 0–2 step action delay, pushes;
  - PPO (unroll 20, 32 minibatches, 4 epochs, LR 1e-4, clip 0.2, entropy 0.005, discount 0.97, GAE 0.95, value
    0.5, gradient norm 1.0, seed 41) and all stops (KL > 0.1, replay, coverage, nonfinite, statistics change);
  - step ceiling from measured throughput (iterations 2–4), deadline at uptime 2,220 s, ≤ 80 M transitions;
  - checkpoints every 20 iterations with snapshots.
- **Budget:** one L4 VM, 45 min hard lifetime, ≤ US$2 (previous identical-shape runs ≈ 33 min, ≈ US$0.70).
  Final accepted checkpoint only; no checkpoint selection.
- **No equal-step control run** [design]: E1 adds conditioning, not an objective. The effect of e is
  measured within one policy (same weights, different e), and regression is measured against the
  starting candidate under identical protocols. The curriculum's equal-step control applies to stage 4.

## 5. Evaluation (Phase C, local, native MuJoCo, paired seeds)

Native evaluation gets the style input through the inference contract (the 103-input ONNX, e appended
after the 101 state values). Same scene, noise, initial joint noise and 20 s trials scored over 5–20 s as
the neutral gate.

1. **Neutral preservation:** the full neutral long motor gate (`neutral_motor_gate.md`, unchanged) at
   e = (0, 0): 160 trials, both latency conditions.
2. **Feature sweeps:** for each axis, e_k ∈ {−1, −0.5, 0, 0.5, 1} with the other axis at 0; seven commands;
   seeds 0–2; no latency. 2 × 5 × 7 × 3 = 210 trials, e = 0 shared. Features: mean base pitch and mean base
   height (qpos z) over 5–20 s.
3. **Task across the style space:** the four corners (±1, ±1) and the held-out point (0.75, 0.75); seven
   commands; seeds 0–2; without and with latency. 5 × 7 × 3 × 2 = 210 trials.
4. **Descriptive:** comparison clips (neutral candidate | E1 at e = (0, 0) | E1 at the corners); within-trial
   style steps (−1 → +1 at t = 10 s) reported, not judged.

## 6. Criteria (all required; fixed now)

1. **Phase A gate passes** (§3); otherwise stop before training.
2. **Neutral gate still passes at e = 0** in both latency conditions (all four gate criteria).
3. **Direction and range:** for each axis, every command and every seed, feature(e_k = +1) − feature(e_k = −1)
   has the sign of the reference change and magnitude ≥ 50% of |Δref| for that command.
4. **Monotonicity:** for each axis and every command, the seed-mean feature is strictly monotonic across the
   five sweep points.
5. **Cross-talk** (normalizer fixed in advance; S1's std-based one failed): for each command, the change
   in the other feature during an axis's −1 → +1 sweep is ≤ 25% of that other feature's own −1 → +1 change
   in its own sweep. Speed cross-talk is covered by criterion 6.
6. **Task across styles:** at the four corners and the held-out point, every command passes the per-trial
   motor rules (`motor_metrics`) in 3/3 seeds without latency, and there are no falls in any trial with or
   without latency.

- **H supported** if 3–5 hold. **E1 passes** only if 1–6 all hold.
- Failures are reported per criterion, command, axis and seed. No threshold, range or checkpoint is
  changed after results.
- **Not claimed:** any emotional reading (that needs human evaluation), deployment, or a default change. A
  pass would justify E2 (tempo, variable-period contract) and, later, stage-4 objectives with an equal-step
  control.

## 7. Implementation scope (after this preregistration, before Phase A/B)

- A styled-reference generator wrapper with preset overrides, plus an admission/interpolation checker.
- `StyledNeutralReference`: bilinear coefficient interpolation, exact grid coverage, hash checks.
- An environment variant with the e observation, interpolated imitation and rest targets, and held-out
  rejection sampling.
- Network widening with zero-initialized e weights, plus a test of identical actions at start.
- ONNX export and native inference with 103 inputs; an E1 evaluator for §5–6. Tests for each.
- No changes to upstream code, the neutral contract defaults, deterministic safety or the affect defaults.
