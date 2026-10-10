# NERVA development log

Newest entry first. Each entry records what was done, what was actually run, and what is still unverified.

---

## 2026-10-10 — Base-origin reward velocity: pivot moves to the base, turn right passes, turn left misses (0.034 vs 0.03)

Third authorized capped run (one L4, 45 min cap, US$2 cap), preregistered at `69a6f0f`; VM ran `0b31524`.
Evidence: `experiments/locomotion_curriculum/results_base_origin_velocity/`.

### What ran
- One g2-standard-8, us-central1-a, ≈ 33 min, ≈ US$0.70. 375 iterations, 60,480,000 transitions, step
  ceiling. Inputs 46/46 OK; max KL 0.0160; replay error 0.
- Fetch verified: 236/236 files, EXIT_CODE 0, checkpoint and ONNX hashes, export parity 1.4e-6. NAT
  removed; audit shows no billable compute or network resources.
- 84 paired native trials; seven three-column videos (B2 | gait-averaged | base-origin).

### Observations [measured]
- No falls. Rest and all four translations 3/3 (84–107% of the request; left overshoot gone).
- Turn translation halved: left 0.067 → 0.034, right 0.054 → 0.020 m/s. **Turn right passes 3/3**, the
  first passing pure turn in any arm. Turn left fails only on translation.
- Pivot moved forward from near the IMU to within 1.4 cm of the base origin (the diagnostic's
  prediction); turn left keeps a 4.6 cm sideways offset.
- Normalized RMSE 0.1346 (untrained 0.4250); 6/7 commands pass.

### Decision (preregistered)
- **H not supported; the pilot fails**, narrowly (turn left). Threshold unchanged; no checkpoint selection.

### Not yet verified
- The cause of the left-turn sideways offset (left/right asymmetry); a second training seed; the long gate.
- No default, deployment, safety or expressive change.

---

## 2026-10-10 — Turn translation is a reward measurement-point mismatch (IMU vs base origin)

Local diagnostic only (no paid compute). Evidence: `experiments/locomotion_curriculum/results_turn_pivot/`
(`turn_pivot_diagnostic.py`, post-hoc on the saved native rollouts).

### Observations [measured]
- The turn metric's translation is almost entirely a constant pivot offset: every evaluated policy,
  including historical B2, turns about a point 4–11 cm behind the base origin (net 20 s displacement
  only 0.01–0.12 m).
- Upstream's tracking and imitation rewards read velocity at the IMU site (−0.08, 0, 0.05 m). Turn
  translation at the IMU is ≈ 0.02 m/s for every arm (one exception), versus 0.034–0.067 m/s at the base
  origin, which the evaluation and the references use. Feet and CoM sit ≈ 3 cm behind the base.
- The reference turn pivots at the base origin; on the gait-averaged tracking term it scores 0.998 at the
  base and 0.782 at the IMU.
- For the gait-averaged candidate, translation (`cross`) is the only failing turn criterion.

### Reading
The rewards pay the policy to pivot about the IMU instead of turning like the reference; the sharper
gait-averaged term strengthened that, which fits the larger base translation [inference; causal test is a run].

### Change (opt-in)
- `BaseOriginGaitAveragedNeutralJoystick`: tracking and imitation use v_base = v_IMU − ω × r_IMU.
  Observations and critic inputs unchanged.
- Tests: the shift matches native free-joint velocity to 1e-5; the reference turn scores ≥ 0.99 only at
  the base; slow MJX environment test. CPU smoke of the trainer passes (2 iterations, no recompilation,
  exact roundtrip).
- Next run preregistered at `69a6f0f` (`docs/base_origin_velocity_pilot.md`); trainer experiment
  `base_origin_velocity`, job `cloud/jobs/base_origin_velocity.sh`, evaluator protocol
  `base_origin_velocity`. **Not launched; needs authorization.**

---

## 2026-10-10 — Gait-averaged tracking: all four translations pass in every seed; turns still fail

Second authorized capped run (one L4, 45 min cap, US$2 cap), preregistered at `a143ae4`; VM ran
`3dad8c8`; evaluator `72326ce` committed before results. Evidence:
`experiments/locomotion_curriculum/results_gait_averaged_tracking/`.

### What ran
- One g2-standard-8 in us-central1-a (first zone, no stockout), ≈ 33 min VM lifetime, ≈ US$0.70.
- 384 accepted iterations, 61,931,520 transitions, step-ceiling stop (35,728 steps/s by the
  preregistered rule). Inputs 47/47 OK; max KL 0.0147; replay error 0; frozen statistics unchanged.
- Fetch verified locally: 242/242 files, EXIT_CODE 0, checkpoint and ONNX hashes match, export parity
  1.1e-6. NAT removed; audit shows no instances, disks, addresses, forwarding rules, routers or snapshots.
- 84 paired native trials and seven three-column videos (B2 | GPU pilot | gait-averaged).

### Observations [measured]
- No falls in any arm. Gait-averaged candidate passes rest, forward (79% of the request), backward
  (108%), left (122%) and right (88%) in 3/3 seeds. The start passed 3/7.
- Normalized RMSE 0.1496 (untrained 0.4250, GPU pilot 0.3104).
- Turns fail: yaw rate 0.59–0.62 rad/s (close to 0.60), but translation 0.053–0.068 m/s, worse than
  the start (0.039–0.044).

### Decision (preregistered)
- **H supported; the pilot fails** (criterion 3, turns). One seed, final checkpoint, no selection.
- Possible causes of the larger turn drift [hypothesis, untested]: the averaged term ignores drift that
  cancels within 0.54 s; the MJX → native turn gap.

### Turn-gap solver check [measured, diagnostic only]
- MJX with 10 solver iterations turns like MJX with 1 (+0.567 / −0.594 vs +0.565 / −0.607 rad/s).
- Native with 10 iterations ≈ native default (+0.720 vs +0.711); native without warmstart barely turns
  (+0.08). The iteration count is not the cause; warmstart/contact differences remain open.
  Recorded in `results_sim_gap/README.md`.

### Not yet verified
- The cause of the turn translation; a second training seed; the long robustness gate.
- No default, deployment, safety or expressive change.

---

## 2026-10-10 — Why the GPU pilot undershoots: the tracking reward prefers standing; turns are a sim gap

Local diagnostics only (no paid compute). Evidence: `experiments/locomotion_curriculum/results_sim_gap/`.

1. **Reference audit [measured]:** all seven verified references have correct means.
   - Translation 0.0738 m/s; turns 0.5986 rad/s with ≤ 0.004 m/s mean translation.
   - The within-stride lateral sway is ≈ 0.15–0.16 m/s RMS.
2. **Tracking reward [measured on the references]:** upstream's 2.5·exp(−|c − v|²/0.01) on instantaneous
   velocity scores:

   | behaviour | score |
   |---|---|
   | perfect reference gait | 0.77–0.80 |
   | **standing still** | **1.45** |
   | half-speed gait | 0.58–0.74 |
   | perfect gait, averaged over one 0.54 s period | 2.49 |

   At 0.074 m/s the tracking term rewards standing over walking correctly. This explains a persistent
   undershoot, and plausibly the S5/S6 standing collapse [hypothesis].
3. **Sim-to-sim [measured]** (`sim_gap_diagnostic.py`; deterministic MJX training-environment rollouts
   vs native MuJoCo, reading fixed in advance):
   - The GPU candidate undershoots in MJX without pushes too: forward +0.039, backward −0.047, left
     +0.047, right −0.021 m/s. That is learned, not a gap.
   - Turns: MJX yaw 0.565 / −0.607 rad/s with drift −0.018 / +0.022; native 0.711 / −0.669 with
     +0.043 / −0.025. The turn failures are largely a MJX → native gap.
   - Pushes make translation worse (forward +0.020).
   - The diagnostic's first run wrote its JSON inside the upstream checkout (relative path after
     `chdir`). The files were moved back, upstream verified clean, and the runner (and `gpu_pilot.py`)
     now resolve output paths first.

**Change (opt-in):** `GaitAveragedTrackingNeutralJoystick`.
- Tracking rewards use the mean body velocity over the last ≤ 27 steps (one gait period); the window
  resets with the episode.
- Tests: the window mean; references prefer walking over standing only under the averaged form; a slow
  MJX step/autoreset test.
- `gpu_pilot.py` now selects named experiments. The original `gpu_neutral_pilot` smoke still reproduces
  KL 0.01112 / 0.00278 exactly.

**Next run preregistered** (`a143ae4`, `docs/gait_averaged_tracking_pilot.md`): continuation from the GPU
candidate with only the reward changed. Job `gait_averaged_tracking`, the same 45 min / ≤ US$2 envelope.
**Not launched: awaiting explicit authorization** (the previous authorization covered exactly one run).
The turn gap is out of scope for that run and needs its own diagnostic.
323 tests pass (slow included); Ruff clean.

---

## 2026-10-10 — Capped GPU neutral continuation: 60 M transitions; backward/left pass, right regresses, turns unchanged; pilot fails

**Preregistration:** `a3ba5ca`, before implementation (`docs/gpu_neutral_pilot.md`). One authorized paid
pilot: ≤ 1 GPU VM, 45 min hard lifetime, ≤ US$2.

**Implementation** (`1d423dd`):
- `experiments/locomotion_curriculum/gpu_pilot.py`: minibatch PPO with upstream's batch structure
  (8,064 environments, unroll 20, 32 minibatches, 4 epochs), LR 1e-4, the pilot's loss settings, and
  frozen B2-cropped statistics.
- Starting point: the local pilot's final checkpoint, hash-verified on the VM.
- Balanced environments: environment i runs `COMMANDS[i mod 7]`, with 1,000-step episodes.
- `PersistentNeutralJoystick` blocks upstream's step-500 command resampling (test added). It also gives
  the "alive" reward a strong dtype: the weak dtype forced one extra collector compilation (values are
  identical, measured in the CPU smoke).
- Stops: deadline at uptime 2,220 s, the throughput-based step ceiling, and integrity stops.
- Launcher `--input-bundle`: a checksummed `inputs.tar.gz` + `MANIFEST.sha256`, only 46 files/6.7 MB (the
  start checkpoint and seven references). Job `cloud/jobs/neutral_gpu_pilot.sh`: GPU assertion, checksum
  verification, 180 s result sync, independent `timeout`.
- The CPU smoke passed end to end before launch.

**Run** [measured]: `neutral_gpu_pilot-20261010-152812`, g2-standard-8 (1× L4), us-central1-c,
on-demand at $0.8536/h (current list price).
- The first attempt hit **STOCKOUT** in all three us-central1 zones (no VM, NAT removed); the retry
  succeeded.
- VM lifetime 04:29:59 → 05:02:39 UTC (32.7 min); bootstrap 236 s; first iteration (compile + reset)
  319 s.
- **374 accepted iterations = 60,318,720 transitions**, stopped at the step ceiling.
  - Rule: 36,291 steps/s over iterations 2–4 → 374 iterations. Steady state 3.77 s per iteration
    (≈ 42,800 steps/s).
- GPU mean utilization 71%. Inputs 46/46 verified. Replay error 0; max KL 0.0192; statistics unchanged;
  exact roundtrip.
- 236/236 files fetched; checkpoint and ONNX hashes re-verified locally.
- Estimated total ≈ US$0.70.
- Training reward 0.33 → 0.49 for every command; true terminations 0.13–0.18% per step.

**Evaluation** [measured]: 84 native-MuJoCo trials (`motor_compare.py`, same protocol as the pilot), no
falls in any arm.

| command | GPU candidate | detail |
|---|---|---|
| rest | 3/3 | |
| forward | 0/3 | +0.031 m/s, 43% |
| backward | **3/3** | 0.065 m/s; pilot 0.017 |
| left | **3/3** | 0.071 m/s; pilot 0.036 |
| right | **0/3** | 0.021 m/s; pilot 0.038, start 3/3 |
| turns | 0/3 | translation 0.039–0.044 m/s > 0.03; over-rotation 0.66–0.71 rad/s |

- Normalized RMSE 0.3104 vs untrained 0.4250 (ratio 0.730) and pilot 0.4130.
- **Criteria 1 and 3 fail; H is not supported; the pilot fails.** No checkpoint selection, no default
  promotion, no expressive objectives. Turn-time lateral drift is also present in historical B2 and the
  pilot.
- Seven three-column clips and a combined 140 s video are local (hashes in the results folder).

**Cleanup (separate commit `d928fb2`):**
- Shared helpers moved unchanged into `nerva/training/b2_warm_start.py`, `nerva/training/motor_artifacts.py`
  and `nerva/analysis/motor_eval.py`, with re-export shims in `learning_support.py`, `gate.py`,
  `normalization_timing.py` and `paired_motor.py`. None of their hashes is recorded in results;
  `neutral_ppo_smoke.py`'s is, and it is untouched.
- Six leaf runners moved byte-identically to `experiments/locomotion_curriculum/archive/`, with a mapping
  README. Runners that other code imports stay in place, marked completed.
- Maintained entry points: `gpu_pilot.py`, `motor_compare.py`, and the cloud job.

**Cloud state** [measured]: no instances, disks, addresses, forwarding rules, routers or snapshots. The
results bucket and runner identity are retained intentionally.

**Next (not run):** diagnose the left/right asymmetry before more training.
- Mirror-symmetric evaluation, and per-command lateral velocity in the training environment vs native
  evaluation.
- Audit why pure turns translate in every arm (reference vs reward).
- Then preregister a bounded continuation addressing the measured cause. No expressive objectives.

---

## 2026-10-10 — Actual balanced neutral learning, paired motor results and videos

Preregistered `8fd4fd3` before implementation/evaluation, baseline `e42735a`.
Convert the retained B2 walking checkpoint (300,482,560 steps) by folding its
three normalized constant-zero style inputs into first-layer biases. Preserve
512/256/128 SiLU policy/value networks and freeze the retained 101/212 input
statistics. This replaces the tiny random 32-wide plumbing network as the
learning initialization; it does not change historical or default controllers.
100 B2 ONNX probes agree within 7.45e-7 before collection.

**Actual training completes:** seven balanced backlash environments, commands
rest, +/-0.074 m/s forward/lateral and +/-0.60 rad/s yaw; seed 27, episode 256,
unroll 32, one clipped-gradient Adam step per 224 new transitions, LR 1e-5.
128 accepted updates/**28,672 new transitions in 300.11 s**, below the 1,200 s
hard cap. Maximum current-batch post-update Gaussian KL 2.485e-5 (limit 0.05),
frozen statistics unchanged, exact 25-leaf checkpoint roundtrip. Parameter warm
start initializes fresh Adam/RNG/environment; those states carry within the run.
Fifteen true training episode terminations (one forward, fourteen left-turn)
and 108 timeouts are reported separately; no claim that stochastic training
survives every episode. Raw checkpoints, optimizer/batch/key snapshots and logs
remain ignored and are not a full physical-environment resume interface.

**Actual motor comparison completes:** 63 independent 20 s native MuJoCo trials,
seven commands, seeds 0/1/2, historical B2 versus untrained converted neutral
clock versus learned candidate, same backlash/noise/zero-head settings, scoring
5–20 s. All trials complete without falls. The candidate passes rest and rightward
motion in all three seeds, but forward/backward/left undershoot the speed limit
and both turns fail horizontal translation RMS (mean 0.03376/0.04428 m/s versus
<=0.03). Yaw tracking passes. Dimensionless moving-axis RMSE: B2 0.430156,
untrained neutral 0.425003, candidate 0.413030. **Overall pilot fails:** 2.82%
error reduction versus the declared control misses >=10%, and only 2/7 commands
pass. Error reduction versus historical B2 is 3.98%; no best-checkpoint selection.
Rest displacement drops from B2's 0.257–0.261 m to 0.016–0.022 m, but the
untrained neutral clock already passes rest. Credit this to rest semantics,
not PPO. No default promotion or expressive objectives; the long gate remains
unmet. These low-speed commands differ from the original B2 gate, which remains
a separate completed negative result.

Seven 20 s B2/candidate comparison clips and a combined 140 s video show actual
native rollouts with requested/measured motion and PASS/FAIL labels. All eight
videos decode to the expected frame counts/durations; rest/forward/left-turn
keyframes visually reviewed. Large artifacts stay in ignored local storage;
public reports/hashes in `results_neutral_learning_corrected/`.

Preserve three implementation/integrity stops: reversed collector arguments
before collection; float32-versus-float64 coverage rejection after 224 collected
transitions with zero SGD; wrong deployment metadata field after one completed
B2 trial. Regression tests cover collector integration, canonical representation
and existing deployment contract. Criteria/tolerances/budget stay fixed. The
retry accounting originally omitted its stopped collected batch; an explicit
companion correction records 224 without rewriting the preserved report.
Evaluation resumes with identical policy hashes and retains/reuses its completed
trial. No raw artifact is deleted or replaced.

Verification: **312 full tests including slow pass in 289.93 s**, same two
historical cast warnings; Ruff passes. Add eight meaningful conversion/export/
balanced-reset/coverage/metric/deployment tests. Align the optional exporter with
the repository's existing ONNX 1.22.0 pin after the initial 1.17.0 export; all
eight focused tests pass again in 7.27 s. The final policy re-export is literally
byte-identical to the evaluated artifact, with zero action difference on 100
probes; originals remain preserved. Other scientific packages and all four
upstream checkouts remain unchanged. Public-content/finite-JSON/size checks pass;
no secrets, identities, personal absolute paths or large artifacts are committed.
No training/evaluation/test worker remains. Final read-only cloud audit has no
instances, disks, addresses, forwarding rules, routers or snapshots; existing
results bucket and runner identity retained. No paid compute or cloud launch.

Next: preregister a longer explicitly capped neutral parameter continuation,
for example a 300k-step local budget, targeting low-speed tracking and turn
translation with held-out motor rollouts. More steps are a hypothesis, not a
promised fix. Preserve this failed pilot and pass the long motor gate before
adding expressive objectives; do not reopen S1–S6 or repeat plumbing milestones.

---

## 2026-10-10 — Carried identity PPO batches pass every post-update stability screen

Preregistered `8c9c16b`, implementation `91356c7`, baseline `9514ad8`.
Admit the exact passing identity warm checkpoint, contract/21 leaves/core/
package/reference hashes. Parameter warm-start initializes fresh Adam/RNG/
environment at seed 19; no full-state resume. Within this run, preserve Adam,
physical environment and PRNG state across batches. Episode 32/action repeat1,
two CPU environments, unroll 4, one Adam update per newly collected eight-step
batch; other PPO/reference/reward/physics settings unchanged. Collector state/
key return is opt-in; existing callers retain their original two-output behavior.
Regression test verifies consecutive observation ticks and advancing keys/count.

**All 16 batches pass**, 128 new transitions/16 optimizer updates in 144.92 s
under 900 s. Every pre-SGD behavior replay and post-SGD finite/KL/value/density
check passes. Maximum deployed KL 0.008009 and value loss 0.126727; control
logits/log-probability max error 9.83e-7/9.06e-6. Unused normalizer count 32->160.
Final 21-leaf literal checkpoint roundtrip, 10 restored deterministic/sampled
probes (error 0) and five statistics-invariance probes (error 0) pass. Archive all
batch datasets, parameter/Adam states and post-update replays in ignored storage.
Public aggregates in results_identity_multibatch; prior failures retained.

Coverage is only lateral -0.074 m/s and yaw +0.6 rad/s. Four zero-discount episode
endings also carry timeout flags; no motor/fall-readiness claim follows. These
128 steps do not establish seven-command coverage, long rest, transition,
push/head robustness or meaningful learned locomotion. Next preregister balanced
seven-command local curriculum/coverage and paired controls before the long
motor gate or larger training. Do not reopen S1–S6 or add expressive objectives.
No physical-safety/affect changes, upstream/dependency edits, default adoption,
deployment, paid/cloud work or artifact deletion.
Verification: **304 full tests including slow MJX checks pass in 262.38 s**;
10 focused carry/identity/control tests pass, Ruff/diff checks pass. Same two
historical cast warnings. Public-content/finite-JSON/size checks pass for all
43 changed/report files; raw batch/Adam/checkpoint artifacts remain ignored.
All four upstream checkouts remain clean; no local test/training worker remains.
Final read-only cloud audit: no instances, disks, addresses, forwarding rules,
routers or snapshots. Existing results bucket and runner identity retained;
no cloud runs, paid compute, resource/artifact deletion or default promotion.

---

## 2026-10-10 — Identity-preprocessing on-policy/restore screen passes

Preregistered `e627af3`, implementation `ed66f4e`, baseline `8175144`.
Exactly one declared change from frozen variance-control settings:
normalize_observations=False. Native Brax records Welford statistics but does
not apply them; counts 16->32 are unused bookkeeping. Shared runner exposes
opt-in flag/initial/stage validation; original defaults remain normalized.
Saved network config and contract retain the actual flag. Fresh/warm initial
parameters pass literal comparison and reference/core/control checkpoint
admission; fresh weights match the original retained first-batch behavior.

**All criteria pass**, two 16-transition stages/four Adam updates in 230.45 s
under 900 s. Training KL fresh/warm 0.000140119/0.000140149; training value loss
0.051039/0.046600. Final deployed replay against original retained behavior
has KL 0.058854/0.243470 and value loss 0.086861/0.084648. Independent densities
and KL pass; all 21 checkpoint leaves and warm initialization are literal-equal;
20 restored-action probes error 0. Changing saved mean/std affects logits/value
by exactly 0 on 10 synthetic probes. Adam/RNG/counters restart on warm load.

Single-update-per-batch training KL is pre-SGD self-KL; deployed checks prevent
mistaking that for update stability. These tiny runs do not establish useful
motor control or general learning stability. Next preregister multi-batch local
checking of post-SGD KL against each batch's behavior, not an ever-growing KL
against the original policy. Command coverage/long readiness remain future gates.
No affect/safety changes, expressive training, upstream/dependency edits,
deployments/default promotions or paid/cloud work. Public aggregate reports
results_identity_preprocessing; all raw artifacts and historical failures retained.

Verification: **303 full tests including slow pass in 283.67 s**, 14 focused
identity/rebase/admission tests pass, Ruff/diff checks pass. Same two historical
cast warnings. Public-content/finite-JSON/size audit passes; raw artifacts ignored.
Final cloud audit clear of instances/disks/addresses/forwarding rules/routers/
snapshots. Results bucket and runner identity retained; no deletion or spending.

---

## 2026-10-10 — Offline optimization/deployment comparison supports fixed preprocessing

Preregistered `e241208`, implementation `d42b541`, baseline `61a0a0b`.
Exactly two independent fresh-state Adam updates on the same hash-admitted
retained batch: one old-statistics, one updated-statistics gradient. Deferred
and rebased deployment reuse the exact old-statistics optimized weights.
No new physical transitions; **11.09 s under the 180 s cap**. All controls,
finite gradients/states, literal four-checkpoint roundtrips, independent
KL/log densities and forty restored-action probes pass (max restore error 0).

Post-update/deployment KL/value loss: native 10.292988/21.143976;
fixed 0.023703/0.088133; deferred 10.288106/20.937843;
rebased 0.023703/0.088133. **Fixed screen and deferred-drift hypothesis pass.**
Moving statistics replacement after optimization alone hides deployment drift.
Affine first-layer rebase passes inference-preservation criteria across captured
current/next observations and five synthetic probes: max logits/value drift
1.91e-6/3.87e-7, deterministic/sampled action drift 7.45e-7/5.37e-7,
pairwise analytic KL <=2.64e-11. Other leaves/dtypes/shapes and independent
float64 formula pass. Adam moments are not transformed: no continued-training
or general learning equivalence claim. Raw-observation preprocessing is the
simpler next local treatment; native Welford ignores until_count.

Common offline input admission extracted without weakening any old checks.
New tests cover affine preservation under unequal scales, literal unaffected
leaves, invalid scales and prospective deployment/rebase decisions. New public
reports `results_normalization_schedule/`; raw gradients/checkpoints/replays
and every earlier failure preserved ignored. No dependency/upstream edits,
paid/cloud work, artifact deletion, deployment or defaults changed.
Next preregister identity-preprocessing on-policy numerical/restore smoke;
neutral learned readiness and expressive objectives remain unestablished.
Verification: **301 full tests including slow MJX checks pass in 220.42 s**,
12 focused comparison/admission tests pass; Ruff/diff and public-content/size/
finite-JSON checks pass. Two existing cast warnings. Final read-only cloud
audit: no instances, disks, addresses, forwarding rules, routers or snapshots.
Results bucket and runner identity intentionally retained; no paid work or
resource/artifact deletion.

---

## 2026-10-10 — Same-batch fixed-weight replay isolates normalization timing drift

Original capture preregistration `0901b11`, implementation `8a2f467` retains
its failed old-control integrity result. Offline retry `619b142`/`37d5312`
retains its admission-only parser abort. Schema-corrected retry preregistered
`75cf8ed`, implementation `63937e7`; completes in **6.41 s under 180 s**.
Correct generated slash-separated metric parsing and keep dataset, weights,
loss/permutation keys, PPO options, epsilon and every threshold unchanged.
Replay normalizer/weights/batch/key are explicit compiled inputs; regression
checks batch changes and independent tanh-Gaussian density. Actual retained
archive preflight validates literal bytes/file/tree hashes without replay.
No dependency or external source changes.

Both replays pass every integrity criterion: hash-admitted original sources,
versions and references; exact frozen weight/batch fingerprints; finite outputs;
statistics count 0->8; literal-byte 21-leaf checkpoint roundtrips; independent
KL/log probabilities. Old logits error 9.54e-7, behavior log-probability error
9.06e-6, conventional self-KL 9.53e-12. **Timing hypothesis supported on this
batch:** official KL 0.000140131 old versus 10.296526 updated, exceeding >1.
Manual stabilized updated KL 10.296525; conventional updated KL 10.296356.

Descriptive measures: deterministic tanh-action max shift 0.957049,
raw-action log-probability max shift 31.303122, value max shift 0.980274.
Value loss 0.089507->20.853298; normalized observation maxima
21.367878->2.644543 in both streams. Exactly eight original transitions,
**zero new physical transitions, gradients or optimizer updates** in the offline
attempt. This isolates preprocessing drift before SGD; it does not explain the
whole prior 16-transition averaged KL, establish motor readiness or validate a
schedule fix. Updating normalization after SGD can also shift deployed policy.
Next preregister normalization schedule/representation comparison including
explicit deployed drift and checkpoint checks before larger neutral learning.
S1–S6 stay closed; affect and deterministic physical safety remain separate.

Public reports: `results_normalization_timing_offline_schema/`; all raw batches,
weights, checkpoints/logs and earlier aborts preserved in ignored storage.
No paid/cloud work, artifact deletion, deployment or default policy promotion.
Verification: **296 tests passed including slow MJX tests in 251.17 s**;
seven focused tests also pass after the slash-key correction, including retained
archive preflight. Ruff and diff checks pass. Same two historical cast-overflow
warnings; no new warnings. Public changed-file checks find finite JSON, no
credentials/account identifiers/personal paths, and every file <1 MB. Raw
artifacts remain ignored. All four external checkouts remain clean; no local
training/test worker remains. Final read-only cloud audit: no instances, disks,
addresses, forwarding rules, routers or snapshots. Results bucket and runner
identity intentionally retained; no cloud spending or resource/artifact deletion.

---

## 2026-10-10 — Fixed-weight timing capture stops on replay integrity

Preregistered `0901b11`, implementation `8a2f467`, baseline `82360a0`.
One CPU attempt captured exactly eight transitions, seed 7, fixed initial
weights, variance epsilon 1e-4; zero optimizer updates. Stops after 98.28 s
under the 600 s cap at the old-statistics control. Counts 0->8, initializer
seed schedule, finite captured data and 21-leaf literal checkpoint roundtrip
pass. Logits error 9.54e-7 and self-KL 0.000139929 pass, but recomputed raw-action
log probabilities differ by 4.315092, exceeding the unchanged <=1e-4 criterion.
Updated-statistics replay was not executed. **Integrity-invalid, timing
hypothesis unevaluated**; preserve partial reports and captured artifacts.

Offline old-control inspection of the retained batch finds independent NumPy
and standalone Brax log probabilities agree with stored behavior. Passing the
batch explicitly into the compiled joint network/loss replay gives error
9.06e-6 and self-KL 0.000140131. The failing implementation closes over the
batch. This is evidence of a compilation-sensitive replay discrepancy, not
proof of a general compiler defect or a normalization timing effect. No new
physical transitions or gradients during inspection. A separately preregistered
offline retry must retain the exact batch/weights and original criteria.

Offline retry preregistered `619b142`, implementation `37d5312`, stopped in
4.62 s before any loss evaluation: archive parser rejected valid slash-separated
episode metric keys. All inputs and abort reports retained. Schema correction
and a fresh offline attempt are separately preregistered; thresholds unchanged.

Verification: **293 tests passed including slow tests in 250.62 s**, Ruff clean;
two existing historical cast warnings. Public reports contain aggregates/hashes;
raw batch/parameters/logs stay ignored. Cloud audit clear of compute/network
resources; results bucket and runner identity retained. No cloud work, paid
spending, upstream edits, artifact deletion or policy promotion.

---

## 2026-10-09 — Variance-floor control reduces amplification but fails the KL screen

Preregistered `1ee56c5`, implementation `d947120`, baseline `22ebf72`.
One change: Welford variance epsilon 0->1e-4, giving minimum std 0.01 after
updates. Frozen comparator is the previous strict epsilon-zero run, not repeated
training. Its PPO/network settings, package versions, core source/reference
hashes and all saved checkpoint hashes match. Source SHA/protocol/report hashes
are retained; one seed/tiny sample limits generalization.

One local CPU attempt, fresh/warm stages of 16 transitions each (32 total),
unchanged candidate rewards/reference/physics and (32,32) networks, seed 7,
hard parent limit 900 s. **Completes in 225.75 s, overall fail.** Both stages
pass finite updates, normalization count 16->32, live/disk 21-leaf literal bytes,
warm initialization and twenty exact restored-action probes. Adam/RNG/counters
restart on warm load. Regression test verifies saved epsilon affects future
statistics updates, with JIT host-array placement matching training.

Amplification criteria pass in both stages: minimum std 0.01, inverse gain 100,
peak synthetic normalized magnitude 81.202/81.903 versus 449070 control, below
the declared <=100 and >=100-fold reduction requirements. Manual/Brax agrees;
mean probes normalize to zero. These probes need not be physically achievable.
Fresh value loss falls from 884221440 to 10.440188; KL from 155226423296 to
1489.401733. Both relative reduction criteria pass, but **fresh KL fails <=1.0**.
Warm value loss 0.058027 and KL 0.132448 pass the fixed screens. Preserve every
criterion; do not retune epsilon, thresholds or claim overall success from
amplification improvement. No default normalization or policy promotion.

The shared runner accepts explicit epsilon/reporting and records mode/variance
epsilon in new checkpoint contracts. Historical default epsilon remains zero.
`comparison.json` is the study decision; `summary.json` is plumbing only.
Public reports in `results_normalization_control/`; raw checkpoints/logs retained
in ignored storage. No new dependencies, upstream edits, deployment, expressive
objectives, paid/cloud work or artifact deletion.

Source inspection shows the constant-LR PPO path collects with old statistics,
then updates statistics before SGD. This identifies a possible normalization
timing confound, not a demonstrated cause. Next: preregister capture/replay of
the same on-policy batch at fixed weights with old versus updated normalizers,
before changing schedules or undertaking larger neutral learning/readiness tests.

Verification: **289 tests passed including slow MJX tests in 276.13 s**, four
additional cases (three prospective-decision/admission checks and nonzero-epsilon
checkpoint continuation). Ruff/diff checks pass; same two existing cast-overflow
warnings in historical training tests. Tests ran alongside the local control;
timing is not a dedicated throughput measurement. Public JSON/content/size checks
find no credentials, account identifiers, personal paths or large artifacts.
External Open Duck checkouts remain clean; no training/test workers remain.
Final read-only cloud audit: no instances, disks, addresses, forwarding rules,
routers or snapshots. Results bucket and runner identity intentionally retained;
no cloud launch, spending or artifact/resource deletion.

---

## 2026-10-09 — Local neutral PPO, warm-start compatibility and normalization diagnosis

Preregistered bounded local PPO smoke `4243988`, implementation `9bffa12`.
Candidate-only autoreset now restores environment-owned motor info alongside
first data/observations; RNG advances and wrapper accounting is retained.
Mixed-row JIT test covers termination, phase/action history and accounting.
Historical wrappers, B2/S policies, affect and deterministic safety are unchanged.

Pinned Brax supports policy/value/normalizer parameter warm start, but initializes
Adam, RNG, environment and counters anew. Contract sidecar checks observation
shapes, network, normalization, reference hashes and source hashes before load.
Preflight caught two compatibility defects without optimizer work: Orbax 0.11.24
expects monitoring APIs absent in JAX 0.5.3, including in UInt64 device leaves;
Brax's config loader tries to look up null initializer names. Save plain Flax
state dictionaries with identical host NumPy arrays and omit default-null
initializer entries. Actual disk/inference tests pass; no dependency upgrades
or edits to external repositories.

Initial smoke: two 16-transition stages, seed 7, two CPU environments, tiny
(32,32) networks, fixed upstream candidate rewards, 900 s parent watchdog;
**32 transitions complete in 193.47 s**. Policy/value update in both stages,
normalization count 16->32, warm initialization and ten restored-action probes
per stage agree exactly. Fresh value loss 884221440 / KL 155226423296 and warm
value loss 0.061235 / KL 0.121367 are preserved. Finite updates are plumbing
evidence, not learning stability or motor quality.

Review found the original live equality check was numeric, not literal-byte
equality for signed zero. Original machine reports remain unchanged; that one
preregistered criterion was incompletely evidenced. Strict comparator/tests
`2218811` distinguish signed zero. A separately preregistered unchanged-settings
retry (`6f10e30`, `neutral_ppo_byte_retry.md`) closes the live-verification gap:
**all criteria pass, 32 transitions in 203.19 s**, including live-before-save
and restored bytes across 21 leaves, warm initialization, both network updates
and 20 exact restored-action probes. Normalization count 16->32. Loss/KL values
match the original run. Total optimizer work this batch: 64 transitions across
two separately documented attempts, no additional optimizer diagnostic. No loss criterion was
tuned or added after observing metrics.

Saved-checkpoint diagnosis: protocol `7a3b7c5`, code `2218811`, **all integrity
checks pass in 2.88 s**, zero rollouts/optimizer updates. Both saved normalizers
have six policy and seventeen privileged slots at the 1e-6 standard-deviation
floor. Fixed synthetic probes reach normalized magnitude 449070 and some
saturated actions; largest value magnitude 12401. NumPy/Brax normalization
agrees, mean probe normalizes to zero, and saved-tree reserialization preserves
all 21 leaves byte-for-byte. Susceptibility is observed; probes need not be
physical states and do not establish causes of the fresh PPO metrics.

Reports: `results_neutral_ppo/`, `results_neutral_normalization/` and the separate
strict retry under `experiments/locomotion_curriculum/`. Original and retry
checkpoints/logs are retained in ignored local storage. No deployment, ONNX
export, continuous-command admission or expressive learning. Next: separately
preregister a normalization-stability control before larger equal-step neutral
learning and its long rest/turn/transition/push gate. No paid work is authorized.

Verification: **285 tests passed including slow MJX tests in 233.15 s**,
four new tests; Ruff/diff checks clean. The same two existing JAX cast-overflow
warnings in historical training tests remain. Public-content/JSON/size scans
find no private paths, account identifiers, credentials or large artifacts.
All external Open Duck checkouts are clean; no test/training workers remain.
Final read-only cloud audit: no instances, disks, addresses, forwarding rules,
routers or snapshots. Results bucket and runner identity intentionally retained;
no launches, cloud spending or artifact/resource deletions.

---

## 2026-10-09 — Reference repairs, geometric turns and opt-in neutral motor environment

Completed the remaining safe-local reference/environment blockers in one batch.
No paid work, optimiser, candidate policy export, upstream edits or expressive
training. Bv4 affect/defaults and deterministic safety remain unchanged; B2's
failed learned motor gate is not overturned by reference/environment checks.

**Repair package:** preregistration `1b86c37`, code `df51807`, results `f8728dc`.
Medium preset with COM 0.215 m, lift 0.020 m, rise ratio 0.30; initial knees
1.2 rad; sorted positive knee range and enabled URDF limits; solver substep
matches actual recorder time. Static support labels require both foot frames
on the floor. Version-2/H12 Fourier positions and interval velocities share
coefficients; runtime joint velocity is analytic, validation compares exact
interval averages. Version-1/H5 remains supported. Two workers, one eight-second
recording per command, 180 s each / 900 s wall cap: **6/7 pass in 83.75 s**.
All derivative fits, contacts, knees and limits pass; left turn alone fails
off-axis motion (body vy +0.037961 m/s >0.020). Worst moving joint-velocity
component RMSE 0.217–0.353 rad/s. No thresholds changed or failed poses clipped.

**Geometric turns:** preregistration `f8728dc`, implementation `3dca496`.
Derive initial base-minus-foot-midpoint offset c in yaw axes and set pure-turn
internal step translation `(I-Rz(theta))c`. Requested commands remain zero
translation and yaw ±0.60; raw poses are not translated afterwards. Installed
Placo 0.6.3 geometry must match the source-derived formula before generation.
First attempt aborted in 3.88 s before recordings: its Footstep.frame binding
cannot return the Eigen transform. Preserved abort; API repair/explicit same-
criteria retry `bc176bb` verifies the exposed support polygon instead.
**Retry 2/2 pass in 18.31 s**, recorded `f92e19e`: body lateral means +0.016680
and +0.014414 m/s; yaw ±0.603689 rad/s; interval-velocity RMSE maxima 0.226951
and 0.219495 rad/s. Both signs retain positive knees and valid joint ranges.
Five passing repair targets plus these two turns form an explicitly mixed-
provenance, hash-verified seven-command subset. Original 0/7, alignment negative,
repair 6/7 and preflight abort stay recoverable and documented.

**Candidate plumbing:** protocol `f92e19e` (`neutral_motor_smoke.md`), code
`25510c8`. `NeutralReference` requires exact seven-command coverage, checks
recording/fit hashes, joint order, schema and all admission criteria. Unsupported
JAX lookup emits nonfinite targets instead of choosing a nearest gait.
`NeutralJoystick` uses upstream physics/control/reset and 101/212 observation
slots, zero head commands, the shared rest/canonical/phase clock, current-body
imitation and symmetric planar tracking. Reward scales stay at upstream
defaults, no foot-height/air-time objectives. Rest pose/velocity targets use
the static reference; imitation is movement-only. Historical StyleJoystick
reference hooks preserve its original clock and transition behavior. Inference
contract opt-in requires exact command coverage, metadata and matching policy
hash; there is no trained candidate artifact to deploy automatically.

Fixed CPU smoke: seed 7, seven eight-step commands plus stand-forward-stop,
zero actions, no optimisation, 600 s parent watchdog. **80/80 pass in 74.88 s**,
finite states/observations/raw and scaled rewards, clock/mode/index parity,
exact reference selection, no terminations or zero-clipped rewards. Reference
NumPy/JAX parity over 189 phases: position <=4.63e-7 rad, velocity <=5.51e-6
rad/s. Weighted imitation -9.711 to +2.348, rest cost -3.533 to 0, alive +20;
reward 0.2143–0.5582 per tick. Weights were not tuned. This is a plumbing check,
not learned locomotion, standing-solution avoidance or continuous-command coverage.

Final review tightened the joint-limit validator from the scored interval to
every recorded frame, as originally preregistered. All 400 raw frames plus
1,000 fitted-cycle samples per admitted target pass; minimum joint margin
0.137816 rad. Regression test covers a bad warmup frame. Admission also requires
all eight named criteria, static/moving semantics and both artifact hashes.

Reports: `results_reference_repair/`, `results_reference_pivot/`,
`results_reference_pivot_retry/`, `results_neutral_smoke/` under
`experiments/locomotion_curriculum/`. Raw motions, coefficients and subprocess
logs are preserved in ignored local storage. Existing Python/WSL environments
reused; dependency versions recorded, no installation or full-grid regeneration.
Next: preregister a bounded PPO/checkpoint-restore smoke, then an equal-step
motor-only comparison and long rest/turn/transition/push readiness gate before
expressive objectives. Any paid pilot still requires its own small hard cap.

Verification: **281 tests passed including slow MJX tests** in 227.98 s,
including upstream transition equality; nine new tests. Sixteen focused
reference/admission tests also pass after the warmup-coverage fix. Ruff and
diff checks clean; the same two existing JAX cast-overflow warnings in legacy
training tests. Public-content/JSON/artifact checks pass; no credentials,
account identifiers, personal absolute paths or large artifacts added.
Upstream checkouts clean; no recorder processes remain. Read-only cloud audit:
no instances, disks, addresses, forwarding rules, routers or snapshots. Results
bucket and runner identity intentionally retained; no launches, spending or deletions.

---

## 2026-10-09 — Corrected reference subset, shared motor contract, derivative alignment

Three consecutive preregistered safe-local phases, no paid work or PPO.
1. Subset protocol `7800728`, code `1e0a42d`, results `009bb5c`.
2. Contract protocol `009bb5c`, code `88c6074`, results `b24aed2`.
3. Alignment protocol `b24aed2`, code `0d42391`.
Reports under `experiments/locomotion_curriculum/results_reference_subset/`,
`results_contract/` and `results_derivative_alignment/`.

**Subset: 0/7 pass, adoption blocked.** Seven eight-second recordings, two
workers, one attempt each, completed in 92.81 s under 180 s per-process / 900 s
wall caps. WSL generator interpreter reused; no dependency installation,
upstream edits, full-grid regeneration, repairs or substitutions. Isolated
wrapper removes unconditional yaw-step bias and records actual timestamps.
New kinematics reconstruct current-body linear/angular derivatives and joint
velocities; known pure/mixed rotation checks pass 12/12. Five-harmonic periodic
schema fits on [2,4), evaluates once on [4,6); no silent nearest-gait lookup.

All held-out position and body-velocity fits pass, but all six moving joint
velocity fits fail: maximum per-component RMSE 1.928–1.974 rad/s (limit 0.5).
Static contact fit is 66% (limit 90%): upstream stand freezes poses but labels
remain walking-phase labels. Left turn also fails positive-knee and lateral
command criteria: right knee min -1.941 rad, lateral mean +0.036 m/s. Its yaw
mean +0.604 tracks correctly. Six moving conditions report knee-limit
exceedances, informational per unchanged protocol. All failed raw artifacts
and fits preserved in ignored storage; public reports include hashes only.

**Shared contract: passes all 2,180 NumPy/JAX ticks.** Rest uses separate planar
0.005 m/s and yaw 0.02 rad/s thresholds; canonical motion slots zero at rest,
head slots retained; phase freezes at [1,0], rest-to-move resets index zero,
subsequent movement advances modulo period. One backend-parametric function
feeds future training and inference adapters. Periods 27 and 20, seed 123,
1,090 commands each; max feature discrepancy 3.07e-7, canonical command error
zero. Symmetric planar tracking penalises vx/vy drift equally. Evaluation
1.55 s under 120 s watchdog. Infrastructure only; not wired to B2 or any
historical policy, reward, safety or default. Dynamic readiness untested.

**Alignment: timing alone insufficient.** Original input/output hashes verified
and endpoint metrics reproduced exactly. Same five-harmonic coefficients:
midpoint derivative RMSE 0.827–0.857, interval-average RMSE 0.814–0.843 rad/s.
Both reduce error by >50%, but neither meets 0.5; all six conditions fail
aggregate timing-accounted support. Four known sinusoid/static checks pass.
No refit, threshold change or new generation. Original subset stays 0/7.

Next: preregister derivative-consistent fitting, static geometry/contact label
validation and deterministic knee-branch initialization before another small
subset. Candidate environment/PPO smoke and checkpoint continuation wait for
valid references; expressive learning waits for a robust neutral motor gate.
Bv4 affect default, B2 reactive role and original failed gate unchanged.
Validation: **272 tests passed**, including slow tests, in 226.11 s; 14 new
kinematics, saved-signal and contract tests. Ruff and diff checks clean. The
same two existing JAX cast-overflow warnings occurred in unchanged training
tests. Public JSON/content checks pass; total small reports about 36 KB,
no secrets, account identifiers, personal paths or large artifacts added.
Upstream generator checkout remains clean; no recorder processes remain.
Versions preserved in subset environment.json; dependencies unchanged.
Final read-only cloud audit: no instances, disks, addresses, forwarding rules,
routers or snapshots. Results bucket and runner identity intentionally retained;
no paid runs, spending or deletions.

---

## 2026-10-09 — Neutral motor audit: tracking tolerance and generator derivative defects

Preregistered `fa22d19` after source inspection; implementation `a3d1220`.
Reports: `experiments/locomotion_curriculum/results_motor_audit/`; candidate
**design only**: `neutral_motor_candidate.md`. Seven actual 27-phase float32
reference lookups, five synthetic reward-dispatch cases, 108 tracking probes,
18 height cases and two known-yaw helper probes. No physics, training, policy,
reward, reference, deterministic safety or gate changes. Bv4 unaffected.

Imitation is exactly zero at zero motion including head-only commands. The
nearest stop reference `(0,-0.037,-0.074)` has mean lateral velocity -0.039775
m/s, but its imitation reward is disabled: this does not establish a direct
zero-command imitation incentive. Neither lateral nor yaw grid contains zero.
At norm exactly 0.010, strict inequality gates enable neither imitation nor
stand-still; 0.011 enables imitation. Stationary pose/velocity cost is active.

Lateral tracking gives identical reward for vy errors 0, 0.05 and 0.10 m/s;
0.11 reduces it. This target tolerance conflicts with the stricter pure-turn
horizontal gate. At stop, isolated vx 0.013 reduces scaled linear reward by
0.041895; yaw 0.033 reduces scaled angular reward by 0.619080. These are formula
sensitivities, not inferred total rewards from B2 trajectories.

Height cost is zero without touchdown and at a 40 mm target touchdown;
a single 20 mm touchdown costs -7.5 at both stop and forward. It is a cost,
not a reward for starting to step or a cost for never stepping. B2 has no
S6 air-time reward. Phase still advances at zero command; no stationary
world-position anchor is active. Behavioural causation remains unproven.

**Inspected generator derivative bug confirmed:** a +0.60 rad/s pure yaw
at dt 0.02 yields +0.0072 rad/s at initial yaw 0 and 1. The helper multiplies
`as_rotvec()` by its angle a second time; quaternion ordering is consistent.
Its relative vector is in prior body axes despite the world-velocity name.
Fitted 40-signal velocity slices match reward indexing, but angular world/body
conventions and world linear targets versus randomized initial yaw need an
explicit frame contract. The mistaken reference-quaternion slice feeds an
inactive orientation term. Historical shipped-pickle provenance and causes
of learned B2 drift cannot be inferred from this source audit.

Proposed motor-only package: verified velocity/frames and exact zero grid
entries, explicit rest/phase semantics shared with inference, symmetric planar
tracking and no height/air-time objectives initially. Design only; next local
phase must preregister a validator and small reference subset, not repeat R0
or S1–S6. Existing B2 readiness gate remains failed; expressive training blocked.
Five new meaningful probe/dispatch tests. Complete suite including slow tests:
**258 passed** in 222.52 s; the same two existing JAX cast-overflow warnings in
unchanged training tests. Ruff, diff and audit-completeness checks clean.
Public-content scan passed; approximately 59 KB reports, no secrets, account
identifiers, personal paths or large artifacts added. Final read-only cloud
audit: no instances, disks, addresses, forwarding rules, routers or snapshots.
Results bucket and runner identity intentionally retained; no launches,
spending or deletions.

---

## 2026-10-09 — Longer B2 turns: aggregate inconclusive; zero-command controls migrate

Preregistered `dec8fba`, implemented `7f18b7f`; reports in
`experiments/locomotion_curriculum/results_long_turn/`. Exactly 15 primary
65-second trials (left/right yaw +/-0.60 and stop, seeds 0–4) completed in
109.62 s against a 900 s wall cap. Same B2 checkpoint, backlash scene, raw
accelerometer, observation noise, initial +/-0.02 rad joint noise, neutral
style and head. No affect, training, cloud run or policy change.

Recorded mass-weighted robot COM and foot sites/contacts via isolated MuJoCo
forward kinematics. Excludes static scene bodies. Rotation-averaged blocks:
five complete rotations per turn; six fixed 10-second blocks per stop. Site
centroid is a support-region proxy, not a force centre or support polygon.

**Aggregate inconclusive, fixed thresholds preserved.** Local-region counts
3/5 left, 2/5 right, 0/5 stop; coherent migration 0/5 left, 1/5 right,
**5/5 stop**. Others inconclusive. All trials upright, no shadow safety
interventions, max tilt 14.59 degrees, minimum block contact coverage 99.81%.
Zero-command COM first-to-last block displacement 0.556–0.588 m; both-foot
midpoint 0.564–0.596 m; contact centroid 0.555–0.586 m. COM endpoint drift
11.12–11.75 mm/s. These are block-average displacements, not raw endpoints.

Every long turn still fails original horizontal-speed criterion: left
0.0356–0.0382, right 0.0437–0.0451 m/s (fixed limit 0.03). Stop horizontal RMS
0.0152–0.0156 m/s still passes the original speed limit despite accumulated
position migration. Do not revise the original gate retrospectively. B2 is
not ready for expressive curriculum; its reactive role and Bv4 remain unchanged.

Recorder COM matches MuJoCo subtree COM and does not alter live simulation
state. Left seed 0 first 20 s base pose/velocity, action and contact arrays
match the original gate exactly; checkpoint SHA256 matches. Six new tests.
Complete suite including slow tests: **253 passed** in 223.65 s; the same two
existing JAX cast-overflow warnings in unchanged training-env tests. Ruff and
diff checks clean. Public-content scan passed; reports total about 63 KB,
15 raw traces remain ignored; no secrets, identifiers or personal paths added.
Final read-only cloud audit: no instances, disks, addresses, forwarding rules,
routers or snapshots. Results bucket and runner identity intentionally retained;
no spending, cloud launches or deletions.

Next: separately preregister a neutral motor-target audit of zero-command
reference lookup, velocity slices and stationary reward activation. Candidate
mechanisms remain unproven; motor-only stopping/turning must precede expressive
objectives. No paid training authorised, no threshold tuning or reruns.

---

## 2026-10-09 — B2 turn-centre diagnostic remains inconclusive

Preregistered at `69dc1e0`, implemented at `661c4be`. Analysed exactly the ten
saved 20-second pure-turn traces; fitted on 5–12.5 s and evaluated on the
held-out 12.5–20 s segment. No new simulation, training, cloud run or policy change.
Small reports and SHA256 provenance: `experiments/locomotion_curriculum/results_turn/`.

**Aggregate inconclusive under unchanged prospective criteria.** Left: 0/5
bounded-orbit, 0/5 sustained-world-drift, 5/5 inconclusive. Right: 1/5 bounded,
0/5 sustained drift, 4/5 inconclusive. All fits full rank and well conditioned.
Orbit held-out position RMSE 13.74–33.46 mm left, 8.66–39.07 mm right; fitted
constant world drift speed 1.88–4.46 / 1.11–9.97 mm/s. Small drift estimates do
not prove bounded movement: nine orbit predictions fail the fixed 10 mm limit.
The original base-velocity robustness gate remains failed [measured].

Shipped reference nearest keys are `(0,-0.037,+0.704)` and
`(0,-0.037,-0.593)`. Mean reference lateral velocity is negative for both;
it disagrees with all five observed left turns and agrees with right turns.
Reference yaw-velocity reward slices average only +0.01425 / -0.00810 rad/s.
These training targets require future scrutiny, but this association does not
establish the cause of policy behaviour. Runtime ONNX does not query references.

Next phase: separately preregister longer turns with direct centre-of-mass and
foot-position logging to measure support-region drift. No threshold tuning,
reference regeneration as a runtime fix, or expressive training is justified.
Validation: complete suite including slow tests **247 passed** in 223.59 s;
four new synthetic geometry tests; Ruff and diff checks clean. The same two
existing JAX cast-overflow warnings occurred in unchanged training-env tests.
Final read-only cloud audit: no instances, disks, addresses, forwarding rules,
routers or snapshots; results bucket and runner identity intentionally retained.
No spending or deletions. Public-content scan passed; small reports only, no
personal paths, account identifiers or credentials; raw traces remain ignored.

---

## 2026-10-09 — Neutral B2 motor gate: pure-turn translation fails; expressive curriculum remains blocked

Preregistered at `14a139c` (`b2_robustness_gate.md`), implemented at `e8e3f34`.
Reports: `experiments/locomotion_curriculum/results_b2/`; raw traces local,
git-ignored. Existing final B2 checkpoint, backlash training scene, neutral
style, raw accelerometer, training observation noise and +/-0.02 rad initial
noise, seeds 0–4. No affect, perception or memory inputs [measured].

**All 115 primary trials completed in 398.95 s.**

| fixed criterion | result |
|---|---|
| unperturbed stability | pass; no falls in any of 115 trials, including pushes; max tilt 15.82 degrees |
| steady tracking, every direction/seed | **fail: pure-turn horizontal RMS exceeds 0.03 m/s in 10/10 turn trials** |
| no standing solution | pass in all steady and moving transition phases |
| starts/stops/transitions | pass, 5/5 trial sequences |
| pushes, no falls + recovery <=5 s in >=4/5 for each command/direction | pass: 5/5 in all eight conditions, 40/40; longest 1.78 s |
| standing/walking head tolerance | pass, 20 standing + 20 walking trials; minimum paired speed retention 86.31% (limit 75%) |

**Steady achieved mean velocity, five seeds per condition:**
- forward command +0.15: +0.1049 m/s; backward -0.15: -0.1126 m/s;
- lateral command +/-0.10: +0.0613 / -0.0598 m/s;
- yaw command +/-0.60: +0.5617 / -0.5864 rad/s.
All command-axis direction and tracking checks pass. The failure is translation
while commanded vx=vy=0: left-turn horizontal RMS 0.0330–0.0379 m/s, right-turn
0.0416–0.0439 m/s (limit 0.03). Mean lateral velocity is +0.0342 / -0.0382 m/s.
The short transition turns pass their axis-tracking criteria; they were not
preregistered to use the steady cross-motion criterion and do not overturn it.

**Decision:** negative motor-readiness verdict; do not proceed to expressive
curriculum training or change the thresholds. B2 retains its established
reactive-scenario role. Affect default Bv4, physical safety and motor control
are unchanged. Completed S1–S6 experiments were not rerun.

**Safety separation:** the unchanged deterministic SafetySupervisor ran in
shadow at 50 Hz, with zero interventions and zero stop time in every primary
trial. No conditional safety-on replay was needed. It neither masked a motor
failure nor supplied the tracking criterion. All work was simulated locally.

**Cause remains open:** heading-frame base translation during yaw does not
establish persistent global drift. Rotation around an offset centre and the
reference lookup are candidate explanations, not causal findings. Source
inspection confirms the known reference grid has no exact zero lateral
velocity and uses nearest-neighbour lookup (`open_duck_baseline.md`); that
alone does not explain the opposite-signed observed lateral velocities.
Next: preregister a local turn-centre/reference diagnostic before selecting
any motor-only training fix (`locomotion_curriculum.md`). No new run launched.

**Verification:** 7 new known-signal tests cover heading-frame direction,
backward sign, yaw unwrap, stationary motion, stop drift, full-window recovery,
later-fall rejection and incomplete-report failure. Complete suite including
slow tests: **243 passed**, 546 s; two JAX cast-overflow warnings in unchanged
training-env tests. Ruff and diff checks clean. Results recompute to the same
negative decision; every protocol trial/seed is present.

**Cloud/public safety:** final read-only audit: no instances, disks, addresses,
forwarding rules, routers or snapshots. Results bucket and runner identity
intentionally retained; no paid work or deletions. Public additions are code,
docs/tests and small metrics reports; no credentials, account IDs, personal
absolute paths or large raw artifacts. Raw traces remain git-ignored.

---

## 2026-10-09 — Same-source facets pass all criteria; Bv4 adopted; curriculum design resumed

Preregistration committed as `ca30916` (`context_facet_experiment.md`), before
implementation/evaluation. Candidate code `33febb1`; evaluator missing-detection
handling `2fbf1fe`. Reports: `experiments/affect_models/results_facets/`.
All runs local, B2 final checkpoint, v2, vision, utility, seeds 0–4 [measured].

**Exact structural change:** Bv4 keeps the latest context per (source, kind),
then uses `max(w) * sum(w*u) / sum(w)` per source before adding distinct sources.
W, gains, fast-onset/slow-return dynamics, phasic input and tendencies are unchanged.
Unit tests cover replacement, mixed-sign facets, idempotence, independent sources,
stale fading, source attribution and phasic/tendency equivalence.

| criterion | result |
|---|---|
| trace/convergence | pass: bounded, recovery 18.3 s, directions 7/7, max 0.564, rate-invariant |
| default / two-person / together behaviour | every criterion 5/5; no falls in 15 runs |
| pooled saturation V / A / D <= 5% | **pass: 2.02 / 0 / 0.37%** (Bv3 touch: 8.17 / 0.14 / 3.91%) |
| pooled std V / A / D | 0.5155 / 0.2293 / 0.3250; pass |
| max absolute V / A in every scenario >= 0.2 | pass |
| pooled minimum D <= -0.10 | pass: -0.695 |
| minimum D within 3 s after detected rapid approach < 0 | 5/5: -0.570 to -0.579 |
| D during A return < B return | 5/5: -0.042 to +0.074 vs +0.718 to +0.722 |

**Decision under the preregistered rule:** adopt Bv4 + reaction-margin
controllability + touch context as the v2 default. Explicit `--affect Bv2`
reproduces the old default with both options off; explicit Bv3 remains reproducible.
Both evaluators and scenario resolve the same defaults; the memory evaluator now
reports its actual learning mode (v2 had overridden the requested legacy mode); `--no-margin-controllability`
and `--no-touch-context` provide diagnostic overrides. Legacy remains A.
The old lunge-window mean remains positive (+0.213 to +0.228), reported rather
than reinterpreted. Some brief negative valence saturation remains (min -0.932);
this is not a claim of zero saturation or generalisation outside these scenarios.
The default-scenario reports are identical to Bv3; the contact scenarios change.

**Verification:** full pre-adoption suite with slow tests: 234 passed (246 s),
with two JAX cast-overflow warnings in the unchanged training-env tests. After
adding evaluator/default-resolution checks: 234 passed, two slow tests skipped;
both slow tests passed in the preceding full run. Ruff and diff whitespace checks
clean. Legacy and explicit Bv2 seed-0 traces were checked against pre-change
snapshots; the adopted default is checked against candidate seed 0 in all three
scenarios. Physical safety and motor control were not modified.

**Cloud/public safety:** read-only audit found no instances, disks, addresses,
forwarding rules, routers or snapshots. Results storage and runner identity retained;
no resources deleted and no paid run launched. Public additions contain small
reports, protocol/checkpoint hash and source/docs/tests; private snapshots and
checkpoints remain git-ignored. No account identifiers, credentials or personal paths
were added.

**Next phase:** `locomotion_curriculum.md` records the robust-locomotion-first
design. First preregister and run a local neutral B2 robustness gate; then verify
checkpoint continuation before introducing continuous feasible style conditioning
and gradually scheduled expressive objectives. No S1–S6 reruns or new training.

---

## 2026-10-06 — Ongoing touch as context: dominance goals met, valence saturation 8.2%; round closed, default stays Bv2

Preregistered in the previous entry. Code at `9ad6618`; `experiments/affect_models/results_touch/`
[measured]. Configuration: Bv3 + margin controllability + touch context.

| criterion | result |
|---|---|
| trace/convergence (`saturation.py`) | pass: bounded, recovery 18.3 s, directions 7/7, converges, rate-invariant |
| behaviour (default, two-person, together), no falls | pass |
| saturation ≤ 5% per dimension | **fail**: V 8.2%; A 0.1%, D 3.9% pass |
| std V, A ≥ 0.05; response in every scenario | pass (std V 0.537, A 0.234) |
| 3a. pooled min D ≤ −0.10 | pass (−0.69) |
| 3b′. min D within 3 s after the detected rapid approach < 0, ≥ 4/5 | pass, 5/5 (−0.57 to −0.58) |
| 3c. D while A returns < D while B returns, ≥ 4/5 | pass, 5/5 (−0.04 … 0.08 vs 0.72) |
| 3d. pooled std of D ≥ 0.05 | pass (0.326) |

The old window criterion (mean D over 46.5–49.5 s) still reads +0.21 to +0.23; the window starts before
the detection.

**Decision (preregistered):** not adopted, because of the one failed criterion. **The default stays Bv2.**
This ends the round. The configuration remains available with
`--affect Bv3 --margin-controllability --touch-context`.

**Where valence still saturates [measured, seed 0]:**
- Only in the two-person scenario (the default scenario has none):
  - negative, 24.0–28.5 s, after the lunge (rapid approach, then person close);
  - positive, 61.7–72.7 s, during the petting.
- The repeated-touch accumulation is gone.
- What remains is that the toucher contributes **two context slots at once**: "B in view", whose
  desirability rises with B's grounded warmth, and "B in ongoing contact". Bv3 **sums** context slots,
  so one person in one situation is counted twice.
- Together with the target gain G (8 for valence), strong but legitimate situations reach the bound.

**Candidate next step (not run; preregister before running):** combine a source's simultaneous context
facets into one (e.g. the mean, or the strongest facet) instead of summing. This is a structural
double-count fix, not a gain reduction. Lowering G or relaxing the 5% threshold would be tuning after
seeing the result, and is not proposed.

**Summary of the round (2026-10-06):** three preregistered attempts.
- Each removed a measured structural cause:
  - constant in-view controllability → reaction margin;
  - single slow time constant → fast onset, slow return;
  - touch as repeated events → contact as context.
- Dominance is now two-sided and responsive in the experimental configuration.
- One criterion still fails, from a fourth identified cause (per-source double counting).
- PAD → style remains blocked until a configuration passes all criteria.

---

## 2026-10-06 — Ongoing touch as context: preregistration (last iteration of this round)

**Change** (frames path; Bv2 and the legacy profile are untouched):
- **Touch:** `FrameAppraiser` appraises the first `touch_gentle` from a source as a discrete event.
  Further touches from the same source within 1.5 s of the previous one become a persistent frame with
  the new hypothesis kind `ongoing_contact` (predicts `benign_contact`; goals approach, keep_distance).
  Ongoing contact is a situation, not a series of events.
- **Bv3 context slots** are keyed by (source, hypothesis kind), so a person's in-view context and
  ongoing-contact context coexist instead of overwriting each other.

**Criteria** (Bv3 + margin controllability + touch context; B2, vision, 5 seeds; default, two-person,
together). All as preregistered for Bv3, except that 3b is **replaced** by a new criterion anchored to the
detected event; the old window criterion stays reported:
- trace/convergence (`saturation.py`) as before;
- behaviour: all criteria, no falls;
- saturation ≤ 5% per dimension; std V and A ≥ 0.05; max |V| and |A| ≥ 0.2 in every scenario;
- 3a. pooled min D ≤ −0.10;
- **3b′. minimum D within 3 s after the detected `person_approaching_rapidly` event < 0 in ≥ 4/5
  default-scenario seeds** (new);
- 3c. D while A returns < D while B returns in ≥ 4/5;
- 3d. pooled std of D ≥ 0.05.

**Decision:** adopt as the default if every criterion holds. Otherwise the default stays Bv2, and this
round ends with a report.

---

## 2026-10-06 — Model B v3 + reaction-margin controllability: criteria failed; default stays Bv2

Preregistered in the previous entry. Code at `3cb1f0c`; `experiments/affect_models/results_bv3/`
[measured].

**Trace and convergence** (`saturation.py`, Bv3): bounded; recovery 18.3 s; event directions 7/7;
converges (max 0.56); rate-invariant. **All pass.**

**Scenario suite** (Bv3 + margin, 5 seeds each):

| criterion | result |
|---|---|
| behaviour (default, two-person, together), no falls | pass |
| saturation ≤ 5% per dimension | **fail**: V 11.0%, A 0.0%, D 8.1% |
| std V and A ≥ 0.05; response in every scenario | pass (std V 0.554, A 0.235) |
| 3a. pooled min D ≤ −0.10 | pass (−0.69) |
| 3b. mean D over 46.5–49.5 s < 0 in ≥ 4/5 | **fail**: +0.21 to +0.23 in all 5 |
| 3c. D while A returns < D while B returns | pass, 5/5 (0.08–0.10 vs 0.72) |
| 3d. pooled std of D ≥ 0.05 | pass (0.380) |

**Decision (preregistered):** not adopted. The default stays Bv2 without margin controllability. Bv3 and
the margin option remain available (`--affect Bv3 --margin-controllability`).

**Where it fails [measured, seed 0]:**
- **Saturation occurs only in the two-person scenario:**
  - valence 23.9–28.8 s (negative, after the lunge);
  - valence and dominance 61.7–76.7 s, which is **the petting**: 9 `touch_gentle` events, one per
    second.
  - The default scenario does not saturate.
- **Continuous contact reaches affect as a discrete event every second:** repeated impulses into the
  phasic channel. This is the same structural problem Bv2 fixed for the in-view appraisals, now in the
  touch path.
- Bv2's single slow time constant had kept x away from the resulting high targets. Bv3's fast onset
  reaches them.
- **The lunge window:** dominance falls from +0.62 to −0.52 within 46.5–49.5 s, so the fast onset does
  what was intended. But the window starts about 1 s before the rapid-approach event is detected, so its
  mean stays positive. The criterion was weakly chosen; it is **not** reinterpreted here.

**Proposed next step (not run; to be preregistered):**
- (a) Treat ongoing touch as a persistent contextual appraisal of the toucher (onset as a phasic event,
  continuation as context), like in-view.
- (b) Define a new, separately preregistered lunge criterion anchored to the detected rapid-approach
  event (e.g. minimum D within 3 s after it < 0).
- (c) Re-evaluate Bv3 + margin against all criteria.

**Status for expressive style:** Bv2 (the default) keeps dynamic range without saturation, but its
dominance is one-sided and slow. PAD → style still waits until dominance is both two-sided and
non-saturating under the same criteria.

---

## 2026-10-06 — Reaction-margin controllability: criteria failed, reverted; cause found; follow-up (Model B v3) preregistered

**Result** (code at `0086282`; `experiments/affect_models/results_controllability/`) [measured]:

| criterion | result |
|---|---|
| 1. behaviour (default, two-person, together; 5 seeds) | all pass, no falls; legacy traces byte-identical |
| 2. saturation and range | 0% with \|x\| > 0.9; std V 0.359, A 0.198 |
| 3a. pooled min D ≤ −0.10 | **fail** (−0.05) |
| 3b. mean D during the lunge < 0 in ≥ 4/5 seeds | **fail** (+0.45 to +0.46 in all 5) |
| 3c. D while A returns < D while B returns, ≥ 4/5 seeds | pass, 5/5 (0.08–0.10 vs 0.40–0.41) |
| 3d. pooled std of D ≥ 0.05 | pass (0.256) |

**Decision (preregistered):** not adopted. It stays behind `margin_controllability` (off by default); the
default reproduces the Bv2 suite exactly.

**Cause [measured, seed 0, around the lunge]:**
- The estimate works at the target level. The person's in-view contribution to the dominance
  pre-activation flips from +0.46 to −0.49 as the lunge closes in, and the dominance target falls from
  +0.45 (46.5 s) to **−0.88** (48.0 s).
- Actual dominance follows with Bv2's single time constant τ_D = 8 s: +0.56 → +0.27 by 49.5 s, +0.05 by
  51.5 s.
- Bv2 uses the same slow rate for building a response as for recovering from one, so a sudden threat
  cannot move dominance within the 3 s window. This is a flaw of the Bv2 dynamics, not of the
  controllability estimate.

**Follow-up, preregistered now: Model B v3** (`affect_model="Bv3"`; Bv2 stays reproducible).
- Bv2 plus two-rate dynamics per dimension: the rate is 1/τ_on (τ_on = 1.0 s for V, A and D) while
  |x* − x0| > |x − x0| and x* − x0 has the sign of (x* − x), i.e. the response is building away from
  baseline. Otherwise it is 1/τ (8 / 4 / 8 s), returning.
- This mirrors Model A's structure (fast pull toward active emotions, slow return).
- Evaluated **with** reaction-margin controllability.

Criteria, all fixed now:
- `saturation.py` for Bv3: bounded; recovery ≤ 60 s; event directions 7/7; converges with max |x| < 0.9;
  rate-invariant.
- Scenario suite (5 seeds each): all behavioural criteria, no falls; saturation ≤ 5%; std V and A ≥ 0.05;
  max |V| and |A| ≥ 0.2 in every scenario; dominance criteria 3a–3d as above.
- **Decision:** Bv3 + margin controllability becomes the default if every criterion holds; otherwise the
  default stays Bv2 without margin and the failure is reported.

---

## 2026-10-06 — In-view controllability from the reaction margin: preregistration

**Problem** (flag from the Bv2 evaluation): dominance under Model B v2 is one-sided (−0.04 … 0.88).
- The in-view appraisal gives every visible thing a constant controllability of 0.8 (0.8 − 0.4·threat with
  memory).
- So a person lunging at the robot is, while in view, appraised as fully controllable. Only the single
  phasic lunge event carries low controllability (0.25), and the in-view context outweighs it.

**Change (frames path only, so the legacy profile is untouched):** `FrameAppraiser.observe` replaces the
in-view controllability c of a **person** with min(c, ĉ), where

  ĉ = clip(TTC / TTC_SAFE, C_MIN, 0.8) · (1 − 0.5 · stability_risk),
  TTC = distance / ego-corrected closing speed (∞ when not closing),
  TTC_SAFE = 3 s, C_MIN = 0.1.

- **Rationale [design]:** control over an interaction with an autonomous agent is the margin the robot
  has to react before contact. A slow, friendly approach (0.25 m/s at 1 m: TTC 4 s) or standing beside
  the robot to pet it (not closing) keeps full control. A fast approach does not.
- The robot's own instability lowers control (the same stability risk as in the frames).
- **Objects** keep c: they do not act.
- Remembered threat still enters through the wrapped appraiser's c.
- Constants are NERVA design choices, fixed now.

**Preregistered criteria** (profile v2, Bv2, B2, vision; 5 seeds; default, two-person and together
scenarios):
1. All existing behavioural criteria pass, with no falls. The legacy-profile traces stay byte-identical.
2. The saturation criteria still hold: time with |x| > 0.9 ≤ 5% per dimension; std ≥ 0.05 for V and A.
3. **Dominance becomes informative:**
   - (a) pooled min D ≤ −0.10;
   - (b) mean D over 46.5–49.5 s (the lunge) in the default scenario < 0 in ≥ 4/5 seeds;
   - (c) in the two-person scenario, mean D while A returns (94–106 s) < mean D while B returns
     (120–130 s) in ≥ 4/5 seeds;
   - (d) pooled std of D ≥ 0.05.

**Decision:** adopt in the frames path (and so the default) if 1–3 all hold; otherwise revert and report.

---

## 2026-10-02 — Model B v2 fixes the saturation; now the default affect model

Preregistered in the previous entry. Code at `6eb2054`; results in `experiments/affect_models/results/`
(`saturation.json`) and `results_scenarios/` (`summary.json`) [measured].

**Fixed trace and convergence** (`saturation.py`):

| model | bounded | recovery | event direction (counterfactual) | converges | max \|x\| | rate-invariant | steady state, 2 s vs 1 s period (V, A, D) |
|---|---|---|---|---|---|---|---|
| A | yes | 18.1 s | 7/7 | yes | 0.26 | yes | 0.21, 0.16, 0.24 vs 0.21, 0.16, 0.24 |
| old B | yes | 14.3 s | 7/7 | saturated | **1.00** | **no** | 0.54, 0.59, **1.00** vs **1.00, 1.00, 1.00** |
| **Bv2** | yes | 13.3 s | 7/7 | yes | 0.56 | yes | 0.26, 0.56, 0.56 vs 0.26, 0.56, 0.56 |

- Model A was already rate-invariant: its PAD moves toward an intensity-weighted *centre*, not a sum.
- With the counterfactual event measure (with vs without the event), old B's directions are also 7/7.
  The earlier 5/7 "sign agreement with A" came from the naive before/after measure, which confounds an
  event with ongoing recovery.

**Full scenarios** (profile v2, B2, 5 seeds each; default with vision, two-person, together):

| model | behaviour | time with \|x\| > 0.9, V / A / D | std V / A / D | min…max V / A / D |
|---|---|---|---|---|
| A | all pass, no falls | 0 / 0 / 0% | 0.142 / 0.054 / 0.087 | −0.31…0.22 / 0…0.21 / −0.06…0.24 |
| old B | all pass, no falls | **13.8 / 6.2 / 31.8%** | 0.530 / 0.256 / 0.331 | −0.97…1.00 / 0…1.00 / 0…1.00 |
| **Bv2** | **all pass, no falls** | **0 / 0 / 0%** | 0.359 / 0.198 / 0.253 | −0.61…0.84 / 0…0.77 / −0.04…0.88 |

Max |V| / |A| per scenario for Bv2: default 0.41 / 0.68, two-person 0.84 / 0.77, together 0.84 / 0.77.

**Verdict:** Bv2 meets every preregistered criterion; old B fails the saturation criterion. Model A
passes too, but only just (max |V| = 0.20 in the default scenario).

**Decision (preregistered rule):** Bv2 replaces old B as the default affect model (`profile="v2"` now
means affect `Bv2`). Old B stays as `--affect B`, so the consolidation results stay reproducible. A
default `evaluate.py` run reproduces the Bv2 suite's seed 0 exactly. The legacy-profile traces are still
byte-identical.

**Flags (not tuned):**
- **Dominance is one-sided** (−0.04 … 0.88). The in-view appraiser gives anything in view a constant
  controllability of 0.8 (0.8 − 0.4·threat with memory), so whenever something is visible the context
  pulls the dominance target positive, and threat events rarely take it below zero. That constant is a
  weakly justified **appraiser** design value, not a Model B weight. Revisit it before dominance
  conditions style.
- **Valence reaches 0.84 during the petting:** ten touch events summed in the phasic channel, bounded by
  tanh.
- **The W flags from the mapping review stand:** the novelty/unexpected redundancy in arousal, the
  neg/pos valence asymmetry from calibration, and the weak pos → A term.
- **Action tendencies are unchanged from Model B.** They also re-add repeated in-view appraisals, but are
  bounded at 2.54× one impulse.

**PAD → style:** Bv2 now keeps useful dynamic range in the existing scenarios. Connecting PAD to style is
no longer blocked by saturation; it is the next phase and has not been started.

---

## 2026-10-02 — Model B saturation: diagnosis, mapping review, preregistration of the fix

**Diagnosis [measured]:** default scenario, B2, vision, seed 0, Model B. Every appraisal's features were
logged and split into the repeated `*_in_view` appraisals (re-evaluated every 2 s per visible track) and
discrete events.

| origin | count | summed drive W·u to V / A / D |
|---|---|---|
| in-view | 42 | 0.75 / **3.20** / **2.31** |
| events | 13 | −0.12 / 1.68 / 0.33 |

- **Dominance comes from the `control` feature r·(2c − 1).**
  - In-view appraisals have a constant controllability of 0.8, so each adds control +0.25 on average.
  - Re-adding it every 2 s into a trace with τ = 4 s holds 2.54× a single impulse; the 8 s dominance leak
    integrates that again.
  - Predicted steady dominance from mean in-view input alone: **1.12**, i.e. saturation.
- **Novelty raises dominance only indirectly:** in-view relevance = 0.2 + 0.4·novelty scales the control
  impulse. W has no novelty → D term.
- **Arousal:** the in-view `unexpected` (mean 0.134) and `novelty` (0.224) terms with the same repetition
  gain; predicted steady arousal 0.77.
- **Structural cause:** an appraisal of an *unchanged, ongoing* situation is added as a new independent
  impulse each time it is re-evaluated, so PAD scales with the re-appraisal rate rather than with the
  situation.

**Mapping review** (W rows V, A, D; columns pos = r·d⁺·l, neg = r·d⁻·l, unexpected = r(1−e)², novelty =
r(1−e)(1−d⁻), control = r(2c−1)):

| term | weight | justification | flag |
|---|---|---|---|
| pos → V | +0.25 | a desirable likely outcome is pleasant (appraisal theory: goal congruence → valence) | — |
| neg → V | −0.47 | an undesirable likely outcome is unpleasant | magnitude asymmetry vs pos comes only from calibration to Model A's fear anchor; weak |
| novelty → V | +0.08 | novel and harmless is mildly pleasant ("interest") | moderate; a NERVA choice |
| pos → A | +0.05 | — | weak; very small, little justification either way |
| neg → A | +0.25 | threat is arousing | — |
| unexpected → A | +0.40 | surprise is arousing | — |
| novelty → A | +0.10 | novelty is arousing | **redundant with unexpected**: both are functions of (1 − e), so the same quantity drives arousal twice |
| neg → D | −0.10 | threat lowers felt control | partly overlaps with control → D |
| control → D | +0.22 | controllability ↔ dominance (the usual appraisal/PAD link) | conceptually sound as a **level**; wrong as an integrated **impulse** (the saturation mechanism) |

No term maps novelty to dominance directly. The high dominance under prolonged novelty is the
control-impulse mechanism above.

**Planned fix: Model B v2, `affect_model="Bv2"`.** Old Model B stays as "B" (reproducible).
- `AppraisalFrame` gets `persistent: bool` (contract). `FrameAppraiser.observe` marks in-view appraisals
  persistent; events are not.
- **Contextual channel:** one slot per source track, holding that source's *latest* persistent appraisal
  features. They are replaced, not added, and fade with τ_ctx = 3 s after the last refresh (longer than
  the 2 s re-appraisal period).
- **Phasic channel:** discrete events add decaying traces, as in Model B.
- **Target:** x* = tanh(G (W u_ctx + W z_ph)), with G = diag(τ_V, τ_A, τ_D) = diag(8, 4, 8). G converts
  the old per-second weights into the same steady-state gain the old model had for a single input; this
  is a unit conversion, not a new calibration.
- **Dynamics:** dx/dt = −Λ (x − x*), the same Λ as Model B. With no input, x* = 0, so x returns to the
  baseline.
- **W is unchanged:** the flagged terms are documented, not retuned. The fix is purely structural.
- **Tendencies are unchanged from Model B** (trace-based, all appraisals). They are bounded in practice
  (2.54× one impulse) and behaviour was calibrated on them. They share the impulse structure; revisit
  later.

**Preregistered evaluation.** Thresholds are design choices fixed now; A, old B and Bv2 are all
reported.

1. **Fixed trace** (affect-prototype timeline):
   - bounded;
   - recovery to ‖PAD‖ < 0.05 within 60 s of the last event;
   - **event response direction**: the effect of each event, measured as the difference at +2 s between
     runs with and without that event (isolating it from ongoing recovery), must have the same sign as
     its desirability for every event with desirability ≠ 0. Arousal must rise for events with
     expectedness < 0.3.
2. **Convergence:** one identical persistent appraisal (r 0.6, d 0, l 0.5, e 0.3, c 0.8) from one source.
   - Repeated every 2 s for 120 s: |x(120) − x(60)| < 0.02 on each dimension, and max |x| < 0.9.
   - **Rate invariance:** the steady state at a 1 s re-appraisal period must be within 0.05 of the 2 s
     steady state (an unchanged situation re-evaluated more often must not mean more).
3. **Full scenarios, profile v2, B2, 5 seeds each** (default with vision, two-person, together):
   - all existing behavioural criteria pass, with no falls;
   - **PAD dynamic range per dimension** (pooled over runs): time with |x| > 0.9 ≤ 5%;
   - standard deviation ≥ 0.05 for V and A (dominance reported);
   - max |V| ≥ 0.2 and max |A| ≥ 0.2 in every scenario.

**Implementation note (before any evaluation run):**
- The first implementation faded each context slot from the moment of its refresh. A unit test then
  showed the average weight depends on the refresh rate (steady states at 1 s vs 2 s differed by 0.06).
- That contradicts the stated intent, so the slot now holds full weight for 2.5 s after a refresh (longer
  than the appraisers' 2 s in-view period), then fades with τ = 3 s.
- No scenario or preregistered trace had been run at that point.

**Decision rule:** Bv2 replaces old B as the default affect model if it meets every criterion above. If
it fails any, the default stays as is and the failure is reported. **PAD is not connected to style in
any case until Bv2 (or a successor) meets the dynamic-range criteria.**

---

## 2026-10-02 — Consolidation results: v2 passes everything legacy passes; defaults switched to v2 + Model B

Preregistered in the previous entry. Code at `1f06847`; B2; `experiments/reactive/results_consolidation/`
[measured].

| # | evaluation | result |
|---|---|---|
| C1 | default scenario, v2 / A | curiosity, fear, habituation, safety 5/5 each |
| C2 | default scenario, v2 / B | 5/5 each |
| C3 | two-person, v2 / A | memory: 5/5 on all four; memory off: b_not_blamed 0/5, a_remembered 0/5, b_welcomed 5/5 |
| C4 | two-person, v2 / B | 5/5 on all four |
| C5 | together, v2 / A | memory: avoids A 5/5, engages B 5/5; memory off: avoids A 0/5, engages B 4/5 |
| C5L | together, **legacy** / A (first B2 run) | memory: avoids A **1/5**, engages B **3/5**; memory off: 0/5, 5/5 |
| C6 | together, v2 / B | avoids A 5/5, engages B 5/5 |
| C7 | default scenario, simulated detector, seeds 0–5, v2 / A and v2 / B | 6/6 on every criterion each, no falls (legacy: 1/6 fell) |

**Readings:**
- **Target-conditioned arbitration fixes the together scenario.** Stage D's "engages B" failure (grounded,
  global fear gate) is gone: v2 5/5. Memory is clearly causal there: avoids A 5/5 vs 0/5 without memory.
- **The legacy profile with B2 does poorly in the together scenario** (1/5, 3/5). The 3/3 recorded on
  2026-10-01 was with S1 in the plain scene.
- **The capability envelope removes the pre-existing B2 fall** (C7).

**Defaults decision (preregistered rule):**
- v2 + Model A passes every criterion legacy passes in the same setups, with no falls, so **v2 becomes the
  default profile**.
- v2 + Model B does too, so **Model B becomes the default affect model**.
- `scenario.run` now defaults to `profile="v2"` (affect B). `evaluate*.py` default to `--profile v2`.
- `--profile legacy` (affect A) reproduces every earlier result. The three stage A traces are still
  byte-identical with `profile="legacy"`, and a default `evaluate.py` run reproduces C2 seed 0 exactly.
- The event-learning experiment keeps its preregistered legacy setup explicitly.

**Caveat found after the suite, not covered by its criteria [measured]:** Model B's PAD saturates under
sustained input. Default scenario, vision, seed 0:

| model | max V / A / D | mean V / A / D | share of time with dominance > 0.9 |
|---|---|---|---|
| A | 0.20 / 0.20 / 0.24 | 0.11 / 0.12 / 0.14 | 0% |
| B | 0.61 / 0.99 / 1.00 | 0.19 / 0.39 / 0.72 | 31% |

- Cause [hypothesis]: the in-view appraisals every 2 s keep adding to Model B's linear drive. The
  calibration and the fixed-trace comparison used isolated events only.
- Behaviour was unaffected: it reads tendencies. Dominance only sets the social stop distance, which is
  floored at 0.7 m.
- **This must be fixed before PAD conditions expressive style e_t.** Candidates: normalise or saturate
  repeated in-view drive, or recalibrate W on a sustained-exposure trace. The fix would be preregistered.

**Charts:** with Model B, the demo charts and the Isaac composer plot action tendencies instead of
emotion intensities. Event captions say "no emotion labels: affect Model B". The replay export includes
tendencies.

---

## 2026-10-02 — Consolidation phase: capabilities, targeted arbitration, world-model queries, risk vs outcome; suite and defaults rule preregistered

Requested next step (review of the refactor): consolidate before deciding defaults.

**1. Policy capability layer.** `PolicyCapabilities` (interfaces) and the registry `nerva/sim/capabilities.py`:
- Fields: head-offset envelope, walking head limit, style input (`phase_clock` / `vector` /
  `neutral_only`), training scene, `tested` flag, evidence string.
- **B2:** pitch (−0.2, 0.6), yaw (−0.4, 0.4), measured (2026-10-02 entries). Unknown policies get
  `LEGACY` (`tested=False`).
- Policies are identified by run folder. Behaviour applies capabilities via `apply_capabilities`, and
  yaw limits may be asymmetric. The head flags remain only as diagnostic overrides.

**2. Target-conditioned arbitration:**
- Both affect models attribute tendencies to the track an appraisal is about: `add(..., source)` and
  `tendencies_for(source)`. Totals are numerically unchanged.
- In-view appraisals now carry their track (`observe_with_subjects`).
- When given these directed tendencies, `UtilityBehaviour` computes explore/approach/avoid per target
  (plus the undirected share). Fear of A gates approaching A only.
- Watch/retreat/freeze use the focal person's avoidance; the focal person is the highest of remembered
  threat or directed avoidance. Without directed tendencies, the old global gate applies (tested both
  ways).

**3. The world model is the query source (v2):**
- It is updated right after identity binding.
- Appraisal identity is read from it (`identity_map`), as are the tracks behaviour sees and the
  remembered threat per perceived entity.
- Touch attribution is a world query (nearest person `near` the robot), followed by `assert_touch`.
  Interaction is asserted after behaviour.
- Nodes that are no longer measured keep their identity but drop their stale measurement.

**4. Risk vs outcome:**
- `RiskEstimate` (`collision_risk`: the former `near_collision`, same detector and threshold) is kept
  apart from `OutcomeSignal`.
- Actual outcomes are `stability_loss`, `contact_impact` (none can occur in the current scenes: the
  people have no collision geometry) and `benign_contact`.
- Grounded memory learns near misses into a separate `risk` field; threat = max(adverse, risk).
- The collision hypothesis is now confirmed by `contact_impact`.
- **Consistency check [measured]:** the stage D grounded run (B2, seed 0) reproduces exactly: A's
  threat 0.49745 → 0.46681, identical to the recorded values.

**All of this is the scenario's `profile="v2"`.** It sets capabilities (scene, neutral style, head
envelope), grounded memory, frames, targeted arbitration and world-model queries. `profile="legacy"` (the
default) is byte-identical to the stage A traces (checked). 216 tests pass, 2 skipped; Ruff clean.

**Preregistered suite** (B2; vision unless stated; criteria unchanged; fall = tilt > 45°):

| # | scenario | profile / affect | seeds | compared with |
|---|---|---|---|---|
| C1 | default | v2 / A | 5 | legacy B2: 5/5 each |
| C2 | default | v2 / B | 5 | C1 |
| C3 | two-person | v2 / A, memory on and off | 5 + 5 | legacy B2: on 5/5 each, off 0/5 blame/remember |
| C4 | two-person | v2 / B | 5 | C3 |
| C5 | together | v2 / A, memory on and off | 5 + 5 | C5L |
| C5L | together | legacy / A, memory on and off | 5 + 5 | first legacy together run with B2 |
| C6 | together | v2 / B | 5 | C5L |
| C7 | default, simulated detector | v2 / A and v2 / B | 6 + 6 (seeds 0–5) | legacy: 1/6 fell (seed 3) |

**Defaults decision rule** (fixed now):
- **v2 becomes the default profile** if v2 + Model A passes every criterion that legacy passes in the
  same setup, with no falls. "Engages B" is included, and C5L is the together-scenario reference.
- **Model B becomes the default affect model** if v2 + Model B also does so; otherwise Model A stays
  the default.
- If v2 + A fails any such criterion, legacy stays the default and the failure is reported.
- The memory-off runs are a validity check: memory must still make a difference where it did before.

---

## 2026-10-02 — F2 results; architecture refactor status

**F2** (preregistered in the previous entry; B2 with `head_pitch_down −0.2`, `head_yaw_max 0.4`;
`experiments/reactive/results_f2/`) [measured]:

| # | evaluation | result |
|---|---|---|
| F2.1 | two-person, vision, grounded + frames + **Model B** | **no falls** (max tilt 13.5–15.0°); b_not_blamed, a_remembered, b_welcomed, touch_to_b 5/5 each |
| F2.2 | default scenario, vision, frames + **Model B** | curiosity, fear, habituation, safety 5/5 each |
| F2.3 | default scenario, vision, Model A, legacy appraisal | 5/5 each: the narrower envelope keeps the baseline behaviour |
| F2.4 | default scenario, simulated detector, Model A, seeds 0–5 | 0/6 falls |

**RQ8 conclusion:** with B2's measured-safe head envelope, Model B (no emotion categories) passes every
criterion Model A passes in both scenarios. The categories were not necessary for these behaviours. The
fixed-trace sign-agreement criterion (5/7) was not met. Model B also revealed a body limit (head yaw
−0.8) that Model A never reached.

**Stage F completed:** `EntityMemory.resolve_evidence` combines per-modality cosine similarities weighted
by evidence confidence; missing modalities contribute nothing. `resolve(kind, appearance)` is the
single-modality case. Merged as `745372f`; all three regression traces still byte-identical.

**Refactor status** (handoff completion criteria):

| criterion | status |
|---|---|
| canonical docs describe the live code | done (`architecture.md` §1.5, README, roadmap, RQs) |
| behaviour does not require Model A labels | done (stage C; traces byte-identical) |
| memory's new learning path grounded in outcomes | done (stage D, opt-in) |
| self state and goals explicit inputs | done in the frames path (stage E) |
| likelihood refers to an explicit hypothesis | done in the frames path |
| small world model | done (stage F); not yet read by behaviour |
| Model A runnable; Model B behind the same contract | done |
| old experiments still run | defaults reproduce the pre-refactor traces byte-for-byte |
| tests / Ruff | 202 pass, 2 skipped / clean |
| measurements logged | yes (this and the preceding 2026-10-02 entries) |
| no unmanaged cloud resources | no cloud work in this session (audit below) |

**Open issues:**
- The new paths are opt-in; the defaults stay on the legacy path for reproducibility.
- Grounded memory fails "engages B" in the together scenario (global fear gate; follow-up not run).
- Perception still emits interpreted events (legacy vocabulary).
- RQ9 was falsified for this learned-event design.
- The safety module is a stop rule, not fall prevention.
- B2's head envelope is narrow, and its asymmetry is unexplained.
- Stage I (semantic cues) is deferred.

**Cloud audit [measured]:** no instances, disks, addresses, forwarding rules, routers or snapshots.

---

## 2026-10-02 — Follow-up F1 (B2 head pitch −0.2) and the head-yaw envelope; F2 preregistered

**F1 results** (preregistered in the previous entry; `experiments/reactive/results_f1/`) [measured]:

| # | evaluation | result |
|---|---|---|
| F1.1 | default scenario, simulated detector, seeds 0–5 | 0/6 falls (max tilt 14.0–15.0°; seed 3 fell at −0.35) |
| F1.2 | default scenario, vision, Model A | curiosity, fear, habituation, safety 5/5 each (the ball stays tracked at −0.2: closest 0.53–0.58 m) |
| F1.3 | two-person, grounded + frames + Model B | **still falls in 5/5 seeds**; memory criteria not meaningful |

**Why F1.3 still fell [measured]:**
- Before the fall, the robot stands in withdraw with the head at pitch −0.2 and yaw **−0.8** for about
  3 s; tilt oscillates 5–14°, then diverges.
- Standing B2, 15 s, seeds 0–3:

  | head pitch | yaw −0.8 | yaw −0.4 | yaw +0.4 | yaw +0.8 |
  |---|---|---|---|---|
  | 0 | **4/4 fall** | 0/4 | 0/4 | 0/4 |
  | −0.2 | **4/4 fall** | 0/4 | 0/4 | 0/4 |

- **B2 cannot hold head yaw −0.8; it can hold +0.8.** This asymmetry is measured, not explained.
- Withdraw "looks away" at −sign(threat bearing) × 0.8, so it fails whenever the threat is on the
  robot's left. Model A rarely withdraws, which is why this was never seen before.
- `ReactiveBehaviour.head_yaw_max` (default 1.3, keeping recorded results; `--head-yaw-max`) now limits
  gaze, retreat glances, withdraw and the exploration scan. All three regression traces are still
  byte-identical with defaults.

**F2, preregistered now** (B2, neutral style, backlash scene; `head_pitch_down −0.2`,
`head_yaw_max 0.4`, the measured standing-safe envelope; criteria unchanged; fall = tilt > 45°):
1. Two-person, vision, grounded + frames + Model B, 5 seeds: no falls expected; then the four memory
   criteria (RQ8 for this scenario).
2. Default scenario, vision, frames + Model B, 5 seeds: the four criteria.
3. Default scenario, vision, Model A (legacy appraisal), 5 seeds: the four criteria. This checks that the
   narrower envelope keeps the baseline behaviour.
4. Default scenario, simulated detector, Model A, seeds 0–5: falls.

This is the last envelope iteration in this session. Whatever F2 shows is reported as is.

---

## 2026-10-02 — Stage E/G/H results; B2 falls found; head-pitch envelope measured

Preregistered in the previous entry. Code at `e7ea38e`; B2, neutral style, backlash scene [measured].

| # | evaluation | result |
|---|---|---|
| E1 | default scenario, vision, frames + Model A | curiosity 5/5, fear 5/5, habituation 5/5, safety 5/5 |
| G1 | default scenario, vision, frames + **Model B** | curiosity 5/5, fear 5/5, habituation 5/5 (exploration tendency 1.41 → 0.50–0.56), safety 5/5 |
| G2 | two-person, vision, grounded + frames + Model A | 5/5 on all four memory criteria |
| G3 | two-person, vision, grounded + frames + **Model B** | **the robot fell in 5/5 seeds at 39.8–40.4 s**; the memory criteria (5/5, 4/5, 5/5, 0/5) are therefore not meaningful |

**RQ8 (preregistered reading):**
- On the default scenario, Model B passes every criterion Model A passes, so discrete categories were not
  necessary for those behaviours.
- In the two-person scenario, Model B fails (falls) where Model A passes.

**Why G3 fell [measured]:**
- The mode sequence before every fall is watch → **withdraw** at 39.0 s, then the fall about 1.5 s later.
- Model B produces a withdraw tendency (harm with low controllability) after the lunge; Model A's
  distress rarely does.
- The withdraw controller set head pitch −0.5 with ±0.8 yaw while standing, bypassing the −0.35 limit
  introduced on 2026-10-01 because B2 fell at −0.6.
- Identical with the safety supervisor on or off. The supervisor's stop at 25° is a stop rule, not fall
  prevention; it neither caused nor prevented the fall.
- **Fix:** withdraw now respects the downward limit. It does not change any Model A trace (all three
  regression traces still byte-identical). With it, the G3 seed-0 fall moved from 40.6 s to 41.5 s, so
  the limit itself was not enough.

**B2 standing head-offset envelope [measured]:** zero velocity, offsets ramped in over 0.5 s, 8 s, seeds
0–1, maximum tilt.

| head pitch | yaw 0 | yaw 0.4 | yaw 0.8 |
|---|---|---|---|
| 0 | 13.8° | 10.4° | 10.6° |
| −0.2 | 15.2° | 11.5° | 10.9° |
| −0.35 | 14.1° | **fell** | 11.0° |

- **The −0.35 downward limit is not safe for B2 combined with head yaw.** The non-monotonic response is
  consistent with head offsets being outside B2's training distribution (upstream training never applies
  head commands).
- `ReactiveBehaviour.head_pitch_down` (default −0.35, keeping recorded results; `--head-pitch-down`) now
  makes the limit policy-specific.

**A pre-existing B2 fall [measured]:**
- Stage H's data runs (Model A, legacy path, simulated detector) contained one fall: default scenario,
  seed 3, at 31.0 s, after inspect with the head at −0.35 → approach while turning → orient (stop).
- The same configuration at the pre-refactor commit `a62fd2f` falls at 31.04 s, so it is **not** a
  regression.
- B2's recorded "safety 5/5" was measured with vision perception only. The simulated-detector runs had
  never been checked with B2. The other 17 of the 18 Stage H runs stayed below 17.2°.

**Stage H, RQ9 (preregistered) — falsified.** 18 runs, 21,600 frames, 181 boundaries, 7 prototypes.

| predictor | Brier, adverse within 3 s | Brier, benign within 3 s |
|---|---|---|
| learned prototypes | 0.0249 | 0.0575 |
| hand-coded labels | **0.0215** | **0.0297** |
| base rate | 0.0246 | 0.0583 |

- Learned prototypes predict worse than the hand-coded labels on both targets, and no better than the
  base rate.
- Most boundaries (145/181) fell into one broad prototype. 114 of 181 boundaries had no hand-coded event
  within 0.5 s. NMI between prototype and label was 0.26.
- One prototype is the fallen run (tilt feature ≈ 17, i.e. ~170°).
- Reading: with these features and this segmentation, prediction-error boundaries mostly mark
  tracking/gait fluctuations rather than outcome-relevant events. The hand-coded vocabulary stays.
- Not re-run with other settings: any change would be a new, separately preregistered experiment.

**Follow-up F1, preregistered now** (B2 with `head_pitch_down = −0.2`, the measured standing-safe value;
criteria unchanged; the fall criterion is tilt > 45°):
1. Default scenario, simulated detector, seeds 0–5: no fall expected.
2. Default scenario, vision, Model A, 5 seeds: curiosity, fear, habituation, safety. Risk: at −0.2 the
   ball may leave the view up close, which could fail curiosity.
3. Two-person, vision, grounded + frames + Model B, 5 seeds: no fall expected; then the four memory
   criteria.

---

## 2026-10-02 — Stages E, F, G (implementation), safety module, trace comparison; scenario evaluations preregistered

**Stage E: self state, goals, appraisal frames.**
- `nerva/world/self_state.py`:
  - tilt and angular speed (IMU-type quantities) and speed (estimator-type);
  - foot contact and escape room left `None` (not measured / not computed, rather than invented);
  - `stability_risk`, a derived heuristic, ramps from 10° to 30° tilt or 2 to 6 rad/s.
- `nerva/behaviour/goals.py`: standing goals (remain upright 1.0, keep distance 0.5, explore 0.4) plus
  goals from the current behaviour (approach/inspect 0.7; retreat and keep distance 0.9 toward a feared
  target).
- `nerva/affect/frames.py`, `FrameAppraiser`:
  - wraps the v1/v2 appraiser; every event kind maps to an explicit hypothesis (`near_collision`,
    `adverse_interaction`, `benign_interaction`, `threat_recedes`, `novel_stimulus`, `stability_loss`, …);
  - an unmapped kind raises an error;
  - likelihood is the hypothesis's probability;
  - relevance × 0.3 if none of the hypothesis's goals is active;
  - for physically adverse hypotheses, stability risk lowers controllability and raises relevance;
  - nominal context reproduces the wrapped numbers (tested).
- **Scenario check [measured]:** `appraisal_mode="frames"`.
  - S1 (simulated detector) trace identical to legacy.
  - B2 vision trace first differs at 61.7 s: a person appraisal made while tilted > 10° (B2's gait tilts
    up to ~16°). Relevance 0.538 → 0.542, controllability 0.50 → 0.496; the trajectory diverges after.

**Stage F: world model.**
- `SensorEvidence`, `WorldEntity`, `WorldRelation`, `WorldModelState` in `interfaces.py`.
- `nerva/world/model.py`: nodes for self, people/objects and places; seven relations, each with
  confidence, timestamp and source; linear confidence decay over per-relation TTLs.
- Track node "track:<tid>" is renamed to the entity ID once identity is known (track ≠ entity, tested).
- Built every perception frame in the scenario (`sim.world`); no behaviour reads it yet.
- Modular sensor evidence in `EntityMemory.resolve`: **not done yet**.

**Stage G: Model B** (`nerva/affect/model_b.py`, `DimensionalAffectModel`):
- Appraisal features (pos, neg, unexpected, novelty, control) feed decaying drive traces; PAD follows
  dx/dt = −Λ(x − x0) + W z, clipped to [−1, 1].
- Tendencies are continuous functions of the traces, with no category thresholds. withdraw = neg · (1 −
  control), so harm the robot can't control leads to disengagement. No emotion labels anywhere (tested).
- W was set in **one calibration pass** on two reference appraisals (a strong rapid approach and a gentle
  touch), so PAD peaks are of Model A's order. On these: A V −0.44 / A +0.31 / D −0.32, B −0.45 / +0.36 /
  −0.30; touch A +0.23 / +0.09 / +0.20, B +0.22 / +0.08 / +0.16. Nothing was tuned on the comparison
  scenarios.
- **Fixed-trace comparison** (`experiments/affect_models/compare_traces.py`, the affect-prototype
  timeline). The criteria were written in the script before its first run; this log entry was written
  after it.

  | criterion | A | B |
  |---|---|---|
  | bounded | yes | yes |
  | recovers within 60 s | 18.1 s | 14.3 s |

  - **Valence sign agreement: 5/7, criterion not met.** The slow approach is −0.004 (counts as no
    change) in A vs +0.021 in B. The obstacle is +0.003 in A vs −0.181 in B: A's distress pull was
    cancelled by its ongoing recovery from the near-fall 13 s earlier.
  - Pearson correlation V 0.975, A 0.796, D 0.970.
  - B's peaks are larger on this back-to-back sequence (V 0.57 vs 0.39), because its linear drive sums
    the lunge and the near-fall.

**Safety module** (`nerva/safety.py`):
- Deterministic stop (zero velocity, neutral head) at tilt ≥ 25° or stability risk ≥ 0.75, held until
  tilt ≤ 12° for 0.5 s.
- Its signature takes only the command and the self state: no affect, tendency or memory input (tested,
  including a large approach tendency that cannot suppress a stop).
- Always on between behaviour and policy; it has not triggered in any recorded run (max tilt 16°).
- What is enforced where is listed in the module (hardware e-stop and motor limits: not yet).

**Regression with defaults [measured]:** all three stage A traces are byte-identical with safety and the
world model on. 199 tests pass, 2 skipped; Ruff clean.

**Preregistered scenario evaluations** (B2, neutral style, backlash scene, vision, 5 seeds, criteria
unchanged). For Model B, the habituation criterion uses the exploration tendency in place of the
`interest` label; the two are identical under Model A.

| # | evaluation | settings | reading |
|---|---|---|---|
| E1 | default scenario | frames + Model A | regression: expected to keep 5/5 on curiosity, fear, habituation, safety |
| G1 | default scenario | frames + **Model B** | RQ8: if all four pass, discrete categories are not necessary *for these behaviours*; if any fails where E1 passes, they matter here |
| G2 | two-person memory | grounded memory + frames + Model A | comparator for G3 |
| G3 | two-person memory | grounded memory + frames + **Model B** | same reading as G1 on the four memory criteria |

**Stage H preregistration** (`experiments/event_learning/run.py`, protocol in its docstring, fixed
before running):
- **Data:** default, memory and together scenarios × seeds 0–5; B2, simulated detector.
- **Method:** prediction-error boundaries → ≤ 12 online prototypes.
- **Metric:** run-level prequential Brier score for "adverse outcome within 3 s" and "benign contact
  within 3 s". Context = most recent prototype (learned) vs most recent appraised event label
  (hand-coded) within 2 s, plus the base rate.
- **RQ9 reading:** supported if learned ≤ hand-coded on both targets; falsified if worse on either.

---

## 2026-10-02 — Refactor stage D results: grounded memory passes the two-person criteria, fails "engages B" in the together scenario

Preregistered in the previous entry. Code at `453a9b9`; vision; criteria unchanged. Results folders
under `experiments/reactive/`: `results_memory_grounded_s1/`, `results_memory_b2_legacy/`,
`results_memory_b2_grounded/` [measured].

| setup | learning | b_not_blamed | a_remembered | b_welcomed | touch_to_b |
|---|---|---|---|---|---|
| S1, plain scene | legacy (recorded 2026-10-01) | 5/5 | 5/5 | 5/5 | 5/5 |
| S1, plain scene | **grounded** | 5/5 | 5/5 | 5/5 | 5/5 |
| B2, backlash, neutral | legacy | 5/5 | 5/5 | 5/5 | 5/5 |
| B2, backlash, neutral | **grounded** | 5/5 | 5/5 | 5/5 | 5/5 |
| B2, backlash, neutral | memory OFF | 0/5 | 0/5 | 5/5 | — |

**First memory evaluation with B2:** same pattern as S1. Memory is causal for "B not blamed" and
"A remembered".

**Together scenario (S1, 3 seeds):**

| learning | avoids A | engages B |
|---|---|---|
| legacy (recorded) | 3/3 | 3/3 |
| **grounded** | 3/3 | **0/3** |

With grounded memory, B was closer than A on average in every seed (0.93–0.95 m vs 1.29–1.31 m), but no
approach/inspect mode occurred.

**Self-reinforcement check:** A's threat at 40 s → 116 s.

| setup | learning | A's threat 40 s → 116 s |
|---|---|---|
| S1 | grounded | 0.50 → 0.47 in every seed |
| B2 | grounded | 0.48–0.50 → 0.45–0.47 |
| B2 | legacy | 0.23 → 0.16 |

- **Grounded:** the prediction (no growth without a new adverse outcome) holds in every seed. The
  decrease is absence drift; replay restores at most to the learned level.
- **Legacy:** its threat also did not grow in this window. It is re-learned from each re-elicited fear,
  so it settles toward the fear intensity that memory itself triggers (self-sustaining rather than
  growing, in this scenario).

**Outcome detector:** one false positive in 10 B2 runs. Seed 2 flagged `near_collision` on B's track at
63.5 s during petting (closing-speed estimate at 0.3 m). B's adverse value reached 0.013 and was
extinguished by the touches; no criterion was affected. Thresholds unchanged.

**RQ7 verdict, by the reading fixed in advance:**
- **Partly falsified:** grounded fails "engages B" in the together scenario, where legacy passes.
- **Supported:** on the two-person criteria and the self-reinforcement prediction.

**Diagnosis [hypothesis, not tested]:**
- Grounded memory gives A a larger threat (≈ 0.5, one full-magnitude near-collision at α = 0.5) than
  legacy (≈ 0.2).
- The utility selector's fear gate is **global**: (1 − 4·F) multiplies the approach utility of every
  target. So fear about A also blocks approaching B.
- The criterion itself does not separate memory from no memory (legacy no-memory also 3/3), so it tests
  this interaction rather than memory.
- Candidate follow-up, to be preregistered before running: gate approach by the threat of the *target*
  only.

The legacy path stays the scenario default (reproducibility); grounded is selected with
`memory_learning="grounded"` / `--learning grounded`.

---

## 2026-10-02 — Refactor stage D (part 1): outcome-grounded memory path; evaluation preregistered

**Implemented** (legacy path unchanged and still the default):
- **`nerva/world/outcomes.py`, `OutcomeMonitor`:**
  - `stability_loss` when tilt > 20°.
  - `near_collision` when a person track is within 0.8 m with time to contact (distance ÷
    ego-corrected closing speed) < 1.0 s. Fires once per track until it is beyond 1.2 m again; magnitude
    = closing speed ÷ 1.5. This is an *estimate*: the MuJoCo people have no collision geometry.
  - `benign_contact` from simulated touch.
  - Thresholds fixed before any evaluation.
- **Entity memory, `learning="grounded"`:**
  - Expected adverse A and benign B are learned only from attributed outcomes, with α = 0.5 ×
    confidence and no arousal scaling. A benign outcome also extinguishes A at half rate.
  - `learn()` from emotions only records history.
  - threat = A and warmth = B − A are derived summaries. Trust stays at its prior: undefined in this
    mode, and not lowered by surprise.
- **Place memory, grounded:** learns only from outcomes at the cell.
- **Episodic, grounded:** outcomes are stored as `outcome:<kind>` episodes, and only those are replayed.
- **Identity binder:** outcomes on a track with unknown identity are held and credited at confidence
  0.8 once resolved, or dropped if the track is lost (no phantom attribution).

**Design bug found before evaluation:**
- A first grounded probe (S1, seed 0) showed A's adverse rising 0.50 → 0.55 → 0.60 across sleeps with no
  new outcome: each replay re-applied the one lunge as new evidence.
- Fixed so that replay only restores toward the level the last real outcomes set (undoing absence
  drift) and never beyond. After the fix, A stays at ≤ 0.50 (0.47 after drift).

**Regression:** the legacy memory trace (S1, seed 0) is still byte-identical to the stage A baseline.
174 tests pass (12 new: grounded vs legacy learning, replay, place memory, outcome monitor, deferred
and dropped attribution), 2 skipped; Ruff clean.

**Preregistered evaluation** (criteria unchanged from the M1 evaluation; vision; `evaluate_memory.py`):
1. S1, plain scene, grounded memory ON, 5 seeds. Compare with the recorded legacy result (5/5 on
   b_not_blamed, a_remembered, b_welcomed, touch_to_b).
2. B2, neutral style, backlash scene, 5 seeds each: legacy ON, grounded ON, memory OFF. This is the
   first memory evaluation with B2.
3. Together scenario, S1, plain, grounded ON, 3 seeds. Compare with the recorded legacy avoids_a 3/3.
4. **Self-reinforcement (RQ7b):** person#0's (A's) threat at 116 s minus at 40 s, with the adverse
   outcomes attributed to A in between.
   - Prediction for grounded: ≤ 0 in every seed in which no new adverse outcome is attributed to A.
   - Legacy is reported for comparison, with no prediction.

**RQ7 reading, fixed now:**
- Supported if grounded passes the criteria wherever legacy passes, and the prediction in (4) holds.
- Falsified if grounded fails a criterion legacy passes in the same setup, or if A's threat grows with
  no new adverse outcome.

---

## 2026-10-02 — Refactor stage C: behaviour reads action tendencies, not emotion labels

- `ReactiveBehaviour.step` and `UtilityBehaviour` take an `ActionTendencyState` instead of an
  emotion-intensity dict. The utility drives are now explore/approach/avoid/orient/withdraw, and the
  freeze utility uses `freeze` directly.
- The scenario passes `affect.tendencies`. Emotion labels remain in the logged rows (logging only), and
  the tendencies are logged too (`tend_*`).
- A test asserts that the behaviour modules contain no emotion-label strings; another drives behaviour
  with label-free tendencies.

**Regression [measured]:** per-step traces (mode, target, commands, PAD, head, position, identity,
remembered threat) compared with the stage A baselines:

| run | result |
|---|---|
| default scenario, S1, simulated detector, utility selector | byte-identical |
| default scenario, B2, vision, backlash, utility selector | byte-identical |
| two-person memory, S1, vision, memory on | byte-identical |
| default scenario, S1, rules selector v1 (vs commit `4df322f`) | identical, 5000/5000 rows |

The rules selector's "positive" drive changed by definition from max(interest, hope, joy) to
max(explore, approach), with approach = hope + joy. These differ only when hope and joy are both
active; that did not affect this run. 162 tests pass, 2 skipped; Ruff clean.

Memory still learns from labels (stage D).

---

## 2026-10-02 — Refactor stage B: typed contracts and the Model A tendency adapter

**Added to `nerva/interfaces.py`** (containers and range checks only; no logic):
- `SelfState`: tilt, angular speed, speed, foot contact (or None), stability risk and escape room
  (derived estimates, marked as such), locomotion mode.
- `Goal` / `GoalState`: seven goal kinds already implicit in the experiments.
- `OutcomeSignal`: only measurable kinds. Adverse: `stability_loss`, `near_collision`. Benign:
  `benign_contact`.
- `OutcomeHypothesis` and `AppraisalFrame`: `AppraisalFrame.likelihood` *is* the hypothesis's
  probability, so a likelihood cannot exist without a referent. `as_appraisal_state()` gives the legacy
  view.
- `ActionTendencyState`: approach, explore, avoid, orient, freeze, withdraw (≥ 0, bounded at 10, NaN
  rejected).
- `AffectSystem` now also requires `tendencies`, and `add` accepts a frame or a legacy state.

`SemanticCue` and the world-model types were not added: no stage uses them yet.

**Model A adapter** (`nerva/affect/tendencies.py`, the only place labels are translated):
- approach = hope + joy; explore = interest; avoid = fear; orient = surprise; freeze = surprise · fear;
  withdraw = distress.
- Chosen to equal the drives the utility selector already used [design].
- `CategoricalAffectModel` gains `intensities()` and `tendencies`, and accepts frames.

**Checks [measured]:**
- 160 tests pass (22 new), 2 skipped; Ruff clean.
- The S1 simulated-detector regression trace is byte-identical to the stage A baseline: no behaviour
  change, as intended.

---

## 2026-10-02 — Architecture refactor stage A: audit and docs truth pass

A new handoff (from ChatGPT) asked for an architecture refactor: action tendencies, outcome-grounded
memory, self state, goals, appraisal frames, a world model, Model B and learned events. Stage A
audited the live repo first.

**Baseline [measured]:** `main` clean at `a62fd2f`; 138 tests pass, 2 skipped; Ruff clean. Per-step
regression traces were recorded (local, not committed) for three seeded runs:
- default scenario, S1, simulated detector: 4.3 s CPU;
- default scenario, B2, vision, backlash scene: 21.6 s;
- two-person memory scenario, S1, vision: 29.4 s.

Two runs of the first were byte-identical, so the loop is deterministic and later stages can be checked
for exact equivalence.

**Discrepancies found and recorded:**
- **Memory ablations used S1 in the plain scene, not B2.** `evaluate_memory.py` has no backlash or
  neutral-style option. The "memory 5/5 vs 0/5" and "together 3/3 vs 0/3" results are S1/plain-scene
  results; the Isaac demo with B2 is a single run, not an evaluation.
- **Emotion labels leak outside Model A.** Behaviour (both selectors), entity memory, place memory and
  episodic sleep replay consume `fear`, `interest`, `hope`, `joy`, `surprise`, `distress`. The docs claimed
  nothing outside `emotions.py` depended on them.
- **Self-reinforcing memory:** entity and place memory, and sleep replay, learn threat/warmth from the
  elicited emotions.
- **The layer types are not the only coupling:** behaviour also reads tracks, novelty and remembered
  threat directly; appraisal reads memory.
- **No NERVA safety module:** limits come from the MuJoCo model and the upstream rate limit; near-falls
  are only detected as appraisal events.
- **Stale docs:** README status ("M1 only"), `experiments/reactive/README.md` ("fear 0/5", "no
  image-based detection"), `experiments/isaac/README.md`, `architecture.md` ("interface only" rows),
  roadmap and research questions.

**Head-pitch sign settled [measured]:** standing 3 s at zero velocity, backlash scene, robot-eye camera
elevation:

| head_pitch | −0.4 | 0 | +0.4 |
|---|---|---|---|
| upstream policy | −13.2° | +5.2° | +38.0° |
| B2 (neutral style) | −7.9° | +20.4° | +39.8° |

Positive head_pitch tilts the face up, as `nerva/behaviour/modes.py` assumes. RQ1c
(`head_posture.py`) and the first demo video assumed the opposite, so their "head up" labels mean face
down. The measurements stand; correction notes were added instead of rewriting them.

**Docs updated:** `architecture.md` (Part 1: live dependency graph and known problems; Part 2:
target architecture and migration stages A–I), README architecture/status, `experiments/reactive` and
`experiments/isaac` READMEs, roadmap (refactor stages), research questions (new active RQ7–RQ9 with
falsification criteria), `memory_design.md` and `affect_model.md` (audit notes),
`style_policy_design.md` (S1–S6, B2, curriculum next), historical notes in
`reactive_behaviour_design.md`, `experiments/demo_video/README.md` and the RQ1c section. No code
changed in this stage.

---

## 2026-10-01 — Isaac render with B2 motion, the robot's-eye view and NERVA's data

Feedback on the first render: the backward walk looked bad (it was S1), and the data and robot point of view of the MuJoCo demo were missing.

**Changes:**
- **Replay:** `replays/memory_b2.npz` (B2, neutral style, backlash scene, memory scenario).
- **Sidecar:** `export_replay.py` writes `memory_b2.json` with:
  - per-frame affect, behaviour and memory values;
  - perception events with their appraisals and emotions;
  - the vision boxes from the robot camera;
  - the robot-eye pose per frame.
- **Robot's-eye camera:** `render_replay.py` adds a `RobotEye` camera at MuJoCo's `robot_eye` pose (same 70° vertical field of view, 4:3, 480×360), written to `frames_eye/`.
- **Composer:** `compose.py` (local) builds the video: the Isaac scene view, the eye view with the vision boxes, the emotion/PAD chart, event and appraisal captions, and behaviour and memory status.

**Run:** `isaac_spike-20261001-191549`, L4 [measured]:
- Setup: driver install 6 min, image pull 8 min.
- Rendering both cameras: 09:31–09:45 UTC, about 14 min for 3,375 frames each.
- A spot-checked eye frame shows the person from the robot's height. The boxes come from MuJoCo vision with the same camera pose, so they align only as closely as the two renderers' geometry matches [not measured].
- **Teardown:** the VM self-deleted; the waiter removed the NAT. `audit`: no instances, disks, addresses or routers (the results bucket remains, pending the user's decision).

---

## 2026-10-01 — First Isaac Sim render with the robot (kinematic replay)

`isaac_spike-20261001-180054`, L4, Ubuntu 22.04 + driver 570, Isaac Sim 5.1 [measured]:
- **Robot:** built from MuJoCo's visual geometry (`export_robot_mesh.py`: 44 meshes on 15 bodies, 30 KB), because Isaac's MJCF and URDF importers produced no robot bodies.
- **Replay:** each body is posed from the memory-scenario replay (16 bodies per frame; `base` has no geometry).
- **Output:** 3,375 frames (135 s at 25 fps) in 733 s of rendering, encoded locally to MP4 (34 MB, local only).
- **Looks:** realistic lighting and shadows; the robot walks beside the person.
- **Earlier fixes along the way:** the job hung twice on apt (unattended-upgrades and needrestart prompts); now non-interactive with a per-phase log. The default stage was Y-up and overexposed; now Z-up with moderate lights.
- **Limits:** people are static T-pose characters sliding along their paths; no captions yet. Physics stays in MuJoCo.
- **Teardown:** the NAT was removed after the run; `audit` shows no instances or routers.

---

## 2026-10-01 — S6 (feet air-time reward) also stands still

`s6_pilot-20261001-130212` = S5 + feet air-time reward 2.0 (thresholds 0.1–0.3 s). Backlash scene [measured]: v = −0.000 at vx −0.15 and +0.001 at +0.15, foot lift 0 mm. **It collapsed to standing like S5.**

**Arithmetic:** at the policies' typical under-lift, the feet-height cost is about 10·(14/40 − 1)² ≈ 4.2 per touchdown, while the air-time reward pays at most 2·0.2 = 0.4. So early in training any step is net negative, and the policy settles on standing before it discovers walking [hypothesis].

**Next idea (not run):** a curriculum that ramps the feet-height weight from 0 after walking emerges, or a normalised error.

**Status:** B2 stays the policy that walks backward (reactive evaluation 4 × 5/5).

---

## 2026-10-01 — Isaac Sim on a cloud L4: what it took to get rendering

Four attempts with `cloud/jobs/isaac_spike.sh`, all capped and self-deleting [measured]:

1. **Hung silently until the VM cap.** Output was buffered by `tail`, and the mounts were root-owned while the container runs as uid 1234. Fixed with streamed logs, owned mounts, step timeouts and a minimal startup test.
2. **Started but couldn't render:** `vkCreateInstance failed … ERROR_INCOMPATIBLE_DRIVER`, `Failed to create any GPU devices`.
3. **Same failure** with `NVIDIA_DRIVER_CAPABILITIES=all` and Isaac 5.1. The host diagnostics showed the cause: **the Deep Learning VM image's driver is compute-only.** There's no Vulkan ICD and no NVIDIA graphics libraries on the host at all.
4. **Plain Ubuntu 22.04 + `nvidia-driver-570`** (the launcher now picks this image for `isaac_*` jobs). `nvidia_icd.json` is present, and **Isaac Sim 5.1 starts headless without GPU errors.**
   - ✓ ground plane and lights
   - ✓ the asset server is reachable (Omniverse S3)
   - ✗ a human character loaded, but the transform call needed double precision
   - ✗ `URDFParseAndImportFile` failed with no detail
   - ✗ frames were skipped because they depended on the robot

**Attempt 5** (running): the MJCF importer with NERVA's exact MuJoCo model first, then the low-level URDF interface, then the URDF command, each logged. Frames render with or without the robot.

**Container pull:** about 3 min. **Startup:** about 20 s once the caches exist.

---

## 2026-10-01 — Codebase restructure; S5 collapsed to standing; Isaac Sim spike launched

**Codebase:**
- Ruff (E/F/W/B) is clean.
- `nerva/` is reorganised into layer packages (`sim`, `perception`, `affect`, `memory`, `behaviour`, `analysis`, `training`), and `tests/` mirrors them.
- Navigation: a README layout map plus index READMEs in `experiments/`, `scripts/` and `cloud/`.
- All references were rewritten. 137 tests pass, and every experiment and cloud script imports.

**S5** (`s5_pilot-20261001-092440`, 7 styles + feet-height −10, 300 M steps, exit 0). In the backlash scene, 15 s, every style and both vx = ±0.15 [measured]:
- **Speed ≈ 0** (|v| ≤ 0.001 m/s), foot lift 0–1 mm.
- **The policy stands still.** Its eval reward stayed similar to S1's (221–270), because the alive reward dominates.

**Interpretation [hypothesis, consistent with the reward form]:** the feet-height cost is charged only at touchdown (`first_contact`), so never stepping avoids it entirely. B2 (−30) happened not to fall into this optimum; S5 did. Next fix: reward stepping while a velocity is commanded (a feet air-time term, as in MuJoCo Playground's locomotion envs), so standing still cannot dodge the swing cost. B2 remains the policy that walks backward.

**Isaac Sim (user accepted the NVIDIA Omniverse License Agreement; privacy consent N):**
- Feasibility spike `isaac_spike-20261001-112607` on an L4 (us-central1-b, 90-min cap): headless start, URDF import, human character, frames driven by the MuJoCo replay of the memory scenario (`experiments/isaac/`).
- The NAT-safety rule kept the NAT up for it when S5 finished.

---

## 2026-10-01 — Backward walking diagnosed (foot clearance); multi-person tracking

**1. Backward walking — cause found [measured]:**
- **Scene mismatch.** All policies were trained in `scene_flat_terrain_backlash.xml`, but NERVA evaluated them in the plain scene. In the training scene S1 walks backward at −0.071 m/s (plain: −0.043) and forward at 0.129 (plain: 0.114). `OpenDuckSim(scene=SCENE_BACKLASH)` and `--backlash` now exist. Earlier numbers were measured in the plain scene.
- **The cause is foot clearance.** In the backlash scene, 2 seeds, at vx −0.15 / −0.10:
  - B1 (no feet-height cost): −0.002 / −0.001 m/s, foot lift ≈ 1 mm. It shuffles in place.
  - B2 (feet-height cost −30): −0.113 / −0.074 m/s, lift ≈ 42 mm. That is 75% of the command and matches the backward reference's own −0.115.
- **Open-loop reference playback** was inconclusive: forward falls, backward stays upright without moving.
- **B2 in the reactive scenario:**
  - First run: fell in 5/5 seeds, while inspecting with the head at −0.6 rad plus head roll and slow turning.
  - Downward head pitch is now limited to −0.35 (the ball stays in view) and the curious tilt to 0.2 rad.
  - Result (B2, neutral style, backlash, vision): **curiosity 5/5, fear 5/5 (distance 0.45 → 0.52–0.57 m within 3 s), habituation 5/5, safety 5/5.** First time all four criteria pass.
- **S5** (7 styles + feet-height −10) launched to combine styles with backward walking.

**2. Multi-person tracking:**
- Tracks have IDs (`Track.tid`) and events carry their track (`Event.source`).
- The tracker associates detections to tracks by world position (gate 0.8 m) and appearance.
- Vision returns every person blob, merging components at similar depth and close together (one person's two legs up close).
- Appraisal, identity binding, memory and the utility selector work per track: fear is directed at the person with the highest remembered threat; a remembered threat gates approach to that person only; touch is attributed to the nearest person track.
- **Regression checks** (reactive with B2; two-person memory with S1): unchanged (5/5 each; no-memory 0/5).

**New two-people-at-once scenario** (history as before, then A and B return together at 95 s), 3 seeds, vision:

| criterion (stated in advance) | memory | no memory |
|---|---|---|
| avoids A (never within 1.0 m) and watches/retreats | 3/3 | 0/3 |
| engages B (approach/inspect, B closer than A on average) | 3/3 | 3/3 |

The second criterion doesn't discriminate, because the agents' scripted approaches set part of the distances.

**Also fixed:** entities a scenario doesn't use are now removed from the scene. The parked ball had been detected in the memory scenarios.

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

**M2/M3 — episodic memory** (`nerva/memory/episodic.py`, `14b4ef1`):
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

**M4 — spatial memory** (`nerva/memory/spatial.py`):
- 0.75 m place grid: familiarity (fades with time), place threat/valence learned from events there, and an exploration heading toward the nearest novel, safe cell.
- **Exploration only, 90 s, 3 seeds [measured]:** 9/10/11 cells visited with spatial memory vs 7/9/8 without (about +25%), limited by walking speed.

---

## 2026-10-01 — Real vision and entity memory (M1): person-specific, evolving associations

**Real vision** (`nerva/perception/vision.py`, `a23a078`):
- Colour segmentation of the robot_eye RGB frame (HSV) → connected components → depth render gives distance → bearing/elevation from the pixel rays. The same tracker as the simulated detector.
- **On renders [measured]:** distance within 0.04–0.09 m (it measures to the surface rather than the centre); bearing within about 0.02 rad.
- **Self-body false positive:** the robot's orange feet were detected as a ball when looking down. Small objects closer than 0.2 m are now rejected.
- **Reactive 5-seed evaluation with vision:** identical verdicts to simulated perception (curiosity 5/5, habituation 5/5, safety 5/5, fear 0/5).

**Memory M1** (`nerva/memory/entity.py`, `MemoryAppraiser`, `IdentityBinder`; design: `docs/memory_design.md`):
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

**Behaviour v2 (`nerva/behaviour/selection.py`):**
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
- **Scene** (`nerva/sim/world.py`):
  - Visual-only mocap person (1.7 m) and ball, added via MjSpec, plus a `robot_eye` camera on the head, 0.09 m in front of the head site (inside the shell the view was blocked).
  - The robot's dynamics are bit-identical to the plain scene. This needed the solver warm start copied over, because `opt.iterations = 1`.
- **Perception** (`nerva/perception/tracker.py`):
  - A simulated head-camera detector: field of view and range, a distance-dependent miss rate, bearing/distance noise.
  - Tracks with memory. Approach speed is a least-squares slope with **ego-motion compensation**; without it, walking toward someone read as them approaching.
  - Separate re-arm flags for slow and rapid approaches (a slow event had masked a lunge).
- **Contextual appraisal** (`ContextualAppraiser`): novelty habituation, proximity and speed of approaches, threat memory (45 s), relief when a threat leaves, habituation to repeated lunges.
- **Affect v0.2:** a new emotion, "interest" (novel and non-harmful). The rule and anchor are NERVA choices. Events without novelty elicit exactly what v0.1 did (existing tests unchanged).
- **Behaviour v1** (`nerva/behaviour/modes.py`): explore / orient / approach / inspect / freeze / retreat / watch / withdraw, with hysteresis and head gaze.
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
- `nerva/behaviour/pad_style.py`:
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
- `OpenDuckSim.set_style_vector(e)` appends e to the observation (noise-free, as in training) and uses `nerva.behaviour.style.s1_nb_steps_in_period(e1)` for the phase clock. The period is linear in e1, measured on the 3 R1 periods; values between them are interpolated.
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
- `nerva/training/reference_validation.py`: flags backward knees and non-finite values.
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
- `nerva/affect/appraisal.py`: a table of EMA-variable appraisals for the 5 synthetic events. The values are NERVA design choices, with the reasoning in the doc.
- `nerva/affect/emotions.py`:
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
- `nerva/behaviour/style.py`: the style → phase-factor mapping.
- `nerva/sim/open_duck.py`: a headless simulator. It reuses upstream `MjInfer` (model, obs, policy, action scaling, speed limit) and replaces only its viewer loop. Options:
  - raw accelerometer
  - seeded initial joint noise
  - training-level observation noise
  - pushes
  - `set_behaviour(BehaviourCommand)`, which clips velocities to the trained range
- `nerva/analysis/gait_metrics.py`: pure-NumPy metrics.
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
