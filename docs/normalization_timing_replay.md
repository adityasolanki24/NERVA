# Fixed-weight same-batch normalization-timing replay

Preregistered 2026-10-09 from clean baseline `82360a0`, before implementation or
capture/replay evaluation. Variance epsilon 1e-4 reduced amplification, but the
previous 16-transition fresh stage failed KL <=1. Source inspection shows the
constant-LR PPO path collects with old statistics, updates them on that batch,
then evaluates the loss. Hypothesis: statistics replacement alone can exceed
the KL screen before a gradient update, on the first collected batch.

Use exactly the prior candidate/reference/physics/wrapper, network (32,32),
policy state 101/value privileged state 212, 14 actions, seed 7, Welford variance
epsilon 1e-4. Verify prior protocol, package/core-source/reference hashes and
normalization-operator hash before capture. The existing negative result stays
unchanged; this is a diagnostic, not a retry or a new optimizer setting.

Initialize using pinned Brax PPO `train(num_timesteps=0)` with the same remaining
options. This initializes parameters and Adam but returns before rollout or SGD;
normalization count must be zero. No optimizer updates anywhere in this replay.
Strip weak types as the pinned training epoch does. Reconstruct the inspected
first-epoch/reset/unroll/minibatch/loss PRNG keys (seed 7, process 0, one CPU
device). Use the same pmap axis and candidate wrapper. Initializer reset and
capture reset may each initialize two environments; neither is a transition.

Capture only one batch: two environments, unroll 4, episode 8, action repeat 1,
no randomization/evaluator rollouts, exactly eight environment transitions.
Use Brax's stochastic inference and `acting.generate_unroll`; mirror the pinned
outer one-unroll scan and [batch,time] reshape. Preserve observations and next
observations, actions/rewards/discounts, truncation and policy extras (raw actions,
behavior log probabilities and distribution parameters). Preserve this simulated
dataset and initial parameters in ignored storage with hashes. No second batch,
new policy, gradient calculation or replay-dependent collection.

Update a copy of the initial normalizer from this batch, with the same pmap
reduction/Welford operator as training; count must become eight. Keep policy and
value weights literally identical. Apply the inspected first-minibatch permutation
and use its same loss key for both loss replays. Evaluate the pinned PPO loss
twice on this one batch with fixed weights: old statistics and updated statistics.
All PPO loss options remain those of `neutral_ppo_restore_smoke.md`.

Report loss components/value predictions, raw distribution means/scales, action
and log-probability drift, normalized observation maxima and per-slot mean/std
changes. Public reports contain small aggregates/hashes only. Compute independent
NumPy float64 Gaussian KL(old behavior || replay) both without and with Brax's
`log(scale_ratio + 1e-5)` term. The latter introduces a small positive self-KL;
do not silently label it zero. Verify the stabilized manual mean against official
loss KL at atol=1e-5, rtol=1e-5. Report conventional analytic KL separately.

All integrity requirements: one CPU device/process; finite initial parameters,
batch, normalizers, metrics and diagnostic outputs; exactly [2,4] batch/time,
eight transitions, statistics count 0->8; same network-weight and batch hashes
before/after replays; literal-byte initial/updated checkpoint roundtrips; old
replay logits max error <=1e-5 and raw-action log probabilities max error <=1e-4
against stored behavior; conventional analytic old self-KL absolute mean <=1e-6
and Brax self-KL <=0.001; manual/official stabilized KL agreement in both cases.

Timing hypothesis support requires integrity plus updated-statistics official
KL >1.0 while old-statistics KL <=0.001. A finite updated KL <=1 is a preserved
negative diagnostic. Report policy/value/drift measures descriptively without
adding criteria afterwards. This isolates a preprocessing contribution on one
frozen batch; it does not reproduce the previous two-update averaged KL, evaluate
SGD timing fixes or prove general motor/learning stability. Updating statistics
after SGD could still change the deployed policy, so a near-zero training KL
alone would not establish a safe solution.

One attempt/worker, CPU only, hard 600 s parent watchdog including initialization,
capture, both replays, saving and checks. Maximum eight physical transitions and
zero optimizer updates. Stop on integrity mismatch, nonfinite output, API/device
error or timeout; preserve raw/partial reports, no unregistered retry. New public
output `results_normalization_timing/`; raw artifacts under ignored
`experiments/cloud_runs/normalization-timing/`. Refuse overwrite. No paid/cloud
work, package installation, upstream edits, default promotion, deployment or
expressive objectives. Complete tests including slow checks, Ruff, docs/log,
public-content audit, commit/push and final cloud resource audit.

## Preserved outcome

The single capture attempt (`8a2f467`) stopped after 98.28 s at old-control
integrity: log-probability error 4.315092 exceeds <=1e-4 despite matching logits
and analytic self-KL. Exactly eight transitions and zero optimizer updates;
updated replay not executed, timing hypothesis unevaluated. Public partial
reports: `results_normalization_timing/`. Raw dataset/weights remain ignored.
Offline old-control inspection with an explicit dynamic batch yields error
9.06e-6. An offline retry requires a new preregistration; the original failure
and criteria stay unchanged.

A separately preregistered schema-corrected offline retry subsequently passes
all integrity criteria on the exact retained batch/weights: updated-statistics
KL 10.296526 versus old 0.000140131, zero new transitions or optimizer updates.
See `normalization_timing_offline_retry.md` for the preserved admission abort,
retry protocol, results and limits. This original capture remains invalid.
