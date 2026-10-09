# On-policy identity-preprocessing numerical and restore screen

Preregistered 2026-10-10 after the passing offline schedule comparison
(`normalization_schedule_comparison.md`, `e241208` / `d42b541`) and before
implementation or on-policy evaluation. Fixed old mean=0/std=1 is raw identity
preprocessing. Hypothesis: the trainer's normalize_observations=False avoids
the observed normalization drift and passes numerical/checkpoint screens on
the same tiny fresh/warm budget. It does not establish locomotion readiness.

Exactly the previous variance-epsilon control configuration, except
normalize_observations=False: seed 7, two CPU environments, (32,32) networks,
state101/privileged212/actions14, candidate references/rewards/physics/wrapper,
epsilon1e-4, 16 fresh plus 16 parameter-warm-start transitions (32 total),
batch2/unroll4/one minibatch/one update per batch, episode8/action repeat1,
LR1e-4 and all PPO loss/options unchanged. Native Brax still updates and saves
unused running statistics: count16->32 is bookkeeping, not applied preprocessing.
No Welford freeze claim or use of normalize_until_count. Adam/RNG/counters
restart on warm load, exactly as before. No native comparator training rerun.

Before training admit existing core/package/reference hashes and frozen prior
control reports/checkpoints; compare exact configuration except the declared
normalization flag. Admit the retained original batch/initial parameters using
the previous checks; its first fresh batch shares the same initial raw-identity
policy, seed and reset/unroll schedule. Keep its original behavior extras and
hash intact. Use it only for post-training deployment replay; no extra physical
transitions or optimizer updates from this diagnostic.

All original plumbing criteria remain: callbacks[0,16] in both stages;
finite initial/final parameters and reported losses; policy/value each update
>1e-12; statistics counts 16->32; literal-byte disk/checkpoint/warm-initial trees;
ten deterministic/same-key restored-action probes per stage error<=1e-6,
finite/bounded actions. Checkpoint configuration and contract explicitly disable
normalization; refuse incompatible warm starts. Existing runners/defaults retain
normalize_observations=True unless this opt-in study requests False.

Additional fixed numerical criteria, all required in both stages: reported
epoch KL<=1 and value loss<=100; replay final deployed policy on the unchanged
retained first batch using identity preprocessing, official KL<=1 and value
loss<=100. Distinguish reported mean pre-update training KL from post-training
KL against the original behavior, particularly for the warm stage's cumulative
drift. Independent NumPy stabilized KL agreement atol/rtol1e-5; raw-action
log probabilities agreement atol1e-4/rtol1e-5. Finite replay/probes. Changing
only saved normalizer mean(+10) and std(*.01) must leave policy logits and
value predictions unchanged within1e-6 on the five existing synthetic probes;
these are identity-preprocessing checks, not physically achievable states.
Preserve failed criteria even if plumbing passes. Compare fresh/warm training
metrics descriptively to frozen results_normalization_control; no new relative
criteria or tuning after outcomes.

One attempt/worker, hard900 s parent watchdog including imports/compilation,
both stages/replays/saving/checks. Exactly32 new environment transitions and
four optimizer updates; no evaluator rollouts, additional calibration, ONNX
export, expressive objectives or dependencies/upstream edits. Stop on integrity,
nonfinite/API error or timeout; stop after a finite stage screen failure before
starting another stage. Preserve checkpoints/partial reports, no automatic retry.
Public results_identity_preprocessing; ignored raw
experiments/cloud_runs/identity-preprocessing. Refuse overwrite. No paid/cloud
work, deployment, default promotion or artifact deletion. Complete full tests
including slow, Ruff, docs/log, public safety checks, commit/push and final audit.

Passing permits separately preregistered local multi-batch neutral-learning
stability/command-coverage work. It does not validate the long motor gate,
replace B2, justify paid training or authorize expressive objectives before
robust locomotion. Affine rebase/Adam continuation remains a separate question.

## Completed outcome

Preregistration `e627af3`, implementation `ed66f4e`: **all criteria pass**,
32 new transitions/four optimizer updates, 230.45 s under 900 s. Counts 16->32
are retained unused statistics; saved normalize flag is False. Fresh initial
weights exactly match the retained behavior; frozen control/source/checkpoint
admission passes. Live/disk/warm initial 21-leaf bytes and 20 restored-action
probes pass, max action error 0. Normalizer perturbation leaves logits/value
exactly unchanged on 10 probes. Independent density/KL checks pass.

| Stage | Training KL | Training value loss | Deployed KL | Deployed value loss |
|---|---:|---:|---:|---:|
| fresh | 0.000140119 | 0.051039 | 0.058854 | 0.086861 |
| warm | 0.000140149 | 0.046600 | 0.243470 | 0.084648 |

Deployed replay is against the original retained behavior; warm drift is
cumulative. Single-update-per-batch epoch KL is measured before SGD, so its
near-self value alone does not establish update stability. No useful locomotion
or general stability claim. Adam/RNG/counters restart on warm load. Public
results_identity_preprocessing; raw checkpoints/replays retained ignored.
Next is a separately preregistered multi-batch local check with post-update KL
against each batch's own behavior, before larger neutral learning/command gate.
