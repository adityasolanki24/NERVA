# Offline normalization schedule and representation comparison

Preregistered 2026-10-10 at clean baseline `61a0a0b`, before implementation or
SGD evaluation. The preceding frozen-weight replay showed KL 10.296526 solely
from replacing statistics on eight observations. Hypotheses: fixed preprocessing
avoids this contribution during and after one optimizer step; deferred updates
alone can hide deployment drift; an affine first-layer rebase preserves the
trained function under an unbounded affine normalizer change.

Admit the exact retained eight-transition batch/initial checkpoint and initial
NPZ using all existing literal-byte/file/tree/source/version/reference/key-ledger
checks in `normalization_timing_offline.py`. One CPU device/process. Same seed 7,
32/32 networks, 101/212 observations, 14 actions, Welford epsilon 1e-4, permutation,
loss key, rewards/physics/references and pinned PPO options. No new transitions,
resets or initializer. Record Optax version/source hashes as well as prior ones.
One fresh Adam state per independent optimization, learning rate 1e-4, default
Adam coefficients, no gradient clipping/schedule. Exactly two independent
optimizer updates on the same batch, one old-normalizer and one new-normalizer
loss gradient. Never differentiate through the statistics update or rebase.
Use explicit compiled parameters, normalizer, batch and key inputs.

| Arm | Optimization statistics | Deployment statistics / weights |
|---|---|---|
| native | updated, count 8 | updated / native one-step weights |
| fixed | initial, count 0 | initial / old-normalizer one-step weights |
| deferred | initial, count 0 | updated / identical fixed-arm weights |
| rebased | initial, count 0 | updated / fixed-arm first layers rebased |

Deferred and rebased reuse the exact fixed-arm optimizer result; no additional
SGD. Report both pre-update loss metrics and post-update/deployment metrics.
KL is always against the retained behavior distribution; also report pairwise
fixed-vs-deployment drift. Keep native as a comparator even if its KL fails.
Frozen behavior/control replay must match the prior successful old/updated
replay (metrics atol=1e-4, rtol=1e-5) before optimization. Stop on integrity
failure; finite threshold failures are preserved outcomes, not tuned retries.

For each policy/value MLP first Dense layer (no input clipping/layer norm),
with input normalizer mean m/std s and new m'/s', rebase using original W,b:
W' = diag(s'/s) W; b' = b + ((m'-m)/s) W. This yields the same preactivation
for raw observations in real arithmetic. Require finite positive stds, exact
first-layer schema/shape, same dtypes; all other leaves literally unchanged.
Check an independent NumPy float64 formula as well as compiled network output.
This inference reparameterization does not transform or resume Adam moments;
no claim of optimization equivalence after rebase or production readiness.

All integrity criteria: unchanged initial weights and batch hashes; finite
inputs, gradients, optimizer states, loss/replay outputs; both independent
updates change policy and value by >1e-12; statistics counts 0/8; independent
NumPy stabilized KL versus official loss atol/rtol=1e-5 and tanh-Gaussian log
probabilities atol=1e-4, rtol=1e-5. Literal-byte roundtrip of all four deployment
checkpoints, matching contracts, and before/after deterministic and same-key
sampled actions (five existing synthetic probes, key 19) error <=1e-6.

Fixed-arm screen: post-update/deployment official KL <=1 and value loss <=100;
no separate preprocessing deployment change (literal same parameters/statistics).
Deferred hypothesis support: its fixed-arm optimization passes the KL screen
but deployed official KL >1. Native has no required passing threshold.
Rebase support: fixed-vs-rebased output logits/value max error <=1e-4, tanh
and same-key sampled action error <=1e-5, pairwise conventional analytic KL
absolute mean <=1e-6 on captured current/next observations plus the five fixed
synthetic probes; independent first-layer formula agreement atol=1e-5, rtol=1e-5.
Rebased deployment also must pass KL <=1/value loss <=100 on the captured batch.
All criteria must pass for the corresponding claim. No overall success is
inferred from only one passing arm; report each separately.

One attempt, hard 180 s parent limit, zero new transitions, two optimizer updates
maximum. Refuse overwrite; stop on API/integrity/nonfinite/timeout, preserve
partial results. Public `results_normalization_schedule/`; ignored raw
`experiments/cloud_runs/normalization-schedule/`. No dependency/upstream edits,
paid/cloud work, deployments, default promotions or artifact deletion.
Complete full tests including slow, Ruff, authoritative docs/log, public audit,
commit/push and final resource audit. Passing fixed preprocessing permits a
new separately preregistered bounded local on-policy identity-preprocessing
smoke using the trainer's normalize_observations=False; it does not authorize
large training, replace B2, or begin expressive objectives. The pinned Welford
operator ignores until_count, so that option is not a freeze implementation.

## Completed outcome

Preregistration `e241208`, implementation `d42b541`: finishes in 11.09 s under
180 s, exactly two independent offline Adam updates, zero new transitions.
All admission/control/gradient/checkpoint/density integrity requirements pass.
Forty deterministic/same-key restored-action probes have zero error.

| Deployment arm | Official KL | Value loss |
|---|---:|---:|
| native | 10.292988 | 21.143976 |
| fixed | 0.023703 | 0.088133 |
| deferred | 10.288106 | 20.937843 |
| rebased | 0.023703 | 0.088133 |

Fixed preprocessing passes its screen. Deferred deployment drift hypothesis is
supported: the same one-step weights pass while optimizing with old statistics
but fail KL <=1 after statistics replacement. Rebase inference hypothesis also
passes: captured current/next observations plus five synthetic probes have max
logits/value error 1.91e-6/3.87e-7, deterministic/sampled action error
7.45e-7/5.37e-7, pairwise analytic KL mean <=2.64e-11. Non-first-layer leaves
remain literal-identical; independent float64 formula and dtypes/shapes pass.
No Adam moment rebase or post-rebase training equivalence is claimed.

Public reports `results_normalization_schedule/`; checkpoints/gradients/replays
retained ignored. No default promotion, expressive objectives or paid work.
Next is a separately preregistered identity-preprocessing on-policy smoke,
using normalize_observations=False with unchanged PPO/reference/physics settings.
This uses the simple fixed-preprocessing treatment instead of claiming the
experimental rebase is ready for continued optimization.
