# Bounded identity-preprocessing multi-batch update stability

Preregistered 2026-10-10 at clean baseline `9514ad8`, before implementation or
collection. Previous32-transition identity screen passes, but pre-SGD self-KL
and cumulative drift against the original policy cannot establish per-update
stability. Hypothesis: identity preprocessing permits finite one-step updates
with post-SGD KL<=1 against each newly collected batch's own behavior.

Admit the exact final warm checkpoint from results_identity_preprocessing by
all file hashes, contract/network/core/package/reference hashes, literal tree
roundtrip, count32 and normalize_observations=False. Start from those parameters,
with fresh Adam, RNG and environment/counters: parameter warm start, not full
resume. Raw identity policy/value preprocessing, (32,32), state101/privileged212,
actions14, Welford epsilon1e-4 retained unused. No existing default promotion.

One CPU process/device, seed19; two environments, action repeat1, unroll4,
one minibatch, one gradient update per new eight-transition batch. Up to16
batches: max128 new transitions and16 Adam updates, LR1e-4, no clipping/schedule;
all PPO loss options/reference/rewards/physics/noise unchanged. Episode length
32 (0.64 s) permits a longer state trajectory than the earlier8-step smoke;
this is a declared horizon change, not a motor-readiness trial. Use candidate
wrapper/reset and pinned first-batch reset/epoch key schedule for seed19.
Carry physical environment state and PRNG between batches. Extend the existing
collector to return next state/key only when explicitly requested; its existing
default output and capture semantics remain unchanged. Record command extras.
Keep unused normalizer updated/count32+8 per batch, but never apply it.

Within each batch use inspected key_sgd permutation/loss keys, explicit dynamic
JIT inputs, one fresh on-policy dataset. Maintain Adam state across batches.
Check pre-SGD replay against saved behavior: logits max error<=1e-5, raw-action
log probabilities<=1e-4, official self-KL<=0.001. Stop before SGD if control fails.
Then take one exact pinned PPO/Adam step; replay final deployed weights on that
same batch with the same loss key. Required per-update screens: official
post-SGD KL<=1 and value loss<=100; finite data/parameters/optimizer/gradients/
outputs; independent NumPy stabilized KL agreement atol/rtol1e-5 and raw-action
log probabilities atol1e-4/rtol1e-5. Both networks update>1e-12; no criteria tuned.
Statistics count exactly32+8*completed_batches. Report pre/post metrics, gradient
norms, raw/identity observation maxima, command counts, terminations/truncations,
and reward descriptively. No fall, tracking or command-coverage success claim.

Archive every batch and post-update parameter/Adam-state snapshot in ignored
storage with hashes; public progress/aggregates only. Save a Brax parameter
checkpoint at completion or finite-screen stop; exact21-leaf literal roundtrip,
matching identity/candidate contract and10 deterministic/same-key restored-action
probes error<=1e-6. Confirm changing saved mean/std has no inference effect
on the five existing probes (<=1e-6). Do not call this a full-state resume API.

All-required study success: all16 batches pass the unchanged integrity and
post-SGD screens and final save/invariance checks. Stop immediately on a failed
per-batch criterion, nonfinite/API/device mismatch or hard900 s parent timeout;
retain partial data/checkpoints/report and preserve the negative result. No
retries or LR/budget/threshold changes. One attempt, no evaluation rollouts,
reference generation, expressive objectives, dependencies/upstream edits,
paid/cloud work, deployment/default promotion or artifact deletion.
Public results_identity_multibatch; ignored raw
experiments/cloud_runs/identity-multibatch. Refuse overwrite. Full tests including
slow, Ruff, docs/log/public audit, commit/push and final cloud audit. Passing only
permits balanced-command local pilot design, then prospective long motor gate;
no learned controller/readiness or paid training authorization.
