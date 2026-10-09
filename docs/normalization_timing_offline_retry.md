# Offline retry of the preserved normalization-timing batch

Preregistered 2026-10-10 at `9a03e63`, before implementation or updated-statistics
evaluation. Original preregistration `0901b11` and invalid attempt `8a2f467`
remain intact: its old-control log probabilities failed integrity. Offline
old-control inspection passed when the compiled function received the batch
explicitly. Retry changes replay input placement only: normalizer, fixed weights,
batch and loss key are explicit dynamic JIT inputs. No change to PPO loss,
networks, epsilon, weights, dataset, permutation, loss key or success criteria.
This avoids relying on the failed graph that captured the batch as constants;
it does not assert a general XLA defect.

Use only retained `normalization-timing/batch.npz`, `initial_params.npz` and
`old/000000000000` checkpoint. Verify file hashes, reconstructed tree hashes,
all checkpoint file hashes and literal-byte checkpoint/initial NPZ equality
against the preserved public reports before replay. Refuse missing or changed
inputs. Check the pinned Python/package/external-source and reference hashes,
one CPU process/device, batch [2,4], eight original transitions, count zero and
initial weight fingerprint. No initializer, environment reset/step, rollout,
optimizer/gradient calculation, policy training or additional transitions.

Reconstruct the same initial-normalizer copy update on unpermuted observations
with the original Welford operator, epsilon 1e-4 and one-device pmap reduction;
count must be eight. Reproduce the same seed-7 minibatch permutation/loss key
and verify the key ledger and minibatch hash. Replay old then updated statistics,
using the exact pinned PPO loss/options and networks. All weights remain
literally fixed. Stop before updated replay if old integrity fails.

Every original integrity and hypothesis criterion in
`normalization_timing_replay.md` remains unchanged: finite inputs/results,
old logits error <=1e-5, old raw-action log-probability error <=1e-4,
conventional old analytic self-KL absolute mean <=1e-6, official self-KL <=0.001,
independent NumPy stabilized KL agreement atol/rtol=1e-5 in both cases,
fixed weight/batch hashes, literal-byte old/updated checkpoint roundtrips,
counts 0->8. Additionally compare both replays' raw-action log probabilities
with independent NumPy tanh-Gaussian probabilities at atol=1e-4, rtol=1e-5.
Report measured errors; no threshold relaxation or cherry-picking retries.

Hypothesis supported only if integrity passes and updated official KL >1.0
while old KL <=0.001. Otherwise preserve a negative or integrity-invalid result.
Report value/loss/distribution/action/log-probability drift and per-slot
normalizer changes descriptively. This diagnostic cannot prove a safe SGD
schedule, explain the whole prior two-update KL or establish motor readiness.

One offline CPU attempt with hard parent watchdog 180 s, zero new physical
transitions and zero optimizer updates. Stop on integrity/API/nonfinite/timeout,
retain all raw/partial outputs. Refuse overwrite. Public small aggregate reports
`results_normalization_timing_offline/`; ignored raw artifacts
`experiments/cloud_runs/normalization-timing-offline/`. No dependencies/upstream
edits, paid cloud, default promotion, expressive training or artifact deletion.
Complete full tests including slow, Ruff, docs/log, public-content checks,
commit/push and final cloud resource audit.

## Admission-only abort and schema correction

The registered offline attempt (`37d5312`) stopped before any loss replay in
4.62 s: the parser rejected existing episode metric keys such as
`cost/action_rate`. No new transitions or optimizer updates; all original
artifacts retained. Public abort reports remain in
`results_normalization_timing_offline/`. This is an implementation defect,
not a negative timing-hypothesis result.

Separately preregistered 2026-10-10: one new offline attempt with identical input
files, all hashes, settings, criteria and 180 s limit. Change only the archive
key parser to accept slash-separated generated metric names; retain no-pickle,
exact reconstructed-name/schema and literal byte/hash checks. Test the actual
retained archive admission and slash-key regression before running. Dynamic
replay implementation remains unchanged. Use fresh output
`results_normalization_timing_offline_schema/` and ignored raw directory
`normalization-timing-offline-schema/`; no overwrites or unregistered retry.
No physical transitions, gradients, optimizer updates, cloud work or promotions.

## Completed schema-corrected replay

Preregistration `75cf8ed`, implementation `63937e7`; completes in 6.41 s under
180 s. All input admission and original integrity criteria pass, including
independent log probabilities and KL, literal 21-leaf old/updated checkpoint
roundtrips and unchanged batch/weight hashes. No new physical transitions or
optimizer updates. Statistics count 0->8 on the eight retained transitions.

**Timing hypothesis supported on this batch:** official KL 0.000140131 with
old statistics versus 10.296526 with updated statistics (>1). Conventional
analytic old self-KL 9.53e-12; updated conventional KL 10.296356, max 37.773238.
Manual stabilized updated KL 10.296525 agrees with the official result. Old
behavior logits max error 9.54e-7, log probabilities max error 9.06e-6.
Independent log probabilities agree in both cases, max error <=9.54e-7.

Descriptive drift: deterministic tanh action maximum change 0.957049, raw-action
log-probability max change 31.303122, value max change 0.980274. Value loss
0.089507->20.853298; observation normalized maxima 21.367878->2.644543 in both
policy/privileged streams. These are not additional acceptance criteria.
Public reports: `results_normalization_timing_offline_schema/`. Raw checkpoints
and replays retained ignored; neither prior abort was overwritten.

No validated schedule fix, general learning stability, motor readiness or
expressive-training result. Next compare normalization treatment under a new
preregistration, requiring both optimization and deployed-policy drift checks;
moving a statistics update after SGD alone can still change deployed behavior.

Completed follow-ups (2026-10-10): same-batch timing replay isolates statistics
replacement drift (`normalization_timing_offline_retry.md`); offline schedule
comparison supports fixed preprocessing and demonstrates deferred deployment
drift (`normalization_schedule_comparison.md`). Next is the bounded on-policy
identity screen (`identity_preprocessing_smoke.md`) before larger neutral
learning. Earlier outcomes/criteria and every abort remain preserved.
