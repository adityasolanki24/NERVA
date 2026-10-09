# Neutral normalization variance-floor control

Preregistered 2026-10-09 from clean baseline `22ebf72`, before implementation
or intervention evaluation. The previous normalization audit suggests severe
amplification outside the tiny observed sample; it does not establish loss
causation. Original epsilon-zero metrics, criteria and artifacts stay intact.

Hypothesis: changing only Welford variance epsilon from 0 to 1e-4 reduces
normalization amplification and the extreme fresh-stage PPO value loss/KL, while
retaining finite updates and exact checkpoint/warm-start behavior. The epsilon
is added to variance, giving a standard-deviation floor of 0.01 after updates
and maximum normalization gain about 100. Initial running-statistics std is
one for both settings in the inspected pinned Brax implementation.

Use the existing strict run `results_neutral_ppo_bytes/` as the frozen matched
control, not a fresh repeat. Require its successful two stages, epsilon zero,
same PPO/network settings, package versions, core module source hashes and
reference hashes. Verify every preserved checkpoint hash before running.
Record public control-report hashes. Source changes are restricted to runner
configuration/metadata, nonintrusive reporting and control evaluation; physics,
reward, reference, wrapper, optimizer and checkpoint implementations stay fixed.
This one-seed historical comparison is limited evidence, not general learning
stability or a motor-quality trial.

One intervention attempt: native CPU/flat terrain, seed 7, two stages (fresh and
parameter warm start), exactly 16 transitions per stage/32 total. Reuse all
`neutral_ppo_restore_smoke.md` settings: 2 environments, episode 8, batch 2,
unroll 4, one minibatch and one update, networks (32,32), unchanged rewards,
learning rate 0.0001 and all other PPO options. No domain randomization or
evaluator rollouts. Only `normalize_observations_std_eps=0.0001` changes.
Adam/RNG/counters restart in warm stage, as before. No old checkpoint transplant.

All plumbing requirements remain: finite initial/final parameters and metrics,
callbacks [0,16] each stage, both networks change >1e-12, normalization count
16 then 32, live/disk and warm-initial literal-byte equality including named
structure/shapes/dtypes, ten deterministic/same-key restored action probes per
stage with error <=1e-6 and finite/bounded actions. Checkpoint contract must also
name normalization mode and variance epsilon; confirm epsilon survives restore
and future statistics updates in a regression test.

At each final checkpoint reuse the five fixed synthetic probes and key 19.
Report std min/median/max and inverse-std maximum by observation key, max absolute
normalized probe values, and agreement with NumPy (x-mean)/std at atol/rtol 1e-6.
Mean probe must normalize to zero. No new rollout or RNG advancement for probes.
Probe states need not be physically achievable; saturation is descriptive.

Separate decisions, all required for overall success:
- Integrity: control admission and every plumbing requirement pass; diagnostics
  finite and manual/Brax normalization agrees.
- Amplification: each final std >=0.01*(1-1e-6), inverse std <=100.0001; max
  normalized magnitude across five probes/both keys <=100 in both stages,
  at least 100-fold lower than the corresponding frozen control peak.
- Optimizer screen: value loss <=100 and KL <=1.0 in each stage; fresh value
  loss and KL each <=1% of the frozen fresh control. These prospective screening
  limits reject runaway scales for this tiny smoke; they are not motor-readiness
  thresholds or proof of converged learning.

Report every criterion separately and retain negative outcomes. A finite fresh
screen failure does not abort warm collection: complete the declared pair so
both stages can be assessed. Stop immediately on integrity/nonfinite/API/CPU
mismatch or timeout; no retry or alternate epsilon is authorized by this protocol.

One worker, hard 900 s parent watchdog including imports/compilation, both stages,
checkpointing and control comparison; no paid/cloud work, dependency install,
deployment or expressive objective. New public output `results_normalization_control/`;
raw logs/checkpoints under ignored `experiments/cloud_runs/normalization-control/`.
Refuse overwrite, preserve partial outputs and raw artifacts. No default policy
or historical normalization setting changes. Passing only permits the next
prospective longer stability/coverage test before useful neutral learning.

Run the complete suite including slow checks and Ruff, update authoritative
docs/log, public-content checks, commit/push, and final cloud resource audit.

Outcome: preregistration `1ee56c5`, implementation `d947120`, **overall fail**,
32 transitions complete in 225.75 s under the 900 s cap. Frozen control admission,
all plumbing and amplification criteria pass in both stages. Minimum std is
0.01 and maximum inverse std 100; probe peaks fall from 449070 to 81.202/81.903.
Fresh value loss falls from 884221440 to 10.440188 and KL from 155226423296 to
1489.401733, but fresh KL fails the fixed <=1 screen. Warm value loss 0.058027
and KL 0.132448 pass. No criterion or epsilon changes, retries or default promotion.
`comparison.json` is the study decision; `summary.json` covers plumbing only.

This supports amplification reduction for this single seed/tiny sample, not
general learning stability or learned locomotion. Source inspection identifies
old-stat collection followed by statistics update before SGD in the constant-LR
path. Preregister a fixed-weight, same-batch old/new-normalizer replay diagnostic
to isolate that timing effect before changing schedules or launching larger work.

Completed follow-ups (2026-10-10): same-batch timing replay isolates statistics
replacement drift (`normalization_timing_offline_retry.md`); offline schedule
comparison supports fixed preprocessing and demonstrates deferred deployment
drift (`normalization_schedule_comparison.md`). The identity on-policy screen
(`identity_preprocessing_smoke.md`) and carried-batch update screen
(`identity_multibatch_stability.md`) subsequently pass. Next balanced seven-command
local coverage/curriculum and paired controls before the long motor gate. Earlier outcomes/criteria and every abort remain preserved.
