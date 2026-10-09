# Strict live-byte PPO smoke retry

Preregistered 2026-10-09 before retry evaluation. Original smoke protocol
`neutral_ppo_restore_smoke.md` (`4243988`) and implementation `9bffa12` completed
32 transitions with numeric parameter equality and exact restored actions.
Review found that `np.array_equal` did not establish the literal-byte criterion
for signed zero. Saved-tree reserialization subsequently passes literal-byte
checks (`neutral_normalization_audit.md`), but cannot recover the discarded
original live arrays. Preserve the first report unchanged as incomplete evidence
for that one requirement.

One explicitly justified retry closes this implementation/verification gap.
All original hypotheses, network/reward/reference settings, seeds, metrics and
criteria remain unchanged. Use the committed strict comparator (`2218811`),
covered by signed-zero comparison and actual Brax roundtrip tests. Compare live
before-save arrays to restored arrays, including literal bytes, named structure,
shape and dtype. The warm initial comparison uses the same strict comparator.

Exact original budget: two stages of 16 optimizer transitions (32 total), CPU,
seed 7, two environments, episode 8, batch 2, unroll 4, one minibatch/update,
networks (32,32), Welford normalization with epsilon 0. No evaluator rollouts.
One worker, 900 s parent watchdog, no paid work. No dependency/weight changes,
normalization tuning or new loss/KL thresholds. Finite but large metrics remain
recorded; a plumbing pass still does not show learning stability or motor quality.

New outputs: `results_neutral_ppo_bytes/`; ignored logs/checkpoints in
`experiments/cloud_runs/neutral-ppo-byte-smoke/`. Never overwrite original
artifacts. Stop on any original failure condition; preserve partial outputs.
No further retry is preregistered here. Normalization-control research remains
a separate prospective phase before larger neutral training.

Outcome: implementation/protocol snapshot `6f10e30`; all criteria pass in
203.19 s, 32 transitions. Both 21-leaf live/restored trees and warm initialization
agree byte-for-byte; twenty deterministic/same-key action probes agree exactly.
Both networks update, normalizer count 16->32, finite metrics match the original
run except timing/throughput. No learned readiness or loss-stability claim.
