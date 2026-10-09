# Initial PPO/warm-start smoke

Protocol `docs/neutral_ppo_restore_smoke.md`, preregistration `4243988`, code
`9bffa12`. Two 16-transition stages complete in 193.47 s (900 s cap).
Finite network/statistics updates, numeric roundtrip, identical actions and
warm initialization pass; normalization counts 16 -> 32. Adam/RNG/counters restart.

Review: original comparison used numeric equality rather than literal bytes,
so the live bitwise criterion is not fully evidenced by this run. Its machine
report is preserved unchanged. Subsequent saved-tree byte audit cannot observe
discarded live arrays. A separate strict retry is preregistered in
`docs/neutral_ppo_byte_retry.md`.

Fresh value loss 884221440 and KL 155226423296 are finite but extreme; warm
value loss 0.061235 and KL 0.121367 are not evidence of learned motor readiness.
No thresholds changed. Normalization diagnosis is in the separate audit report.
Checkpoints/logs remain ignored; public results contain only small measurements
and hashes. No cloud work or expressive training.
