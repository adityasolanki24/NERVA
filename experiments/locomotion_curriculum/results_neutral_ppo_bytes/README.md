# Strict PPO/warm-start verification retry

Original protocol `4243988`; explicit retry preregistration and implementation
snapshot `6f10e30` (`docs/neutral_ppo_byte_retry.md`). All criteria pass in
203.19 s under a 900 s parent cap, two 16-transition stages, CPU/seed 7.

Both live-before-save/restored parameter trees and warm initial parameters agree
in literal bytes, shape, dtype and named structure. Each roundtrip covers 21
leaves. Both networks update; normalization count 16 -> 32. Ten action probes per
stage agree exactly and remain finite/bounded. Adam/RNG/counters restart on warm
load; this is parameter warm start, not full-state continuation.

Loss/KL values repeat the first run (timings differ), including the extreme fresh
metrics. No post-hoc loss threshold, dependency change, reward tuning or expressive
objectives. The first report and checkpoint audit remain preserved separately.
Total optimizer work across the two declared attempts: 64 transitions.

No learned motor-readiness claim or cloud work. Checkpoints/logs are retained in
ignored local storage; public reports contain small measurements and hashes.
Next: separately preregister normalization-stability control before larger neutral
training and its prospective long rest/turn/transition/push gate.
