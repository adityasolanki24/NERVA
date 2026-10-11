# E1′ training attempt 1: stopped by the replay-integrity check before any update (no training)

Run `e1_prime-20261011-120126` (one L4, 90 min cap authorized; code `6c1df31`; preregistration `0c221cf`).
- The VM started 01:03:32 UTC, the job at uptime 289 s with the 90-minute settings (trainer hard timeout 4,991 s).
- The trainer stopped at iteration 1 with `collection_integrity`. Command coverage was correct, but replaying the
  collected batch through the same policy differed from the behaviour policy by up to 0.0101 in logits (tolerance
  1e-3) and 0.083 in log-probability (tolerance 1e-2).
- **No update was accepted: 0 transitions trained.** The final "checkpoint" is the unchanged widened start.
  The VM exited normally and deleted itself (≈ 10 min lifetime, ≈ US$0.2 including NAT).
- Audit afterwards: no instances, disks, addresses, forwarding rules, routers or snapshots.

**Diagnosis [inference]:** every earlier GPU run replayed with exactly zero error; the CPU smoke replays within
≈ 1e-6 for both the neutral and the styled trainer. A 1e-2 discrepancy that appears only on the GPU and only with the
103-input first layer is consistent with float32 matmuls running at reduced (TF32) precision on different code
paths in collection and replay. **Fix:** the E1′ trainer forces full float32 matmul precision
(`jax_default_matmul_precision = "highest"`); earlier experiments are unchanged. Tolerances, protocol and
criteria are unchanged. Not verifiable locally (no GPU).
