# Preserved offline normalization-timing result

Protocol: `docs/normalization_timing_offline_retry.md`, preregistration `75cf8ed`,
implementation `63937e7`. Completed in 6.41 s under 180 s. Both prior aborts
remain preserved. Exactly eight retained transitions, zero new transitions,
gradients or optimizer updates. One seed/first batch; no readiness claim.

| Fixed weights / same batch | Old statistics | Updated statistics |
|---|---:|---:|
| Normalizer count | 0 | 8 |
| Official KL against stored behavior | 0.000140131 | 10.296526 |
| Conventional analytic KL mean | 9.53e-12 | 10.296356 |
| Value loss | 0.089507 | 20.853298 |
| Largest normalized observation magnitude | 21.367878 | 2.644543 |

All integrity criteria pass. Timing hypothesis supported by updated KL >1,
old KL <=0.001, unchanged weight/batch hashes, literal checkpoint roundtrips,
finite results and independent KL/log probabilities. Deterministic action max
shift 0.957049. This demonstrates preprocessing drift on this frozen batch,
not a validated learning schedule or explanation of all prior training KL.

`protocol.json`: admission/source/version/report fingerprints.
`cases.json`: both replays, independent comparisons and checkpoint fingerprints.
`summary.json`: preregistered decision and descriptive drift.
`normalizer_shift.json`: per-slot mean/std changes.
`parent_completion.json`: successful bounded worker completion.
Raw replays/checkpoints stay in ignored storage. No cloud work or promotion.
