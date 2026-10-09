# Reference derivative time-alignment diagnostic: preregistration

The seven-reference subset failed; keep its criteria/results unchanged. This
follow-up uses exactly those saved recordings and five-harmonic fits, no new
generation, tuning, increased harmonics, repairs, policy changes or training.

Question: a backward difference represents an interval-average derivative,
whereas the original fit test evaluated an instantaneous derivative at the
interval endpoint. Can consistent timing account for the six moving failures?

For all seven existing conditions, retain held-out endpoints 4 ≤ t[k] <6.
Use the original fit coefficients unchanged and actual timestamps. Compare
recorded joint backward differences against three fixed predictions:
1. instantaneous derivative at t[k] (reproduce original metric);
2. instantaneous derivative at `(t[k]+t[k-1])/2`;
3. exact fitted interval average `(q_fit(t[k])-q_fit(t[k-1]))/(t[k]-t[k-1])`.

Report per-joint RMSE and maximum component RMSE for all three; ratios relative
to original endpoint metric. Input raw/fit hashes must match the recorded
subset report before any calculation. Known sinusoid: prove interval-average
prediction matches differences to 1e-8 and midpoint error is smaller than
endpoint error, for actual dt 0.01/0.02/0.04 and period 0.54. Static pose yields
exactly zero (tolerance 1e-8). No threshold selection using these results.

Per moving condition, timing-accounted support requires BOTH midpoint and
interval-average RMSE ≤0.5 rad/s AND ≤50% of original endpoint RMSE. Aggregate
support requires all six moving conditions. Otherwise classify timing alone
insufficient. The original failed experiment stays failed either way; this
diagnostic cannot repair contact labels or invalid knees, establish dynamic
feasibility, or authorise candidate learning.

Stop on hash mismatch, malformed/nonfinite input or missing samples. Cap the
local subprocess at 120 s wall time; preserve partial/failure reports. Commit
protocol then code before evaluation. Tests, full suite, Ruff, docs/log,
public checks, read-only cloud audit and final commit/push. Raw artifacts remain
ignored; only small reports public. No cloud run, spending or deletion.
