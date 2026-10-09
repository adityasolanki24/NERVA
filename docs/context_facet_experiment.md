# Same-source context facets: preregistration (2026-10-09)

Status: preregistered at `ca30916` before implementation or evaluation; completed
2026-10-09, all criteria pass. Bv4 + margin + touch adopted. Results and limitations
are recorded in `development_log.md` and `experiments/affect_models/results_facets/`.

Hypothesis [NERVA design]: summing in-view and ongoing-contact context for one source
double-counts one interaction. Combining its facets will reduce valence saturation
without losing two-sided dominance or the existing behaviours.

## Exact rule

Candidate Bv4 inherits Bv3's fast onset/slow return, unchanged W, gains, phasic
channel and action tendencies. Store the latest persistent features u per
(source, hypothesis kind), using tuple keys. For each retained facet f, compute
w_f = 1 while age <= 2.5 s, otherwise exp(-(age - 2.5)/3).
For each source s, its context contribution is

`c_s = max_f(w_f) * sum_f(w_f * u_f) / sum_f(w_f)`.

Sum c_s across sources; absent sources contribute zero. Retain the existing
pruning threshold w > 0.001. Empty subject is the existing undirected source;
an explicit source argument overrides the frame subject. This rule preserves
single-facet decay exactly, makes identical simultaneous facets idempotent,
and gives stale facets decreasing influence without cancelling the source's
overall fade. Distinct sources still add. No feature-wise maxima or weight tuning.

## Fixed protocol and criteria

Use Bv4 + reaction-margin controllability + touch context, profile v2, B2 final
300,482,560-step checkpoint, training/backlash scene, neutral style, utility
selector, vision, paired seeds 0–4. Default / two-person / together durations:
100 / 135 / 125 s. Compare with the preserved Bv3 touch-context reports.

- Fixed trace: bounded; norm(PAD) < 0.05 within 60 s after last event;
  counterfactual event directions 7/7.
- Repeated single context: 2 s refresh for 120 s; per-dimension mean drift
  (110–120 vs 50–60 s) < 0.02, max absolute PAD < 0.9; 1 s vs 2 s
  final-window means differ by < 0.05.
- All existing behavioural criteria pass in all five seeds per scenario; no
  falls (tilt > 45 degrees). Safety remains deterministic and separate.
- Sample-weighted pooled time with |x| > 0.9 <= 5% per dimension;
  pooled std V and A >= 0.05; max |V| and |A| >= 0.2 in every scenario.
- Pooled min D <= -0.10; minimum D in the 3 s after first detected rapid
  approach < 0 in >= 4/5 default seeds (missing detection fails).
- Mean D during A's return (94–106 s) < B's return (120–130 s) in
  >= 4/5 two-person seeds; pooled std D >= 0.05.
- Report the old 46.5–49.5 s lunge-window mean without using it for adoption.

Tests cover single-facet equivalence, same-source duplicate invariance,
different-source addition, mixed-sign averaging, stale-facet fading, tuple-key
identity, refresh replacement, unchanged phasic/tendency signals and contract.
Check legacy and Bv2 seed-0 traces against pre-change snapshots.

Stopping/decision: one fixed candidate, no coefficient or threshold tuning.
Finish all 15 local runs even if an early criterion fails, unless execution is
broken or unsafe; preserve failures. Adopt Bv4 + margin + touch as defaults
only if every criterion passes; otherwise Bv2 stays default. No paid work.
Then return to robust-locomotion-first curriculum design, without rerunning S1–S6.
