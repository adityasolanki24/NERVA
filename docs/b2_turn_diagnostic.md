# B2 turn-centre diagnostic

Completed: aggregate **inconclusive**. Fixed criteria unchanged. See
[results](../experiments/locomotion_curriculum/results_turn/README.md).

Preregistered before computing geometry metrics. This is a descriptive follow-up
to the failed B2 robustness gate, not a replacement gate or a policy change.

## Hypotheses and fixed analysis

H1: the base traces a bounded orbit about an offset turn centre. H2: a constant
world-frame translation improves prediction. Neither model covers changing
centres, heading-locked drift, or all possible locomotion errors.

Use exactly the ten saved primary steady-turn traces, seeds 0–4, both ±0.60 rad/s.
Fit on 5 ≤ t < 12.5 s; evaluate once on 12.5 ≤ t ≤ 20 s. Use raw horizontal
base positions and measured quaternion yaw. Fit ordinary least squares:

`p(t) = c + R(yaw(t)) r` (orbit), and
`p(t) = c + R(yaw(t)) r + v (t - 5)` (orbit plus constant world drift).

Here c, r and v each have two components. Report training design condition
numbers, body-frame offset r, drift v, and held-out Euclidean position RMSE.
Reject deficient rank or condition number >10,000 as inconclusive. No smoothing,
outlier removal, parameter tuning, reruns or new physics trials in this phase.

Per trace, support bounded orbit if orbit held-out RMSE ≤0.01 m and drift-model
speed ≤0.01 m/s. Support sustained world drift if speed ≥0.02 m/s and the drift
model reduces held-out RMSE by ≥50%. Otherwise classify inconclusive. Support
an aggregate hypothesis only if ≥4/5 traces support it in **each** direction.
These thresholds are prospective diagnostic choices, not revised gate criteria.

Audit the shipped neutral reference grid used by B2: exact nearest-neighbour
keys for `(0,0,±0.60)`, phase-average linear/angular reference velocities over
50 equally spaced phases in [0,1), and lateral-sign agreement with each saved
trace's existing 5–20 s heading-frame velocity. This is association only;
reference lookup is a training target, not a runtime input to the fixed ONNX.
Do not infer causation or modify the upstream reference.

## Provenance, stopping and safety

Record SHA256 of each input and reference, upstream Git revision, analysis
implementation revision and this preregistration revision. Stop on missing,
nonfinite or malformed input; never overwrite results. Commit small JSON
reports and interpretation, retain raw traces in ignored storage. No training,
cloud calls, spending, changes to deterministic safety or affect. Keep public
reports free of personal paths and private identifiers. Run meaningful synthetic
geometry tests, the complete suite including slow tests, and Ruff.

The original pure-turn base-velocity requirement remains failed regardless of
classification. Next steps must separately address that requirement or
preregister a physically justified future gate; this diagnosis cannot adopt B2
as a robust expressive-training baseline.
