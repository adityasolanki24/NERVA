# B2 neutral motor gate — negative result (2026-10-09)

Preregistered at `14a139c`; implementation `e8e3f34`.
**B2 does not pass the all-required motor-readiness gate.** Steady in-place
turns exceed the fixed 0.03 m/s horizontal translation RMS limit in 10/10
trials. Every other criterion passes. No threshold was changed after evaluation.

| steady command | achieved commanded-axis mean | full tracking pass | horizontal RMS in pure turns |
|---|---:|---:|---:|
| forward +0.15 m/s | +0.1049 m/s | 5/5 | — |
| backward -0.15 m/s | -0.1126 m/s | 5/5 | — |
| left +0.10 m/s | +0.0613 m/s | 5/5 | — |
| right -0.10 m/s | -0.0598 m/s | 5/5 | — |
| turn left +0.60 rad/s | +0.5617 rad/s | **0/5** | **0.0330–0.0379 m/s** |
| turn right -0.60 rad/s | -0.5864 rad/s | **0/5** | **0.0416–0.0439 m/s** |

The yaw direction and yaw tracking thresholds themselves pass. The failing
quantity is uncommanded translation while the requested vx and vy are zero.
These body/heading-frame measurements do not establish persistent global drift,
a reference defect, or its cause; turning about an offset centre is a candidate
explanation for a separate diagnostic. The short transition turn phases pass
their preregistered axis-tracking checks; their criteria did not include this
steady-turn cross-motion limit, so they do not cancel the steady failure.

- All 115 primary trials complete and remain upright; maximum tilt 15.82 degrees.
- All five start/stop/reversal trials pass; no standing solution detected.
- All 40 velocity-kick trials recover within 5 s (5/5 per command/direction);
  longest recovery 1.78 s, including the required full 1 s confirmation window.
- All 20 standing-head and 20 walking-head trials pass. Minimum paired
  forward-speed retention is 86.31% (limit 75%). Axis head limits were tested
  separately while walking; combined offsets were not validated.
- Shadow safety: zero interventions and zero stop time; no conditional
  safety-on diagnostic replays were needed. No affect or safety code was changed.

`trials.json` preserves each primary trial and its metrics; `summary.json`
contains the unchanged criteria; `condition_summary.json` aggregates recorded
metrics only; `protocol.json` records checkpoint SHA256 and code provenance.
Raw traces are local in the git-ignored cloud_runs directory. No paid work.

Reproduce from a clean checkout in the pinned Open Duck inference environment
using a NEW report directory (the runner refuses to overwrite prior results):

```sh
python experiments/locomotion_curriculum/gate.py --policy "$B2_POLICY" --out experiments/locomotion_curriculum/results_b2_repeat --raw-dir experiments/cloud_runs/b2-gate-repeat
```

Set B2_POLICY locally to the final B2 checkpoint; never commit its absolute path.
Protocol and thresholds: `docs/b2_robustness_gate.md`.

**Decision:** retain B2's established reactive role, but do not call it ready
for the proposed expressive curriculum. Next, preregister a local pure-turn
diagnostic separating base motion around a turning centre, sustained global
translation, and reference/command lookup. The known reference grid has no
exact zero lateral velocity; that is a possible confound, not a demonstrated
cause. Preserve this gate result before proposing any motor-only training fix.
