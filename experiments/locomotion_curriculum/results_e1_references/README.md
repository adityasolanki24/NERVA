# E1 Phase A, full range: fails; all failures involve |e_height| = 1 (range rule applies)

Protocol: `docs/expressive_posture_experiment.md` §3 (`630610f`); runner `e1_references.py` (`77b1223`).
91 recordings in 695 s (local WSL, Placo 0.6.3). Attempt 1 was an infrastructure abort (see
`../results_e1_references_abort_pathconv/`).

| criterion | result |
|---|---|
| 1. grid recordings pass all eight admission criteria | **fail** |
| 2. regenerated (0, 0) matches the admitted neutral set (≤ 0.05 rad) | pass (0.000 rad for all seven: deterministic) |
| 3. interpolation validity at (±0.5, ±0.5) | **fail**: (+0.5, +0.5) turn right 0.051 rad; (+0.5, −0.5) stand not evaluable (its (+1, −1) stand recording crashed) |

Per grid style (walking commands; stand passes everywhere except one recorder crash):
- pitch-only styles (±1, 0): **all seven commands pass**;
- (0, −1), (−1, −1), (+1, −1) (COM 0.203 m): joint limits fail for every walking command; (+1, −1) stand: recorder crash;
- (0, +1), (−1, +1), (+1, +1) (COM 0.227 m): positive knees, joint-velocity fit and joint limits fail.
- Checkpoints at e_height = +0.5 (COM 0.221 m) themselves fail the joint-velocity fit; at e_height = −0.5 they pass
  and interpolate within 0.012 rad.

Reference feature changes between e = −1 and +1: pitch 0.209 rad (12.0°) for every command; base height
0.028–0.029 m.

**Range rule (preregistered):** every failure involves a point with |e_height| = 1 (or a checkpoint whose
interpolation needs one), none involves the pitch axis alone, so the height range is halved once (COM
0.209–0.221 m) and Phase A is rerun (`results_e1_references_p1.0_h0.5/`). If that fails, E1 stops before training.
