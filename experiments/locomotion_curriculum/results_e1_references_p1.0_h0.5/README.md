# E1 Phase A, halved height range: fails; E1 stops before training (preregistered rule)

Protocol: `docs/expressive_posture_experiment.md` §3 (`630610f`), range rule applied once after
`../results_e1_references/`: height scale 0.5 (COM 0.209–0.221 m), pitch unchanged (−10°…+2°). Runner
`e1_references.py --height-scale 0.5`; 91 recordings in 640 s, local WSL.

| criterion | result |
|---|---|
| 1. grid recordings pass all eight admission criteria | **fail**: all three e_height = +1 styles (COM 0.221 m) fail the joint-velocity fit for the walking commands (17 of 18; (+1, +1) right passes) |
| 2. regenerated (0, 0) matches the admitted neutral set | pass (0.000 rad) |
| 3. interpolation validity at (±0.5, ±0.5) | pass (max leg-joint RMSE 0.0074 rad, contacts 100%) |

Every style with e_height ≤ 0 passes all criteria, at every pitch: (±1, −1), (0, −1), (±1, 0) and the
neutral set. The failing check is the reference's joint-velocity fit (limit 0.5 rad/s), which degrades
steadily as the COM is raised and the knees straighten (walking commands, worst component):

| COM height | joint-velocity fit RMSE | knee minimum |
|---|---|---|
| 0.209 m | 0.18–0.19 rad/s | ≈ 1.0 rad |
| 0.215 m (neutral) | 0.22–0.24 | ≈ 0.8 |
| 0.218 m (checkpoint) | 0.31–0.37 | ≈ 0.7 |
| 0.221 m | 0.53–0.61 (fails) | ≈ 0.5 |
| 0.227 m (full range) | 0.46–0.77, knees reach ≈ 0 | ≈ 0 |

**Decision (preregistered):** the range may be halved only once; it still fails, so **E1 stops and is
reported. No training, no paid compute.** Torso pitch over −10°…+2° and body height *below* neutral
(down to 0.209 m) are feasible references; raising the body above ≈ 0.218 m is not, with this generator and
these criteria. Any experiment using that knowledge needs its own preregistration.
