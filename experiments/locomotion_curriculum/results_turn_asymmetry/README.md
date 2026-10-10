# Left-turn miss: a MJX → native gap, about half of it from training-only delays (2026-10-10)

Runner: `turn_asymmetry_diagnostic.py`. Parts 1–2 had their reading committed before running (`83905f0`);
part 3 (delays) was added afterwards, post-hoc (`5ccb195`). Diagnostic only: no criteria, no training,
CPU. Files: `summary.json`, `rollouts.json`. The upstream checkout stayed clean.

Correction: the first run of part 2 used the wrong joint mirror convention (hip pitch not negated) and
reported a spurious 0.56 rad mirror error. It was fixed before this write-up (`e04b728`); the numbers below are from the corrected runner.

## 1. The base-origin candidate in MJX vs native (deterministic, pushes off, seeds 0/1/2, 5–20 s)

| | turn left: translation (m/s), pivot forward/left (m) | turn right |
|---|---|---|
| MJX, training noise and delays | **0.022–0.026**, pivot +0.006, −0.022 (2.2 cm to the right) | 0.025–0.030, pivot +0.023, −0.019 |
| native evaluation | **0.033–0.035**, pivot −0.015, **−0.046** | 0.019–0.020, pivot +0.002, −0.014 |

By the reading fixed in advance: the left turn is within the 0.03 limit in MJX but not natively, so the
miss is a **MJX → native gap**, not learned behaviour. Both MJX turns pivot ≈ 2 cm to the robot's right
(not mirror-symmetric); natively the left turn's offset grows to 4.6 cm and the right turn's shrinks.

## 2. The reference turns are symmetric

With the correct mirror (legs swapped; hip yaw, roll and pitch negated), turn left matches turn right at
a half-period shift with joint RMS 0.041 rad, the same as the forward walk's own left/right mirror error
(0.040 rad). Contacts disagree on 1 of 27 frames; mean body velocities are ≤ 0.004 m/s. The reference is
not the cause.

## 3. Delays explain about half of the gap (post-hoc)

Training delays actions by a random 0–2 control steps; native evaluation does not. Training also
"delays" the IMU, but only a gravity vector that is not part of the observation, so that delay is inert
(corrected after first writing; disabling it in the runner changes nothing the policy sees). Native evaluation
adds the training observation noise. MJX rollouts with the delays disabled:

| | MJX, delays (as trained) | MJX, no delays | native (no delays) |
|---|---|---|---|
| base-origin, turn left: translation / lateral pivot | 0.022–0.026 / −0.022 m | 0.028 / −0.036 m | 0.033–0.035 / −0.046 m |
| base-origin, turn right: translation / lateral pivot | 0.025–0.030 / −0.019 m | 0.022–0.025 / −0.018 m | 0.019–0.020 / −0.014 m |
| GPU pilot candidate, yaw rate left / right (rad/s) | 0.572 / −0.610 | 0.616 / −0.654 | 0.711 / −0.669 |
| GPU pilot candidate, lateral pivot left / right | −0.001 / +0.012 m | −0.013 / +0.034 m | −0.007 / +0.043 m |

Removing the delays moves every MJX quantity toward native, by roughly half the distance for the left
turn and the yaw rates. The remainder is unexplained.

## Other checks (scratch, not committed)

- Foot contacts: on identical native states, MJX and native MuJoCo produce the same contact points and
  depths (MJX occasionally one extra). Collision geometry is not the cause.
- Solver: with one iteration from a cold start the two engines' accelerations differ (yaw-acceleration RMS
  ≈ 14 rad/s² between them); warm-started MJX matches the converged native solution closely. Earlier,
  10 iterations changed neither engine's turns, so the solver is not established as a cause.

## Reading

- The left-turn failure is a sim-to-sim gap, about half of which is the evaluation's lack of the
  training-time delays [measured]. A real robot has latency, so the delay-free native protocol is itself
  a modelling choice; changing the evaluation protocol would need its own preregistration and cannot
  rescue past results.
- A small rightward pivot bias exists in MJX for both turns [measured]; its cause is unknown.
