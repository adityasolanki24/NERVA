# Neutral long motor gate: fails only on the steady left turn; robust everywhere else, with and without latency

Protocol: `docs/neutral_motor_gate.md`, preregistered at `b6ec4c5`; runner `neutral_gate.py` at `d29bf8e`
(recorded in `protocol.json`). Local CPU, no training or paid work; 160 trials in 287 s. Policy: the
base-origin candidate (SHA256 `9394fa5d…`), evaluated knowing it had failed the 3-seed pilot on the left
turn. Nothing was changed after results.

## Decision

| criterion | no latency | training-like latency (0–2 step action delay) |
|---|---|---|
| 1. stability (all complete, no falls in steady/transitions) | pass | pass |
| 2. steady, every command 5/5 | **fail**: turn left 0/5 | **fail**: turn left 2/5 |
| 3. transitions (every phase, 5 seeds) | pass | pass |
| 4. pushes (no falls; ≥ 4/5 recover per command/direction) | pass (40/40) | pass (40/40) |

**The gate fails**, as expected. The only failing check anywhere is steady turn-left translation (`cross`).

## Details

| | no latency | latency |
|---|---|---|
| turn left horizontal RMS (limit 0.03 m/s), seeds 0–4 | 0.0345, 0.0343, 0.0325, 0.0340, 0.0334 | 0.0312, 0.0332, **0.0298, 0.0299**, 0.0312 |
| turn right horizontal RMS | 0.019–0.021 | 0.024–0.026 |
| forward / backward / left / right signed mean (m/s; request 0.074) | 0.062 / 0.079 / 0.073–0.075 / 0.078–0.080 | 0.057–0.060 / 0.072–0.073 / 0.080–0.087 / 0.079–0.084 |
| rest | 5/5 | 5/5 |
| transitions: 8 phases × 5 seeds | all pass | all pass |
| push recovery (0.30 m/s, 4 directions × forward/backward × 5 seeds) | 40/40, mean 1.11 s, max 1.48 s | 40/40, mean 1.09 s, max 1.54 s |
| max tilt: unperturbed / pushed | 8.2° / 15.5° | 8.4° / 14.8° |
| falls; shadow-safety interventions | 0; 0 | 0; 0 |

- No-latency steady results reproduce the 3-seed pilot exactly for seeds 0–2 (same simulator and seeds),
  and seeds 3–4 agree.
- Latency improves the left turn (closer to the MJX training behaviour) but not enough for 5/5, and makes
  the right turn and forward speed slightly worse; both still pass.
- Compared with the B2 gate (`results_b2/`, different command magnitudes and with head checks), this
  candidate also recovers every push and handles every transition, and is the first to pass a pure turn
  in the native evaluation.

## Reading

Robust within this protocol except for one remaining issue: the left turn's sideways pivot offset
(`results_turn_asymmetry/`). One training seed; simulation only. No default, deployment, safety or
expressive change. Raw rollouts (160) are local and ignored: `experiments/cloud_runs/neutral-gate-local/`.

Files: `protocol.json`, `trials.json`, `summary.json`.
