# B2 longer turns: aggregate inconclusive; zero-command controls migrate

Protocol [preregistered](../../../docs/b2_long_turn_experiment.md) at `dec8fba`;
implementation `7f18b7f`. Fifteen 65-second primary trials, seeds 0–4,
completed in 109.62 s wall time (cap 900 s). No training or cloud work.
Per-trial metrics and raw trace hashes: trials.json. Checkpoint/model hashes
and upstream revision: protocol.json. Raw NPZ traces remain local and ignored.

| Fixed classification | Left turns | Right turns | Zero command |
|---|---:|---:|---:|
| local region | 3/5 | 2/5 | 0/5 |
| coherent migration | 0/5 | 1/5 | 5/5 |
| inconclusive | 2/5 | 2/5 | 0/5 |

Neither aggregate hypothesis passes: neither turn direction reaches 4/5 for
one description, and all stationary controls fail local-region support.
All 15 trials were upright, no shadow safety interventions, maximum tilt
14.59 degrees. Every turn produced five complete rotations; stationary trials
used six fixed blocks. Minimum per-block contact-centroid coverage was 99.81%.

| Measurement | Left turns | Right turns | Zero command |
|---|---:|---:|---:|
| maximum COM block-centre separation | 38.01–64.97 mm | 37.79–124.02 mm | 555.94–587.86 mm |
| COM first-to-last block displacement | 14.93–57.64 mm | 33.72–124.02 mm | 555.94–587.86 mm |
| both-foot midpoint first-to-last displacement | 15.86–58.61 mm | 33.13–124.29 mm | 563.78–595.91 mm |
| contact-centroid first-to-last displacement | 14.67–58.35 mm | 34.51–124.77 mm | 554.82–586.12 mm |
| COM block-endpoint drift speed | 0.332–1.286 mm/s | 0.788–2.887 mm/s | 11.12–11.75 mm/s |
| smoothed base horizontal RMS | 35.59–38.16 mm/s | 43.68–45.06 mm/s | 15.21–15.63 mm/s |

Displacements compare time-averaged centres of blocks, not raw start/end
positions. Turn averaging covers complete measured yaw rotations; final
partial rotations are excluded as preregistered. COM and feet migrate together
in every zero-command control, so this effect is not merely an offset base
origin rotating about a stationary foot region. The foot centroid is a
site-based proxy, not a force centre or a support polygon.

The original gate's instantaneous stop-speed criteria still pass over this
longer interval; low residual speed can nevertheless accumulate substantial
position error. Do not retroactively fail or relax those original criteria.
The original pure-turn horizontal-speed criterion remains failed in all ten
long turns. B2 is not adopted as a robust expressive-training baseline.

Validation: isolated mass-weighted COM matches MuJoCo subtree COM; recorder
leaves live qpos/qvel/qacc/warm-start/site state unchanged. The first 20 s of
left seed 0 match the original gate's base pose, velocity, actions and contacts
exactly, and checkpoint hashes match. Six new synthetic/integration tests
cover fixed rotation/stop blocks, local versus migrating motion, incomplete,
airborne and safety-intervention exclusion, and isolated kinematics. Complete suite including slow tests: 253 passed;
Ruff clean. Two existing JAX cast warnings in unchanged training tests.

Next local phase: preregister a neutral motor-target audit, including zero
command reference selection, stored velocity slices, and reward terms active
while stationary. These are candidate mechanisms, not established causes of
creep. Design a motor-only candidate addressing stopping and in-place turns
before checkpoint continuation or expressive objectives. Any future training
still needs an explicitly authorised small deterministic cloud cap.
