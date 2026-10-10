# Turn-translation tracking pilot: the neutral long motor gate PASSES, with and without latency

Protocol: `docs/turn_translation_pilot.md`, preregistered at `366aec9` (pinned `deae89e`); the VM ran `a009a3a`.
Decision: the neutral long motor gate (`docs/neutral_motor_gate.md`, `b6ec4c5`, protocol unchanged) on the
final accepted checkpoint only. Nothing was changed after results. One authorized run; no other run.

## Run

| item | value |
|---|---|
| VM | one g2-standard-8 (1× NVIDIA L4), us-central1-a, on-demand, first zone |
| VM lifetime | ≈ 13:08 → 13:41 UTC ≈ **33 min** (hard cap 45 min); job start at uptime 229 s, trainer exit at uptime 1,960 s. A first launch attempt was refused locally (uncommitted file) before any VM existed |
| training | **367 accepted iterations = 59,189,760 transitions**, stop: step ceiling (35,443 steps/s measured over iterations 2–4; steady ≈ 3.8 s/iteration); first iteration (compile + reset) 197 s |
| GPU | `cuda:0`; mean utilization 72% (57 samples), 17 GB |
| integrity | inputs 47/47 OK on the VM; all iterations accepted; max KL 0.0149; frozen statistics unchanged; exact roundtrip; local checkpoint (11 files) and ONNX hashes match; 236/236 files fetched; export parity 1.2e-6; 18 recoverable snapshots + 19 checkpoints |
| estimated cost | ≈ US$0.70 (VM ≈ 34 min × $0.8536/h ≈ $0.48; NAT ≈ $0.2; disk/storage < $0.03) |
| after the run | no instances, disks, addresses, forwarding rules, routers or snapshots; results bucket and runner identity retained |

## Decision: full neutral gate (160 trials, 5 seeds, both latency conditions)

| criterion | no latency | training-like latency |
|---|---|---|
| 1. stability | pass (0 falls) | pass (0 falls) |
| 2. steady, every command 5/5 | **pass**: all seven 5/5 | **pass**: all seven 5/5 |
| 3. transitions (8 phases × 5 seeds) | pass | pass |
| 4. pushes (40) | pass: 40/40, mean 1.08 s, max 1.82 s | pass: 40/40, mean 1.07 s, max 1.58 s |
| shadow-safety interventions | 0 | 0 |

**H supported and the pilot passes:** steady turn left 0.021–0.023 m/s (no latency) and 0.022–0.025 m/s
(latency), limit 0.03, in 5/5 seeds each; the whole gate passes in both conditions. The previous candidate
failed only this check (0/5 and 2/5).

Steady details (5 seeds; request 0.074 m/s or 0.60 rad/s):

| command | no latency | latency |
|---|---|---|
| rest: horizontal RMS / displacement | 0.0003 m/s / 0.005–0.010 m | 0.0004–0.0005 / 0.007–0.011 m |
| forward signed mean | 0.072–0.074 (97–99%) | 0.065–0.066 |
| backward | 0.082–0.084 | 0.084–0.088 |
| left | 0.091–0.096 (**overshoot, 123–130%**) | 0.082–0.093 |
| right | 0.076–0.078 | 0.075–0.084 |
| turn left: translation / yaw | **0.021–0.023** / 0.64–0.65 rad/s | **0.022–0.025** / 0.64–0.66 |
| turn right: translation / yaw | 0.025–0.027 / −0.55 to −0.56 | 0.023–0.027 / −0.59 |

Margins are real but modest: turn right is the closest to its limit (0.027 vs 0.03), the left step overshoots
(still within the tracking rule) and the left turn runs ≈ 8% fast. The descriptive paired evaluation's
aggregate tracking error is slightly higher than the previous candidate's (normalized RMSE 0.155 vs 0.135),
mainly from these speed errors.

## The left turn (descriptive; `turn_left_report.json`, `turn_left_trajectories.png`)

Commanded yaw +0.60 rad/s. Means over 5–20 s:

| | achieved yaw | vx | vy | translation RMS | pivot (forward, left) | stance L/R | touchdowns/s L/R |
|---|---|---|---|---|---|---|---|
| historical B2 (3 seeds) | 0.564 | −0.007 | +0.034 | 0.033–0.038 | −0.060, −0.013 m | 0.49 / 0.55 | 1.9 / 1.8 |
| current (base-origin), 5 seeds, no latency | 0.608 | −0.027 | +0.009 | 0.033–0.035 | −0.014, **−0.045** m | 0.55 / 0.63 | 1.9 / **3.5** |
| current, latency | 0.622 | −0.024 | +0.001 | 0.030–0.033 | −0.001, −0.038 m | 0.57 / 0.65 | 2.1 / 3.7 |
| **trained**, 5 seeds, no latency | 0.647 | −0.011 | +0.005 | **0.021–0.023** | −0.007, **−0.017** m | 0.53 / 0.68 | 2.4 / 2.9 |
| **trained**, latency | 0.649 | −0.012 | −0.003 | **0.022–0.025** | +0.005, −0.018 m | 0.56 / 0.68 | 2.6 / 3.1 |

- B2 translates mostly sideways (vy) with its pivot 6 cm behind the base (the IMU-point pivot).
- The current candidate pivots 4.5 cm to its right, so its base drifts backward in the heading frame
  (vx −0.027); its right foot touches down almost twice as often as its left (3.5 vs 1.9 per s).
- The trained candidate's pivot offset fell to 1.7 cm, vx to −0.011, and the touchdown rates are more even
  (2.4 / 2.9). The trajectory figure shows the smaller drift circle.

## Videos (local, ignored)

`experiments/cloud_runs/turn_translation-20261011-000711/`: `comparison_turn_left.mp4` (historical B2 |
current base-origin candidate | trained candidate) and the six other commands, plus the 140 s
`comparison_all_commands.mp4`; hashes in `paired/videos.json` and `paired/combined_video.json`.

## Scope

One training seed, simulation only, final checkpoint, thresholds unchanged. This is the first candidate to
pass the neutral long motor gate. It opens the preregistration of expressive-objective work. It does **not**
approve hardware deployment, a default change or claims of general robustness.

## Files

`gate/` (`protocol.json`, `trials.json`, `summary.json`), `paired/` (descriptive 84-trial evaluation and video
hashes), `turn_left_report.json`, `turn_left_trajectories.png`, `training_protocol.json`, `training.json`,
`step_ceiling.json`, `training_summary.json`, `input_manifest.sha256`, `gpu_model.txt`.
