# Base-origin reward velocity: pivot moves to the base, turn right passes; turn left misses narrowly (H not supported)

Protocol: `docs/base_origin_velocity_pilot.md`, preregistered at `69a6f0f`. The VM ran `0b31524` (recorded in
`vm.log`); the evaluator protocol was committed in the same commit, before any result (recorded in
`evaluation_protocol.json`). Nothing was changed after results.

## Run

| item | value |
|---|---|
| VM | one g2-standard-8 (1× NVIDIA L4), us-central1-a, on-demand; first zone, no stockout |
| VM lifetime | ≈ 09:20 → 09:53 UTC ≈ **33 min** (hard cap 45 min, delete-on-expiry); job start at uptime 195 s, trainer exit at uptime 1,960 s |
| training | **375 accepted iterations = 60,480,000 transitions**, stop: step ceiling; first iteration (compilation + reset) 201 s |
| throughput | preregistered rule over iterations 2–4: 35,648 steps/s → ceiling 375 iterations; steady state ≈ 3.8 s/iteration ≈ 42,400 steps/s |
| GPU | `jax.devices() = [CudaDevice(id=0)]`; mean utilization 70% (58 samples), 17 GB memory |
| integrity | inputs 46/46 checksums OK on the VM; all 375 iterations accepted; replay error 0; max post-update KL 0.0160 (limit 0.1); frozen statistics unchanged; final checkpoint roundtrip exact; local re-hash of the 11 checkpoint files and the ONNX match; local file count equals the bucket (236) |
| export parity | 100 probes, max action error 1.4e-6 (limit 1e-5) |
| estimated cost | ≈ US$0.70: VM ≈ 34 billed min × $0.8536/h ≈ $0.48, NAT data processing ≈ $0.2, disk/NAT gateway/storage < $0.03 |
| cloud state after run | no instances, disks, addresses, forwarding rules, routers or snapshots; results bucket and runner identity retained |

Training reward rose from 0.35–0.41 to 0.49–0.56 for every command; true terminations ≈ 0.12–0.14% of
environment-steps per command. Neither is a motor-readiness claim.

## Paired native-MuJoCo evaluation (84 trials, no falls in any arm)

| command | historical B2 | untrained neutral | gait-averaged (start) | **base-origin** | base-origin: signed axis mean / translation |
|---|---:|---:|---:|---:|---|
| rest | 0/3 | 3/3 | 3/3 | **3/3** | displacement 0.007–0.008 m |
| forward +0.074 m/s | 0/3 | 0/3 | 3/3 | **3/3** | 0.062 m/s (84%); cross 0.015–0.016 |
| backward −0.074 m/s | 0/3 | 0/3 | 3/3 | **3/3** | 0.079 m/s (107%); cross 0.016–0.018 |
| left +0.074 m/s | 1/3 | 1/3 | 3/3 | **3/3** | 0.073–0.075 m/s (100%; start 122%); cross 0.023–0.024 |
| right −0.074 m/s | 1/3 | 1/3 | 3/3 | **3/3** | 0.078–0.079 m/s (106%); cross 0.013–0.017 |
| turn left +0.60 rad/s | 0/3 | 0/3 | 0/3 | **0/3** | 0.60–0.61 rad/s; translation **0.033–0.035** m/s (> 0.03; start 0.066–0.068) |
| turn right −0.60 rad/s | 0/3 | 0/3 | 0/3 | **3/3** | 0.57 rad/s; translation **0.019–0.020** m/s (start 0.053–0.054) |

Normalized moving-axis RMSE: B2 0.4302, untrained 0.4250, gait-averaged 0.1496, **base-origin 0.1346**
(ratio to untrained 0.317; to the start 0.900). Turn left fails only on translation (`cross`).

## Pivot (same functions as `turn_pivot_diagnostic.py`)

| | turn left: base / IMU translation (m/s), pivot forward, left (m) | turn right |
|---|---|---|
| gait-averaged (start) | 0.067 / 0.024, pivot −0.107, +0.002 | 0.054 / 0.022, pivot −0.080, +0.023 |
| base-origin | 0.034 / 0.053, pivot **−0.014, −0.046** | 0.020 / 0.049, pivot **+0.002, −0.014** |

The pivot moved forward from near the IMU to within 1.4 cm of the base origin in both turns, as the
diagnostic predicted. Turn left's remaining translation is a 4.6 cm sideways offset (pivot to the
robot's right), not a forward one.

## Preregistered decision

| criterion | result |
|---|---|
| 1. both turns translate ≤ 0.03 m/s at the base origin in 3/3 seeds | **fail** (turn right 0.019–0.020 passes; turn left 0.033–0.035) |
| 2. rest, forward, backward, left, right still 3/3; no added falls | **pass** |
| 3. all seven commands pass and RMSE ≤ 90% of untrained | **fail** (6/7; RMSE ratio 0.317) |

**H not supported; the pilot fails**, narrowly, on turn left. Measuring the rewards at the base origin
halved turn translation (left 0.067 → 0.034, right 0.054 → 0.020), moved the pivot to the base as
predicted, made turn right the first passing pure turn in any arm, and kept or improved every
translation (left overshoot gone). The remaining failure is a left/right asymmetry: a sideways pivot
offset in the left turn. One training seed, final checkpoint, no selection; the threshold was not
touched. The long robustness gate remains unmet; no default, deployment or expressive change.

## Videos (local, ignored)

`experiments/cloud_runs/base_origin_velocity-20261010-201858/`: seven 20 s clips `comparison_<command>.mp4`
(historical B2 | gait-averaged candidate | base-origin candidate) and a combined 140 s
`comparison_all_commands.mp4`. Hashes in `videos.json` and `combined_video.json`.

## Files

`training_protocol.json`, `training.json`, `step_ceiling.json`, `training_summary.json`,
`input_manifest.sha256`, `gpu_model.txt`, `export_parity.json`, `evaluation_protocol.json`,
`evaluation.json`, `motor_summary.json`, `videos.json`, `combined_video.json`.
