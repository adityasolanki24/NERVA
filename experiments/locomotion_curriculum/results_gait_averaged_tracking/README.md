# Gait-averaged tracking continuation: H supported (all four translations pass); pilot fails on turns

Protocol: `docs/gait_averaged_tracking_pilot.md`, preregistered at `a143ae4`. Implementation `3dad8c8`
(the code the VM ran, recorded in `vm.log`); evaluator `72326ce`, committed before any result was
seen (recorded in `evaluation_protocol.json`). Nothing was changed after results.

## Run

| item | value |
|---|---|
| VM | one g2-standard-8 (1× NVIDIA L4), us-central1-a, on-demand; first zone tried, no stockout |
| VM lifetime | ≈ 06:02 → 06:35 UTC ≈ **33 min** (hard cap 45 min, delete-on-expiry); job start at uptime 164 s, trainer exit at uptime 1,965 s |
| training | **384 accepted iterations = 61,931,520 transitions**, stop: step ceiling; first iteration (compilation + reset) 198 s |
| throughput | preregistered rule over iterations 2–4: 35,728 steps/s → ceiling 384 iterations; steady state ≈ 3.8 s/iteration ≈ 42,400 steps/s |
| GPU | `jax.devices() = [CudaDevice(id=0)]`; mean utilization 77% (60 samples, max 100%), 17 GB memory |
| integrity | inputs 47/47 checksums OK on the VM; all 384 iterations accepted; behaviour replay error 0; max post-update KL 0.0147 (limit 0.1); frozen statistics unchanged; final 25-leaf checkpoint roundtrip exact; local re-hash of the 10 checkpoint files and the ONNX match; local file count equals the bucket (242) |
| export parity | 100 probes, max action error 1.1e-6 (limit 1e-5) |
| estimated cost | ≈ US$0.70: VM ≈ 34 billed min × $0.8536/h ≈ $0.48, NAT data processing ≈ $0.2, disk/NAT gateway/storage < $0.03 |
| cloud state after run | no instances, disks, addresses, forwarding rules, routers or snapshots; results bucket and runner identity retained |

Training reward (now with the gait-averaged tracking term, so not comparable with the GPU pilot's) rose
from 0.35–0.40 to 0.49–0.55 for every command; true terminations stayed ≈ 0.12–0.14% of
environment-steps per command. Neither is a motor-readiness claim.

## Paired native-MuJoCo evaluation (84 trials, no falls in any arm)

Passing trials per command (a command passes with 3/3):

| command | historical B2 | untrained neutral | GPU pilot (start) | **gait-averaged** | gait-averaged: signed axis mean / cross or turn translation |
|---|---:|---:|---:|---:|---|
| rest | 0/3 | 3/3 | 3/3 | **3/3** | displacement 0.015–0.019 m |
| forward +0.074 m/s | 0/3 | 0/3 | 0/3 | **3/3** | 0.058–0.059 m/s (79%; start 0.031–0.032); cross 0.013–0.014 |
| backward −0.074 m/s | 0/3 | 0/3 | 3/3 | **3/3** | 0.079–0.080 m/s (108%); cross 0.018–0.022 |
| left +0.074 m/s | 1/3 | 1/3 | 3/3 | **3/3** | 0.089–0.091 m/s (122%, overshoot); cross 0.012–0.014 |
| right −0.074 m/s | 1/3 | 1/3 | 0/3 | **3/3** | 0.064–0.068 m/s (88%; start 0.019–0.023); cross 0.012–0.013 |
| turn left +0.60 rad/s | 0/3 | 0/3 | 0/3 | **0/3** | 0.61–0.62 rad/s (start 0.71); translation **0.066–0.068** m/s (> 0.03; start 0.041–0.044) |
| turn right −0.60 rad/s | 0/3 | 0/3 | 0/3 | **0/3** | 0.59–0.60 rad/s (start 0.66–0.67); translation **0.053–0.054** m/s (start 0.039–0.042) |

Normalized moving-axis RMSE: B2 0.4302, untrained 0.4250, GPU pilot 0.3104, **gait-averaged 0.1496**
(ratio to untrained 0.352; to the GPU pilot 0.482).

## Preregistered decision

| criterion | result |
|---|---|
| 1. forward/backward/left/right signed axis mean ≥ 50% of the request in 3/3 seeds each | **pass** (79–122%) |
| 2. no added falls, and rest 3/3 | **pass** (0 falls in 84 trials; rest 3/3) |
| 3. all seven commands pass and RMSE ≤ 90% of untrained | **fail** (5/7; both turns fail on translation, although the RMSE ratio is 0.352) |

**H is supported; the pilot fails** on criterion 3, as the preregistration expected for the turns.

- Changing only the tracking term (instantaneous → gait-period-averaged velocity) removed the
  translation undershoot and the right-side regression: all four translations now pass in every seed,
  with lower cross-axis drift than the start on forward, left and right (backward slightly higher, 0.018–0.022
  vs 0.016).
- Left overshoots (122%). It still passes the preregistered RMSE rule; reported, not tuned.
- Yaw rate is now close to the command natively (0.59–0.62 rad/s), but the robot translates **more**
  while turning (0.053–0.068 m/s vs 0.039–0.044 at the start). The averaged tracking term only scores
  mean planar velocity over 0.54 s, so drift that cancels within a window is not penalized, and the
  MJX → native turn gap (`results_sim_gap/`) is unaddressed. Both are explanations, not tested
  [hypothesis].
- One training seed, final checkpoint only, no selection. The long robustness gate remains unmet; no
  default, deployment or expressive change.

## Videos (local, ignored)

`experiments/cloud_runs/gait_averaged_tracking-20261010-170130/`: seven 20 s clips
`comparison_<command>.mp4` (historical B2 | GPU pilot candidate | gait-averaged candidate, labelled with
command, mean velocity and PASS/FAIL) and a combined 140 s `comparison_all_commands.mp4`. Hashes are in
`videos.json` and `combined_video.json`. Raw rollouts (84), checkpoints, snapshots and logs are in the
same folder.

## Files

`training_protocol.json`, `training.json` (per iteration: KL, per-command reward and terminations,
losses, timing), `step_ceiling.json`, `training_summary.json`, `input_manifest.sha256`, `gpu_model.txt`,
`export_parity.json`, `evaluation_protocol.json`, `evaluation.json`, `motor_summary.json`, `videos.json`,
`combined_video.json`.
