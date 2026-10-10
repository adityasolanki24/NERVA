# Capped GPU neutral continuation: overall fail; backward and left now pass, right regresses

Protocol: `docs/gpu_neutral_pilot.md`, preregistered at `a3ba5ca` before implementation. Implementation
`1d423dd` (the code the VM ran), evaluator `908cd6a` (recorded in
`evaluation_protocol.json`). Nothing was changed after results.

## Run

| item | value |
|---|---|
| VM | one g2-standard-8 (1× NVIDIA L4), us-central1-c, on-demand; first attempt hit L4 stockout in all three zones (no VM created), second attempt succeeded |
| VM lifetime | 04:29:59 → 05:02:39 UTC = **32.7 min** (hard cap 45 min, delete-on-expiry) |
| bootstrap | job started at uptime 236 s; first iteration (compilation + reset) 319 s |
| training | **374 accepted iterations = 60,318,720 transitions** (≈ 2,100× the local pilot), stop: step ceiling |
| throughput | preregistered rule over iterations 2–4: 36,291 steps/s (iteration 2 was slower) → ceiling 374 iterations; steady state 3.77 s/iteration ≈ 42,800 steps/s |
| GPU | `jax.devices() = [cuda:0]`; mean utilization 71% (59 samples, max 100%), 17 GB memory |
| integrity | inputs 46/46 checksums OK on the VM; behaviour replay error 0; max post-update KL 0.0192 (limit 0.1); frozen statistics unchanged; final 25-leaf checkpoint roundtrip exact; local re-hash of checkpoint and ONNX match |
| checkpoints | 18 intermediate + final parameter checkpoints with Adam/key/iteration snapshots (every 20 iterations); parameter + optimizer restart only, not physical-environment resume |
| estimated cost | ≈ US$0.70: VM ≈ 34 billed min × $0.8536/h ≈ $0.48, NAT data processing ≈ $0.2, disk/NAT gateway/storage < $0.03 |
| cloud state after run | no instances, disks, addresses, forwarding rules, routers or snapshots; results bucket and runner identity retained |

Training reward rose from 0.33 to 0.49 for every command; true episode terminations were 0.13–0.18%
of environment-steps per command. Neither is a motor-readiness claim.

## Paired native-MuJoCo evaluation (84 trials, no falls in any arm)

Passing trials per command (a command passes with 3/3):

| command | historical B2 | untrained neutral | local pilot (start) | **GPU candidate** | GPU candidate: signed axis mean / translation |
|---|---:|---:|---:|---:|---|
| rest | 0/3 | 3/3 | 3/3 | **3/3** | displacement 0.010–0.014 m |
| forward +0.074 m/s | 0/3 | 0/3 | 0/3 | **0/3** | +0.031–0.032 m/s (< 0.037); cross 0.027–0.030 |
| backward −0.074 m/s | 0/3 | 0/3 | 0/3 | **3/3** | 0.065–0.066 m/s (pilot 0.016) |
| left +0.074 m/s | 1/3 | 1/3 | 1/3 | **3/3** | 0.070–0.071 m/s (pilot 0.035–0.037) |
| right −0.074 m/s | 1/3 | 1/3 | 3/3 | **0/3** | 0.019–0.023 m/s (pilot 0.037–0.040), regressed |
| turn left +0.60 rad/s | 0/3 | 0/3 | 0/3 | **0/3** | 0.71 rad/s; translation 0.041–0.044 m/s (> 0.03; pilot 0.032–0.035) |
| turn right −0.60 rad/s | 0/3 | 0/3 | 0/3 | **0/3** | 0.66–0.67 rad/s; translation 0.039–0.042 m/s (pilot 0.043–0.045) |

Normalized moving-axis RMSE: B2 0.4302, untrained 0.4250, local pilot 0.4130, **GPU candidate 0.3104**
(ratio to untrained 0.730; to pilot 0.752).

## Preregistered decision

| criterion | result |
|---|---|
| 1. forward/backward/left ≥ 50% speed in 3/3 and both turns ≤ 0.03 m/s translation in 3/3 | **fail** (forward 43%; both turns 0.039–0.044) |
| 2. no added falls | pass (0 falls in all 84 trials) |
| 3. all seven commands pass and RMSE ≤ 90% of untrained | **fail** (3/7 pass, although the RMSE ratio is 0.730) |

**H not supported; the pilot fails.** 60 M transitions changed the policy substantially (backward and
left now pass; aggregate error −27%), but forward still undershoots, rightward motion regressed, and
pure-turn translation was not reduced. The improvements are asymmetric: left up/right down, plus
lateral drift while walking forward. This is consistent with a learned lateral bias [hypothesis, not
tested]. Turn-time lateral drift is already present in historical B2 and the local pilot. No checkpoint
was selected after evaluation; the long robustness gate remains unmet; no default or expressive change.

## Videos (local, ignored)

`experiments/cloud_runs/neutral_gpu_pilot-20261010-152812/`: seven 20 s clips
`comparison_<command>.mp4` (historical B2 | local pilot candidate | GPU candidate, labelled with command,
mean velocity and PASS/FAIL) and a combined 140 s `comparison_all_commands.mp4`. Hashes are in
`videos.json` and `combined_video.json`. Raw rollouts (84), checkpoints, snapshots and logs are in the
same folder.

## Files

`training_protocol.json`, `training.json` (per iteration: KL, per-command reward and terminations,
losses, timing), `step_ceiling.json`, `training_summary.json`, `input_manifest.sha256`, `gpu_model.txt`,
`export_parity.json`, `evaluation_protocol.json`, `evaluation.json`, `motor_summary.json`, `videos.json`,
`combined_video.json`.
