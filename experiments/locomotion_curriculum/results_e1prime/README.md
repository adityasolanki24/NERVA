# E1′ pitch × crouch conditioning: FAILS — pitch is controllable but too small, crouch is ignored; neutral skill retained

Protocol: `docs/expressive_posture_e1prime.md`, preregistered at `0c221cf`. Phase A′ passed
(`../results_e1prime_references/`). VM code `9cb0b29`. Evaluator `6c1df31` and gate candidate, committed before
results. Final accepted checkpoint only; no thresholds changed.

## Training [measured]

| item | value |
|---|---|
| attempt 1 | `e1_prime-20261011-120126`: stopped at iteration 1 by the replay-integrity check (GPU logits error 0.0101 > 1e-3), **0 transitions**, ≈ 10 min VM, ≈ US$0.2 (`../results_e1prime_attempt1_integrity_stop/`). Fix: full float32 matmul precision for E1′; relaunch authorized by the user |
| attempt 2 (this result) | `e1_prime-20261011-132718`, one L4, us-central1-a, 90 min hard cap; VM ≈ 02:27 → 03:18 UTC ≈ 51 min; trainer exit at uptime 2,844 s |
| training | **496 iterations, 79,994,880 transitions**, stop: step ceiling; 30,950 steps/s measured (full precision ≈ 25% slower than before); first iteration 139 s |
| **deviation** | the preregistration allowed ≤ 160 M with the ceiling from throughput (≈ 133 M available). The code took min(80 M default, 160 M) = 80 M. **E1′ trained ≈ 40% fewer transitions than the preregistered rule allowed**; the curriculum spanned that 80 M ceiling. Fixed for the future in `33bbe74` |
| curriculum | scale 0.25 → 1.0 at iteration 249, then the full domain; neutral-anchor fraction 0.19–0.20; held-out region sampled 0 times |
| integrity | inputs 411/411; all iterations accepted; max KL 0.0149; max replay error 4.2e-5; frozen statistics unchanged; exact roundtrip; checkpoint (10 files) and ONNX hashes match; 306/306 files fetched; 24 recoverable snapshots |
| GPU | `cuda:0`, mean utilization 77% (83 samples), 17 GB |
| cost | attempt 2 ≈ 51 min × $0.8536/h ≈ $0.73 + NAT ≈ $0.2 → **≈ US$0.95**; both attempts ≈ **US$1.15** (cap US$3) |
| cloud after | no instances, disks, addresses, forwarding rules, routers or snapshots |

## Decision (preregistered criteria)

| criterion | result |
|---|---|
| 1. Phase A′ | pass |
| 2. neutral gate at e = 0, both latency conditions | **pass** (`gate/`): every command 5/5, no falls in 160 trials, all transitions, 80/80 pushes, no shadow interventions |
| 3. direction and range (≥ 50% of the reference change, every command and seed) | **fail**: pitch 9–30% of the reference; crouch ≈ 0% |
| 4. monotonicity of seed means | **fail**: pitch monotonic for all six moving commands, not at rest; crouch not monotonic |
| 5. cross-talk ≤ 25% | **fail**: the pitch axis moves height 3–24× more than the crouch axis does; crouch → pitch 0.12–0.28 |
| 6. task across the style space incl. held-out | **pass**: all commands 3/3 at the four trained and two held-out points; 0 falls with and without latency |
| 7. task interference (speed ratio within 0.75–1.33) | **pass**: 0.93–1.15 |
| 8. continuity of ramps | **fail**: pitch ramps Spearman 0.93–0.99, but 2 of 3 have one window jump of 0.26–0.38 (> 0.25); crouch ramps show no relation (ρ −0.46…0.10) |

**H not supported; E1′ fails.**

## What the style input does [measured]

**Pitch axis** (e_pitch −1 → +1 at e_crouch 0; reference change 0.209 rad = 12.0°):

| command | base pitch, seed means at e_pitch = −1, −0.5, 0, 0.5, 1 (rad) | change / reference |
|---|---|---|
| forward | 0.004, 0.014, 0.026, 0.038, 0.050 | 0.21–0.23 |
| backward | 0.000, 0.006, 0.012, 0.016, 0.019 | 0.09 |
| left | −0.020, −0.012, −0.002, 0.010, 0.021 | 0.19–0.20 |
| right | −0.014, 0.000, 0.016, 0.031, 0.046 | 0.27–0.30 |
| turn left | −0.006, 0.001, 0.008, 0.017, 0.029 | 0.16–0.18 |
| turn right | 0.010, 0.020, 0.026, 0.034, 0.043 | 0.15–0.17 |
| rest | −0.209 throughout | 0.01 |

The pitch effect is real, monotonic and in the reference's direction while moving (1.1°–3.4°), at roughly the
fraction S1 reached (≈ 1/3). It is **not a speed artefact**: between the pitch extremes, translation speed changes
by < 0.01 m/s, and the effect is unchanged after removing the within-trial speed–pitch slope
(`style/speed_adjusted_pitch.json`). Command calibration for speed matching is not possible under the neutral
contract, so this is the speed check. At rest the style has no effect.

**Crouch axis** (reference −7.2 mm): base height changes ≤ 0.2 mm, often with the wrong sign, for every command.
**The input is ignored.** Over the same sweep, pitch changes height by −1.5 to −2.7 mm (cross-talk).

**Held-out combinations** (0.75, 0.75) and (1, 1): every command passes the motor rules in 3/3 seeds, with no
falls with or without latency. They are stable; their posture carries only the small pitch effect (and no crouch).

## Neutral retention vs the validated neutral candidate (5 seeds, no latency; `gate/neutral_posture_comparison.json`)

| | neutral candidate | E1′ at e = 0 |
|---|---|---|
| turn translation left / right (m/s) | 0.021–0.023 / 0.025–0.027 | **0.020–0.021 / 0.019–0.020** |
| yaw left / right (rad/s) | 0.647 / −0.558 | 0.613 / −0.582 |
| forward / right speed (m/s) | 0.073 / 0.077 | 0.074 / **0.101** (right overshoot, within the rules) |
| walking torso pitch | +1.2…+2.6° | −0.1…+1.5° (≈ 1° lower) |
| push recovery max, no latency / latency | 1.82 / 1.58 s | 1.58 / 2.10 s |

The neutral controller is kept, and the turns are slightly better.

## Videos (local, ignored: `experiments/cloud_runs/e1_prime-20261011-132718/`)

Side camera, seed 0, same start and duration; hashes in `style/videos.json`.
- `e1prime_forward_extremes.mp4`, `e1prime_rest_extremes.mp4`, `e1prime_turn_left_extremes.mp4`: neutral
  candidate | E1′ e = 0 | pitch −1 | pitch +1 | crouch 1.
- `e1prime_held_out.mp4`: E1′ e = 0 | (0.75, 0.75) | (1, 1).
- `e1prime_ramp_pitch.mp4`, `e1prime_ramp_crouch.mp4`: continuous 30 s ramps.
- The visible differences are small (a few degrees of pitch); the crouch clips show no visible change. That is the
  failure, kept visible.

## Reading

- [measured] A continuous style input on top of the validated neutral controller gives monotonic, speed-independent
  torso-pitch control without destabilizing locomotion, but at only 10–30% of the reference range, and the crouch
  input is ignored.
- [hypothesis, untested] Likely contributors: the imitation term's joint targets for a 7 mm COM change are small
  relative to the reward's other terms; pitch was learned at 1/3 magnitude in S1 too; and this run trained 80 M
  rather than the ≈ 133 M the preregistered rule allowed.
- No PAD/affect connection; no default change; no emotional claim.

## Files

`gate/` (neutral retention, posture comparison), `style/` (protocol, trials, ramps, summary, speed-adjusted
pitch, videos), `training_protocol.json`, `training.json`, `step_ceiling.json`, `training_summary.json`,
`input_manifest.sha256`, `gpu_model.txt`.
