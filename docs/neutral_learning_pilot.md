# Balanced neutral learning pilot (preregistration, 2026-10-10)

Baseline: `e42735a`. This is a local motor-learning pilot, not a repeat of S1–S6
or a replacement for the long robustness gate. No affect, physical-safety,
hardware deployment, default promotion or paid cloud changes.

Hypothesis: a small PPO refinement of the already walking B2 weights under the
verified neutral reference/reward/clock contract can reduce seven-command error
without increasing falls. The previous 32-wide identity network is a plumbing
control, not a useful walking initialization. Use the retained B2 final checkpoint
(300,482,560 training steps), whose ONNX SHA256 is
`6ebfe384f4bcb8f06657ba597825744a01a69146ab07583b5f993f15b7f2c9a1`.

## Fixed training rule

Remove only the three constant-zero style inputs from each B2 network. For
first-layer kernel W and bias b, keep W[:-3] and set
b' = b + (-mean[-3:]/std[-3:]) @ W[-3:]. Keep the original 512/256/128 SiLU
networks and frozen B2 mean/std on the retained 101/212 observation entries.
Statistics are never updated or rebased. Before collecting training data, verify
deterministic action parity against B2 ONNX on 100 seeded finite probes (zero
style appended), max absolute error <=1e-5. Reject incompatible checkpoints.
The fixed-preprocessing offline result motivates this choice; the raw identity
experiment remains preserved and is not claimed equivalent to this pilot.

Use NeutralJoystick with the existing seven hash-verified references, backlash
scene, unchanged neutral reward and physics/noise settings, no new domain
randomization. Seven environments, exactly one per command in canonical order:
rest, forward/backward 0.074 m/s, left/right 0.074 m/s, left/right yaw 0.60 rad/s.
Force command before computing reset observations/reference; episode length
256 control steps (5.12 s) keeps commands fixed before the upstream 500-step
resampling boundary. Autoreset restores the matching command/clock/observation.
Training seed 27. Warm parameters only: fresh Adam, RNG, environment/counters.
One newly collected 32-step trajectory per environment per batch, one full-batch
Adam step, learning rate 1e-5, global gradient norm clipped to 1.0, PPO clip 0.2,
discount 0.97, GAE 0.95, entropy coefficient 0.005, reward scale 1, value
coefficient 0.5, normalized advantage. Adam/RNG/physical states carry across
batches. Maximum 128 batches = 28,672 new transitions/128 optimizer steps.

Stop on nonfinite data/gradient/output, command-coverage mismatch, behavior
replay discrepancy (logits >1e-4 or log probability >1e-3), or mean post-update
Gaussian KL >0.05 on the current batch. Preserve the rejected update and use
the last accepted parameters for evaluation. Soft stop at 1,100 wall seconds;
hard worker kill at 1,200 seconds. Save the initial policy and accepted checkpoints
every 16 batches and at exit, preserving partial work. Never extend budget or
adjust criteria after seeing results. Report value loss and true terminations
separately from timeouts; no fixed value-loss limit because the reward changed.

## Paired motor evaluation and visible output

Three arms: historical B2 with its original gait clock; converted B2 with the
neutral clock and no optimizer update; final accepted learned candidate with
the neutral clock. Same backlash scene, raw accelerometer, observation noise,
initial joint noise +/-0.02 rad, seeds 0/1/2, zero head offsets, seven commands
above, 20 s per independent trial (63 trials). Score seconds 5–20. No safety
override in primary motor trials; deterministic safety stays separate.
Abort an individual trial at nonfinite state or tilt >45 degrees; retain the
failure and exclude incomplete tracking from aggregate means. Never score a
shorter survival interval as successful tracking. Evaluation wall cap 600 s.

Report per-command survival, heading-frame mean velocity, trailing 1 s tracking
RMSE, orthogonal/yaw drift, rest displacement and foot lift. A command passes
only if all three trials complete without falling. Movement also requires
signed mean >=50% of request, tracking RMSE <=60% of request and stationary
fraction <=10% (<0.01 m/s or <0.10 rad/s). Translation cross RMS <=0.05 m/s and
yaw RMS <=0.20 rad/s; pure-turn horizontal RMS <=0.03 m/s. Rest horizontal
RMS <=0.02 m/s, yaw RMS <=0.15 rad/s and displacement <=0.10 m over 20 s.

Pilot improvement requires all seven candidate commands pass, no additional
falls versus either paired control, and mean dimensionless axis RMSE across
six moving commands <=90% of the untrained converted control. This criterion
is exploratory evidence from one training seed, not robust locomotion approval.
Report both comparisons even on failure; do not select the best checkpoint.
Render seed-0 B2/candidate comparisons for every command, with labels and fall
timestamps (hold the final failed pose through the remaining video). Rendering
cap 600 s; retain raw full states/rollouts/videos in ignored local storage and
publish only small aggregate reports/hashes. If rendering fails, preserve the
rollouts and report that limitation. No cloud run or artifact deletion.

After a failed pilot, recommend the cause supported by motor data and a concrete
bounded next run. After a passing pilot, preregister the full long motor gate
before adding expressive objectives. The original B2 gate failure remains valid.
