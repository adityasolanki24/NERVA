# Capped GPU neutral continuation (preregistration, 2026-10-10)

Baseline: `b22d531`. One authorized paid pilot: at most one GPU VM, hard lifetime 45 min including
bootstrap and compilation, total incremental cost at most US$2, no extension, replacement or second
run. Neutral motor skill only: no expressive objectives, affect, safety, default or deployment changes.

## Hypothesis

The local pilot (`neutral_learning_pilot.md`) failed five commands: forward/backward/left undershoot
half the requested speed, and both pure turns exceed 0.03 m/s horizontal translation RMS. Its maximum
post-update KL was 2.5e-5, so the policy barely changed. That pilot used 28,672 transitions, 128 single
full-batch steps at LR 1e-5, and 5.12 s episodes, shorter than the 20 s evaluation.

**H:** with the reward, references, observation contract and frozen preprocessing unchanged, a
continuation with ~10³× more transitions, standard minibatch PPO epochs, LR 1e-4 and 20 s episodes
changes the policy enough to:
- (a) raise forward, backward and left signed speed to at least half the request; and
- (b) bring pure-turn translation RMS to ≤ 0.03 m/s;

without added falls. More steps are a hypothesis, not a promised fix.

## Starting point and restoration semantics

- **Starting checkpoint:** the local pilot's final accepted checkpoint
  `experiments/cloud_runs/neutral-learning-pilot-corrected/checkpoints/000000028672`: B2 walking
  weights (300,482,560 steps) converted to 101/212 inputs, plus 128 accepted updates.
  - It is the end of the preregistered pilot trajectory, not a selected intermediate.
  - Every file must match the `checkpoint_hashes` in
    `results_neutral_learning_corrected/training_summary.json` before use (verified on the VM).
- **Parameters only:** policy, value and normalizer leaves. Adam, RNG, counters and environment states
  are fresh (seed 41), and carry within this run.
- **Frozen preprocessing:** B2's cropped observation mean/std, never updated or rebased. The
  normalizer fingerprint is checked at start, at every checkpoint and at exit; any change stops the run.
- **Snapshots:** each checkpoint also saves an Adam/key/iteration snapshot. This allows a parameter +
  optimizer restart, but **not** a full physical-environment resume: per-environment physics and
  episode states are not saved.

## Environment

- `NeutralJoystick`, backlash scene, the seven hash-verified references; reward, noise and push
  settings unchanged; no domain randomization.
- **8,064 environments**, 1,152 per command. Environment i always uses `COMMANDS[i mod 7]` (rest,
  ±0.074 m/s forward, ±0.074 m/s lateral, ±0.60 rad/s yaw). The command is forced before the reset
  observation and reference are computed.
- **Episode length 1,000 control steps (20 s).**
  - Upstream resamples commands once `step > 500`, which would silently change the declared
    curriculum. A persistent-command wrapper restores each environment's command after every step.
  - A test asserts the command survives step 501. Every collected batch is checked against the
    expected per-environment command; a mismatch stops the run.
  - Autoreset restores the matching command, clock and observation.

## PPO

| setting | value |
|---|---|
| unroll length | 20 (161,280 new transitions per iteration) |
| minibatches | 32 |
| epochs per batch | 4 |
| learning rate | Adam 1e-4, constant |
| gradient clipping | global norm 1.0 |
| PPO clip | 0.2 |
| discount / GAE | 0.97 / 0.95 |
| entropy coefficient | 0.005 |
| value coefficient | 0.5 |
| advantage | normalized |
| reward scale | 1 |

These are the pilot's loss settings with upstream's batch structure (the B2 training configuration:
8192 environments, unroll 20, 32 minibatches, 4 epochs). LR is 10× the pilot's, because its updates did
not move the policy, and ⅓ of B2's original 3e-4, because this is a fine-tune.

## Budget, step ceiling and stops

- **Hard VM lifetime:** `--max-run-duration 45m`, deleted on expiry. The VM also self-deletes at
  job end.
- **Training deadline:** VM uptime 2,220 s (37 min). This leaves time to save, export and sync before
  the cap.
- **Step ceiling, fixed rule:** after the first three timed iterations following compilation, the
  ceiling is min(80,000,000, throughput × seconds remaining to the deadline), rounded down to whole
  iterations. The measured throughput and ceiling are recorded.
- **Checkpoints:** every 20 iterations (3,225,600 transitions) and at exit, each with its snapshot.
  Results sync to the bucket every 180 s. The final sync runs on any exit path.
- **Stops** (the last accepted parameters are kept and the rejected batch archived):
  - nonfinite data, gradients or outputs;
  - command-coverage mismatch;
  - behaviour replay error (logits > 1e-3 or log-probability > 1e-2; GPU float tolerance);
  - mean post-update Gaussian KL on the batch > 0.1;
  - normalizer change.
- **No extension, relaunch or criteria change after results.**

## Evaluation (local, after fetch and checksum verification)

- **Paired native-MuJoCo protocol of the pilot:** backlash scene, raw accelerometer, observation
  noise, initial joint noise ±0.02 rad, seeds 0/1/2, zero head offsets, the seven commands, 20 s
  trials scored over 5–20 s, no safety override. A trial aborts on a nonfinite state or tilt > 45°.
- **Four arms (84 trials):**
  - historical B2;
  - untrained converted control (the pilot's `initial.onnx`);
  - local pilot candidate (the starting point);
  - **GPU candidate:** the final accepted checkpoint, exported to ONNX with action parity ≤ 1e-5 on
    100 probes.
- **No checkpoint selection after evaluation.**
- **Per command:** falls, mean velocity, axis tracking RMSE, signed axis mean, cross/turn translation
  RMS, stationary fraction, rest displacement and foot lift. The pass rules are unchanged from the
  pilot.

## Success criteria (fixed now)

1. **Hypothesis-specific:** forward, backward and left signed axis mean ≥ 50% of the request in 3/3
   seeds each, **and** both turns' horizontal translation RMS ≤ 0.03 m/s in 3/3 seeds each.
2. **No added falls** compared with the untrained control and the pilot candidate.
3. **Pilot improvement:** all seven commands pass (3/3), and normalized moving-axis RMSE ≤ 90% of the
   untrained control.

- **H is supported** if 1 and 2 hold.
- **The pilot passes** if 1–3 hold.
- Even a pass is one training seed: it does not approve robust locomotion or expressive training. It
  would justify preregistering the long robustness gate.

## Visible output

Seven 20 s comparison clips (seed 0): historical B2 | local pilot candidate | GPU candidate, labelled
with command, measured velocity and PASS/FAIL, plus a combined video. Raw artifacts stay in ignored
local storage; small reports and hashes are committed.
