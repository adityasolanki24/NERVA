# Local neutral PPO and parameter-restore smoke

Preregistered 2026-10-09 before implementation or training. Repository baseline
`9c31255`, existing seven-target admission and 80-step dynamics smoke passed.
No paid/cloud work, ONNX export, efficacy comparison or expressive objectives.

Hypothesis: the opt-in neutral environment can collect batched transitions,
perform finite PPO updates, serialize policy/value/normalization parameters,
restore identical action behavior and continue updating from those parameters.
This is a plumbing smoke; 32 transitions cannot establish useful locomotion.

Pinned local Brax source inspection: `restore_checkpoint_path` restores the
normalizer, policy and (with `restore_value_fn=True`) value weights. Adam state,
RNG, environment state and env-step counter are initialized anew. Consequently
call this parameter warm start, not exact optimizer/trajectory continuation.
The saved artifact and report must explicitly record that distinction.

Candidate-only autoreset correction: retain the existing batched EpisodeWrapper
and reuse first data/observations on done, but also restore the environment-owned
motor info to the corresponding initial snapshot. Preserve advancing RNG and
wrapper-owned episode accounting/truncation. Include command, reference,
rest/phase clock, action/IMU history and contact state. Historical wrappers and
environments stay unchanged. Test mixed done/not-done rows and RNG/accounting
preservation before training. Reusing a sampled initial pose/command each episode
is the existing autoreset convention, not fresh domain randomization.

Exact configuration: native CPU only, flat terrain, candidate rewards/reference
contract unchanged, no domain randomization; default sensor/action-delay noise
and push configuration retained. Seed 7 for both fresh and warm-start stages.
Each stage exactly 16 environment transitions: 2 environments, batch size 2,
unroll length 4, one minibatch, one gradient update per batch, two training
batches. Action repeat 1, episode length 8, one reported epoch, no evaluator
rollouts (`run_evals=False`, 2 eval-env placeholder). Policy and value networks
(32,32), swish, tanh-normal, policy reads state (101), value reads privileged
state (212), 14 actions. Normalize observations, learning rate 0.0001,
entropy 0.0001, discount 0.9, reward scaling 1, clipping 0.3, GAE 0.95,
value-loss coefficient 0.5, normalize advantage, no LR schedule or gradient clip.
Other Brax options use the inspected defaults, captured in the public protocol.
Fresh and warm stages have equal budgets; they are not a motor-quality control.

Save fresh and final warm parameters through the inspected Brax PPO checkpoint
API into a new ignored directory. Require a candidate contract sidecar matching
network shape/config, reference hashes, normalization and package source hashes
before restore. A checkpoint from B2/S policies is not admitted by this smoke.
No automatic overwrite, dependency installation or checkpoint deletion.

All-required success criteria: initial/final parameter trees and reported losses
finite; both stages' callbacks report exactly 16 steps; policy and value weights
each change by >1e-12 in both stages; fresh normalization count >0 and increases
in warm stage; disk roundtrip preserves every parameter/normalizer leaf bitwise
including shapes/dtypes; warm initial callback exactly equals the saved fresh
parameters. Deterministic and same-key sampled actions before/after checkpoint
restore agree within 1e-6 and are finite/in [-1,1]. Probe five fixed observation
dictionaries (zeros, +0.1, -0.1, linspace [-0.2,0.2], saved normalizer mean),
key 19. These are inference probes, not physical-state evaluations.

One attempt, one worker process, 900 s parent wall watchdog including imports,
compilation, both stages, saving and comparisons; max 32 optimization transitions.
Stop on mismatch, nonfinite values, API error, timeout or CPU-device mismatch.
Write progress/results as each stage finishes; preserve checkpoint/log/partial
results on any failure. Report failure without relaxing criteria or escalating
to paid hardware. Public outputs contain only small measurements, configurations
and artifact/source hashes, never personal paths or credentials.

Afterward run complete tests including slow legacy equality checks and Ruff,
update authoritative docs/log and commit/push verified work. Passing permits
planning an equal-step neutral motor comparison and prospective long motor gate;
it does not replace B2 or authorize paid training. A true full-state resume API
remains a separate implementation if the research needs exact continuation.
