# Robust locomotion before expressive objectives

## Maintained neutral motor workflow (supported entry points)

| step | entry point | notes |
|---|---|---|
| train (GPU, capped) | `cloud/jobs/<experiment>.sh` → `cloud/jobs/_neutral_gpu.sh` → `python -m experiments.locomotion_curriculum.gpu_pilot --experiment NAME --out DIR` (named, preregistered experiments in `EXPERIMENTS`) | launch: `python cloud/launch.py launch --job <experiment> --hw l4 --max-minutes 45 --input-bundle cloud/inputs/<experiment>.txt --wait --cleanup-network` (requires explicit authorization) |
| smoke (CPU, tiny) | `python -m experiments.locomotion_curriculum.gpu_pilot --smoke` | checks the whole path; not a result |
| evaluate + videos | `python -m experiments.locomotion_curriculum.motor_compare --run RUN --out DIR [--render] [--protocol NAME]` | paired native-MuJoCo protocol (3 seeds); `--protocol` picks the preregistered arms and decision rule |
| readiness gate | `python -m experiments.locomotion_curriculum.neutral_gate --candidate NAME [--run RUN] --out DIR --raw-dir DIR` | `docs/neutral_motor_gate.md`: 160 trials, 5 seeds, steady/transitions/pushes, each without and with training-like action latency |
| left-turn report | `python -m experiments.locomotion_curriculum.turn_left_report --run RUN --out DIR` | descriptive (yaw, vx/vy, pivot, contacts, trajectory) |

**Current validated neutral candidate:** `experiments/cloud_runs/turn_translation-20261011-000711/candidate.onnx`
(SHA256 `610b1c59…`; local, ignored), the first to pass the neutral gate (`results_turn_translation/`). One
training seed, simulation only; no default or deployment change.

Library code lives in the `nerva` package:
- `nerva/training/b2_warm_start.py`: B2 conversion, frozen preprocessing, balanced resets, ONNX export.
- `nerva/training/neutral_joystick.py`: neutral environment and its opt-in variants: persistent command,
  gait-averaged tracking, base-origin reward velocity, tighter pure-turn translation tracking.
- `nerva/training/neutral_reference.py`: hash-verified references.
- `nerva/training/parameter_checkpoint.py`: checkpoints.
- `nerva/training/motor_artifacts.py`: JSON/archive/fingerprint/KL helpers.
- `nerva/analysis/motor_eval.py`: velocities, motor pass rules and push recovery.
- `nerva/sim/open_duck.py`: native simulator; `action_delay=True` adds training-matched latency.

`learning_support.py`, `gate.py` and `normalization_timing.py` re-export these for the completed runners.

## Completed experiments (frozen; reproduce at the recorded commit)

Runners still imported by other code stay here unchanged: `gate.py`, `normalization_*.py`,
`identity_*.py`, `neutral_ppo_smoke.py`, `neutral_learning.py`, `paired_motor.py`, `reference_subset.py`
and `record_reference.py`. Leaf runners nothing else imports are in `archive/` (byte-identical, with a
mapping). Every result directory, preregistration and raw artifact is preserved.

## History

`gate.py` implements the fixed neutral B2 readiness gate, preregistered in
`docs/b2_robustness_gate.md`. It runs 115 local motor trials, with affect absent
and deterministic safety observed separately. No training or cloud launch.

`results_b2/` is the completed negative result: in-place-turn translation
exceeds the preregistered limit in 10/10 turning trials, despite no falls in
115 trials and passing push recovery, transitions and head tolerance.

Curriculum design and next-phase decision: `docs/locomotion_curriculum.md`.
S1–S6 are completed historical experiments and are not rerun here.

The bounded learned-controller pilot is preregistered in
`docs/neutral_learning_pilot.md`. `neutral_learning.py` converts the retained
B2 walking checkpoint to 101/212 motor observations, freezes its preprocessing,
and performs balanced seven-command PPO updates with a 20-minute hard limit.
`paired_motor.py` compares historical B2, the converted policy before training,
and the final accepted candidate, then renders seven side-by-side videos.

Use the existing Open Duck environment; the exporter additionally requires
`pip install -r env/neutral_learning_extra.txt`. From the NERVA repository root:

```sh
python -m experiments.locomotion_curriculum.neutral_learning --corrected
python -m experiments.locomotion_curriculum.paired_motor --corrected
python -m experiments.locomotion_curriculum.paired_motor --corrected --render
```

Each command refuses to overwrite existing run artifacts. `--corrected` selects
the admitted attempt after the two preserved setup/integrity stops; it does not
change experimental criteria. The archived attempts remain in
`results_neutral_learning/` and `results_neutral_learning_retry/`. Checkpoints,
raw rollouts and videos stay under ignored `experiments/cloud_runs/`.
Success in this short pilot never substitutes for the long robustness gate.
