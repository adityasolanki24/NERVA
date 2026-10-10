# Robust locomotion before expressive objectives

## Maintained neutral motor workflow (supported entry points)

| step | entry point | notes |
|---|---|---|
| train (GPU, capped) | `cloud/jobs/<experiment>.sh` → `cloud/jobs/_neutral_gpu.sh` → `python -m experiments.locomotion_curriculum.gpu_pilot --experiment NAME --out DIR` (named experiments in `EXPERIMENTS`) | launch: `python cloud/launch.py launch --job neutral_gpu_pilot --hw l4 --max-minutes 45 --input-bundle cloud/inputs/neutral_gpu_pilot.txt --wait --cleanup-network` |
| smoke (CPU, tiny) | `python -m experiments.locomotion_curriculum.gpu_pilot --smoke` | checks the whole path; not a result |
| evaluate + videos | `python -m experiments.locomotion_curriculum.motor_compare --run RUN --out DIR [--render] [--protocol NAME]` | paired native-MuJoCo protocol; `--protocol` picks the preregistered arms and decision rule |

Library code lives in the `nerva` package:
- `nerva/training/b2_warm_start.py`: B2 conversion, frozen preprocessing, balanced resets, ONNX export.
- `nerva/training/neutral_joystick.py`: neutral environment and persistent-command variant.
- `nerva/training/neutral_reference.py`: hash-verified references.
- `nerva/training/parameter_checkpoint.py`: checkpoints.
- `nerva/training/motor_artifacts.py`: JSON/archive/fingerprint/KL helpers.
- `nerva/analysis/motor_eval.py`: velocities and motor pass rules.

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
