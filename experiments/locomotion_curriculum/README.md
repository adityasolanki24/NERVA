# Robust locomotion before expressive objectives

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
