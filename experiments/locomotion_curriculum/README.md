# Robust locomotion before expressive objectives

`gate.py` implements the fixed neutral B2 readiness gate, preregistered in
`docs/b2_robustness_gate.md`. It runs 115 local motor trials, with affect absent
and deterministic safety observed separately. No training or cloud launch.

`results_b2/` is the completed negative result: in-place-turn translation
exceeds the preregistered limit in 10/10 turning trials, despite no falls in
115 trials and passing push recovery, transitions and head tolerance.

Curriculum design and next-phase decision: `docs/locomotion_curriculum.md`.
S1–S6 are completed historical experiments and are not rerun here.
