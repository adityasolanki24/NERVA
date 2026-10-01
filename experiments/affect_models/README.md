# Affect models: Model A vs Model B (RQ8)

Are discrete emotion categories necessary for the behaviours NERVA is tested on? Model A
(`nerva/affect/emotions.py`) goes appraisal → emotion instances (labels) → PAD anchors → persistent PAD,
and its labels are translated to action tendencies (`nerva/affect/tendencies.py`). Model B
(`nerva/affect/model_b.py`) maps appraisal features to PAD with a linear leaky system and to tendencies
with continuous functions, with no labels. Both satisfy `nerva.interfaces.AffectSystem`, so behaviour and
grounded memory run unchanged with either.

- `compare_traces.py`: both models on the scripted affect-prototype timeline; criteria (bounded, recovery,
  valence-sign agreement) in its docstring. Results: `results/traces.json`.
- Scenario comparisons use `experiments/reactive/evaluate.py --affect B` and
  `evaluate_memory.py --affect B --learning grounded`.

Neither model is a model of human emotion; the comparison only asks which one produces the tested
behaviours. Results and their preregistration: `docs/development_log.md`.
