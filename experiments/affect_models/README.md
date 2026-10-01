# Affect models: Model A vs Model B (RQ8)

Are discrete emotion categories necessary for the behaviours NERVA is tested on? Model A
(`nerva/affect/emotions.py`) goes appraisal → emotion instances (labels) → PAD anchors → persistent PAD,
and its labels are translated to action tendencies (`nerva/affect/tendencies.py`). Model B
(`nerva/affect/model_b.py`) maps appraisal features to PAD with a linear leaky system and to tendencies
with continuous functions, with no labels. Both satisfy `nerva.interfaces.AffectSystem`, so behaviour and
grounded memory run unchanged with either.

- `compare_traces.py`: both models on the scripted affect-prototype timeline; criteria (bounded, recovery,
  valence-sign agreement) in its docstring. Results: `results/traces.json`.
- Scenario comparisons use `experiments/reactive/evaluate.py --affect A|B|Bv2` and `evaluate_memory.py`.
- **Model B saturation and Model B v2:**
  - `diagnose_model_b.py` attributes old B's PAD drive to features and appraisal origins;
  - `saturation.py` runs the fixed-trace and convergence criteria (`results/saturation.json`);
  - `summarise_scenarios.py` applies the scenario criteria to `results_scenarios/`.

Neither model is a model of human emotion; the comparison only asks which one produces the tested
behaviours. Results and their preregistration: `docs/development_log.md`.
