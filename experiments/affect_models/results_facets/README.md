# Bv4 same-source context facets (2026-10-09)

All preregistered criteria pass. v2 now defaults to Bv4 with reaction margin
and ongoing-touch context. Pooled saturation V/A/D: 2.02/0/0.37%; no falls in
15 local B2/vision runs. Brief negative valence saturation remains.

Preregistration: `docs/context_facet_experiment.md`, commit `ca30916`.
Candidate: `33febb1`. Missing rapid detections fail under summariser `2fbf1fe`.
Checkpoint hash and exact configuration: `protocol.json`. Historical comparator:
`../results_touch/`. No thresholds, W or gains were tuned after evaluation.

Reproduce from the repository root in the pinned Open Duck inference environment
(set POLICY to the local B2 final checkpoint; do not commit that path):

```sh
python experiments/affect_models/saturation.py --out experiments/affect_models/results_facets
python experiments/reactive/evaluate.py --policy "$POLICY" --seeds 5 --perception vision --affect Bv4 --margin-controllability --touch-context --out experiments/affect_models/results_facets/default_Bv4
python experiments/reactive/evaluate_memory.py --policy "$POLICY" --seeds 5 --affect Bv4 --margin-controllability --touch-context --no-ablation --learning grounded --out experiments/affect_models/results_facets/memory_Bv4
python experiments/reactive/evaluate_memory.py --policy "$POLICY" --seeds 5 --affect Bv4 --margin-controllability --touch-context --no-ablation --learning grounded --together --out experiments/affect_models/results_facets/together_Bv4
python experiments/affect_models/summarise_scenarios.py --root results_facets --models Bv4
```

The explicit flags keep this protocol independent of future defaults. No paid
work or new policy training was part of this experiment.
