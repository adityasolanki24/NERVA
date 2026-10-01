# Reactive behaviour and memory

The robot in a scene with people and a ball. It perceives them through its head camera (colour + depth
vision, or a ground-truth "simulated detector"), appraises events in context, updates its affect,
remembers people, places and episodes, and chooses what to do by emotion-modulated action selection.
Only the people and the ball follow scripts. Design: `docs/reactive_behaviour_design.md`,
`docs/memory_design.md`; current vs target architecture: `docs/architecture.md`.

- `scenario.py`: the whole closed loop in one function (`run`), plus the scenarios:
  `default_scenario` (ball, friendly approach, lunge, return), `memory_scenario` (two people: A lunges,
  B pets the robot, both return), `together_scenario` (A and B return at the same time)
- `evaluate.py`: 5-seed evaluation of the default scenario against criteria stated in advance
  (curiosity, fear, habituation, safety)
- `evaluate_memory.py`: memory ON vs OFF on the two-person scenarios (criteria stated in advance)
- `render.py`: video with scene view, robot's-eye detections and live affect charts
  (cloud: `cloud/jobs/reactive_demo.sh`)

## Latest results [measured, see `docs/development_log.md`]

| Evaluation | Setup | Result |
|---|---|---|
| default scenario | B2, neutral style, backlash scene, vision, utility selector | curiosity 5/5, fear 5/5, habituation 5/5, safety 5/5 |
| default scenario (older) | S1, plain scene, simulated detector or vision | curiosity 5/5, habituation 5/5, safety 5/5, **fear 0/5** (retreat too slow) |
| two-person memory | **S1, plain scene**, vision | memory: 5/5 on all four criteria; no memory: 0/5 on "B not blamed" and "A remembered" |
| together scenario | S1, plain scene, vision, 3 seeds | avoids A: memory 3/3, no memory 0/3; "engages B" does not discriminate (3/3 both) |
| two-person memory | B2, backlash, neutral, vision | legacy: 5/5 on all four; no memory 0/5 on "B not blamed" and "A remembered" |
| two-person memory, **grounded** learning | S1 plain and B2 backlash | 5/5 on all four in both |
| together, **grounded** learning | S1, plain, 3 seeds | avoids A 3/3; **engages B 0/3** (legacy 3/3) |

`evaluate_memory.py` options: `--learning legacy|grounded` (refactor stage D), `--backlash
--neutral-style` (B2), `--no-ablation`. Results: `results_memory_b2_legacy/`,
`results_memory_b2_grounded/`, `results_memory_grounded_s1/`.

Results folders: `results_eval_*` (default scenario, per policy) and `results_memory/`.
