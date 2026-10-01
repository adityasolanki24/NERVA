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

The memory evaluations have **not** been repeated with B2 in the backlash scene (`evaluate_memory.py`
has no backlash/neutral-style option yet). The Isaac demo replays the memory scenario with B2, but that
is a single demo run, not an evaluation.

Results folders: `results_eval_*` (default scenario, per policy) and `results_memory/`.
