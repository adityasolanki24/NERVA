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

## Defaults

Since 2026-10-02 `scenario.run` and the evaluators default to **profile v2 with affect Model B** (policy
capabilities set the scene, style handling and head envelope). Use `--profile legacy` (affect A) to
reproduce anything recorded earlier.

## Consolidation suite (v2, B2) [measured]

| scenario | v2 + A | v2 + B | legacy |
|---|---|---|---|
| default (vision) | 5/5 on all four | 5/5 on all four | 5/5 on all four |
| two-person memory | 5/5 on all four; memory off 0/5 blame/remember | 5/5 on all four | 5/5; off 0/5 |
| together | avoids A 5/5, engages B 5/5; off 0/5, 4/5 | 5/5, 5/5 | 1/5, 3/5 |
| default, simulated detector, 6 seeds | 6/6, no falls | 6/6, no falls | 1/6 fell |

Results: `results_consolidation/`.

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

| default scenario, frames + **Model B** | B2, vision | 5/5 on all four |
| two-person, grounded + frames + **Model B** | B2, vision, default head limits | **fell in 5/5 seeds** (withdraw: head yaw −0.8) |
| same, **B2 head envelope** (pitch ≥ −0.2, \|yaw\| ≤ 0.4) | B2, vision | no falls; 5/5 on all four |
| default scenario, Model A, B2 head envelope | B2, vision | 5/5 on all four |
| default scenario, simulated detector, Model A | B2, seeds 0–5 | default limits: 1/6 fell (seed 3, pre-existing); B2 envelope: 0/6 |

**For B2 runs use `--head-pitch-down -0.2 --head-yaw-max 0.4`** (measured standing-safe envelope,
development log 2026-10-02). The defaults keep the earlier, recorded behaviour.

`evaluate.py` / `evaluate_memory.py` options: `--appraisal legacy|frames`, `--affect A|B`,
`--head-pitch-down`, `--head-yaw-max`; `evaluate_memory.py` also: `--learning legacy|grounded` (refactor stage D), `--backlash
--neutral-style` (B2), `--no-ablation`. Results: `results_memory_b2_legacy/`,
`results_memory_b2_grounded/`, `results_memory_grounded_s1/`.

Results folders: `results_eval_*` (default scenario, per policy) and `results_memory/`.
