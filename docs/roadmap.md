# NERVA roadmap

This is the long-term research trajectory. It is **not** a task list. Items are implemented one justified step at a time, each through an experiment we understand. The status column is the only part that changes often.

| # | Stage | Status |
|---|---|---|
| 1 | Expressive locomotion parameter study (gait-clock style) | done: `experiments/expressive_locomotion/` |
| 2 | Speed-matched evaluation of that style | done (RQ1b): differences persist at 0.045 m/s; coupled single axis |
| 3 | Expressive style representation: scalar vs small style vector, justified by measurement and expressive-motion research | done: style vector e = (tempo, step height, torso pitch) |
| 4 | PAD → expressive style | hand-designed v0 (`nerva/behaviour/pad_style.py`); not validated |
| 5 | Context-aware appraisal: appraisal = f(perception, goals, self-state, expectations, history, available actions) | v1: proximity, approach speed, novelty/habituation, threat memory (`ContextualAppraiser`) |
| 6 | Physical experience (near-falls, slips, saturation) feeding appraisal and PAD | near-fall from tilt feeds appraisal in the reactive loop |
| 7 | Multiple locomotion skills | not started |
| 8 | Failure-aware behaviour ("I fell" vs "I am becoming unstable") | not started |
| 9 | Style-conditioned RL policy π(s, c, e) | S1 trained and evaluated (tempo, torso pitch work; step height does not); S2/S3/S4 variants |
| 10 | Expressive reference motion (animation, mocap, acted, designed) | not started |
| 11 | Imitation learning: "move like this" + RL "while staying stable" | via Placo references per style (R1) in S1 |
| 12 | Social perception (presence, distance, approach velocity, orientation, interaction state) | simulated detector (ground-truth positions); real vision next |
| 13 | Human proximity and interaction behaviour | approach, inspect, watch, freeze, retreat (`nerva/behaviour/selection.py`); retreat limited by weak backward walking |
| 14 | Appraisal based on context and goals (extends 5) | not started |
| 15 | Human evaluation of perceived expression | not started; the only route to emotional labels |
| 16 | Sim-to-real | not started |
| 17 | Physical NERVA hardware | not started |
| 18 | Expressive head/neck/body mechanisms | not started |
| 19 | Learned or probabilistic semantic appraisal, compared against rules | not started; never in the safety loop |
| 20 | Longer-term behavioural persistence / personality | memory design proposed (`memory_design.md`) |

## Planned comparisons

- **Expressive control:** parameter modulation (stage 1) vs a style-conditioned learned policy (9) vs reference/imitation-based control (10–11).
- **Affect models:** Model A, appraisal → discrete emotions → PAD (current v0.x), vs Model B, appraisal → PAD directly, vs others. See `affect_model.md`.

## Constraints that hold at every stage

- **Physical safety stays deterministic and separate:** fall detection, torque and joint limits, emergency stop. Physical experiences may *also* feed affect, which is a slower pathway, but the two are never merged.
- **Timescales stay separated:** motor/PD fastest, then the locomotion policy and state estimation, then behaviour, then event-driven appraisal, with PAD slowest.
- **No PAD → joint angles.** Affect reaches the body only through behaviour: a functional command plus an expressive condition.
- **No LLM in motor control or safety.**
