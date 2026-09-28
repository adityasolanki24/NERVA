# NERVA roadmap

This is the long-term research trajectory. It is **not** a task list. Items are implemented one justified step at a time, each through an experiment we understand. The status column is the only part that changes often.

| # | Stage | Status |
|---|---|---|
| 1 | Expressive locomotion parameter study (gait-clock style) | done: `experiments/expressive_locomotion/` |
| 2 | Speed-matched evaluation of that style | done (RQ1b): differences persist at 0.045 m/s; coupled single axis |
| 3 | Expressive style representation: scalar vs small style vector, justified by measurement and expressive-motion research | next decision |
| 4 | PAD → expressive style | not started; deliberately *after* 3 |
| 5 | Context-aware appraisal: appraisal = f(perception, goals, self-state, expectations, history, available actions) | v0 is a fixed lookup table (`nerva/appraisal.py`) |
| 6 | Physical experience (near-falls, slips, saturation) feeding appraisal and PAD | not started |
| 7 | Multiple locomotion skills | not started |
| 8 | Failure-aware behaviour ("I fell" vs "I am becoming unstable") | not started |
| 9 | Style-conditioned RL policy π(s, c, e) | not started; needs Linux + NVIDIA GPU |
| 10 | Expressive reference motion (animation, mocap, acted, designed) | not started |
| 11 | Imitation learning: "move like this" + RL "while staying stable" | not started |
| 12 | Social perception (presence, distance, approach velocity, orientation, interaction state) | not started |
| 13 | Human proximity and interaction behaviour | not started |
| 14 | Appraisal based on context and goals (extends 5) | not started |
| 15 | Human evaluation of perceived expression | not started; the only route to emotional labels |
| 16 | Sim-to-real | not started |
| 17 | Physical NERVA hardware | not started |
| 18 | Expressive head/neck/body mechanisms | not started |
| 19 | Learned or probabilistic semantic appraisal, compared against rules | not started; never in the safety loop |
| 20 | Longer-term behavioural persistence / personality | not started; only after affect-conditioned motion works |

## Planned comparisons

- **Expressive control:** parameter modulation (stage 1) vs a style-conditioned learned policy (9) vs reference/imitation-based control (10–11).
- **Affect models:** Model A, appraisal → discrete emotions → PAD (current v0.x), vs Model B, appraisal → PAD directly, vs others. See `affect_model.md`.

## Constraints that hold at every stage

- **Physical safety stays deterministic and separate:** fall detection, torque and joint limits, emergency stop. Physical experiences may *also* feed affect, which is a slower pathway, but the two are never merged.
- **Timescales stay separated:** motor/PD fastest, then the locomotion policy and state estimation, then behaviour, then event-driven appraisal, with PAD slowest.
- **No PAD → joint angles.** Affect reaches the body only through behaviour: a functional command plus an expressive condition.
- **No LLM in motor control or safety.**
