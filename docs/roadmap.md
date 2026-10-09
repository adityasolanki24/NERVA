# NERVA roadmap

The long-term research trajectory. It is **not** a task list. Items are implemented one justified step
at a time, each through an experiment we understand. The status column is the only part that changes
often.

## Research stages

| # | Stage | Status |
|---|---|---|
| 1 | Expressive locomotion parameter study (gait-clock style) | done: `experiments/expressive_locomotion/` |
| 2 | Speed-matched evaluation of that style | done (RQ1b): differences persist at 0.045 m/s; coupled single axis |
| 3 | Expressive style representation: scalar vs small style vector | done: style vector e = (tempo, step height, torso pitch) |
| 4 | PAD → expressive style | hand-designed v0 (`nerva/behaviour/pad_style.py`); not validated |
| 5 | Context-aware appraisal | v1 contextual and v2 memory-based (`nerva/affect/appraisal.py`); goals and self state still implicit |
| 6 | Physical experience (near-falls, slips, saturation) feeding appraisal and PAD | near-fall from tilt feeds appraisal; self state (tilt, angular speed, stability risk) modulates appraisal frames; outcomes feed grounded memory |
| 7 | Multiple locomotion skills | not started |
| 8 | Failure-aware behaviour ("I fell" vs "I am becoming unstable") | deterministic stop on instability (safety module, not fall prevention); B2 head-offset envelope measured; behaviour-level adaptation not started |
| 9 | Style-conditioned RL policy π(s, c, e) | S1–S6 trained and evaluated: tempo and torso pitch work, step height does not; S5/S6 collapsed to standing; next: curriculum design (separate motor-learning thread) |
| 10 | Expressive reference motion (animation, mocap, acted, designed) | not started |
| 11 | Imitation learning: "move like this" + RL "while staying stable" | via Placo references per style (R1) in S1 |
| 12 | Social perception (presence, distance, approach velocity, identity) | colour + depth vision, multi-person tracking, clothing-colour identity; faces not yet |
| 13 | Human proximity and interaction behaviour | approach, inspect, watch, freeze, retreat, withdraw; utility selection; retreat works with B2 |
| 14 | Appraisal relative to explicit goals and self state | planned: refactor stage E |
| 15 | Human evaluation of perceived expression | not started; the only route to emotional labels |
| 16 | Sim-to-real | not started |
| 17 | Physical NERVA hardware | not started; sensor/payload direction proposed only |
| 18 | Expressive head/neck/body mechanisms | not started |
| 19 | Learned or probabilistic semantic appraisal, compared against rules | not started; never in the safety loop (refactor stage I) |
| 20 | Longer-term persistence / relationships | memory M1–M4 implemented; grounding in outcomes planned (stage D) |
| 21 | Realistic rendering and perception development | Isaac Sim kinematic replay with robot-eye camera; people not animated |

## Architecture refactor (current priority)

Defined in `architecture.md` §2. Each stage: tests, Ruff, a short simulation check, a development-log
entry, docs, commit.

| Stage | Content | Status |
|---|---|---|
| A | Audit + docs describe the live code | done 2026-10-02 |
| B | Typed contracts + adapters (`SelfState`, `GoalState`, `OutcomeSignal`, `OutcomeHypothesis`, `AppraisalFrame`, `ActionTendencyState`) | done 2026-10-02 |
| C | Behaviour consumes action tendencies, not emotion labels | done 2026-10-02 |
| D | Outcome-grounded memory learning (legacy path kept as baseline) | done 2026-10-02; together-scenario "engages B" fails (diagnosis: global fear gate) |
| E | Self state, goals, appraisal frames over explicit hypotheses | done 2026-10-02 |
| F | Compact world model / scene graph, modular sensor evidence | done 2026-10-02; not yet read by behaviour |
| G | Model B (appraisal → PAD directly) vs Model A | done 2026-10-02; see RQ8 |
| H | Learned event prototypes vs hand-coded events (simulation) | done 2026-10-02; RQ9 falsified for this design |
| I | Optional typed semantic cues from speech | deferred |

**Consolidation (2026-10-02), done:**
- policy capability layer;
- target-conditioned arbitration;
- world model as the query source;
- RiskEstimate vs OutcomeSignal;
- suite re-run and defaults switched to v2 + Model B.

**Next, in this order:**
1. ~~Fix Model B's PAD saturation~~ Done 2026-10-02: Model B v2 (bounded attractor), default.
   **Two-sided dominance:** completed 2026-10-09. Bv4 combines same-source context facets
   by a fade-weighted mean: all preregistered criteria pass, 15 runs without falls;
   saturation V/A/D 2.02/0/0.37%. Bv4 + reaction margin + touch context is now the
   default; Bv2 and failed Bv3 experiments remain reproducible. See
   `context_facet_experiment.md` and the development log.
2. Return to expressive movement: π(s, c, e) with continuous expressive conditioning / reference motion
   (curriculum design: robust locomotion first, then expressive objectives;
   `locomotion_curriculum.md`: B2 gate failed pure-turn translation, 2026-10-09;
   short/long turn diagnostics completed, aggregate inconclusive; long zero-command
   controls migrate in COM/foot region; motor audit and candidate design complete;
   original subset fails 0/7, alignment insufficient; smooth/joint-limited repair
   passes 6/7 and geometric correction passes both turns, giving seven verified
   candidate targets; opt-in motor environment passes 80-step CPU smoke;
   32-transition PPO/warm-start plumbing completed; saved-checkpoint audit finds
   normalization amplification; next preregister a normalization-control smoke,
   then a neutral policy comparison/gate
   before expressive training), evaluated on monotonic
   control, cross-talk and task performance.
3. Then: learned vs utility action selection; optional semantic cues; Isaac person animation and
   detectors; human evaluation.

## Planned comparisons

- **Expressive control:** parameter modulation (stage 1) vs style-conditioned policy (9) vs
  reference/imitation-based control (10–11).
- **Affect models:** Model A (appraisal → discrete emotions → PAD) vs Model B (appraisal → PAD directly),
  on fixed traces and fixed scenarios with criteria stated in advance (stage G).
- **Memory learning signal:** emotion-label learning (legacy) vs outcome-grounded learning (stage D).
- **Event vocabulary:** hand-coded events vs learned prototypes (stage H).

## Constraints that hold at every stage

- **Physical safety stays deterministic and separate.** Physical experiences may *also* feed affect, a
  slower pathway; the two are never merged.
- **Timescales stay separated:** motor/PD, then policy and state estimation, then behaviour, then
  event-driven appraisal, with PAD slowest.
- **No PAD → joint angles.** Affect reaches the body only through behaviour.
- **No LLM or foundation model in motor control or safety.**
