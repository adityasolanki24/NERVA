# NERVA architecture

This document has two parts. **Part 1** is what the code does today (verified against the live code).
**Part 2** is the target architecture and the staged migration toward it. Nothing in Part 2 is
implemented unless its stage is marked done in §2.3.

## 1. Current implementation (as of the 2026-10-02 audit)

### 1.1 The loop that actually runs

The whole loop is one function, `experiments/reactive/scenario.py::run`. Every 0.1 s:

```
robot-eye camera (MuJoCo render)  ──▶ PERCEPTION  nerva/perception/vision.py + tracker.py
                                       tracks (with track IDs) + events ("person_approaching_rapidly", …)
simulated touch, body tilt         ──▶ extra events ("touch_gentle", "near_fall")
                                            │
             ┌──────────────────────────────┼──────────────────────────────┐
             ▼                              ▼                              ▼
   IDENTITY BINDING              APPRAISAL (v1 contextual /           BEHAVIOUR reads tracks,
   scenario.IdentityBinder ───▶  v2 memory-based)                     per-track novelty and
   track ID → entity record      nerva/affect/appraisal.py            remembered threat directly
             │                              │ AppraisalState
             │                              ▼
             │                   AFFECT, Model A  nerva/affect/emotions.py
             │                   appraisal → emotion instances (labels) → PAD anchors → persistent PAD
             │                              │ PAD + emotion intensities by label
             │        ┌─────────────────────┼─────────────────────┐
             ▼        ▼                     ▼                     ▼
   ENTITY MEMORY learns from      PLACE MEMORY learns     BEHAVIOUR (utility selection)
   the elicited emotion labels    from emotion labels     nerva/behaviour/selection.py
   nerva/memory/entity.py         nerva/memory/spatial.py  WHAT: mode + target; HOW: style from PAD
   EPISODIC MEMORY encodes them; sleep replay re-learns     │ BehaviourCommand + head offsets
   them into entity memory        nerva/memory/episodic.py  ▼
                                                  LOCOMOTION POLICY (B2 / S1 …) in nerva/sim/open_duck.py
                                                            ▼
                                                  MuJoCo physics (upstream model, unmodified)
```

So the cognitive side is **already a graph**, not the strict chain the older docs described.

### 1.2 Actual dependencies between modules

| Consumer | Reads | Notes |
|---|---|---|
| Appraisal v2 (`MemoryAppraiser`) | events, tracks, entity records (familiarity, threat, warmth, episodes) | memory → appraisal link |
| Affect Model A | `AppraisalState` | the only part that should know emotion labels |
| Behaviour (`UtilityBehaviour`, `ReactiveBehaviour`) | PAD, **emotion intensities by label** (`fear`, `interest`, `hope`, `joy`, `surprise`, `distress`), tracks, per-track novelty, per-track remembered threat | **abstraction leak**, see 1.3 |
| Entity memory `learn()` | **emotion labels** (`fear` → threat; `joy`/`hope`/`interest` − `fear`/`distress` → warmth), arousal, "negative surprise" | **abstraction leak**, and self-reinforcing (1.3) |
| Place memory `learn()` | **emotion labels**, arousal | same leak |
| Episodic memory | emotion labels (stored), relevance, arousal, expectedness; sleep replay calls entity `learn()` with the stored labels | same leak; replay re-learns from emotions |
| Locomotion policy | velocity command, style vector (S-policies only), head offsets | PAD never reaches joints [fact] |

### 1.3 Known architectural problems (motivate Part 2)

1. **Model A is not replaceable.** `architecture.md` used to say "nothing outside `emotions.py` may
   depend on emotion labels". That was false: behaviour, entity memory, place memory and episodic
   replay all consume the labels. A Model B without labels could not drive behaviour or memory.
2. **Memory learns from its own emotional output.** A remembered threat makes the next sighting
   appraise as threatening, which elicits fear, which entity memory learns as more threat. The
   development log recorded this ("A's fear partly re-generates itself", 2026-10-01). There is no
   grounded outcome (no contact, no loss of balance) behind the association.
3. **No explicit self state or goals.** Appraisal uses goals only implicitly (inside rules), and the
   robot's own condition only through the `near_fall` event.
4. **`likelihood` has no referent.** `AppraisalState.likelihood = 0.7` does not say *which* outcome is
   0.7 likely.
5. **Perception emits interpretations.** Events such as `person_approaching_rapidly` mix a measurement
   (approach speed) with an interpretation (threshold "rapid").
6. **No world model.** Tracks, entity identities and places are separate structures; relations
   (who is near whom, who touched the robot) are implicit in scenario code.

### 1.4 What exists, by area

| Area | State [fact unless marked] | Where |
|---|---|---|
| Robot, physics, low-level control | Open Duck Mini v2, upstream, unmodified | external checkout (`open_duck_baseline.md`) |
| Locomotion policies | upstream `BEST_WALK_ONNX_2`; NERVA-trained B0/B1/B2 (neutral) and S1–S6 (style-conditioned, mostly negative results) | `nerva/sim/open_duck.py`, `nerva/training/`, `style_policy_design.md` |
| Perception | colour + depth vision on the robot-eye camera, multi-object tracker with track IDs and ego-motion-compensated approach speed; a ground-truth "simulated detector" alternative | `nerva/perception/` |
| Appraisal | v0 table, v1 contextual, v2 memory-based | `nerva/affect/appraisal.py` |
| Affect | Model A v0.2 (incl. "interest") | `nerva/affect/emotions.py`, `affect_model.md` |
| Memory | entity (M1), episodic + consolidation (M2–M3), place (M4) | `nerva/memory/`, `memory_design.md` |
| Behaviour | utility arbitration (current) and rule-based v1; PAD → style vector | `nerva/behaviour/` |
| Safety | no NERVA safety module; limits come from the MuJoCo model (force/control ranges) and the upstream control loop (target rate limit) | see §3 |
| Rendering | MuJoCo renders; Isaac Sim kinematic replay with robot-eye camera | `experiments/reactive/render.py`, `experiments/isaac/` |

## 2. Target architecture [design]

### 2.1 Diagram

```
                         WORLD → SENSORS
                                   ▼
             PERCEPTION FRONT-ENDS (vision / touch / audio / proprioception)
             report measurements, not interpretations
                                   ▼
             WORLD MODEL  entities • places • self • relations, each with confidence + provenance
                 │                                   │
                 ▼                                   ▼
             MEMORY (episodic, entity, spatial)   EVENT LEARNING (prediction error, segmentation,
             learns from grounded OUTCOMES          prototypes; research path)
                 │                                   │
                 └───────────────┬───────────────────┘
   SELF STATE ─────────────────▶ APPRAISAL ◀───────────────── GOALS
   SEMANTIC CUES (optional,       appraisal frames over explicit
   slow, typed) ───────────────▶  outcome hypotheses
                                  │                 │
                                  ▼                 ▼
                           AFFECT (PAD)       ACTION TENDENCIES
                           Model A or B       approach / explore / avoid / orient / freeze / withdraw
                                  └────────┬────────┘
                                           ▼
                                BEHAVIOUR: WHAT + HOW
                                           ▼
                                MOTION POLICY π(s, c, e)
                                           ▼
                                ROBOT → physical/task OUTCOMES → memory / appraisal

   DETERMINISTIC SAFETY / RECOVERY: independent, faster override; affect never gates it.
```

The cognitive side is a stateful loop; the motor side stays hierarchical (behaviour request →
motion policy → actuation), with safety able to override it.

### 2.2 Interface rules (target)

- Behaviour consumes `PADState` + `ActionTendencyState` + world/self/goal context, never emotion labels.
- Any affect model (A, B, …) produces `PADState` and `ActionTendencyState`. Labels exist inside Model A
  (and in logs) only.
- Memory's primary learning signal is `OutcomeSignal` (measurable physical/task/social outcomes);
  emotions may be stored as history but do not teach threat or warmth.
- Every appraisal likelihood belongs to an explicit `OutcomeHypothesis`.
- Track ID (temporary perceptual continuity) ≠ entity ID (persistent identity).
- An optional language model may produce typed `SemanticCue`s; it never outputs emotions,
  appraisal values, actions or joint commands, and never enters safety or motor control.

### 2.3 Migration stages

| Stage | Content | Status |
|---|---|---|
| A | Audit; docs describe the live code; abstraction leaks documented | done (this document) |
| B | Typed contracts: `SelfState`, `GoalState`, `OutcomeSignal`, `OutcomeHypothesis`, `AppraisalFrame`, `ActionTendencyState` + adapters | planned |
| C | Behaviour consumes `ActionTendencyState`; Model A produces it; labels for logging only | planned |
| D | Outcome-grounded memory learning path; legacy path kept as a baseline | planned |
| E | Explicit self state, goals and appraisal frames | planned |
| F | Compact world model / scene graph + modular sensor evidence | planned |
| G | Model B (appraisal → PAD directly), compared with Model A | planned |
| H | Learned event prototypes (simulation experiment), compared with hand-coded events | planned |
| I | Optional semantic cues from speech | deferred until a concrete experiment exists |

## 3. Safety rule (non-negotiable)

Physical safety never depends on appraisal, affect or any learned or probabilistic high-level model.
Fall detection, torque and joint limits, motor speed limits and emergency stop stay deterministic and
sit below behaviour.

**Today [fact]:** force and control ranges come from the MuJoCo model, and the 5.24 rad/s target
rate limit from the upstream control loop. The reactive scenario only *detects* near-falls (tilt >
20°) as an event for appraisal. A dedicated NERVA safety module is required before hardware.

**Dual pathway for physical events.** A near-fall triggers deterministic recovery or stop (safety
pathway: fast, rule-based, never learned) and may also be appraised (affective pathway: slower,
behavioural). The two are never merged; affect never delays, suppresses or gates safety.

## 4. Boundary between Open Duck and NERVA

- **Open Duck owns** the robot model, physics, actuator model, trained upstream walking policy and its
  observation/action conventions.
- **NERVA owns** everything above the locomotion command, plus its own trained policies and the
  experiments.
- Only `nerva/sim/open_duck.py` (simulation) and `nerva/training/` (JAX/MJX training env) import Open
  Duck. `nerva/training/style_joystick.py` subclasses upstream's env and is tested to reproduce it with
  one neutral style. The simulator's control loop is tested to reproduce upstream `mujoco_infer.py`.
- Upstream code is never edited; differences are overridden from NERVA code and documented.

## 5. Timescales

Fastest first: motor/PD control → locomotion policy and state estimation → behaviour → event-driven
appraisal → PAD, the slowest. PAD is never updated or used like a motor controller.

## 6. How claims are labelled

1. **Established robotics/control theory:** PD control, PPO, domain randomisation.
2. **Established affective-computing theory:** appraisal theory and EMA (Marsella & Gratch), PAD
   (Mehrabian & Russell), action tendencies (Frijda).
3. **NERVA design assumptions:** PAD as the output of EMA-inspired appraisal (EMA does not use PAD); the
   appraisal subset and ranges; all coefficients; the discrete-emotion bridge of Model A; the
   tendency mapping; the outcome definitions.
4. **Measured in this project:** only what `development_log.md` records as measured.
5. **Hypotheses:** everything else, including that any behaviour or style reads as an emotion to people.
6. **Future research directions:** `roadmap.md`. These are not claims.
