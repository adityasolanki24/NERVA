# NERVA architecture

This document has two parts. **Part 1** is what the code does today (verified against the live code).
**Part 2** is the target architecture and the staged migration toward it. Nothing in Part 2 is
implemented unless its stage is marked done in §2.3.

## 1. Current implementation

§1.1–1.4 describe the code as audited on 2026-10-02, before the refactor; §1.5 lists what the refactor
changed; §1.6 the consolidation. Since 2026-10-02 the scenario default is the v2 profile with affect
Model B; `profile="legacy"` runs the audited path.

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
| Behaviour (`UtilityBehaviour`, `ReactiveBehaviour`) | PAD, `ActionTendencyState` (since stage C; before: emotion intensities by label), tracks, per-track novelty, per-track remembered threat | label leak removed in stage C |
| Entity memory `learn()` | **emotion labels** (`fear` → threat; `joy`/`hope`/`interest` − `fear`/`distress` → warmth), arousal, "negative surprise" | **abstraction leak**, and self-reinforcing (1.3) |
| Place memory `learn()` | **emotion labels**, arousal | same leak |
| Episodic memory | emotion labels (stored), relevance, arousal, expectedness; sleep replay calls entity `learn()` with the stored labels | same leak; replay re-learns from emotions |
| Locomotion policy | velocity command, style vector (S-policies only), head offsets | PAD never reaches joints [fact] |

### 1.3 Known architectural problems (motivate Part 2)

1. **Model A is not replaceable.** `architecture.md` used to say "nothing outside `emotions.py` may
   depend on emotion labels". That was false: behaviour, entity memory, place memory and episodic
   replay all consumed the labels. *Behaviour fixed in stage C* (it reads `ActionTendencyState`;
   `nerva/affect/tendencies.py` is Model A's only label translation). Memory: stage D.
2. **Memory learns from its own emotional output.** A remembered threat makes the next sighting
   appraise as threatening, which elicits fear, which entity memory learns as more threat. The
   development log recorded this ("A's fear partly re-generates itself", 2026-10-01). There is no
   grounded outcome (no contact, no loss of balance) behind the association. *Stage D adds a grounded
   path* (`memory_learning="grounded"`: entity/place memory learn only from `OutcomeSignal`s from
   `nerva/world/outcomes.py`); the legacy path remains the default for reproducibility.
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

### 1.5 After the refactor (stages B–H, 2026-10-02)

| Problem (§1.3) | Status | How |
|---|---|---|
| 1. Model A not replaceable | **fixed** | behaviour reads `ActionTendencyState`; Model A translates labels in `affect/tendencies.py` only; Model B (`affect/model_b.py`) runs behaviour and grounded memory without labels |
| 2. memory learns from its own emotions | **alternative path** | `memory_learning="grounded"`: entity/place memory learn only from `OutcomeSignal`s (`world/outcomes.py`); sleep replay restores but adds no evidence. Legacy remains the default |
| 3. no explicit self state / goals | **added** | `world/self_state.py`, `behaviour/goals.py`; used by appraisal frames and the safety supervisor |
| 4. likelihood without referent | **fixed in the frames path** | `appraisal_mode="frames"`: every appraisal is an `AppraisalFrame` over an explicit `OutcomeHypothesis`; unmapped events are errors |
| 5. perception emits interpretations | **unchanged** | legacy events (`person_approaching_rapidly`, …) remain the baseline vocabulary; outcomes and the world model work from measurements |
| 6. no world model | **added** | `world/model.py`: self, people/objects (track ≠ entity ID), places, seven relations with confidence and provenance; built every frame, not yet read by behaviour |

Also added: a deterministic safety supervisor (`nerva/safety.py`, stop on instability, no affect inputs),
and identity from modular sensor evidence (`EntityMemory.resolve_evidence`).

**Selecting the new paths** in `experiments/reactive/scenario.run` (and `evaluate*.py`):
`memory_learning="grounded"`, `appraisal_mode="frames"`, `affect_model="B"`, `head_pitch_down`.

**Measured outcomes of the refactor:**
- With defaults, all regression traces are byte-identical to the pre-refactor code.
- The new paths pass most preregistered criteria. Negative results: grounded memory fails "engages B" in
  the together scenario; Model B made the robot fall in the two-person scenario (withdraw posture); learned
  event prototypes predict outcomes worse than hand-coded events.
- Found during the audit: B2 falls with some head offsets (pitch −0.35 with yaw), including one
  pre-existing fall.
- Details: `development_log.md`, 2026-10-02 entries.

### 1.6 Consolidation (profile "v2", 2026-10-02)

- **Policy capabilities.** Behaviour takes its head envelope and walking head limit from the policy's
  measured `PolicyCapabilities` (`nerva/sim/capabilities.py`), and the scenario takes the training scene
  and style handling from it. B2: pitch (−0.2, 0.6), yaw (−0.4, 0.4).
- **Target-conditioned tendencies.** Affect models attribute tendencies to the track an appraisal is
  about (`tendencies_for`). Behaviour gates approach to *k* by avoidance directed at *k*, and
  watch/retreat by the focal person's avoidance.
- **World model as query source.** It is updated right after identity binding. Appraisal identity, the
  tracks behaviour sees, remembered threat per entity and touch attribution are all queries on it.
- **Risk vs outcome.** `RiskEstimate` (`collision_risk`: an estimated near miss) is separate from
  `OutcomeSignal` (`stability_loss`, `contact_impact`, `benign_contact`: what actually happened).
  Grounded memory keeps `risk` and `adverse` apart; threat = max(adverse, risk).

`profile="v2"` turns all of this on, together with grounded memory and appraisal frames. **Since the
consolidation suite (2026-10-02), v2 with affect Model B is the default.** `profile="legacy"` (affect A)
is byte-identical to the pre-refactor code and reproduces every earlier result.

Model B's PAD saturation (dominance > 0.9 for 31% of the default scenario) is fixed by **Model B v2**
(`Bv2`, bounded attractor; the default since 2026-10-02): 0% saturation in all scenarios, with all
behavioural criteria kept. Open: dominance is mostly positive, because the appraiser gives anything in
view a constant controllability of 0.8.

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
| A | Audit; docs describe the live code; abstraction leaks documented | done 2026-10-02 |
| B | Typed contracts: `SelfState`, `GoalState`, `OutcomeSignal`, `OutcomeHypothesis`, `AppraisalFrame`, `ActionTendencyState` + adapters | done 2026-10-02 |
| C | Behaviour consumes `ActionTendencyState`; Model A produces it; labels for logging only | done 2026-10-02 |
| D | Outcome-grounded memory learning path; legacy path kept as a baseline | done 2026-10-02 (grounded passes the two-person criteria; fails "engages B" in the together scenario, see log) |
| E | Explicit self state, goals and appraisal frames | done 2026-10-02 (frames path; nominal context reproduces legacy numbers) |
| F | Compact world model / scene graph + modular sensor evidence | done 2026-10-02 (not yet consumed by behaviour) |
| G | Model B (appraisal → PAD directly), compared with Model A | done 2026-10-02 (passes default scenario; fell in two-person scenario; see log) |
| H | Learned event prototypes (simulation experiment), compared with hand-coded events | done 2026-10-02 (RQ9 falsified for this design) |
| I | Optional semantic cues from speech | deferred until a concrete experiment exists |

## 3. Safety rule (non-negotiable)

Physical safety never depends on appraisal, affect or any learned or probabilistic high-level model.
Fall detection, torque and joint limits, motor speed limits and emergency stop stay deterministic and
sit below behaviour.

**Today [fact]:** force and control ranges come from the MuJoCo model, and the 5.24 rad/s target
rate limit from the upstream control loop. `nerva/safety.py` (stage B–H refactor) is a deterministic
stop-on-instability override between behaviour and the policy: zero velocity and neutral head at tilt ≥
25° or stability risk ≥ 0.75. It takes no affect, tendency or memory input. It is a stop rule, not fall
prevention: in the recorded B2 falls it neither caused nor prevented the fall. Hardware still needs an
emergency stop, motor torque/temperature limits and fall-triggered motor disable.

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
