# NERVA architecture

## Canonical architecture

This is the reference architecture for NERVA unless evidence gives a reason to change it:

```
WORLD → SENSORS
  → PERCEPTION                 "What happened?"  (observable facts only)
  → CONTEXT-AWARE APPRAISAL    "What does this mean relative to my goals, expectations,
                                capabilities and history?"
  → PERSISTENT AFFECT (PAD)    valence, arousal, dominance; slow, continuous, recovers to baseline
  → BEHAVIOUR                  WHAT to do (functional command) + HOW to do it (expressive condition)
  → LEARNED MOTION POLICY      a_t = π(s_t, c_t, e_t): body state, command, expressive condition
  → DETERMINISTIC SAFETY + LOW-LEVEL CONTROL
  → ROBOT → new perception / physical experience ↺
```

The types passed between layers are defined in `nerva/interfaces.py`: `PerceptionState` → `AppraisalState` → `PADState` → `BehaviourCommand` (functional command + `ExpressiveStyle`). Any affect implementation must satisfy the `AffectSystem` protocol. A layer only consumes the type produced by the layer directly before it. For example, the locomotion policy never sees events or PAD, only a command and an expressive condition, and **PAD never maps directly to joint angles**.

**Affect model v0.x is one implementation, not an architectural layer.** The current prototype (`nerva/affect.py`, `CategoricalAffectModel`, "Model A") goes appraisal → discrete emotion labels (fear, joy, …) → PAD anchors → persistent PAD. The discrete-label step belongs to that model only. A future "Model B" (appraisal → PAD directly), or any other model, can replace it behind the same `AffectSystem` interface. Nothing outside `nerva/affect.py` may depend on emotion labels.

**Appraisal v0 is a lookup table, not the target design.** Long term, appraisal = f(perception, goals, self/physical state, expectations, history, available actions). The same event, e.g. "person approaching rapidly", must be able to mean different things in different contexts. The v0 table sits behind `appraise(Event) → AppraisalState` so it can be replaced without touching affect or behaviour.

**Timescales** (fastest first): motor/PD control → locomotion policy and state estimation → behaviour → event-driven appraisal → PAD, the slowest. PAD is never updated or used like a motor controller.

Long-term stages and comparisons: `roadmap.md`.

## What exists today

| Layer | Status | Where |
|---|---|---|
| Robot, simulation, low-level control | **Open Duck Mini v2, upstream, unmodified** | `C:\Users\24adi\dev\open_duck\` (see `open_duck_baseline.md`) |
| Locomotion policy | **Open Duck `BEST_WALK_ONNX_2.onnx`**, 50 Hz, velocity-commanded | upstream |
| Behaviour → locomotion adapter | **`OpenDuckSim.set_behaviour`**: clips velocities to the trained range, style → phase-clock rate | `nerva/open_duck_sim.py`, `nerva/style.py` |
| ExpressiveStyle | **used**: one scalar → gait-clock rate (method A) | `nerva/style.py` |
| Gait measurement | pure-NumPy metrics, unit-tested | `nerva/gait_metrics.py` |
| Behaviour selection | interface only (`BehaviourCommand`) | `nerva/interfaces.py` |
| Affect (PAD dynamics) | **v0.1 prototype (Model A), simulation only**: simplified EMA-inspired emotion rules → ALMA PAD anchors → per-dimension decaying pull and return to baseline; not connected to movement | `nerva/affect.py`, `docs/affect_model.md` |
| Appraisal | **v0 prototype**: fixed EMA-variable appraisals for 5 synthetic events | `nerva/appraisal.py` |
| Perception | interface only (`PerceptionState`, `Event`) | `nerva/interfaces.py` |

No LLM or foundation model is part of the plan, and none will ever be in the motor-control or safety path.

## Boundary between Open Duck and NERVA

- **Open Duck owns** the robot model, the physics, the actuator model, the trained walking policy and its observation/action conventions.
- **NERVA owns** everything above the locomotion command: style, behaviour, affect, appraisal, perception, and the experiments and evaluation.
- Only one module, `nerva/open_duck_sim.py`, imports Open Duck and MuJoCo, and only when a simulator object is created. Everything else in `nerva` (interfaces, style mapping, metrics, appraisal, affect) needs only NumPy. The simulator's control loop is tested to reproduce upstream `mujoco_infer.py` exactly.
- Upstream code is never edited. Where we need different behaviour (for example the accelerometer offset in `open_duck_baseline.md` §10), we override it from NERVA code and document why.

## Safety rule (non-negotiable)

Physical safety never depends on appraisal, affect or any learned or probabilistic high-level model. The following stay deterministic and sit below the behaviour layer:
- fall detection
- torque and joint limits
- motor speed limits
- emergency stop

Today these are provided by the MuJoCo model (force and control ranges) and by the upstream control loop (the 5.24 rad/s target rate limit). A dedicated NERVA safety layer will be added before anything runs on hardware.

**Dual pathway for physical events.** A near-fall must trigger *deterministic* recovery immediately, which is the safety pathway: fast, rule-based, never learned. It *may also* produce an appraisal, for example high relevance, undesirable, low controllability. That changes PAD and can make later behaviour more conservative, which is the affective pathway: slower and behavioural. The two pathways are never merged; affect never gates safety.

## How claims are labelled

Every design document distinguishes between:
1. **Established robotics/control theory:** PD control, PPO, domain randomisation.
2. **Established affective-computing theory:** appraisal theory and EMA (Marsella & Gratch), PAD (Mehrabian & Russell).
3. **NERVA design assumptions:**
   - using PAD as the output of an EMA-inspired appraisal (EMA itself does not use PAD)
   - the chosen subset and ranges of appraisal variables
   - any appraisal→PAD mapping coefficients
   - a single style scalar for locomotion
   - the discrete-emotion bridge of affect model v0.x
4. **Measured in this project:** only what `development_log.md` records as measured.
5. **Hypotheses:** everything else, including that any style setting will read as a particular emotion to people.
6. **Future research directions:** `roadmap.md`. These are not claims.
