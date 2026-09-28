# NERVA architecture

## Pipeline (long-term)

```
WORLD → SENSORS → PERCEPTION ─PerceptionState─▶ APPRAISAL ─AppraisalState─▶ AFFECT (PAD, persistent)
      ─PADState─▶ BEHAVIOUR ─BehaviourCommand + ExpressiveStyle─▶ LOCOMOTION POLICY
      → SAFETY + LOW-LEVEL CONTROL → ROBOT → (feedback to sensors)
```

The types between layers are defined in `nerva/interfaces.py`. A layer may only consume the type produced by the layer directly before it. For example, the locomotion policy never sees events or PAD, only a velocity command and an expressive style. This keeps each layer replaceable and testable on its own.

## What exists today

| Layer | Status | Where |
|---|---|---|
| Robot, simulation, low-level control | **Open Duck Mini v2, upstream, unmodified** | `C:\Users\24adi\dev\open_duck\` (see `open_duck_baseline.md`) |
| Locomotion policy | **Open Duck `BEST_WALK_ONNX_2.onnx`**, 50 Hz, velocity-commanded | upstream |
| Behaviour → locomotion adapter | **`OpenDuckSim.set_behaviour`**: clips velocities to the trained range, style → phase-clock rate | `nerva/open_duck_sim.py`, `nerva/style.py` |
| ExpressiveStyle | **used**: one scalar → gait-clock rate (method A) | `nerva/style.py` |
| Gait measurement | pure-NumPy metrics, unit-tested | `nerva/gait_metrics.py` |
| Behaviour selection | interface only (`BehaviourCommand`) | `nerva/interfaces.py` |
| Affect (PAD dynamics) | interface only (`PADState`) | `nerva/interfaces.py` |
| Appraisal | interface only (`AppraisalState`) | `nerva/interfaces.py` |
| Perception | interface only (`PerceptionState`, `Event`) | `nerva/interfaces.py` |

Build order: locomotion style experiment (Phase 5) first, then a simulation-only appraisal → PAD prototype with synthetic events. No LLM or foundation model is part of the plan.

## Boundary between Open Duck and NERVA

- **Open Duck owns** the robot model, the physics, the actuator model, the trained walking policy and its observation/action conventions.
- **NERVA owns** everything above the locomotion command: style, behaviour, affect, appraisal, perception, and the experiments and evaluation.
- Only one module, `nerva/open_duck_sim.py`, imports Open Duck and MuJoCo, and only when a simulator object is created. Everything else in `nerva` (interfaces, style mapping, metrics) is dependency-free apart from NumPy for metrics. The simulator's control loop is tested to reproduce upstream `mujoco_infer.py` exactly.
- Upstream code is never edited. Where we need different behaviour (for example the accelerometer offset in `open_duck_baseline.md` §10), we override it from NERVA code and document why.

## Safety rule (non-negotiable)

Physical safety never depends on appraisal, affect or any learned or probabilistic high-level model. The following stay deterministic and sit below the behaviour layer:
- fall detection
- torque and joint limits
- motor speed limits
- emergency stop

Today these are provided by the MuJoCo model (force and control ranges) and by the upstream control loop (the 5.24 rad/s target rate limit). A dedicated NERVA safety layer will be added before anything runs on hardware.

## How claims are labelled

Every design document distinguishes between:
1. **Established robotics/control theory:** PD control, PPO, domain randomisation.
2. **Established affective-computing theory:** appraisal theory and EMA (Marsella & Gratch), PAD (Mehrabian & Russell).
3. **NERVA design assumptions:**
   - using PAD as the output of an EMA-inspired appraisal (EMA itself does not use PAD)
   - the chosen subset and ranges of appraisal variables
   - any appraisal→PAD mapping coefficients
   - a single style scalar for locomotion
4. **Empirically validated in this project:** only what `development_log.md` records as measured.
5. **Hypotheses:** everything else, including that any style setting will read as a particular emotion to people.
