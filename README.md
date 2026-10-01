# NERVA

**Neural Expressive Robot with Variable Affect.** A research project on an expressive, emotionally reactive biped: can a robot's internal affective state, driven by what it perceives and remembers, change *what* it does and *how* it moves, while its locomotion stays stable?

NERVA is built **on top of** [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini), which supplies the robot, the MuJoCo simulation and the locomotion baseline. Open Duck code is used unmodified from its upstream repositories; this repository contains only NERVA's own work. The internal state is an engineered representation for studying expressive behaviour, not a claim that the robot has emotions.

## Architecture

Today the loop is a graph rather than a chain: perception feeds appraisal, behaviour and identity
binding; appraisal reads memory; memory and behaviour read the affect model's output.

```
camera images, touch, body tilt ─▶ PERCEPTION (vision + multi-object tracker) ─▶ tracks + events
        ─▶ APPRAISAL (contextual, memory-based) ─▶ AFFECT (Model A: emotions → persistent PAD)
        ─▶ BEHAVIOUR (utility selection: WHAT + HOW) ─▶ LOCOMOTION POLICY π(s, c, e) ─▶ MuJoCo
   MEMORY (entity, episodic, place) is read by appraisal and behaviour, and learns from events
```
PAD never drives joints directly, and no LLM is in the control or safety loop. A staged refactor
toward a target architecture (world model, self state, goals, grounded outcomes, action tendencies,
a replaceable affect model) is under way. Current vs target, and the known problems that motivate
it: `docs/architecture.md`.

## Status (October 2026, simulation only)

| Area | State |
|---|---|
| Baseline | Open Duck policy reproduced bit for bit in a NERVA harness (`docs/open_duck_baseline.md`) |
| Expressive locomotion on the pretrained policy | RQ1/RQ1b/RQ1c done (`experiments/expressive_locomotion/`); RQ1c's head-posture labels are inverted (positive head_pitch tilts the face up, measured 2026-10-02) |
| Style-conditioned policies (PPO + imitation, cloud GPUs) | S1 expresses tempo and torso pitch, not step height (preregistered verdict: fail); S2–S6 variants did not solve it; S5/S6 collapsed to standing (`docs/style_policy_design.md`) |
| Backward walking | B2 (neutral, feet-height cost) walks backward at −0.113 m/s commanded −0.15, in the backlash training scene |
| Perception | colour + depth vision from the robot's head camera; multi-person tracking with track IDs (`nerva/perception/`) |
| Affect | Model A v0.2: EMA-inspired appraisal → emotions (incl. interest) → persistent PAD (`docs/affect_model.md`) |
| Reactive behaviour | emotion-modulated utility selection; B2 + vision + backlash: curiosity, fear, habituation, safety 5/5 seeds each (`experiments/reactive/`) |
| Memory | M1–M4: entity, episodic with consolidation, place memory; two-person ablation memory 5/5 vs none 0/5, measured with S1 in the plain scene (`docs/memory_design.md`) |
| Isaac Sim | kinematic replay of MuJoCo runs with realistic rendering and the robot's-eye camera (`experiments/isaac/`); not used for physics |
| Human evaluation | not started; no behaviour is claimed to *look* emotional |

The detailed record of every run, measurement and correction is `docs/development_log.md`. Long-term trajectory: `docs/roadmap.md`.

**Licence:** `LICENSE` is currently empty, so no licence has been chosen yet. That is the author's decision and still open.

## Setup (inference only)

Tested on Windows 11 with Python 3.12, CPU only. Training runs on Linux with an NVIDIA GPU (`cloud/`).

```bash
# 1. upstream code, unmodified, outside this repo
git clone https://github.com/apirrone/Open_Duck_Playground.git
git clone -b v2 https://github.com/apirrone/Open_Duck_Mini.git

# 2. environment (exact versions)
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r <NERVA>/env/open_duck_inference.lock.txt
.venv/Scripts/python -m pip install -e Open_Duck_Playground --no-deps

# 3. run the pretrained walking policy (click the window, then press the arrow keys)
cd Open_Duck_Playground
../.venv/Scripts/python playground/open_duck_mini_v2/mujoco_infer.py -o ../Open_Duck_Mini/BEST_WALK_ONNX_2.onnx
```

On Windows, enable long-path support first; a transitive dependency has paths longer than 260 characters. Set `OPEN_DUCK_ROOT` if the upstream repos are not in `<user-profile>/dev/open_duck`.

## Layout

The package has one folder per layer. The typed contracts between layers live in `nerva/interfaces.py`;
where layers still exchange other data (e.g. emotion labels), `docs/architecture.md` §1.2 lists it.

```
nerva/                      the NERVA package
  interfaces.py               typed contracts between modules (start here)
  sim/                        simulation (the only code that imports Open Duck)
    open_duck.py                headless Open Duck robot: policy loop, style vector, head offsets, scene extension
    world.py                    scene extension: people, ball, robot-eye camera
  perception/                 what is out there (measurements)
    tracker.py                  multi-object tracker; simulated (ground-truth) detector
    vision.py                   detection from the robot's camera images (colour + depth)
  world/                      the robot's current situation
    self_state.py               self state from IMU-type quantities (tilt, angular speed, stability risk)
    outcomes.py                 measured outcomes: near-collision estimate, loss of stability, benign touch
    model.py                    compact world model / scene graph: self, people, objects, places, relations
  affect/                     what it means and how it feels
    appraisal.py                appraisal: v0 table, v1 contextual, v2 memory-based
    frames.py                   appraisal frames over explicit outcome hypotheses, relative to goals and self state
    emotions.py                 Model A: appraisal → emotions → persistent PAD
    tendencies.py               Model A's emotions → action tendencies (the only label translation)
    model_b.py                  Model B: appraisal → PAD and tendencies directly, no emotion labels
  memory/                     what the robot remembers (legacy or outcome-grounded learning)
    entity.py                   people/objects: identity, familiarity, threat, warmth, trust / expected outcomes
    episodic.py                 significant events and outcomes, forgetting, sleep consolidation
    spatial.py                  places: familiarity and affect; exploration heading
  events/                     learned event discovery (research path)
    segmentation.py             prediction-error event boundaries
    prototypes.py               bounded online prototypes with outcome statistics
  behaviour/                  what to do and how (reads action tendencies, never emotion labels)
    selection.py                utility-based action selection (current)
    modes.py                    behaviour modes and their controllers (rule-based v1)
    goals.py                    explicit goal state
    pad_style.py                PAD → expressive style vector
    style.py                    style definitions (gait clock, S1 style vector)
  safety.py                   deterministic safety supervisor (independent override; no affect inputs)
  analysis/gait_metrics.py    gait measurements on simulation logs
  training/                   learning the locomotion policy (JAX/MJX; cloud only)
    style_joystick.py           style-conditioned training environment
    train_style.py              training script
    reference_validation.py     checks for generated reference gaits
experiments/                runnable studies, one folder each, with results and a README (see experiments/README.md)
cloud/                      Google Cloud runner: capped, self-deleting VMs and the job scripts (see cloud/README.md)
scripts/                    small tools around the Open Duck baseline (see scripts/README.md)
tests/                      pytest suite, same folders as nerva/
docs/                       architecture, design documents, research questions, roadmap, development log
```

**Where to start reading:** `docs/architecture.md` → `nerva/interfaces.py` →
`experiments/reactive/scenario.py` (the whole loop in one function) → the layer you care about.

## Tests

```bash
.venv/Scripts/python -m pip install -e "<NERVA>[dev,experiments]"
.venv/Scripts/python -m pytest <NERVA>
```
