# NERVA

**Neural Expressive Robot with Variable Affect.** A research project on expressive bipedal locomotion: can a robot's internal state change *how* it moves while its locomotion controller keeps it stable?

NERVA is built **on top of** [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini), which supplies the robot, the MuJoCo simulation and the locomotion baseline. Open Duck code is used unmodified from its upstream repositories; this repository contains only NERVA's own work.

## Status

| Phase | Goal | State |
|---|---|---|
| 1 | Run the Open Duck baseline in MuJoCo | done (see `docs/development_log.md`) |
| 2 | Document how the baseline works | done (see `docs/open_duck_baseline.md`) |
| 3–4 | NERVA package and layer interfaces | done (`nerva/interfaces.py`, `docs/architecture.md`) |
| 5 | First expressive-locomotion experiment | first result: `experiments/expressive_locomotion/README.md` |

Affect v0 exists as a simulation-only prototype: synthetic events → EMA-inspired appraisal → emotions → persistent PAD (`docs/affect_model.md`). It is **not** connected to the robot's movement yet.

## Setup (inference only)

Tested on Windows 11 with Python 3.12, CPU only. Training needs Linux and an NVIDIA GPU (upstream uses JAX/MJX on CUDA).

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

On Windows, enable long-path support first; a transitive dependency has paths longer than 260 characters.

Headless check (stability and velocity tracking for five commands; add `--raw-accel` to feed the policy the raw accelerometer, matching training and hardware):
```bash
.venv/Scripts/python <NERVA>/scripts/check_open_duck_baseline.py
```
Set `OPEN_DUCK_ROOT` if the upstream repos are not in `C:\Users\24adi\dev\open_duck`.

## Layout

```
docs/overview.md                 project intent
docs/development_log.md          what was done, what ran, what is unverified
docs/open_duck_baseline.md       how the Open Duck baseline works (obs, actions, control, training, sim vs hardware)
docs/papers/                     reference papers
env/open_duck_inference.lock.txt exact package versions
nerva/interfaces.py              data types passed between layers (no logic yet)
nerva/style.py                   style → gait-clock rate (Phase 5, method A)
nerva/open_duck_sim.py           headless Open Duck sim driven by BehaviourCommand (only module importing Open Duck)
nerva/gait_metrics.py            gait measurements (speed, cadence, stride, posture, effort, stability)
experiments/expressive_locomotion/ RQ1 protocol, results and write-up
nerva/appraisal.py               synthetic events → EMA appraisal variables (NERVA design values)
nerva/affect.py                  appraisal → emotions (EMA) → PAD (ALMA points) with dynamics
docs/affect_model.md             affect model design, sources, and open design questions
experiments/affect_prototype/    scripted event timeline → PAD plot
tests/                           pytest suite
docs/architecture.md             layers, Open Duck/NERVA boundary, safety rule
docs/research_questions.md       RQ1 (active) and later questions
scripts/                         NERVA tooling around the baseline
```

## Tests

```bash
.venv/Scripts/python -m pip install -e "<NERVA>[dev,experiments]"
.venv/Scripts/python -m pytest <NERVA>
```
