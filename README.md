# NERVA

**Neural Expressive Robot with Variable Affect.** A research project on an expressive, emotionally reactive biped: can a robot's internal affective state, driven by what it perceives and remembers, change *what* it does and *how* it moves, while its locomotion stays stable?

NERVA is built **on top of** [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini), which supplies the robot, the MuJoCo simulation and the locomotion baseline. Open Duck code is used unmodified from its upstream repositories; this repository contains only NERVA's own work. The internal state is an engineered representation for studying expressive behaviour, not a claim that the robot has emotions.

## Architecture

```
PERCEPTION → CONTEXT-AWARE APPRAISAL → AFFECT (emotions → PAD) → BEHAVIOUR (what + how)
          → LEARNED LOCOMOTION POLICY π(s, c, e) → DETERMINISTIC SAFETY / CONTROL
```
PAD never drives joints directly, and no LLM is in the control or safety loop. Details: `docs/architecture.md`.

## Status (October 2026, simulation only)

| Area | State |
|---|---|
| Baseline | Open Duck policy reproduced bit for bit in a NERVA harness (`docs/open_duck_baseline.md`) |
| Expressive locomotion on the pretrained policy | RQ1/RQ1b/RQ1c done (`experiments/expressive_locomotion/`) |
| Style-conditioned policy (PPO + imitation, cloud GPUs) | S1 expresses tempo and torso pitch, not step height; S2/S3 variants evaluated (`docs/style_policy_design.md`) |
| Affect | Affect model v0.2: EMA-inspired appraisal → emotions (incl. interest) → persistent PAD (`docs/affect_model.md`) |
| Reactive behaviour | Scene with a person and a ball, simulated head-camera perception, contextual appraisal, emotion-modulated action selection (`docs/reactive_behaviour_design.md`, `experiments/reactive/`) |
| Memory | Design proposal: evolving, brain-inspired, hardware-bounded (`docs/memory_design.md`) |
| Human evaluation | Not started; no behaviour is claimed to *look* emotional |

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

```
nerva/                  the NERVA package
  interfaces.py           data types passed between layers
  open_duck_sim.py        headless Open Duck sim (only module importing Open Duck); style vector, head offsets, scene extension
  world.py                scene extension: person, ball, robot-eye camera
  perception.py           simulated head-camera detector and tracker
  appraisal.py            appraisal v0 (table) and v1 (contextual)
  affect.py               affect model v0.2 (Model A)
  behaviour.py            PAD → style vector
  reactive_behaviour.py   behaviour modes and their controllers (v1 rules)
  action_selection.py     emotion-modulated action selection (v2)
  gait_metrics.py         gait measurements
  reference_validation.py checks for generated reference gaits
  training/               style-conditioned training env (JAX/MJX) and training script
cloud/                  Google Cloud runner: capped, self-deleting VMs; job scripts; training lockfile
experiments/            expressive_locomotion, style_policy, reactive, affect_prototype, demo_video
docs/                   architecture, designs, research questions, roadmap, development log
tests/                  pytest suite
scripts/                tooling around the baseline
```

## Tests

```bash
.venv/Scripts/python -m pip install -e "<NERVA>[dev,experiments]"
.venv/Scripts/python -m pytest <NERVA>
```
