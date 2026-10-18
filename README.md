# NERVA

A simulated biped research platform built on [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) (MuJoCo).
It develops:
- a **stable locomotion controller** that can carry expressive modulation;
- a **robot-side loop**: perception → appraisal and memory → behaviour → locomotion, under a deterministic safety
  layer.

Open Duck code is used unmodified; this repository holds only NERVA's own work. Everything is simulation only. No
claim is made that the robot has emotions.

**Read first:** [docs/STATE.md](docs/STATE.md), which says what works, what does not, and which controller is
current. All docs: [docs/README.md](docs/README.md).

## What works today (simulation)

- **Validated neutral walking controller**, the turn-translation candidate. It passes the neutral long motor gate with
  and without action latency: 0 falls in 160 trials, every command 5/5, 80/80 pushes recovered
  (`experiments/locomotion_curriculum/results_turn_translation/`).
- **The full reactive loop**: vision, tracking, memory, affect Model Bv4, behaviour. It meets every preregistered
  scenario criterion (`experiments/reactive/`).
- **Expressive limits**, measured honestly:
  - style policies express tempo and torso pitch, not step height;
  - continuous posture on the neutral controller is weak (E1′: pitch 9–30%, crouch ignored).
- **Not yet:**
  - continuous speed control;
  - robustness to repeated pushes;
  - exteroceptive sensing in the deployed loop;
  - hardware;
  - human evaluation.

NERVA is the first embodiment of the separate **APM** project (`affective-processing-module`). APM runs its
experience-learning experiments on NERVA's frozen controller, read-only. Nothing in NERVA depends on APM.

## Setup (inference, CPU)

Tested on Windows 11 with Python 3.12. Training runs on Linux with an NVIDIA GPU (`cloud/`).

```bash
git clone https://github.com/apirrone/Open_Duck_Playground.git      # upstream, unmodified, outside this repo
git clone -b v2 https://github.com/apirrone/Open_Duck_Mini.git
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r <NERVA>/env/open_duck_inference.lock.txt
.venv/Scripts/python -m pip install -e Open_Duck_Playground --no-deps
.venv/Scripts/python -m pip install -e "<NERVA>[dev,experiments]"
.venv/Scripts/python -m pytest <NERVA>
```

On Windows, enable long-path support. Set `OPEN_DUCK_ROOT` if the upstream repositories are not in
`<user-profile>/dev/open_duck`.

## Demonstrations

| what | where |
|---|---|
| the upstream walking policy, keyboard-driven | `Open_Duck_Playground/playground/open_duck_mini_v2/mujoco_infer.py -o Open_Duck_Mini/BEST_WALK_ONNX_2.onnx` |
| the reactive loop in a scene with people and a ball | `experiments/reactive/` (README) |
| the neutral gate evaluation of the validated controller | `experiments/locomotion_curriculum/` (README, `neutral_gate.py`) |

## Layout

```
nerva/            the package: sim (only code importing Open Duck), perception, world, affect, memory, events,
                  behaviour, safety.py, analysis, training (JAX/MJX); typed contracts in interfaces.py
experiments/      one folder per study, with README and small results (experiments/README.md)
cloud/            capped, self-deleting Google Cloud runner (cloud/README.md)
docs/             STATE, ARCHITECTURE, ROADMAP, RESEARCH_HISTORY, development log, design references,
                  experiment records, archive
tests/            pytest, same folders as nerva/
```

**Licence:** `LICENSE` is currently empty. No licence has been chosen; that is the author's decision.
