# NERVA development log

Newest entry first. Each entry records what was done, what was actually run, and what is still unverified.

---

## 2026-09-28 — Phase 2: understanding the baseline

Wrote `docs/open_duck_baseline.md` from the upstream source code, the compiled MuJoCo model and the ONNX file. Nothing upstream was modified.

### What was checked directly (not just read)
- **Effective actuator parameters come from the compiled model:** kp 13.37, ±3.23 N·m, damping 0.56. `xmls/joints_properties.xml` (kp 17.8) is not included by any model and is stale.
- **The ONNX graph** has `onnx` installed to a scratch folder only, not the project venv. The network is 101→512→256→128→28 with swish activation, normalisation built into the graph, and `tanh` on the means. That matches the Berkeley Humanoid PPO config the runner uses.
- **Reference-motion pickle:** 240 gaits on a 6×4×10 grid (vx, vy, ωz), 40 signals each, degree-15 polynomials, period 0.54 s (27 policy steps), nearest-neighbour lookup.
- **Reference-motion speeds:** the reference gaits' own forward speed matches their command (0.155 m/s for the 0.148 gait), so the reference is not why the policy under-tracks forward.

### Finding: accelerometer offset mismatch
- `joystick.py:502` (training) intends to add 1.3 to accelerometer x, but uses `.at[0].set()` without assigning the result. In JAX that's a no-op, so training used the raw value.
- The hardware runtime also uses the raw value (`tare_x()` is disabled).
- Only `mujoco_infer.py:74` adds +1.3.
- Added `--raw-accel` to `scripts/check_open_duck_baseline.py` to cancel it (upstream untouched). Result, 20 s per command:
  - forward 0.097 → **0.112 m/s**
  - lateral 0.093 → 0.102
  - backward and turn unchanged
  - no falls
- Conclusion: a real but minor effect. **NERVA experiments will use the raw accelerometer.**

### Command tracking, current understanding
- **Lateral:** explained. The reward's 0.1 m/s dead-band plus a reference grid limited to ±0.111 m/s means about 0.10 m/s earns full reward.
- **Forward vs turning:** a likely explanation is that the absolute `tracking_sigma = 0.01` enforces m/s errors much more weakly than rad/s errors. That's inferred, not tested.
- **Backward:** about 21%, unexplained.

### For Phase 5 (noted, not acted on)
The phase clock `[cos φ, sin φ]` is the only trace of the reference motion at runtime. Its rate is already adjustable (`P`/`;` in sim; LB and a per-robot offset on hardware). That's a zero-retraining candidate for option A (gait-parameter modulation), but its effect on the gait hasn't been measured yet.

---

## 2026-09-28 — Phase 1: Open Duck Mini v2 baseline running in MuJoCo

### Machine
- Windows 11 Home, Intel i7-1355U (10 cores), 15.6 GB RAM, **Intel Iris Xe only (no NVIDIA GPU)**.
- Python 3.12 and 3.13 installed (`py -0`). 37 GB free on C:.
- Windows long-path support enabled by the user (`LongPathsEnabled = 1`, verified).

**Consequence:** Open Duck *training* uses JAX + MJX on CUDA (`jax[cuda12]` in the Playground `pyproject.toml`). CUDA JAX does not exist for native Windows and this machine has no NVIDIA GPU, so training cannot run here. Inference (running a trained ONNX policy in CPU MuJoCo) works fine. Training will need a Linux + NVIDIA machine (cloud, lab PC or Colab). That decision is deferred until we actually need to train.

### Upstream code used (unmodified)
Cloned to `C:\Users\24adi\dev\open_duck\`, outside OneDrive (OneDrive sync interferes with git repos, LFS meshes and venvs).

| Repo | Branch (default) | Commit |
|---|---|---|
| apirrone/Open_Duck_Playground | main | b9be205 (2025-08-05) |
| apirrone/Open_Duck_Mini | v2 | b23317a (2026-01-31) |
| apirrone/Open_Duck_Mini_Runtime | v2 | 376de65 (2026-09-08) |
| apirrone/Open_Duck_reference_motion_generator | main | 3d7bc6f (2025-04-03) |

- Policy: `Open_Duck_Mini/BEST_WALK_ONNX_2.onnx` (uploaded 2025-04-02). This is the policy the hardware runtime README tells users to deploy.
- Model: `Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml` (Open Duck Mini v2 MJCF).
- `git status` in all upstream repos is clean.

### Environment
Venv: `C:\Users\24adi\dev\open_duck\.venv` (Python 3.12). Exact versions: `env/open_duck_inference.lock.txt`.

Setup steps, reproducible:
```bash
py -3.12 -m venv C:/Users/24adi/dev/open_duck/.venv
C:/Users/24adi/dev/open_duck/.venv/Scripts/python -m pip install -r env/open_duck_inference.lock.txt
C:/Users/24adi/dev/open_duck/.venv/Scripts/python -m pip install -e C:/Users/24adi/dev/open_duck/Open_Duck_Playground --no-deps
```

Decisions and why:
1. **Did not use `uv sync`** (the upstream-recommended installer). The upstream deps include `jax[cuda12]`, `tensorflow` and `tf2onnx`, which are for training and ONNX export only, and the CUDA wheel cannot install on Windows. Installed only what the inference path imports (checked file-by-file): `playground` (DeepMind MuJoCo Playground), `mujoco`, `mujoco-mjx`, `jax` (CPU), `onnxruntime`.
2. **Pinned to the era of the Duck code**, not the newest packages. The upstream `pyproject.toml` says `playground>=0.0.3` with no lockfile, so pip resolves today's 0.2.0 (2026-03), a year newer than the Duck env code. Pinned instead to `playground==0.0.4` (2025-03-07, when `base.py` was last changed), `mujoco==3.3.0` (2025-02-27, before the policy was uploaded), `jax==0.5.3`, `onnxruntime==1.21.0`. **Not pinned to that era:** `numpy` (2.5.3), `brax`, `flax`, `orbax-checkpoint`. They are transitive deps, and the inference path works with them. Revisit if physics or behaviour discrepancies appear.
3. **Editable install of Open_Duck_Playground with `--no-deps`.** Equivalent to what `uv run` does. Without it, `mujoco_infer.py` fails with `ModuleNotFoundError: No module named 'playground'`. The two similarly named packages don't collide: the Duck repo provides import name `playground`, and DeepMind's distribution `playground` provides import name `mujoco_playground`.
4. On first import, `mujoco_playground` 0.0.4 git-clones the whole of `mujoco_menagerie` (about 1.7 GB, commit 14ceccf) into `site-packages/mujoco_playground/external_deps/`. It takes several minutes and happens once per venv. Open Duck doesn't use menagerie models; the clone is just a side effect of the import.

### Problems hit (and causes)
- `pip install playground` failed with `OSError: No such file or directory` on a very long path. Cause: `orbax-checkpoint` ships test fixtures deeper than Windows' 260-character `MAX_PATH`. Fixed properly by the user enabling long paths. It was not worked around in the final environment.
- `ModuleNotFoundError: playground` when launching the upstream script: see decision 3.
- `cp -r src dest/` put the copied menagerie contents one level too high because `dest/` did not exist yet. Corrected, and the commit was verified.

### What ran
1. **GUI (upstream, unmodified):**
   ```bash
   cd C:/Users/24adi/dev/open_duck/Open_Duck_Playground
   ../.venv/Scripts/python playground/open_duck_mini_v2/mujoco_infer.py -o ../Open_Duck_Mini/BEST_WALK_ONNX_2.onnx
   ```
   The MuJoCo viewer opened and rendered the duck standing on flat ground (screenshot checked). Keys in `key_callback` (`mujoco_infer.py`): arrows for forward/back/lateral, `Q`/`E` to turn, `H` to toggle head-control mode, `P`/`;` to raise/lower the gait phase frequency. These are GLFW keycodes 81/69/72/80/59. The upstream comments say "a" and "m" because the author uses an AZERTY keyboard; on QWERTY the keys are Q and ;. A keypress latches the command; it doesn't reset on release.

2. **Headless check** `scripts/check_open_duck_baseline.py`. It calls the upstream `MjInfer.run()` loop unchanged, with the viewer replaced by a recorder. 20 s simulated per command, with steady-state velocity measured over the last 10 s in the robot's heading frame:

   | Command | Fell | Max tilt | Base height | Measured |
   |---|---|---|---|---|
   | zero | no | 6.3° | 0.150–0.167 m | 0.000 m/s |
   | fwd 0.15 m/s | no | 5.0° | 0.150–0.169 m | fwd +0.097, lat +0.017 m/s |
   | back 0.15 m/s | no | 5.1° | 0.150–0.169 m | fwd −0.032 m/s |
   | left 0.2 m/s | no | 4.2° | 0.150–0.168 m | lat +0.093 m/s |
   | turn 1.0 rad/s | no | 4.0° | 0.150–0.169 m | yaw +0.81 rad/s |

   Two runs gave identical numbers, so the simulation is deterministic.

### Observations (measured, not interpreted)
- Stable in every tested command: no falls, tilt ≤ 6.3°.
- **Command tracking is incomplete:** about 65 % of the forward command, about 21 % of the backward command, about 47 % of the lateral command and about 81 % of the yaw command. We don't yet know whether this is expected for this policy, a sim-version effect, or an observation mismatch. Worth understanding in Phase 2 before any style experiment uses velocity as a metric.
- Policy loop: `sim_dt = 0.002 s`, `decimation = 10`, so the policy runs at 50 Hz. Observation dimension 101, action dimension 14 (ONNX I/O `obs[1,101] → continuous_actions[1,14]`).

### Follow-up: "I didn't see it move"
On the first GUI launch the user saw the duck standing still. Cause: the upstream script starts with a **zero command** (`self.commands = [0]*7`), so the policy stands and sways slightly until a key press reaches the viewer window. The window must have focus: click it, then press ↑.

To rule out other explanations, I ran a diagnostic (the upstream `MjInfer.run()` with the real viewer and `commands[0] = 0.15` preset):
- The duck walked **1.44 m in 20 s wall-clock** in the GUI. The user confirmed visually that it walks.
- **The GUI runs at about 0.75× real time on this laptop.** The upstream loop calls `viewer.sync()` after every 2 ms physics step, and its sleep only compensates when a step is faster than real time. This affects viewing speed only, not the physics. Headless runs are unaffected.
- The GUI velocity matched the headless measurement (about 0.097 m/s simulated), so the viewer and headless runs behave the same.

### Not yet verified
- Whether results would differ under the newest `playground`/`mujoco` versions.
