# How the Open Duck Mini v2 baseline works

This document explains the upstream baseline NERVA builds on: what the robot model is, what the walking policy sees and outputs, how it was trained, and how simulation differs from hardware. Every claim points to source code or to a measurement we ran.

**Code versions.** Open_Duck_Playground `b9be205`, Open_Duck_Mini `b23317a` (for `BEST_WALK_ONNX_2.onnx`), Open_Duck_Mini_Runtime `376de65`, and DeepMind `playground` 0.0.4 (MuJoCo Playground). Paths below are relative to `Open_Duck_Playground/playground/` unless stated otherwise.

**How to read the labels:**
- **[code]**: read directly from source.
- **[measured]**: we ran it and observed it.
- **[inferred]**: our reasoning from the code, not tested.
- **[unknown]**: cannot be determined from what upstream published.

---

## 1. The big picture

```
                 50 Hz (every 10 physics steps)
 ┌──────────────────────────────────────────────────────┐
 │  build observation (101 numbers)                     │
 │        │                                             │
 │        ▼                                             │
 │  ONNX policy  π: R^101 → R^14   (MLP, deterministic) │
 │        │  action a ∈ [-1, 1]^14                      │
 │        ▼                                             │
 │  motor target = home_pose + 0.25 · a                 │
 │  then rate-limit to 5.24 rad/s                       │
 └────────┼─────────────────────────────────────────────┘
          ▼  data.ctrl (14 joint position targets)
 ┌──────────────────────────────────────────────────────┐
 │ MuJoCo, 500 Hz (dt = 2 ms)                           │
 │  position actuators: τ = 13.37·(q_target − q)        │
 │  clipped to ±3.23 N·m, plus passive joint damping    │
 │  0.56·q̇, friction loss and rotor armature            │
 └──────────────────────────────────────────────────────┘
```

The policy is a **joint-position-target policy**. It never outputs torques. MuJoCo's position actuators do the low-level control, playing the role of the servo's internal controller.

---

## 2. The robot model

**Files [code]:**
- `open_duck_mini_v2/xmls/open_duck_mini_v2.xml`: the robot, generated from Onshape CAD by `onshape-to-robot` (see `xmls/config.json`).
- `open_duck_mini_v2/xmls/scene_flat_terrain.xml`: includes the robot and adds a floor (friction 0.6) and the `home` keyframe.
- Other scenes: `scene_flat_terrain_backlash.xml` and `scene_rough_terrain_backlash.xml` add backlash joints.

**Format:** MJCF (MuJoCo XML), with STL meshes in `xmls/assets/`.

**Structure [measured from the compiled model]:**
- 18 bodies. Total mass **2.107 kg**.
- `nq = 21`: 7 for the floating base (3 position + 4 quaternion) and 14 joint angles. `nv = 20`: 6 base + 14 joints.
- **14 actuated hinge joints**, in this order. The order matters, because every 14-long vector in the observation and action uses it:

| idx | joint | range (rad) | role |
|---|---|---|---|
| 0 | left_hip_yaw | ±0.524 | leg |
| 1 | left_hip_roll | ±0.436 | leg |
| 2 | left_hip_pitch | −1.222 … 0.524 | leg |
| 3 | left_knee | ±1.571 | leg |
| 4 | left_ankle | ±1.571 | leg |
| 5 | neck_pitch | −0.349 … 1.134 | head |
| 6 | head_pitch | ±0.785 | head |
| 7 | head_yaw | ±2.793 | head |
| 8 | head_roll | ±0.524 | head |
| 9–13 | right_hip_yaw … right_ankle | mirrored | leg |

So the robot has **10 leg DoF (5 per leg) and 4 head/neck DoF**. The head matters for NERVA: it is an expressive channel that is already actuated and already controlled by the policy.

- **Home pose** (`scene_flat_terrain.xml`, keyframe `home`): base at z = 0.15 m, knees bent about 1.37 rad, hips pitched about ±0.63 rad, head joints at 0. This is the "default pose" that actions are offsets from.
- **Sensors** (`xmls/sensors.xml`), all on the `imu` site on the trunk:
  - `gyro`, `accelerometer`, `local_linvel` (a velocimeter)
  - `upvector` (the IMU frame's z-axis expressed in world coordinates)
  - global linear and angular velocity
  - foot-site positions and velocities
  - Foot contact is not a sensor. It is computed by collision checking between `left/right_foot_bottom_tpu` and the floor.
- **Solver options** (`open_duck_mini_v2.xml`): Euler integrator with `eulerdamp` disabled, `iterations=1`, `ls_iterations=5`. These are low iteration counts, typical for fast GPU (MJX) training.

### Watch out: two conflicting servo definitions
`xmls/joints_properties.xml` defines class `sts3215` with `kp=17.8, forcerange ±3.35, damping 0.60`, but that file is **not included** by any model. The values actually used are the ones pasted into `open_duck_mini_v2.xml` (around line 45). Verified from the compiled model:

| parameter | value used | meaning |
|---|---|---|
| actuator type | `position`, `kv=0` | force = kp·(ctrl − q) |
| kp | **13.37 N·m/rad** | stiffness |
| forcerange | **±3.23 N·m** | torque saturation |
| joint damping | **0.56 N·m·s/rad** | passive, always on |
| frictionloss | 0.068 N·m | dry friction |
| armature | 0.027 kg·m² | reflected rotor inertia of the geared motor |
| ctrlrange | = joint range (`inheritrange=1`) | targets clipped to joint limits |

---

## 3. The low-level controller ("PD")

You might expect τ = Kp(q_des − q) − Kd·q̇. Here is what actually happens [code + measured]:

- The **P term** is the MuJoCo `position` actuator: `τ_act = 13.37 · (q_target − q)`, saturated at ±3.23 N·m. Its `kv` (the velocity term) is **0**.
- The **D-like term** comes from **passive joint damping**, `τ_damp = −0.56 · q̇`. It is always present and is not subject to the actuator's force limit.
- On top of that come dry friction (0.068 N·m) and armature (rotor inertia 0.027).

So in effect τ ≈ 13.37·(q_target − q) − 0.56·q̇, with the P part clipped. The parameters were chosen to imitate a **Feetech STS3215** servo. The damping and armature are an identified model of the motor and gearbox, not a tuned D gain.

This runs at the physics rate, **500 Hz**. The policy only changes `q_target` every 10 physics steps.

---

## 4. What the policy outputs (the action)

**[code]:** `mujoco_infer.py:203–231` for sim inference; `joystick.py:404–420` for training.

1. The policy outputs `a ∈ R^14`, one value per actuator in the order above. Each value is in [−1, 1] because the network ends in `tanh` (see §7).
2. `motor_target = home_ctrl + 0.25 · a`. `action_scale = 0.25`, so one unit of action is 0.25 rad (about 14°) away from the home pose.
3. **Rate limit:** each target may move at most `5.24 rad/s × 0.02 s = 0.105 rad` per policy step (`USE_MOTOR_SPEED_LIMITS`). 5.24 rad/s is the servo's maximum speed.
4. `data.ctrl = motor_target`, held constant for the next 10 physics steps.

The policy outputs **joint position targets as offsets from the home pose, for all 14 joints including the head**.

---

## 5. What the policy sees (the observation)

**[code]:** `joystick.py:487–589` (training) and `mujoco_infer.py:67–101` (sim inference). **Dimension 101 [measured]**: it matches the ONNX input `obs[1,101]`.

| index | size | content | notes |
|---|---|---|---|
| 0–2 | 3 | gyro (rad/s, IMU frame) | |
| 3–5 | 3 | accelerometer (m/s², IMU frame) | see the accelerometer-offset note in §10 |
| 6–12 | 7 | **command**: [vx, vy, ωz, neck_pitch, head_pitch, head_yaw, head_roll] | the joystick input |
| 13–26 | 14 | joint angles − home pose | |
| 27–40 | 14 | joint velocities × 0.05 | scaled to similar magnitude |
| 41–54 | 14 | last action | |
| 55–68 | 14 | action from 2 steps ago | |
| 69–82 | 14 | action from 3 steps ago | |
| 83–96 | 14 | current motor targets (after rate limit) | |
| 97–98 | 2 | foot contacts [left, right] | 0/1 |
| 99–100 | 2 | **gait phase** [cos φ, sin φ] | the clock from the reference motion (§8) |

Notes:
- The inline comments in `joystick.py` say `# 10` next to the 14-long entries. They are stale, left over from before the head joints were added.
- The policy **does not see** base linear velocity, base height or orientation directly. Orientation reaches it only implicitly through the accelerometer, which senses gravity. That is deliberate: those quantities are hard to measure on the real robot.
- `privileged_state` (`joystick.py:596–615`) is a larger vector that includes true velocity, base height, actuator forces, foot velocities, air time and the current reference frame. **Only the value network (critic) sees it during training.** The deployed policy never does. This setup is called an *asymmetric actor–critic*.
- During training, noise is added to gyro, accelerometer, joint angles and velocities. Actions are randomly delayed by 0 to 2 steps, and IMU readings by 0 to 2 steps (`noise_config`, `joystick.py:60–76`; `randint`'s max is exclusive). Sim inference adds no noise.

---

## 6. Timing

**[code + measured]:**
- `sim_dt = 0.002 s`: physics at 500 Hz (`mujoco_infer_base.py:15`; `joystick.py:52`).
- `ctrl_dt = 0.02 s`, `decimation = 10`: **policy at 50 Hz** (`mujoco_infer_base.py:16`; `joystick.py:51`).
- Gait period 0.54 s, which is 27 policy steps (`nb_steps_in_period`, from the reference data).
- Training episode length 1000 steps, which is 20 s (`joystick.py:53`).
- In the viewer on this laptop, the GUI loop runs at about 0.75× real time [measured]. Headless runs are unaffected.

---

## 7. The network and how it is loaded

**[measured from `BEST_WALK_ONNX_2.onnx`]:**
- Exported with `tf2onnx 1.16.1`, opset 11.
- Input normalisation is **built into the graph**: `(obs − mean) / std`, using statistics saved from training.
- MLP layer sizes **101 → 512 → 256 → 128 → 28**, with swish activation (Sigmoid·x, three times).
- The 28 outputs are split into 14 means and 14 spread (scale) parameters of Brax's tanh-normal action distribution. Only the **means** are used, through `tanh`. So deployment is **deterministic**: the exploration noise used during PPO training is dropped.

**Loading [code]:** `common/onnx_infer.py` wraps `onnxruntime.InferenceSession(..., CPUExecutionProvider)`. `infer(obs)` returns the 14 actions. The same class is copied into the hardware runtime.

**How the ONNX file is made [code]:**
1. During training, `common/runner.py:68–84` (`policy_params_fn`) saves an Orbax checkpoint of the JAX parameters at every evaluation.
2. `common/export_onnx.py` rebuilds the same MLP in TensorFlow/Keras, copies the weights over, and converts the result with `tf2onnx`.
3. This is why the Playground depends on TensorFlow even though training is JAX.

---

## 8. Reference motion (imitation) — the part most relevant to NERVA

**Source of the data [code]:**
- `Open_Duck_reference_motion_generator` uses **Placo** (a whole-body kinematics and gait library from Rhoban) to generate walking cycles procedurally.
- It covers a grid of commanded velocities. Each cycle is fitted with polynomials and saved to `open_duck_mini_v2/data/polynomial_coefficients.pkl`.

**What is in that file [measured]:**
- 240 gaits on a grid:

| axis | values |
|---|---|
| vx (6) | −0.148 … +0.222 m/s |
| vy (4) | ±0.037, ±0.111 m/s (no exact zero) |
| ωz (10) | −1.111 … +1.222 rad/s |

- Each gait has 40 signals: 16 joint positions, 16 joint velocities, 2 foot contacts, 3 linear velocities and 3 angular velocities. Each signal is a **degree-15 polynomial in phase t ∈ [0, 1)**.
- Period 0.54 s, sampled at 50 fps.
- Lookup is **nearest neighbour** on the grid, not interpolation (`common/poly_reference_motion_numpy.py`, `vel_to_index`).

**How it is used [code]:**
1. **During training only**, the reward `reward_imitation` (`custom_rewards.py:4–148`, weight 1.0) compares the robot's leg joint positions and velocities, base velocities and foot contacts to the reference frame for the current command and phase. Joint positions dominate, with weight 15. It pays nothing when the command is zero.
2. **At training and at runtime**, the policy observes only the **phase clock** [cos φ, sin φ] (obs 99–100), not the reference joint angles. The clock advances one step per policy step and wraps every 27 steps.

**Consequence [inferred, important for NERVA]:**
- After training, the reference motion is gone. What is left is a policy that has learned *"walk in a Placo-like way, in sync with this clock"*.
- The runtime can change **how fast the clock ticks**:
  - Sim inference: keys `P` / `;` change `phase_frequency_factor` (`mujoco_infer.py:176`).
  - Hardware: holding **LB** on the gamepad sets the factor to 1.3, and each robot has a calibration offset (`v2_rl_walk_mujoco.py:226, 257`).
- That is an existing "gait-timing" knob that needs no retraining. We haven't yet measured what it does to the gait.

Two oddities [code]:
- `reward_imitation` reads the torso quaternion from slice 3:7 of the 40-long reference. In the 40-long layout those are joint positions, not a quaternion. It is harmless only because the orientation term is excluded from the sum.
- The reference's angular-velocity slice is nearly zero even for turning gaits (0.026 rad/s for a 0.96 rad/s gait) [measured]. We haven't determined whether that is a layout mismatch or a property of the generator.

---

## 9. Training

**Algorithm [code]:**
- **PPO** from **Brax** (`brax.training.agents.ppo`), run on **MJX** (MuJoCo in JAX) so thousands of robots simulate in parallel on a GPU.
- Entry point: `open_duck_mini_v2/runner.py` → `common/runner.py:BaseRunner.train`.
- Environment: `open_duck_mini_v2/joystick.py:Joystick`, which subclasses `base.py:OpenDuckMiniV2Env`, which subclasses MuJoCo Playground's `MjxEnv`.

**Hyperparameters [code]:**
- These are borrowed from DeepMind's **Berkeley Humanoid** config: `runner.py:87` calls `brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")`, defined in `mujoco_playground/config/locomotion_params.py`.

| parameter | value |
|---|---|
| parallel envs | 8192 |
| timesteps | 150 M (default); the README's "current win" used 300 M on `flat_terrain_backlash` |
| discount γ | 0.97 |
| learning rate | 3e-4 |
| entropy cost | 0.005 |
| clip ε | 0.2 |
| unroll length | 20 |
| minibatches | 32 |
| updates per batch | 4 |
| network sizes | policy and value both (512, 256, 128) |
| observation normalisation | on |

**Episode [code]:**
- Reset (`joystick.py:206`) starts from the home pose with:
  - random yaw
  - ±5 cm xy offset
  - joint angles scaled by U(0.5, 1.5)
  - small base velocity
- A command is sampled uniformly in vx ±0.15, vy ±0.2, ωz ±1.0 plus random head targets. With 10% probability it is all zeros. It is resampled every 500 steps (10 s).
- **Pushes:** a random velocity kick of 0.1–1.0 m/s every 5–10 s.
- **Termination** (`joystick.py:483–485`): the IMU up-vector's world-z component goes negative (tilt past 90°), or NaNs appear.

**Reward** (`joystick.py:622–669`; functions in `common/rewards.py`). Per step, `reward = clip(Σ scaleₖ·termₖ × dt, 0, ∞)`:

| term | scale | formula |
|---|---|---|
| alive | +20 | 1 per step survived |
| tracking_ang_vel | +6.0 | exp(−(ωz_cmd − gyro_z)² / 0.01) |
| tracking_lin_vel | +2.5 | exp(−[(vx_cmd − vx)² + max(0, abs(vy − vy_cmd) − 0.1)²] / 0.01) |
| imitation | +1.0 | see §8 |
| stand_still | −0.2 | when the command is about 0: pose deviation + joint speed |
| action_rate | −0.5 | ‖aₜ − aₜ₋₁‖² |
| torques | −0.001 | ‖τ‖² |

Head commands are sampled and observed, but **no reward term uses them** in this config (`cost_head_pos` exists but is not wired in), and they are not applied to the head motors in training.

**Domain randomisation** (`common/randomize.py:26`), per environment:
- floor friction U(0.5, 1.0)
- joint friction loss ×U(0.9, 1.1)
- armature ×U(1.0, 1.05)
- torso centre-of-mass ±5 cm
- all body masses ×U(0.9, 1.1), plus torso ±0.1 kg
- `qpos0` ±0.03 rad (upstream's comment suggests joint calibration offsets; in MuJoCo `qpos0` is the model's reference configuration, so the practical effect is [unknown])
- **kp ×U(0.9, 1.1)**

The point is sim-to-real robustness.

**[unknown]** Which scene, how many timesteps and which exact code version produced `BEST_WALK_ONNX_2.onnx`. Upstream didn't publish that training config. The observation size is 101 with or without backlash joints, so the file itself can't tell us.

**This machine cannot train.** Brax/MJX training is designed for CUDA GPUs. See the development log.

---

## 10. Simulation vs hardware

Hardware loop: `Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py`. Despite the name, it drives the real robot. It runs on a Raspberry Pi Zero 2W (`Open_Duck_Mini_Runtime/README.md`).

| aspect | simulation (`mujoco_infer.py`) | hardware (`v2_rl_walk_mujoco.py`) |
|---|---|---|
| actuators | MuJoCo position actuator, kp 13.37 N·m/rad, ±3.23 N·m | Feetech STS3215 servos over serial (`rustypot`). Servo-register kp **30** for legs and **8** for head, kd 0 (line 179). These are servo units, not N·m/rad. |
| IMU | MuJoCo `gyro` / `accelerometer` sensors | Bosch **BNO055** (`adafruit_bno055`, `raw_imu.py`) |
| accelerometer x | **+1.3 added** (`mujoco_infer.py:74`) | raw (`tare_x()` is commented out, so `x_offset = 0`) |
| foot contacts | geometry collision test | two switches on Pi GPIO D22/D27 (`feet_contacts.py`) |
| motor speed limit | applied (5.24 rad/s clip) | commented out (line 292); an optional low-pass filter instead |
| head | policy targets only; head keys change only the observation | **command added on top** of the policy's head targets (line 310), so the gamepad drives the head directly |
| loop | 50 Hz policy over 500 Hz physics | 50 Hz wall-clock (`time.sleep`), warns when over budget |
| phase clock | factor 1.0, `P`/`;` keys | factor 1.0, 1.3 while LB held, plus per-robot offset |

### The accelerometer offset mismatch [code + measured]
- **Training** has `accelerometer.at[0].set(accelerometer[0] + 1.3)` (`joystick.py:502`). JAX arrays are immutable and the result isn't assigned, so **the line has no effect**. The policy was trained on the raw accelerometer.
- **Hardware** also feeds the raw value.
- **The sim inference script alone adds +1.3 m/s²**, so it feeds the policy an input it never saw in training.

We measured the effect by cancelling the offset (`scripts/check_open_duck_baseline.py --raw-accel`, 20 s per command):

| command | upstream sim (+1.3) | raw accel (matches training) |
|---|---|---|
| forward 0.15 m/s | 0.097 m/s | **0.112 m/s** |
| backward 0.15 m/s | −0.032 | −0.031 |
| lateral 0.2 m/s | 0.093 | 0.102 |
| turn 1.0 rad/s | 0.81 rad/s | 0.82 rad/s |
| falls | none | none |

The offset is real and costs some forward tracking, but it isn't the main reason for under-tracking. **For NERVA experiments we should use the raw accelerometer**, because that matches both training and hardware.

---

## 11. Why the robot under-tracks velocity commands

Measured with the raw accelerometer: forward about 75%, backward about 21%, lateral about 51%, turning about 82% of the command.

- **Lateral [inferred from code, consistent with measurement].**
  - The lateral reward ignores errors below 0.1 m/s.
  - The reference grid only goes to ±0.111 m/s.
  - So about 0.10 m/s sideways already earns full reward when 0.2 is commanded.
- **Forward and turning [inferred].**
  - Both tracking rewards use the same absolute tolerance, `tracking_sigma = 0.01`.
  - At 0.112 m/s vs 0.15 m/s commanded, the forward reward only drops about 13%.
  - An 0.18 rad/s turn error drops the turn reward about 96%.
  - Linear velocities are about ten times smaller in number than angular ones, so they are weakly enforced. That fits turning being tracked much better.
  - We haven't tested this by retraining.
- **The reference motion is not the cause [measured].** The reference gaits' own speed matches their command (0.155 m/s for the 0.148 gait).
- **Backward [unknown].**
  - Its error would be strongly penalised, and the backward reference moves at 0.116 m/s, yet the policy only manages 0.031 m/s.
  - Not explained yet.

**Why this matters for NERVA:** if style changes the gait, we can't take "commanded speed" as the actual speed. Every experiment must log the **measured** velocity.

---

## 12. Where to look in the code

| question | file |
|---|---|
| robot geometry, joints, actuators | `open_duck_mini_v2/xmls/open_duck_mini_v2.xml` |
| sensors | `open_duck_mini_v2/xmls/sensors.xml` |
| home pose, floor | `open_duck_mini_v2/xmls/scene_flat_terrain.xml` |
| names of feet, IMU and sensors used by code | `open_duck_mini_v2/constants.py` |
| env base: joint/sensor accessors | `open_duck_mini_v2/base.py` |
| training env: obs, reward, reset, commands | `open_duck_mini_v2/joystick.py` |
| reward functions | `common/rewards.py`, `open_duck_mini_v2/custom_rewards.py` |
| reference motion lookup | `common/poly_reference_motion.py` (JAX), `…_numpy.py` (NumPy) |
| domain randomisation | `common/randomize.py` |
| PPO training loop and checkpoint/ONNX export | `common/runner.py`, `open_duck_mini_v2/runner.py`, `common/export_onnx.py` |
| sim inference (viewer) | `open_duck_mini_v2/mujoco_infer.py`, `mujoco_infer_base.py` |
| ONNX runtime wrapper | `common/onnx_infer.py` |
| hardware loop | `Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py` |
| standing-only task (separate policy) | `open_duck_mini_v2/standing.py` (not examined in detail) |

---

## 13. Summary in one paragraph

Open Duck Mini v2 is a 2.1 kg biped with 5-DoF legs and a 4-DoF head, modelled in MJCF with servo-like position actuators: kp 13.37, torque limit ±3.23 N·m, plus identified damping, friction and armature. A 101→512→256→128→14 MLP runs at 50 Hz and outputs joint targets as offsets (×0.25 rad) from a crouched home pose, rate-limited to the servo's top speed. MuJoCo simulates at 500 Hz. The policy sees the IMU, the command, joint states, its last three actions, foot contacts and a gait-phase clock. It never sees its own velocity or height. It was trained with Brax PPO on MJX, using a reward that mixes survival, velocity tracking and imitation of Placo-generated reference gaits, under heavy noise, delays, pushes and domain randomisation. After training, the reference motion survives only as the phase clock, whose speed can be changed at runtime.
