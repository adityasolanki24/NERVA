"""Headless Open Duck Mini v2 simulation driven by NERVA behaviour commands.

This is the only NERVA module that imports Open Duck (lazily, inside
OpenDuckSim). It reuses upstream `MjInfer` for everything that defines the
baseline: model loading, the home pose, observation construction, the ONNX
policy, action scaling and the motor speed limit. It replaces only
`MjInfer.run()`, the viewer loop, with `_control_step()`, because we need to
inject style, pushes and logging. `tests/test_open_duck_sim.py` checks that with
default settings this loop reproduces upstream `run()` exactly.

Differences from upstream `mujoco_infer.py`, all opt-in:
  raw_accel=True       cancel the +1.3 on accelerometer x that only upstream sim
                       inference adds (docs/open_duck_baseline.md §10)
  init_joint_noise>0   seeded uniform noise on initial joint angles, to get
                       run-to-run variation from a deterministic simulator
  obs_noise=True       add the observation noise the policy was TRAINED with
                       (joystick.py noise_config), resampled every control step
  action_delay=True    apply the action from 0, 1 or 2 control steps ago (uniform, resampled every control
                       step, its own seeded stream), as the training env's noise_config does
                       (action_min_delay=0, action_max_delay=3 exclusive; the history starts at zero).
                       The observation's last actions stay undelayed, as in training. Training's IMU
                       delay acts only on an unobserved gravity vector, so it has no counterpart here.
  push()               add a velocity to the base, as the training env does
  set_style_vector()   S1 policies only: append the style vector e to the observation
                       and use that style's gait period for the phase clock, as
                       nerva/training/style_joystick.py does in training
  set_head_offset()    head joint offsets added on top of the policy's head targets,
                       exactly as the hardware runtime adds gamepad head commands
                       (Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py:310): the
                       offset goes into the observation's command slots 3:7 and is
                       added to motor targets 5:9 AFTER the speed limit. Default zero.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nerva.interfaces import BehaviourCommand
from nerva.behaviour.style import s1_nb_steps_in_period, style_to_phase_factor

OPEN_DUCK_ROOT = Path(os.environ.get("OPEN_DUCK_ROOT", Path.home() / "dev" / "open_duck"))
PLAYGROUND = OPEN_DUCK_ROOT / "Open_Duck_Playground" / "playground" / "open_duck_mini_v2"
SCENE = PLAYGROUND / "xmls" / "scene_flat_terrain.xml"
# The scene NERVA's policies (B0-B2, S1-S4) were trained in: joints with backlash. Evaluating in the
# plain scene underestimates them (2026-10-01: S1 backward -0.043 plain vs -0.071 backlash).
SCENE_BACKLASH = PLAYGROUND / "xmls" / "scene_flat_terrain_backlash.xml"
REFERENCE = PLAYGROUND / "data" / "polynomial_coefficients.pkl"
POLICY = OPEN_DUCK_ROOT / "Open_Duck_Mini" / "BEST_WALK_ONNX_2.onnx"

# Command ranges the policy was trained on (joystick.py default_config).
TRAINED_VX = (-0.15, 0.15)
TRAINED_VY = (-0.2, 0.2)
TRAINED_YAW_RATE = (-1.0, 1.0)

ACCEL_X_OFFSET = 1.3  # added by upstream mujoco_infer.get_obs only


def training_obs_noise_scale() -> np.ndarray:
    """Half-widths of the uniform noise joystick.py adds to each of the 101 obs entries.

    Reproduces the training env exactly, including a quirk: joint-angle noise
    indices are computed on the 10-joint no-head list (constants.JOINTS_ORDER_NO_HEAD)
    but applied to the 14-actuator vector. So head joints receive hip/knee noise
    and right hip roll/pitch, knee and ankle receive none. We keep it because
    it is the distribution the policy was trained on.
    """
    joint_scale = np.zeros(14)
    no_head = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
    for i, name in enumerate(no_head):
        joint_scale[i] = 0.03 if "_hip" in name else 0.05 if "_knee" in name else 0.08
    scale = np.zeros(101)
    scale[0:3] = 0.1  # gyro
    scale[3:6] = 0.05  # accelerometer
    scale[13:27] = joint_scale  # joint angles (rad)
    scale[27:41] = 2.5 * 0.05  # joint velocity noise 2.5 rad/s, then × dof_vel_scale
    return scale


@dataclass
class StepLog:
    """One row per 50 Hz control step, recorded after the control update."""

    t: float
    base_pos: np.ndarray  # (3,) world
    base_quat: np.ndarray  # (4,) w x y z
    base_linvel: np.ndarray  # (3,) world
    base_angvel: np.ndarray  # (3,) body frame (MuJoCo free-joint convention)
    foot_z: np.ndarray  # (2,) left, right foot site height
    contacts: np.ndarray  # (2,) bool
    joint_pos: np.ndarray  # (14,)
    action: np.ndarray  # (14,) raw policy output
    tau_sq: float  # mean over the 10 physics steps of Σ τ²
    power: float  # mean over the 10 physics steps of Σ abs(τ·q̇)  [W]


def to_arrays(log: list[StepLog], start: int = 0) -> dict[str, np.ndarray]:
    """Stack StepLog rows from index `start` into arrays keyed by field name."""
    rows = log[start:]
    return {name: np.array([getattr(r, name) for r in rows]) for name in StepLog.__dataclass_fields__}


STYLED_CONTRACT = "e1_prime_pitch_crouch"  # docs/expressive_posture_e1prime.md
ACTION_DELAY_STEPS = 3  # training: randint(action_min_delay=0, action_max_delay=3) → 0, 1 or 2 steps


class OpenDuckSim:
    def __init__(self, raw_accel: bool = True, init_joint_noise: float = 0.0,
                 obs_noise: bool = False, seed: int = 0, action_delay: bool = False,
                 policy_path: str | Path = POLICY, scene_extender=None, scene: str | Path = SCENE):
        import mujoco  # noqa: F401  (imported here so `nerva` core never needs it)
        from playground.open_duck_mini_v2.mujoco_infer import MjInfer

        self.scene = Path(scene)
        self.policy_path = Path(policy_path)
        self._neutral_motor_clock = None
        self._styled_contract = False
        self.inf = MjInfer(str(self.scene), str(REFERENCE), str(policy_path), standing=False)
        if scene_extender is not None:
            self._extend_scene(scene_extender)
        self.model, self.data = self.inf.model, self.inf.data
        self.phase_factor = 1.0
        self.inf.commands = [0.0] * 7
        self._counter = 0  # physics steps, as `counter` in upstream run()

        if raw_accel:
            upstream_get_acc = self.inf.get_accelerometer

            def raw(data):
                a = np.array(upstream_get_acc(data), copy=True)
                a[0] -= ACCEL_X_OFFSET  # get_obs adds it back, so the policy sees the raw value
                return a

            self.inf.get_accelerometer = raw

        rng = np.random.default_rng(seed)
        self._rng = rng
        # Separate stream: enabling the delay leaves the paired observation-noise sequence unchanged.
        self._delay_rng = np.random.default_rng([seed, 1]) if action_delay else None
        self._action_history = None
        self._obs_noise_scale = training_obs_noise_scale() if obs_noise else None
        if init_joint_noise > 0:
            addr = self.inf.get_actuator_joints_addr()
            self.data.qpos[addr] += rng.uniform(-init_joint_noise, init_joint_noise, len(addr))

        self._foot_sites = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
                            for n in ("left_foot", "right_foot")]
        self._qvel_addr = self.inf.actuator_qvel_addr
        self._mj_step = mujoco.mj_step
        self._last_action = np.zeros(self.model.nu)
        self.head_offset = np.zeros(4)  # neck_pitch, head_pitch, head_yaw, head_roll [rad]
        self.applied_command: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.style_vector: np.ndarray | None = None  # S1 policies only
        self.nb_steps_in_period = self.inf.PRM.nb_steps_in_period

    def _extend_scene(self, extender) -> None:
        """Rebuild the model with extra (DOF-free) elements, e.g. nerva.sim.world.extend_scene.

        MjInfer looked up all robot indices by name at construction; extensions must only
        append bodies/geoms/cameras without degrees of freedom, so those indices stay valid.
        The robot's state is copied over unchanged.
        """
        import mujoco

        spec = mujoco.MjSpec.from_file(str(self.scene))
        extender(spec)
        model = spec.compile()
        model.opt.timestep = self.inf.model.opt.timestep  # MjInfer overrides it after loading
        old = self.inf.data
        if (model.nq, model.nv, model.nu, model.nsensordata) != (self.inf.model.nq, self.inf.model.nv,
                                                                  self.inf.model.nu, self.inf.model.nsensordata):
            raise ValueError("scene extensions must not add joints, actuators or sensors")
        data = mujoco.MjData(model)
        data.time = old.time
        data.qpos[:], data.qvel[:], data.ctrl[:] = old.qpos, old.qvel, old.ctrl
        mujoco.mj_forward(model, data)
        # The solver warm start matters here (opt.iterations = 1 in the upstream scene).
        data.qacc_warmstart[:] = old.qacc_warmstart
        self.inf.model, self.inf.data = model, data

    # ── inputs ──────────────────────────────────────────────────────────────

    def set_neutral_motor_contract(self, metadata) -> None:
        """Explicit deployment opt-in for a matching future candidate ONNX artifact."""
        import hashlib
        from nerva.motor_contract import NeutralPhaseClock
        from nerva.training.neutral_reference import COMMANDS, CONTRACT
        styled = metadata.get("style") == STYLED_CONTRACT
        if (metadata.get("contract") != CONTRACT or metadata.get("observation_size") != (103 if styled else 101)
                or metadata.get("style") not in (None, STYLED_CONTRACT)
                or metadata.get("period_steps") != 27 or metadata.get("head_commands_zero") is not True
                or {tuple(c) for c in metadata.get("commands", ())} != set(COMMANDS)
                or metadata.get("policy_sha256") != hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
                or self.style_vector is not None or np.any(self.head_offset != 0)):
            raise ValueError("candidate deployment metadata mismatch")
        self._neutral_motor_clock = NeutralPhaseClock(27)
        if styled:
            self.style_vector = np.zeros(2)  # e = (0, 0): the neutral anchor until set_neutral_style()
        self._styled_contract = styled
        self.inf.imitation_i = 0
        self.inf.imitation_phase = self._neutral_motor_clock.reset()
        self.nb_steps_in_period = 27

    def set_neutral_style(self, e_pitch: float, e_crouch: float) -> None:
        """E1′ styled candidates only: e = (e_pitch ∈ [−1, 1], e_crouch ∈ [0, 1]), appended noise-free after the
        101 observation values (as in training). The phase clock and period are unchanged."""
        if self._neutral_motor_clock is None or not self._styled_contract:
            raise ValueError("style requires a styled neutral contract")
        e = np.array([e_pitch, e_crouch], dtype=np.float64)
        if not np.isfinite(e).all() or abs(e[0]) > 1 or not 0 <= e[1] <= 1:
            raise ValueError("style outside the admitted E1' domain")
        self.style_vector = e

    def _validate_neutral_command(self, command):
        from nerva.motor_contract import canonical_command
        from nerva.training.neutral_reference import COMMANDS
        command = canonical_command(command)
        if (np.shape(command) != (7,) or not np.isfinite(command).all() or np.any(command[3:] != 0)
                or not any(np.allclose(command[:3], known, atol=1e-8, rtol=0) for known in COMMANDS)):
            raise ValueError("unsupported candidate motor command")
        return command

    def set_behaviour(self, cmd: BehaviourCommand) -> None:
        """Apply a BehaviourCommand. Velocities are clipped to the trained range."""
        if self._neutral_motor_clock is not None:
            if cmd.style_vector is not None or cmd.style.style != 0:
                raise ValueError("neutral candidate has no expressive conditioning")
            command = self._validate_neutral_command([cmd.vx, cmd.vy, cmd.yaw_rate, *self.head_offset])
            self.inf.commands = command.tolist()
            self.applied_command = tuple(command[:3])
            return
        vx = float(np.clip(cmd.vx, *TRAINED_VX))
        vy = float(np.clip(cmd.vy, *TRAINED_VY))
        wz = float(np.clip(cmd.yaw_rate, *TRAINED_YAW_RATE))
        self.inf.commands = [vx, vy, wz, *self.head_offset.tolist()]
        self.applied_command = (vx, vy, wz)
        self.phase_factor = style_to_phase_factor(cmd.style)
        if cmd.style_vector is not None:
            self.set_style_vector(cmd.style_vector.as_tuple())

    def set_style_vector(self, e) -> None:
        """S1 policy input e = (tempo, step height, torso pitch) in [-1, 1]; appended to obs.

        Style enters the observation noise-free (StyleJoystick appends it after
        upstream noise) and the phase clock uses the style's own gait period.
        """
        if self._neutral_motor_clock is not None:
            raise ValueError("neutral candidate has no style vector")
        e = np.asarray(e, dtype=np.float64)
        if e.shape != (3,) or np.any(np.abs(e) > 1.0):
            raise ValueError("style vector must be three values in [-1, 1]")
        self.style_vector = e
        nb = s1_nb_steps_in_period(float(e[0]))
        if nb != self.nb_steps_in_period:
            # NERVA deployment choice: keep the gait phase FRACTION when the period changes, so
            # a continuously varying tempo does not make the clock jump (training resampled
            # style rarely and did not rescale; with fixed styles this has no effect).
            self.inf.imitation_i = self.inf.imitation_i * nb / self.nb_steps_in_period
            self.nb_steps_in_period = nb

    def set_head_offset(self, neck_pitch: float = 0.0, head_pitch: float = 0.0,
                        head_yaw: float = 0.0, head_roll: float = 0.0) -> None:
        """Head posture offsets [rad] on top of the policy (hardware-runtime convention)."""
        offset = np.array([neck_pitch, head_pitch, head_yaw, head_roll], dtype=float)
        if self._neutral_motor_clock is not None and np.any(offset != 0):
            raise ValueError("neutral candidate requires zero head offsets")
        self.head_offset = offset
        self.inf.commands = list(self.inf.commands[:3]) + self.head_offset.tolist()

    def push(self, dvx: float, dvy: float) -> None:
        """Instant base velocity change in world x/y [m/s], like the training env's pushes."""
        self.data.qvel[0] += dvx
        self.data.qvel[1] += dvy

    # ── simulation ──────────────────────────────────────────────────────────

    def _control_step(self) -> None:
        """Mirror of the body of `if counter % decimation == 0:` in upstream run()."""
        inf = self.inf
        if self._neutral_motor_clock is not None:
            tick = self._neutral_motor_clock.tick(self._validate_neutral_command(inf.commands))
            inf.imitation_i, inf.imitation_phase, inf.commands = tick["index"], tick["phase"], tick["command"].tolist()
        else:
            inf.imitation_i += 1.0 * self.phase_factor
            inf.imitation_i = inf.imitation_i % self.nb_steps_in_period
            ang = inf.imitation_i / self.nb_steps_in_period * 2 * np.pi
            inf.imitation_phase = np.array([np.cos(ang), np.sin(ang)])

        obs = inf.get_obs(self.data, inf.commands)
        if self._obs_noise_scale is not None:
            obs = obs + self._rng.uniform(-1.0, 1.0, obs.shape) * self._obs_noise_scale
        if self.style_vector is not None:
            obs = np.concatenate([obs, self.style_vector])
        action = inf.policy.infer(obs)

        inf.last_last_last_action = inf.last_last_action.copy()
        inf.last_last_action = inf.last_action.copy()
        inf.last_action = action.copy()

        applied = action
        if self._delay_rng is not None:
            if self._action_history is None:
                self._action_history = np.zeros((ACTION_DELAY_STEPS, action.shape[0]))
            self._action_history = np.roll(self._action_history, 1, axis=0)
            self._action_history[0] = action
            applied = self._action_history[self._delay_rng.integers(0, ACTION_DELAY_STEPS)]
        inf.motor_targets = inf.default_actuator + applied * inf.action_scale
        max_delta = inf.max_motor_velocity * (inf.sim_dt * inf.decimation)
        inf.motor_targets = np.clip(inf.motor_targets,
                                    inf.prev_motor_targets - max_delta,
                                    inf.prev_motor_targets + max_delta)
        inf.prev_motor_targets = inf.motor_targets.copy()
        inf.motor_targets[5:9] = inf.motor_targets[5:9] + self.head_offset  # hardware order
        self.data.ctrl = inf.motor_targets.copy()
        self._last_action = action

    def step_physics(self, n: int) -> None:
        """Advance n physics steps (2 ms each), running the policy every 10th, as upstream."""
        for _ in range(n):
            self._mj_step(self.model, self.data)
            self._counter += 1
            if self._counter % self.inf.decimation == 0:
                self._control_step()

    def run(self, seconds: float, pushes: dict[int, tuple[float, float]] | None = None) -> list[StepLog]:
        """Simulate `seconds`, logging every control step.

        pushes: {control_step_index: (dvx, dvy)}, applied just before that step.
        """
        pushes = pushes or {}
        n_ctrl = int(round(seconds / (self.inf.sim_dt * self.inf.decimation)))
        log: list[StepLog] = []
        d = self.data
        for k in range(n_ctrl):
            if k in pushes:
                self.push(*pushes[k])
            tau_sq = power = 0.0
            for _ in range(self.inf.decimation):
                self.step_physics(1)
                tau = d.actuator_force
                tau_sq += float(np.sum(tau * tau))
                power += float(np.sum(np.abs(tau * d.qvel[self._qvel_addr])))
            log.append(StepLog(
                t=float(d.time),
                base_pos=d.qpos[0:3].copy(),
                base_quat=d.qpos[3:7].copy(),
                base_linvel=d.qvel[0:3].copy(),
                base_angvel=d.qvel[3:6].copy(),
                foot_z=np.array([d.site_xpos[i][2] for i in self._foot_sites]),
                contacts=np.array(self.inf.get_feet_contacts(d), dtype=bool),
                joint_pos=self.inf.get_actuator_joints_qpos(d.qpos).copy(),
                action=self._last_action.copy(),
                tau_sq=tau_sq / self.inf.decimation,
                power=power / self.inf.decimation,
            ))
        return log
