"""Style-conditioned version of the Open Duck Mini v2 joystick training env (experiment S1).

Design: docs/style_policy_design.md. Upstream code is NOT modified; this module
subclasses `playground.open_duck_mini_v2.joystick.Joystick` and reuses its
`_get_obs` (extended), `_get_reward`, `_get_termination`, `sample_command`.

`reset` and `step` are re-implemented because upstream's are monolithic. They are
line-for-line copies of upstream (Open_Duck_Playground b9be205, joystick.py
reset/step) except for the lines marked `# NERVA:`:
  1. the imitation reference is looked up for (command, style) in a StyledReference;
  2. the phase clock uses that style's gait period;
  3. the style vector is appended to both observations;
  4. a style is sampled at reset and resampled whenever the command is resampled.
Optional (S2, off by default): a per-style feet-height cost, `feet_height_scale`,
added in `_get_reward` (see FeetHeight below). With scale 0 nothing changes.
Optional (S4, off by default): `backward_fraction` replaces the forward-velocity command with a
backward one (uniform in BACKWARD_RANGE) with that probability, drawn from a derived key, because
every policy walks backward at only ~25% of the command (2026-10-01).
Optional (S3, off by default): `apply_head_commands` adds the sampled head commands to the head
motor targets after the speed limit, exactly as the hardware runtime does
(Open_Duck_Mini_Runtime v2_rl_walk_mujoco.py); upstream training leaves them in the observation
only (joystick.py, commented-out line in step). Without it, policies stop walking when the head
is moved at run time (measured 2026-09-30: 0.2 rad of head yaw → 0.012 m/s).
Style randomness is drawn from keys DERIVED with jax.random.fold_in, so the
upstream random stream is consumed exactly as upstream consumes it. With a single
neutral style built from upstream's reference file, this env therefore reproduces
upstream transitions exactly (tests/test_style_joystick.py).

Requires jax, mujoco_playground and the Open Duck `playground` package (training
environment only). Upstream's _post_init loads its own reference file from a path
relative to the Open_Duck_Playground directory, so run from there.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jp
from jax import vmap
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.open_duck_mini_v2 import joystick as upstream

from nerva.style import s1_foot_height

STYLE_KEY_SALT = 0x5717E  # fold_in constant: style keys never consume upstream randomness
BACKWARD_KEY_SALT = 0xBAC4  # fold_in constant for the S4 backward-emphasis draw
BACKWARD_RANGE = (-0.15, -0.05)  # m/s, forward-velocity command used for backward-emphasis draws

# S2 feet-height cost (docs/development_log.md, 2026-09-30). Same form as MuJoCo
# Playground's Berkeley Humanoid _cost_feet_height, with a per-style target:
#   Σ_feet ((swing_peak − STANCE_FOOT_SITE_Z) / walk_foot_height(e2) − 1)² · first_contact
STANCE_FOOT_SITE_Z = 0.003  # m; median foot-site z in stance, measured in MuJoCo (B0 policy, 2026-09-30)


class StyledReference:
    """Several upstream reference sets (one per style) stacked for jit-friendly lookup.

    Each pickle is parsed by upstream's own PolyReferenceMotion. All must have the
    same grid shape (same number of dx, dy, dtheta values and polynomial size);
    the velocity VALUES and the gait period may differ per style.
    """

    def __init__(self, pickle_paths: Sequence[str], styles: Sequence[Sequence[float]]):
        if len(pickle_paths) != len(styles) or not pickle_paths:
            raise ValueError("need one pickle per style, at least one")
        prms = [PolyReferenceMotion(p) for p in pickle_paths]
        shape = prms[0].data_array.shape
        for p, path in zip(prms, pickle_paths):
            if p.data_array.shape != shape:
                raise ValueError(f"{path}: grid shape {p.data_array.shape} != {shape}")
        self.n_styles = len(prms)
        self.data = jp.stack([p.data_array for p in prms])  # (S, nx, ny, nth, dims, coeffs)
        self.dxs = jp.array([p.dxs for p in prms])
        self.dys = jp.array([p.dys for p in prms])
        self.dthetas = jp.array([p.dthetas for p in prms])
        self.ranges = jp.array([[p.dx_range, p.dy_range, p.dtheta_range] for p in prms])  # (S, 3, 2)
        self.nb_steps = jp.array([p.nb_steps_in_period for p in prms], dtype=jp.int32)
        self.periods = [p.period for p in prms]
        self.styles = jp.array(styles, dtype=jp.float32)  # (S, style_dim)
        self.foot_heights = jp.array([s1_foot_height(float(e[1])) for e in styles], dtype=jp.float32)
        self.style_dim = int(self.styles.shape[1])

    def nb_steps_in_period(self, s):
        return self.nb_steps[s]

    def get_reference_motion(self, dx, dy, dtheta, i, s):
        """Same computation as upstream PolyReferenceMotion.get_reference_motion, for style s."""
        r = self.ranges[s]
        dx = jp.clip(dx, r[0, 0], r[0, 1])
        dy = jp.clip(dy, r[1, 0], r[1, 1])
        dtheta = jp.clip(dtheta, r[2, 0], r[2, 1])
        ix = jp.argmin(jp.abs(self.dxs[s] - dx))
        iy = jp.argmin(jp.abs(self.dys[s] - dy))
        itheta = jp.argmin(jp.abs(self.dthetas[s] - dtheta))
        nb = self.nb_steps[s]
        t = i % nb / nb
        t = jp.clip(t, 0.0, 1.0)
        coeffs = self.data[s, ix, iy, itheta]
        return vmap(lambda c: jp.polyval(c, t))(coeffs)


def feet_height_cost(swing_peak: jax.Array, first_contact: jax.Array, target: jax.Array) -> jax.Array:
    """Squared relative error of each foot's swing peak lift, counted on touchdown."""
    error = (swing_peak - STANCE_FOOT_SITE_Z) / target - 1.0
    return jp.sum(jp.square(error) * first_contact)


class StyleJoystick(upstream.Joystick):
    """Joystick env whose policy observes a style vector and imitates that style's reference."""

    def __init__(self, reference: StyledReference, task: str = "flat_terrain",
                 config=None, config_overrides=None, feet_height_scale: float = 0.0,
                 apply_head_commands: bool = False, backward_fraction: float = 0.0):
        self.SREF = reference
        config = config or upstream.default_config()
        self.feet_height_scale = float(feet_height_scale)
        self.apply_head_commands = bool(apply_head_commands)
        self.backward_fraction = float(backward_fraction)
        if self.feet_height_scale != 0.0:  # NERVA S2; absent from the reward otherwise
            config.reward_config.scales.feet_height = self.feet_height_scale
        super().__init__(task=task, config=config, config_overrides=config_overrides)

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        if self.feet_height_scale != 0.0:
            target = self.SREF.foot_heights[info["style_idx"]]
            rewards["feet_height"] = feet_height_cost(info["swing_peak"], first_contact, target)
        return rewards

    # ── style helpers ────────────────────────────────────────────────────────

    def sample_command(self, rng: jax.Array) -> jax.Array:
        cmd = super().sample_command(rng)  # consumes rng exactly as upstream
        if self.backward_fraction <= 0.0:
            return cmd
        k1, k2 = jax.random.split(jax.random.fold_in(rng, BACKWARD_KEY_SALT))
        backward = jax.random.uniform(k2, minval=BACKWARD_RANGE[0], maxval=BACKWARD_RANGE[1])
        return cmd.at[0].set(jp.where(jax.random.bernoulli(k1, self.backward_fraction), backward, cmd[0]))

    def _sample_style_idx(self, key: jax.Array) -> jax.Array:
        return jax.random.randint(key, (), 0, self.SREF.n_styles)

    def _style_key(self, rng: jax.Array) -> jax.Array:
        return jax.random.fold_in(rng, STYLE_KEY_SALT)

    # ── observation: upstream + style ────────────────────────────────────────

    def _get_obs(self, data, info: dict[str, Any], contact):
        obs = super()._get_obs(data, info, contact)
        style = info["style"]  # NERVA
        return {"state": jp.hstack([obs["state"], style]),
                "privileged_state": jp.hstack([obs["privileged_state"], style])}

    # ── reset: upstream copy + NERVA lines ───────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        style_idx = self._sample_style_idx(self._style_key(rng))  # NERVA (derived key)

        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2] + dxy
        )
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7], quat
        )
        base_qpos = base_qpos.at[3:7].set(new_quat)
        qpos = self.set_floating_base_qpos(base_qpos, qpos)

        rng, key = jax.random.split(rng)
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key, (self._actuators,), minval=0.5, maxval=1.5
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)

        rng, key = jax.random.split(rng)
        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05), qvel
        )
        ctrl = self.get_actuator_joints_qpos(qpos)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        current_reference_motion = self.SREF.get_reference_motion(
            cmd[0], cmd[1], cmd[2], 0, style_idx
        )  # NERVA

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            "action_history": jp.zeros(self._config.noise_config.action_max_delay * self._actuators),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            "imitation_i": 0,
            "current_reference_motion": current_reference_motion,
            "imitation_phase": jp.zeros(2),
            "style_idx": style_idx,  # NERVA
            "style": self.SREF.styles[style_idx],  # NERVA
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id)
                            for geom_id in self._feet_geom_id])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ── step: upstream copy + NERVA lines ────────────────────────────────────

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        s_idx = state.info["style_idx"]  # NERVA
        nb = self.SREF.nb_steps_in_period(s_idx)  # NERVA: per-style gait period

        state.info["imitation_i"] += 1
        state.info["imitation_i"] = state.info["imitation_i"] % nb
        state.info["imitation_phase"] = jp.array([
            jp.cos((state.info["imitation_i"] / nb) * 2 * jp.pi),
            jp.sin((state.info["imitation_i"] / nb) * 2 * jp.pi),
        ])
        state.info["current_reference_motion"] = self.SREF.get_reference_motion(
            state.info["command"][0], state.info["command"][1], state.info["command"][2],
            state.info["imitation_i"], s_idx,
        )  # NERVA

        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(state.info["rng"], 4)

        action_history = (
            jp.roll(state.info["action_history"], self._actuators).at[: self._actuators].set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng, (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._actuators))[action_idx[0]]

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2].set(
            push * push_magnitude + qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
        )
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        motor_targets = self._default_actuator + action_w_delay * self._config.action_scale
        if upstream.USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]
            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets - self._config.max_motor_velocity * self.dt,
                prev_motor_targets + self._config.max_motor_velocity * self.dt,
            )

        applied = motor_targets
        if self.apply_head_commands:  # NERVA S3: head offsets after the speed limit, as on hardware
            applied = motor_targets.at[5:9].add(state.info["command"][3:7])
        data = mjx_env.step(self.mjx_model, state.data, applied, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id)
                            for geom_id in self._feet_geom_id])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, state.info, state.metrics, done, first_contact, contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        resample = state.info["step"] > 500
        state.info["command"] = jp.where(resample, self.sample_command(cmd_rng), state.info["command"])
        new_style_idx = self._sample_style_idx(self._style_key(cmd_rng))  # NERVA (derived key)
        state.info["style_idx"] = jp.where(resample, new_style_idx, s_idx)  # NERVA
        state.info["style"] = self.SREF.styles[state.info["style_idx"]]  # NERVA
        state.info["step"] = jp.where(done | (state.info["step"] > 500), 0, state.info["step"])
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v
                else:
                    state.metrics[f"cost/{k}"] = -v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)
