"""Opt-in neutral motor candidate, no affect/style or historical policy adoption."""
import jax
import jax.numpy as jp
from playground.open_duck_mini_v2 import joystick as upstream

from nerva.motor_contract import at_rest, planar_tracking, training_motor_tick
from nerva.training.neutral_reference import CONTRACT
from nerva.training.style_joystick import StyleJoystick


def body_imitation(joints, joint_velocity, linear_body, angular_body, contact, target, command):
    leg = jp.concatenate([joints[:5], joints[9:]])
    leg_velocity = jp.concatenate([joint_velocity[:5], joint_velocity[9:]])
    ref_leg = jp.concatenate([target[:5], target[11:16]])
    ref_velocity = jp.concatenate([target[16:21], target[27:32]])
    linear_error = linear_body - target[34:37]
    angular_error = angular_body - target[37:40]
    reward = (jp.exp(-8 * jp.sum(linear_error[:2] ** 2)) + jp.exp(-8 * linear_error[2] ** 2)
              + .5 * jp.exp(-2 * jp.sum(angular_error[:2] ** 2)) + .5 * jp.exp(-2 * angular_error[2] ** 2)
              - 15 * jp.sum((leg - ref_leg) ** 2) - .001 * jp.sum((leg_velocity - ref_velocity) ** 2)
              + jp.sum(contact == (target[32:34] > .5)))
    return jp.where(at_rest(command, jp), 0., reward)


class NeutralJoystick(StyleJoystick):
    contract = CONTRACT

    def __init__(self, reference, task="flat_terrain"):
        config = upstream.default_config()
        super().__init__(reference, task=task, config=config)
        static = reference.get_reference_motion(0., 0., 0., 0)
        self.neutral_pose = jp.concatenate([static[:9], static[11:16]])

    def sample_command(self, rng):
        command = self.SREF.commands[jax.random.randint(rng, (), 0, 7)]
        return jp.concatenate([command, jp.zeros(4)])

    def _get_obs(self, data, info, contact):
        return upstream.Joystick._get_obs(self, data, info, contact)

    def _initialize_reference(self, info):
        info.update(motor_phase_index=jp.int32(0), motor_previous_rest=jp.bool_(True),
                    imitation_phase=jp.array([1., 0.]))

    def _advance_reference(self, info):
        info.update(training_motor_tick(info, info["command"], 27))
        info["imitation_i"] = info["motor_phase_index"]
        info["current_reference_motion"] = self.SREF.get_reference_motion(
            info["command"][0], info["command"][1], info["command"][2], info["imitation_i"])

    def reward_linvel(self, data):
        """Body-frame linear velocity used by the tracking and imitation rewards (upstream: the IMU site)."""
        return self.get_local_linvel(data)

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        del metrics, done, first_contact
        joints = self.get_actuator_joints_qpos(data.qpos)
        velocity = self.get_actuator_joints_qvel(data.qvel)
        linear, angular = self.reward_linvel(data), self.get_gyro(data)
        command = info["command"]
        return {"tracking_lin_vel": planar_tracking(command, linear, self._config.reward_config.tracking_sigma, jp),
                "tracking_ang_vel": upstream.reward_tracking_ang_vel(command, angular, self._config.reward_config.tracking_sigma),
                "torques": upstream.cost_torques(data.actuator_force),
                "action_rate": upstream.cost_action_rate(action, info["last_act"]),
                "alive": upstream.reward_alive(),
                "imitation": body_imitation(joints, velocity, linear, angular, contact,
                                             info["current_reference_motion"], command),
                "stand_still": jp.where(at_rest(command, jp),
                                       jp.sum(jp.abs(joints - self.rest_pose(info))) + jp.sum(jp.abs(velocity)), 0.)}

    def rest_pose(self, info):
        """Actuator-order rest target of `stand_still` (the static neutral reference)."""
        del info
        return self.neutral_pose


class PersistentCommand:
    """Mixin: keep each environment's command across upstream's resampling boundary.

    Upstream (and StyleJoystick) resample the command once `step > 500`. Balanced curricula with episodes
    longer than 500 control steps would silently change their declared per-environment command there
    (docs/gpu_neutral_pilot.md). The command at the start of a step is restored afterwards; resets still
    set it through the balanced reset wrapper.
    """

    def step(self, state, action):
        command = state.info["command"]
        state = super().step(state, action)
        state.info["command"] = command
        return state


class PersistentNeutralJoystick(PersistentCommand, NeutralJoystick):
    """NeutralJoystick whose per-environment command never changes within an episode.

    Also gives the "alive" reward a strong float32 type: upstream's weakly typed scalar changes the
    state's abstract signature after the first step, forcing one extra compilation of the collector
    (values are identical; measured 2026-10-10)."""

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        rewards["alive"] = jp.asarray(rewards["alive"], dtype=jp.float32)
        return rewards


TRACKING_WINDOW = 27  # one gait period (0.54 s at 50 Hz)


def window_mean(window, count):
    """Mean of the newest `count` rows of a rolling window (newest first)."""
    rows = jp.arange(window.shape[0]) < count
    return jp.sum(window * rows[:, None], axis=0) / jp.maximum(count, 1)


class GaitAveragedTrackingNeutralJoystick(PersistentNeutralJoystick):
    """Tracking rewards on body velocity averaged over one gait period instead of instantaneous velocity.

    At the neutral 0.074 m/s commands the gait's own within-stride sway (≈0.15 m/s RMS) dominates an
    instantaneous comparison: the upstream-form tracking reward pays a robot standing still 1.45 but one
    following its reference gait perfectly only 0.77; on the period average the reference scores 2.49
    (development log 2026-10-10). Everything else (imitation, costs, alive, noise, pushes) is unchanged.
    """

    def _initialize_reference(self, info):
        super()._initialize_reference(info)
        info.update(velocity_window=jp.zeros((TRACKING_WINDOW, 3)), velocity_count=jp.int32(0))

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        current = jp.concatenate([self.reward_linvel(data)[:2], self.get_gyro(data)[2:3]])
        window = jp.roll(info["velocity_window"], 1, axis=0).at[0].set(current)
        count = jp.minimum(info["velocity_count"] + 1, TRACKING_WINDOW)
        info["velocity_window"], info["velocity_count"] = window, count
        mean = window_mean(window, count)
        sigma = self._config.reward_config.tracking_sigma
        rewards["tracking_lin_vel"] = planar_tracking(info["command"], mean[:2], sigma, jp)
        rewards["tracking_ang_vel"] = upstream.reward_tracking_ang_vel(info["command"], jp.zeros(3).at[2].set(mean[2]), sigma)
        return rewards


def imu_offset(model):
    """IMU site position in the base frame; the site must sit on the floating base with identity orientation."""
    import mujoco
    import numpy as np
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    if model.body(model.site_bodyid[site]).name != "base" or not np.allclose(model.site_quat[site], (1, 0, 0, 0)):
        raise ValueError("IMU site is not an unrotated site on the floating base")
    return np.asarray(model.site_pos[site], dtype=np.float32)


def base_origin_velocity(imu_linvel, gyro, offset):
    """Rigid-body shift of the IMU velocimeter reading to the base origin (both in the base frame)."""
    return imu_linvel - jp.cross(gyro, offset)


class BaseOriginVelocity:
    """Mixin: tracking and imitation rewards use body velocity at the base origin instead of the IMU site.

    The IMU sits 8 cm behind the base origin. The verified references and the native evaluation measure
    the base origin; upstream's rewards measure the IMU. Turning in place about the base therefore reads as
    0.6 rad/s × 0.08 m ≈ 0.048 m/s of IMU translation in training, and the trained turns pivot near the IMU
    instead (results_turn_pivot/; development log 2026-10-10). Observations, the critic's privileged inputs,
    costs, alive, noise and pushes are unchanged.
    """

    def reward_linvel(self, data):
        if not hasattr(self, "_imu_offset"):
            self._imu_offset = imu_offset(self._mj_model)  # NumPy constant: never cache a traced value
        return base_origin_velocity(self.get_local_linvel(data), self.get_gyro(data), self._imu_offset)


class BaseOriginGaitAveragedNeutralJoystick(BaseOriginVelocity, GaitAveragedTrackingNeutralJoystick):
    """Gait-averaged tracking with rewards measured at the base origin (docs/base_origin_velocity_pilot.md)."""


TURN_TRACKING_SIGMA = 0.0025  # pure-turn commands only; upstream tracking_sigma is 0.01


def is_pure_turn(command):
    return jp.logical_and(jp.all(command[:2] == 0), command[2] != 0)


class TurnTranslationTracking:
    """Mixin: for pure-turn commands, the (gait-averaged, base-origin) linear tracking term uses a 4× tighter
    width, 0.0025 instead of 0.01 (tolerance 0.05 instead of 0.1 m/s).

    With upstream's width, 0.03 m/s of turn translation costs only 9% of the term; the trained turns sit at
    0.022–0.030 m/s in MJX, and native MuJoCo adds ≈ 0.01 m/s to the left turn (results_turn_asymmetry/,
    results_neutral_gate/). At 0.03 m/s the tighter width costs 30%. Translation and rest commands, every other
    term and every scale are unchanged (docs/turn_translation_pilot.md).
    """

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        mean = window_mean(info["velocity_window"], info["velocity_count"])
        command = info["command"]
        tight = planar_tracking(command, mean[:2], TURN_TRACKING_SIGMA, jp)
        rewards["tracking_lin_vel"] = jp.where(is_pure_turn(command), tight, rewards["tracking_lin_vel"])
        return rewards


class TurnTranslationNeutralJoystick(TurnTranslationTracking, BaseOriginGaitAveragedNeutralJoystick):
    """Base-origin gait-averaged tracking with a tighter pure-turn translation width."""


STYLE_DIM = 2  # E1′: (e_pitch ∈ [−1, 1], e_crouch ∈ [0, 1]); docs/expressive_posture_e1prime.md


class StyleConditioning:
    """Mixin (E1′): e = info["e1_style"] is appended to the policy and privileged observations, selects the imitation
    target by bilinear interpolation of the styled grid, and sets the `stand_still` rest pose. Requires a
    `StyledNeutralReference`; e = (0, 0) reproduces the neutral targets exactly. Sampling and the curriculum live in
    `nerva.training.style_curriculum.StyleCurriculumWrapper`."""

    def _initialize_reference(self, info):
        super()._initialize_reference(info)
        if "e1_style" not in info:
            info["e1_style"] = jp.zeros(STYLE_DIM)

    def _advance_reference(self, info):
        super()._advance_reference(info)
        info["current_reference_motion"] = self.SREF.get_styled_motion(
            info["command"][0], info["command"][1], info["command"][2], info["imitation_i"], info["e1_style"])

    def rest_pose(self, info):
        static = self.SREF.get_styled_motion(0., 0., 0., 0, info["e1_style"])
        return jp.concatenate([static[:9], static[11:16]])

    def _get_obs(self, data, info, contact):
        obs = super()._get_obs(data, info, contact)
        style = jp.asarray(info["e1_style"], dtype=obs["state"].dtype)
        return {"state": jp.concatenate([obs["state"], style]),
                "privileged_state": jp.concatenate([obs["privileged_state"], style])}


class StyledTurnTranslationNeutralJoystick(StyleConditioning, TurnTranslationNeutralJoystick):
    """E1′ environment: the validated turn-translation reward with continuous posture conditioning."""
