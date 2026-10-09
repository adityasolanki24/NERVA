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

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        del metrics, done, first_contact
        joints = self.get_actuator_joints_qpos(data.qpos)
        velocity = self.get_actuator_joints_qvel(data.qvel)
        linear, angular = self.get_local_linvel(data), self.get_gyro(data)
        command = info["command"]
        return {"tracking_lin_vel": planar_tracking(command, linear, self._config.reward_config.tracking_sigma, jp),
                "tracking_ang_vel": upstream.reward_tracking_ang_vel(command, angular, self._config.reward_config.tracking_sigma),
                "torques": upstream.cost_torques(data.actuator_force),
                "action_rate": upstream.cost_action_rate(action, info["last_act"]),
                "alive": upstream.reward_alive(),
                "imitation": body_imitation(joints, velocity, linear, angular, contact,
                                             info["current_reference_motion"], command),
                "stand_still": jp.where(at_rest(command, jp),
                                       jp.sum(jp.abs(joints - self.neutral_pose)) + jp.sum(jp.abs(velocity)), 0.)}
