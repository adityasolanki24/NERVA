"""Checks that NERVA's simulation loop is the upstream baseline, not an approximation.

Skipped automatically when the Open Duck environment is not installed.
"""

import time

import numpy as np
import pytest

pytest.importorskip("playground.open_duck_mini_v2.mujoco_infer")

import mujoco.viewer  # noqa: E402
from playground.open_duck_mini_v2 import mujoco_infer  # noqa: E402

from nerva.interfaces import BehaviourCommand, ExpressiveStyle  # noqa: E402
from nerva.sim.open_duck import POLICY, REFERENCE, SCENE, OpenDuckSim  # noqa: E402

N_PHYSICS_STEPS = 1500  # 3 s: enough for the walk to start and diverge if anything differs


class _StopAfter:
    def __init__(self, n):
        self.n, self.i = n, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sync(self):
        self.i += 1
        if self.i >= self.n:
            raise KeyboardInterrupt


def _upstream_qpos_after(n, command, phase_factor, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # upstream writes mujoco_saved_obs.pkl to cwd
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *a, **k: _StopAfter(n))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    m = mujoco_infer.MjInfer(str(SCENE), str(REFERENCE), str(POLICY), standing=False)
    m.commands = list(command) + [0.0] * 4
    m.phase_frequency_factor = phase_factor
    m.run()
    return m.data.qpos.copy()


@pytest.mark.parametrize("style", [0.0, 1.0])
def test_loop_reproduces_upstream_run_exactly(style, monkeypatch, tmp_path):
    cmd = BehaviourCommand(vx=0.15, style=ExpressiveStyle(style))
    sim = OpenDuckSim(raw_accel=False)  # upstream behaviour, including its +1.3 offset
    sim.set_behaviour(cmd)
    sim.step_physics(N_PHYSICS_STEPS)

    expected = _upstream_qpos_after(N_PHYSICS_STEPS, sim.applied_command, sim.phase_factor,
                                     monkeypatch, tmp_path)
    assert sim.data.qpos[0] > 0.1  # the robot really walked forward, so the check is not vacuous
    np.testing.assert_array_equal(sim.data.qpos, expected)


def test_comparison_detects_a_small_difference(monkeypatch, tmp_path):
    """Negative control: a 1% different clock must NOT reproduce the upstream trajectory."""
    sim = OpenDuckSim(raw_accel=False)
    sim.set_behaviour(BehaviourCommand(vx=0.15))
    sim.step_physics(N_PHYSICS_STEPS)
    other = _upstream_qpos_after(N_PHYSICS_STEPS, sim.applied_command, 1.01, monkeypatch, tmp_path)
    assert not np.array_equal(sim.data.qpos, other)


def test_raw_accel_removes_exactly_the_upstream_offset():
    a = OpenDuckSim(raw_accel=False)
    b = OpenDuckSim(raw_accel=True)
    obs_a = a.inf.get_obs(a.data, a.inf.commands)
    obs_b = b.inf.get_obs(b.data, b.inf.commands)
    diff = obs_a - obs_b
    assert diff[3] == pytest.approx(1.3)
    assert np.all(np.delete(diff, 3) == 0)


def test_behaviour_command_is_clipped_to_trained_range_and_sets_clock():
    sim = OpenDuckSim()
    sim.set_behaviour(BehaviourCommand(vx=0.5, vy=-1.0, yaw_rate=3.0, style=ExpressiveStyle(-1.0)))
    assert sim.applied_command == (0.15, -0.2, 1.0)
    assert sim.phase_factor == pytest.approx(0.7)


def test_seeded_initial_noise_is_reproducible_and_seed_dependent():
    q1 = OpenDuckSim(init_joint_noise=0.02, seed=3).data.qpos.copy()
    q2 = OpenDuckSim(init_joint_noise=0.02, seed=3).data.qpos.copy()
    q3 = OpenDuckSim(init_joint_noise=0.02, seed=4).data.qpos.copy()
    assert np.array_equal(q1, q2)
    assert not np.array_equal(q1, q3)


def test_obs_noise_scale_matches_training_env_assignment():
    """Recompute the joint-angle noise exactly as joystick.py._post_init does."""
    from playground.open_duck_mini_v2 import constants
    from nerva.sim.open_duck import training_obs_noise_scale

    expected = np.zeros(14)  # joystick.py: np.zeros(self._actuators), actuators = 14
    order = constants.JOINTS_ORDER_NO_HEAD
    expected[[i for i, j in enumerate(order) if "_hip" in j]] = 0.03
    expected[[i for i, j in enumerate(order) if "_knee" in j]] = 0.05
    expected[[i for i, j in enumerate(order) if "_ankle" in j]] = 0.08
    np.testing.assert_array_equal(training_obs_noise_scale()[13:27], expected)


def test_obs_noise_is_seeded():
    def x_after(seed):
        sim = OpenDuckSim(obs_noise=True, seed=seed)
        sim.set_behaviour(BehaviourCommand(vx=0.15))
        sim.step_physics(500)
        return sim.data.qpos.copy()

    assert np.array_equal(x_after(1), x_after(1))
    assert not np.array_equal(x_after(1), x_after(2))


def test_head_offset_reaches_observation_and_head_targets_only():
    base = OpenDuckSim()
    base.set_behaviour(BehaviourCommand(vx=0.15))
    head = OpenDuckSim()
    head.set_behaviour(BehaviourCommand(vx=0.15))
    head.set_head_offset(head_pitch=0.3)
    assert head.inf.commands[4] == 0.3 and base.inf.commands[4] == 0.0
    base.step_physics(10)
    head.step_physics(10)  # first control step: same state, so policy leg outputs differ only via obs
    assert head.data.ctrl[6] - base.data.ctrl[6] == pytest.approx(0.3, abs=0.12)
    # set_behaviour after set_head_offset keeps the offset
    head.set_behaviour(BehaviourCommand(vx=0.1))
    assert head.inf.commands[4] == 0.3


def test_style_vector_is_appended_to_obs_and_sets_gait_period():
    sim = OpenDuckSim()
    sim.set_behaviour(BehaviourCommand(vx=0.15))
    assert sim.nb_steps_in_period == sim.inf.PRM.nb_steps_in_period == 27
    seen = []
    sim.inf.policy.infer = lambda obs: (seen.append(obs.copy()), np.zeros(14))[1]
    sim.set_style_vector((1.0, -0.5, 0.25))
    assert sim.nb_steps_in_period == 20  # 0.405 s gait at 50 Hz, as the R1 e1+ reference
    sim.step_physics(10)
    assert seen[0].shape == (104,)
    np.testing.assert_array_equal(seen[0][-3:], [1.0, -0.5, 0.25])
    with pytest.raises(ValueError):
        sim.set_style_vector((1.5, 0.0, 0.0))


def test_behaviour_style_vector_reaches_sim_and_keeps_phase_fraction():
    from nerva.interfaces import StyleVector

    sim = OpenDuckSim()
    sim.set_behaviour(BehaviourCommand(vx=0.15, style_vector=StyleVector(tempo=0.0)))
    sim.inf.imitation_i = 13.5  # half-way through the 27-step neutral cycle
    sim.set_behaviour(BehaviourCommand(vx=0.15, style_vector=StyleVector(tempo=1.0, torso_pitch=0.5)))
    assert sim.nb_steps_in_period == 20 and sim.inf.imitation_i == pytest.approx(10.0)
    np.testing.assert_array_equal(sim.style_vector, [1.0, 0.0, 0.5])


class _Scripted:
    """Policy stub returning 0, 1, 2, ... on successive calls (all 14 actions equal)."""

    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return np.full(14, float(self.calls - 1))


def test_action_delay_applies_a_recent_action_and_keeps_obs_history_undelayed():
    from nerva.sim.open_duck import ACTION_DELAY_STEPS
    sim = OpenDuckSim(seed=3, action_delay=True)
    sim.inf.policy = _Scripted()
    sim.inf.max_motor_velocity = 1e9  # isolate the delay from the speed limit
    delays = []
    for _ in range(60):
        sim.step_physics(sim.inf.decimation)
        latest = sim.inf.policy.calls - 1
        applied = (sim.inf.motor_targets[0] - sim.inf.default_actuator[0]) / sim.inf.action_scale
        delays.append(latest - round(applied) if latest >= 2 else None)
        assert sim.inf.last_action[0] == latest  # the observation history is the undelayed action
    seen = {d for d in delays if d is not None}
    assert seen == set(range(ACTION_DELAY_STEPS))


def test_action_delay_does_not_change_the_observation_noise_stream():
    a = OpenDuckSim(seed=5, obs_noise=True)
    b = OpenDuckSim(seed=5, obs_noise=True, action_delay=True)
    assert np.array_equal(a._rng.uniform(size=8), b._rng.uniform(size=8))
