"""Tests for the style-conditioned training env (experiment S1, docs/style_policy_design.md)."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playground.common.poly_reference_motion")
jax = pytest.importorskip("jax")
import jax.numpy as jp  # noqa: E402

from nerva.training.style_joystick import StyledReference  # noqa: E402

N_DIMS, DEG = 3, 2


def _write_pickle(path, dxs, period, value_of):
    """Synthetic reference file in upstream's format; dim d of gait dx is value_of(dx, d) + slope·t."""
    data = {}
    for dx in dxs:
        coeffs = {f"dim_{d}": [value_of(dx, d), 1.0 if d == 0 else 0.0] + [0.0] * (DEG - 1)
                  for d in range(N_DIMS)}  # lowest order first, as fit_poly stores them
        data[f"{dx}_0.0_0.0"] = {"coefficients": coeffs, "period": period, "fps": 50,
                                 "frame_offsets": {}, "startend_double_support_ratio": 1.0}
    with open(path, "wb") as f:
        pickle.dump(data, f)


@pytest.fixture
def two_styles(tmp_path):
    a, b = tmp_path / "a.pkl", tmp_path / "b.pkl"
    _write_pickle(a, [-0.1, 0.0, 0.1], 0.54, lambda dx, d: 100 + 10 * dx + d)  # 27 steps
    _write_pickle(b, [-0.2, 0.0, 0.2], 0.40, lambda dx, d: 200 + 10 * dx + d)  # 20 steps
    return StyledReference([str(a), str(b)], [[0, 0, 0], [1, 0, -1]])


def test_per_style_period_and_style_vectors(two_styles):
    assert two_styles.n_styles == 2 and two_styles.style_dim == 3
    assert [int(x) for x in two_styles.nb_steps] == [27, 20]
    np.testing.assert_array_equal(np.asarray(two_styles.styles[1]), [1, 0, -1])


def test_lookup_uses_the_style_grid_and_phase(two_styles):
    # style 0, command 0.1 → gait dx=0.1, t = 5/27; dim 0 = 100 + 1 + t
    r = two_styles.get_reference_motion(0.1, 0.0, 0.0, 5, 0)
    assert float(r[0]) == pytest.approx(101 + 5 / 27, rel=1e-5)
    assert float(r[2]) == pytest.approx(103, rel=1e-5)
    # style 1 has its own grid {-0.2, 0, 0.2}: command 0.15 → nearest gait 0.2, t = 5/20
    r1 = two_styles.get_reference_motion(0.15, 0.0, 0.0, 5, 1)
    assert float(r1[0]) == pytest.approx(202 + 5 / 20, rel=1e-5)


def test_phase_wraps_with_the_style_period(two_styles):
    r = two_styles.get_reference_motion(0.0, 0.0, 0.0, 23, 1)  # 23 % 20 = 3
    assert float(r[0]) == pytest.approx(200 + 3 / 20, rel=1e-5)


def test_commands_are_clipped_to_the_style_range(two_styles):
    r = two_styles.get_reference_motion(5.0, 0.0, 0.0, 0, 0)  # clipped to 0.1
    assert float(r[0]) == pytest.approx(101, rel=1e-5)


def test_mismatched_grids_are_rejected(tmp_path):
    a, b = tmp_path / "a.pkl", tmp_path / "b.pkl"
    _write_pickle(a, [-0.1, 0.0, 0.1], 0.54, lambda dx, d: 0.0)
    _write_pickle(b, [0.0, 0.1], 0.54, lambda dx, d: 0.0)
    with pytest.raises(ValueError):
        StyledReference([str(a), str(b)], [[0, 0, 0], [1, 0, 0]])


# ── slow: full env on CPU MJX ────────────────────────────────────────────────

PLAYGROUND = Path(os.environ.get("OPEN_DUCK_ROOT", Path.home() / "dev" / "open_duck")) / "Open_Duck_Playground"
UPSTREAM_PKL = str(PLAYGROUND / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
N_STEPS = 8


@pytest.fixture
def in_playground(monkeypatch):
    monkeypatch.chdir(PLAYGROUND)  # upstream _post_init loads its reference by relative path


@pytest.mark.slow
def test_single_neutral_style_reproduces_upstream_exactly(in_playground):
    from playground.open_duck_mini_v2 import joystick
    from nerva.training.style_joystick import StyleJoystick

    up = joystick.Joystick(task="flat_terrain")
    ours = StyleJoystick(StyledReference([UPSTREAM_PKL], [[0.0, 0.0, 0.0]]), task="flat_terrain")
    key = jax.random.PRNGKey(7)
    s_up, s_ours = jax.jit(up.reset)(key), jax.jit(ours.reset)(key)
    step_up, step_ours = jax.jit(up.step), jax.jit(ours.step)
    actions = jax.random.uniform(jax.random.PRNGKey(1), (N_STEPS, up.action_size), minval=-1, maxval=1)
    for k in range(N_STEPS + 1):
        np.testing.assert_array_equal(np.asarray(s_ours.data.qpos), np.asarray(s_up.data.qpos))
        np.testing.assert_array_equal(np.asarray(s_ours.obs["state"][:101]), np.asarray(s_up.obs["state"]))
        np.testing.assert_array_equal(np.asarray(s_ours.obs["privileged_state"][:212]),
                                      np.asarray(s_up.obs["privileged_state"]))
        np.testing.assert_array_equal(np.asarray(s_ours.obs["state"][101:]), [0.0, 0.0, 0.0])
        assert float(s_ours.reward) == float(s_up.reward)
        np.testing.assert_array_equal(np.asarray(s_ours.info["current_reference_motion"]),
                                      np.asarray(s_up.info["current_reference_motion"]))
        if k < N_STEPS:
            s_up, s_ours = step_up(s_up, actions[k]), step_ours(s_ours, actions[k])
    assert float(jp.abs(s_up.data.qpos[:3] - up._init_q[:3]).sum()) > 0  # something actually happened


@pytest.mark.slow
def test_style_is_observed_and_resampled_with_the_command(in_playground):
    from nerva.training.style_joystick import StyleJoystick

    styles = [[0.0, 0.0, 0.0], [1.0, -1.0, 0.5], [-1.0, 1.0, -0.5]]
    env = StyleJoystick(StyledReference([UPSTREAM_PKL] * 3, styles), task="flat_terrain")
    assert env.observation_size["state"] == (104,)
    s = jax.jit(env.reset)(jax.random.PRNGKey(3))
    idx = int(s.info["style_idx"])
    np.testing.assert_array_equal(np.asarray(s.obs["state"][101:]), styles[idx])
    step = jax.jit(env.step)
    seen = {idx}
    for _ in range(6):
        # Force a command resample. As upstream does for the COMMAND, the observation of this
        # step is built before resampling, so the new style (like the new command) shows up in
        # the observation of the NEXT step.
        s.info["step"] = jp.array(501)
        prev_cmd, prev_idx = np.asarray(s.info["command"]), int(s.info["style_idx"])
        s = step(s, jp.zeros(env.action_size))
        np.testing.assert_array_equal(np.asarray(s.obs["state"][101:]), styles[prev_idx])
        np.testing.assert_array_equal(np.asarray(s.obs["state"][6:13]), prev_cmd)
        idx = int(s.info["style_idx"])
        seen.add(idx)
        s = step(s, jp.zeros(env.action_size))  # no resample: step counter was reset to 0
        np.testing.assert_array_equal(np.asarray(s.obs["state"][101:]), styles[idx])
        np.testing.assert_array_equal(np.asarray(s.obs["state"][6:13]), np.asarray(s.info["command"]))
    assert len(seen) > 1  # resampling actually changes the style


def test_foot_height_targets_follow_the_r1_mapping(tmp_path):
    a = tmp_path / "a.pkl"
    _write_pickle(a, [0.0, 0.1], 0.54, lambda dx, d: 0.0)
    ref = StyledReference([str(a)] * 3, [[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    np.testing.assert_allclose(np.asarray(ref.foot_heights), [0.04, 0.02, 0.06], rtol=1e-6)


def test_feet_height_cost_counts_touchdowns_only():
    from nerva.training.style_joystick import STANCE_FOOT_SITE_Z, feet_height_cost

    target = jp.float32(0.04)
    on_target = jp.array([STANCE_FOOT_SITE_Z + 0.04, STANCE_FOOT_SITE_Z + 0.02])
    # left foot lands exactly at target (0 cost), right lands at half (0.25); right only if it touched down
    assert float(feet_height_cost(on_target, jp.array([1.0, 1.0]), target)) == pytest.approx(0.25, rel=1e-5)
    assert float(feet_height_cost(on_target, jp.array([1.0, 0.0]), target)) == pytest.approx(0.0, abs=1e-6)


def test_feet_height_scale_can_be_added_to_upstream_config():
    from playground.open_duck_mini_v2 import joystick as upstream

    config = upstream.default_config()
    config.reward_config.scales.feet_height = -30.0
    assert config.reward_config.scales.feet_height == -30.0
    assert "feet_height" not in upstream.default_config().reward_config.scales


def test_backward_emphasis_keeps_upstream_draws_and_forces_backward_commands():
    from playground.open_duck_mini_v2 import joystick as upstream

    from nerva.training.style_joystick import BACKWARD_RANGE, StyleJoystick

    class Probe(StyleJoystick):  # only sample_command is exercised; no model is built
        def __init__(self, fraction):
            self.backward_fraction = fraction
            self._config = upstream.default_config()

    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    plain = [Probe(0.0).sample_command(k) for k in keys]
    emph = [Probe(1.0).sample_command(k) for k in keys]
    for p, e in zip(plain, emph):
        np.testing.assert_array_equal(np.asarray(p[1:]), np.asarray(e[1:]))  # other commands untouched
        assert BACKWARD_RANGE[0] <= float(e[0]) <= BACKWARD_RANGE[1]


def test_feet_air_time_rewards_steps_only_when_moving():
    from nerva.training.style_joystick import feet_air_time_reward

    air = jp.array([0.2, 0.05])
    landed = jp.array([1.0, 1.0])
    moving, still = jp.array([0.1, 0, 0, 0, 0, 0, 0]), jp.zeros(7)
    assert float(feet_air_time_reward(air, landed, moving)) == pytest.approx(0.1 - 0.05, abs=1e-6)  # short hop penalised
    assert float(feet_air_time_reward(air, landed, still)) == 0.0
    assert float(feet_air_time_reward(jp.array([0.9, 0.0]), jp.array([1.0, 0.0]), moving)) == pytest.approx(0.2)
