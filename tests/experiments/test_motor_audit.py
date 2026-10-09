from types import SimpleNamespace

import numpy as np
import pytest

jp = pytest.importorskip("jax.numpy")
pytest.importorskip("playground.open_duck_mini_v2.joystick")

from experiments.locomotion_curriculum.motor_audit import (  # noqa: E402
    angular_probes, extract_angular_helper, make_probe, synthetic_rewards, tracking_probes,
)


def fixture_reward(vx=0., head=0., lift=.02, first_contact=(1, 0)):
    frame = jp.array([.1] * 32 + [1., 1.] + [0.] * 6)
    probe = make_probe(SimpleNamespace(foot_heights=jp.array([.04])))
    return synthetic_rewards(probe, frame, [vx, 0, 0, 0, 0, head, 0], lift, first_contact)


def test_zero_and_head_only_disable_imitation_with_active_stationary_cost():
    for head in (0., .4):
        r = fixture_reward(head=head)
        assert r["imitation"]["raw"] == 0
        assert r["stand_still"]["raw"] == pytest.approx(2.8)
    assert fixture_reward(.009)["imitation"]["raw"] == 0
    assert fixture_reward(.010)["imitation"]["raw"] == 0
    assert fixture_reward(.010)["stand_still"]["raw"] == 0  # exact boundary has neither term
    assert fixture_reward(.011)["imitation"]["raw"] > 0
    assert fixture_reward(.011)["stand_still"]["raw"] == 0


def test_height_is_touchdown_cost_not_an_incentive_to_start_stepping():
    for vx in (0., .15):
        assert fixture_reward(vx, lift=0., first_contact=(0, 0))["feet_height"]["raw"] == 0
        assert fixture_reward(vx, lift=.04)["feet_height"]["raw"] == pytest.approx(0, abs=1e-6)
        assert fixture_reward(vx, lift=.02)["feet_height"]["scaled"] == pytest.approx(-7.5)


def test_tracking_deadband_and_forward_loss_are_distinct():
    rows = tracking_probes()
    assert len(rows) == 108
    stop = [r for r in rows if r["command"] == "stop" and r["velocity_vx_vy_wz"][2] == 0]
    for vx in (0., .013, .1):
        vals = {r["velocity_vx_vy_wz"][1]: r["linear_raw"] for r in stop if r["velocity_vx_vy_wz"][0] == vx}
        assert vals[0] == vals[.05] == vals[.1]
        assert vals[.11] < vals[0]
        assert vals[0] == pytest.approx(np.exp(-vx ** 2 / .01), abs=1e-6)


def test_angular_probe_detects_double_angle_error(tmp_path):
    source = tmp_path / "generator.py"
    source.write_text('raise RuntimeError("top-level must not execute")\n'
                      'def compute_angular_velocity(quat, prev_quat, dt):\n'
                      '    rotvec = (R.from_quat(prev_quat).inv() * R.from_quat(quat)).as_rotvec()\n'
                      '    return rotvec * np.linalg.norm(rotvec) / dt\n', encoding="utf-8")
    rows = angular_probes(extract_angular_helper(source))
    assert all(not r["matches_expected"] for r in rows)
    for r in rows:
        np.testing.assert_allclose(r["actual_xyz_rad_s"], [0, 0, .0072], atol=1e-10)
    from scipy.spatial.transform import Rotation
    correct = angular_probes(lambda q, p, dt: (Rotation.from_quat(p).inv() * Rotation.from_quat(q)).as_rotvec() / dt)
    assert all(r["matches_expected"] for r in correct)


def test_changed_helper_signature_is_rejected(tmp_path):
    source = tmp_path / "generator.py"
    source.write_text('def compute_angular_velocity(quat):\n    return quat\n', encoding="utf-8")
    with pytest.raises(ValueError):
        extract_angular_helper(source)
