import numpy as np
import pytest

from nerva.analysis import gait_metrics as gm

DT = 0.02  # 50 Hz


def _yaw_quat(yaw):
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], axis=1)


def test_quat_to_rpy_recovers_known_angles():
    # pitch of 0.2 rad about y
    q = np.array([[np.cos(0.1), 0.0, np.sin(0.1), 0.0]])
    np.testing.assert_allclose(gm.quat_to_rpy(q)[0], [0.0, 0.2, 0.0], atol=1e-12)
    np.testing.assert_allclose(gm.quat_to_rpy(_yaw_quat(np.array([1.0])))[0], [0, 0, 1.0], atol=1e-12)


def test_tilt_ignores_yaw_and_measures_pitch():
    assert gm.tilt_deg(_yaw_quat(np.array([2.0])))[0] == pytest.approx(0.0, abs=1e-9)
    q = np.array([[np.cos(np.radians(30) / 2), 0.0, np.sin(np.radians(30) / 2), 0.0]])
    assert gm.tilt_deg(q)[0] == pytest.approx(30.0)


def test_heading_frame_velocity_rotates_world_velocity():
    # robot facing +y (yaw 90°) moving along world +y → pure forward motion
    v = gm.heading_frame_velocity(np.array([[0.0, 0.1, 0.0]]), np.array([np.pi / 2]))
    np.testing.assert_allclose(v[0], [0.1, 0.0], atol=1e-12)


@pytest.mark.parametrize("f", [1.1, 1.85, 2.6])
def test_dominant_frequency_of_foot_like_signal(f):
    t = np.arange(0, 15, DT)
    z = 0.01 * np.clip(np.sin(2 * np.pi * f * t), 0, None)  # half-rectified: foot lifts, then rests
    assert gm.dominant_frequency(z, DT) == pytest.approx(f, abs=0.02)


def test_lift_height_of_sine_is_close_to_peak_to_peak():
    z = 0.005 * np.sin(np.linspace(0, 40 * np.pi, 4000))
    assert gm.lift_height(z) == pytest.approx(0.01, rel=0.03)


def test_summarise_on_synthetic_straight_walk():
    t = np.arange(0, 10, DT)
    n = len(t)
    f, v = 1.85, 0.1
    arrays = {
        "base_pos": np.stack([v * t, 0 * t, 0.15 + 0 * t], axis=1),
        "base_quat": _yaw_quat(0 * t),
        "base_linvel": np.tile([v, 0.0, 0.0], (n, 1)),
        "base_angvel": np.zeros((n, 3)),
        "foot_z": np.stack([0.01 * np.clip(np.sin(2 * np.pi * f * t), 0, None)] * 2, axis=1),
        "joint_pos": np.zeros((n, 14)),
        "action": np.zeros((n, 14)),
        "tau_sq": np.ones(n),
        "power": np.full(n, 2.0),
    }
    s = gm.summarise(arrays, DT, mass=2.0)
    assert s["v_fwd"] == pytest.approx(v)
    assert s["gait_hz"] == pytest.approx(f, abs=0.02)
    assert s["cadence_steps_per_s"] == pytest.approx(2 * f, abs=0.04)
    assert s["stride_m"] == pytest.approx(v / f, rel=0.02)
    assert s["cost_of_transport"] == pytest.approx(2.0 / (2.0 * gm.G * v))
    assert s["fell"] == 0.0
    assert s["base_height_m"] == pytest.approx(0.15)
