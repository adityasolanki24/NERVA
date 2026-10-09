import numpy as np
import pytest

from nerva.training.reference_kinematics import exact_reference, fit_reference, pose_velocities, sample_reference


def test_derivatives_are_rotation_covariant_and_use_real_timestamps():
    Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    t = np.array([0, .01, .031, .06])
    xyz = t[:, None] * [.1, -.2, .3]
    q = Rotation.from_rotvec(t[:, None] * [.2, -.3, .4])
    a = pose_velocities(t, xyz, q.as_quat(), t[:, None] * np.ones((1, 16)))
    np.testing.assert_allclose(a["angular_world"], np.tile([.2, -.3, .4], (3, 1)), atol=1e-10)
    np.testing.assert_allclose(a["joint_velocity"], 1, atol=1e-10)
    frame = Rotation.from_euler("xyz", [.5, -.2, 1.])
    b = pose_velocities(t, frame.apply(xyz), (frame * q).as_quat(), t[:, None] * np.ones((1, 16)))
    np.testing.assert_allclose(a["linear_body"], b["linear_body"], atol=1e-10)
    np.testing.assert_allclose(a["angular_body"], b["angular_body"], atol=1e-10)


def test_fourier_positions_and_analytic_joint_derivatives_share_clock():
    period = .54
    t = np.arange(200) * .02
    joints = np.sin(2 * np.pi * t[:, None] / period) * np.ones((1, 16))
    ref = fit_reference(t, joints, np.ones((200, 2)), np.zeros((200, 3)), np.zeros((200, 3)), period)
    samples = sample_reference(ref, t)
    np.testing.assert_allclose(samples["joint_position"], joints, atol=1e-10)
    expected = 2 * np.pi / period * np.cos(2 * np.pi * t[:, None] / period) * np.ones((1, 16))
    np.testing.assert_allclose(samples["joint_velocity"], expected, atol=1e-9)


def test_static_reference_and_unsupported_commands():
    t = np.arange(100) * .02
    ref = fit_reference(t, np.ones((100, 16)), np.ones((100, 2)), np.zeros((100, 3)),
                        np.zeros((100, 3)), .54, static=True)
    assert np.all(sample_reference(ref, t)["joint_velocity"] == 0)
    assert exact_reference([{"command": [0, 0, 0], "reference": ref}], [0, 0, 0]) is ref
    with pytest.raises(ValueError):
        exact_reference([{"command": [0, 0, 0], "reference": ref}], [.01, 0, 0])


def test_invalid_timestamps_are_rejected():
    with pytest.raises(ValueError):
        pose_velocities([0, 0], np.zeros((2, 3)), np.tile([0, 0, 0, 1], (2, 1)), np.zeros((2, 16)))


def test_v2_joint_positions_and_interval_derivatives_share_coefficients():
    t = np.arange(1, 401) * .02
    start = t - .02
    omega = 2 * np.pi * 11 / .54
    def q(times):
        return .1 * np.sin(omega * times[:, None]) * np.ones((1, 16))
    velocity = (q(t) - q(start)) / .02
    ref = fit_reference(t, q(t), np.ones((400, 2)), np.zeros((400, 3)), np.zeros((400, 3)),
                        .54, interval_start=start, joint_velocity=velocity)
    assert ref["version"] == 2 and ref["harmonics"] == 12
    values = sample_reference(ref, t)
    np.testing.assert_allclose(values["joint_position"], q(t), atol=1e-10)
    np.testing.assert_allclose(values["joint_velocity"], .1 * omega * np.cos(omega * t[:, None]) *
                               np.ones((1, 16)), atol=1e-8)
    interval = (values["joint_position"] - sample_reference(ref, start)["joint_position"]) / .02
    np.testing.assert_allclose(interval, velocity, atol=1e-8)


def test_invalid_intervals_and_nonfinite_coefficients_fail_closed():
    t = np.arange(100) * .02
    inputs = (t, np.ones((100, 16)), np.ones((100, 2)), np.zeros((100, 3)), np.zeros((100, 3)), .54)
    with pytest.raises(ValueError):
        fit_reference(*inputs, interval_start=t, joint_velocity=np.zeros((100, 16)))
    ref = fit_reference(*inputs)
    ref["coefficients"]["joint_position"][0][0] = float("nan")
    with pytest.raises(ValueError):
        sample_reference(ref, t)
