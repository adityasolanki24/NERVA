import numpy as np
import pytest

from experiments.locomotion_curriculum.record_reference import adapt, adapt_engine, pivot_step, static_support
from experiments.locomotion_curriculum.reference_subset import known_pose_checks, validate


def recording(knee=1.2):
    t = np.arange(1, 401) * .02
    f = np.zeros((400, 61))
    f[:, 0] = .074 * t
    f[:, 6] = 1  # SciPy xyzw identity
    f[:, 7:23] = .1 * np.sin(2 * np.pi * t[:, None] / .54)
    f[:, 10], f[:, 21] = knee, knee
    f[:, 59:61] = 1
    return {"Frames": f.tolist(), "FrameTimes": t.tolist(), "Placo": {"period": .54},
            "Frame_offset": [{"joints_pos": 7, "foot_contacts": 59}]}


def test_known_recording_passes_and_negative_knees_remain_failed():
    report, _ = validate(recording(), [.074, 0, 0])
    assert report["all_pass"]
    bad, _ = validate(recording(knee=-1.2), [.074, 0, 0])
    assert not bad["all_pass"] and not bad["criteria"]["positive_knees"]


def test_known_rotations_pass_at_all_fixed_headings_and_intervals():
    rows = known_pose_checks()
    assert len(rows) == 12 and all(r["pass"] for r in rows)


def test_recorder_adaptation_is_fail_closed():
    with pytest.raises(ValueError):
        adapt("unexpected upstream source")


def test_static_support_requires_both_feet_on_floor():
    left, right = np.eye(4), np.eye(4)
    assert static_support(left, right) == [1, 1]
    right[2, 3] = .01
    with pytest.raises(ValueError):
        static_support(left, right)
    with pytest.raises(ValueError):
        adapt_engine("unexpected engine")


def test_repaired_validator_enforces_bounds_and_joint_order():
    from nerva.training.reference_validation import REFERENCE_JOINTS
    data = recording()
    data["Joints"] = list(REFERENCE_JOINTS)
    limits = {name: (-2., 2.) for name in REFERENCE_JOINTS}
    result, ref = validate(data, [.074, 0, 0], joint_limits=limits)
    assert result["all_pass"] and ref["version"] == 2
    data["Frames"][0][10] = 3.  # Warmup outside fitting/scoring must still respect physical bounds.
    result, _ = validate(data, [.074, 0, 0], joint_limits=limits)
    assert not result["criteria"]["joint_limits"]
    data["Frames"][0][10] = 1.2
    limits["left_knee"] = (.01, 1.)
    result, _ = validate(data, [.074, 0, 0], joint_limits=limits)
    assert not result["criteria"]["joint_limits"]
    data["Joints"].reverse()
    with pytest.raises(ValueError):
        validate(data, [.074, 0, 0], joint_limits=limits)


def test_analytic_pivot_is_fixed_for_both_turn_signs_without_pose_clipping():
    for theta in (-.162, 0, .162):
        c = np.array([.04, -.01])
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        shift = pivot_step(c, theta)
        np.testing.assert_allclose(shift + rotation @ c, c, atol=1e-12)
    with pytest.raises(ValueError):
        pivot_step([float("nan"), 0], .1)
