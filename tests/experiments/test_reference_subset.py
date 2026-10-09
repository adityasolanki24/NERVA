import numpy as np
import pytest

from experiments.locomotion_curriculum.record_reference import adapt
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
