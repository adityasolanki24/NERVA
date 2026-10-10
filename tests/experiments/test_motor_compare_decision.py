"""Gait-averaged tracking decision rule (docs/gait_averaged_tracking_pilot.md) on synthetic summaries."""
import pytest

pytest.importorskip("mujoco")
from experiments.locomotion_curriculum.motor_compare import NAMES, gait_averaged_decision  # noqa: E402


def arm(speed=.04, falls=0, rest_passes=3, passing=7, rmse=.5):
    commands = {n: {"signed_axis_mean": [speed] * 3, "passes": 3} for n in NAMES}
    commands["rest"]["passes"] = rest_passes
    return {"commands": commands, "falls": falls, "commands_passing": passing, "normalized_tracking_rmse": rmse}


def decide(new, **controls):
    return gait_averaged_decision({"gait_candidate": new, "untrained_neutral": arm(rmse=1.),
                                   "gpu_pilot_candidate": arm(rmse=.7), **controls})


def test_passes_when_all_criteria_hold():
    d = decide(arm())
    assert d["hypothesis_supported"] and d["pilot_passes"]
    assert d["gait_to_untrained_error_ratio"] == pytest.approx(.5)


def test_speed_below_half_request_fails_hypothesis():
    new = arm()
    new["commands"]["right"]["signed_axis_mean"] = [.04, .036, .04]
    assert not decide(new)["hypothesis_supported"]


def test_rest_or_added_falls_fail():
    assert not decide(arm(rest_passes=2))["hypothesis_supported"]
    assert not decide(arm(falls=1))["hypothesis_supported"]


def test_turn_failures_block_only_the_overall_pilot():
    d = decide(arm(passing=5))
    assert d["hypothesis_supported"] and not d["pilot_passes"]
