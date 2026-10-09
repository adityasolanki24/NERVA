"""Known-signal checks for directional tracking, stationary motion and recovery."""

import numpy as np
import pytest

from experiments.locomotion_curriculum.gate import (
    DT, protocol_trials, recovery_metrics, segment_metrics, summarise,
)


def trace(seconds=15.0, forward=0.15, lateral=0.0, yaw_rate=0.0):
    t = (np.arange(round(seconds / DT)) + 1) * DT
    yaw = yaw_rate * t
    quat = np.column_stack([np.cos(yaw / 2), 0 * t, 0 * t, np.sin(yaw / 2)])
    # Heading-frame motion transformed to world frame, including multiple yaw wraps.
    v = np.column_stack([forward * np.cos(yaw) - lateral * np.sin(yaw),
                         forward * np.sin(yaw) + lateral * np.cos(yaw), 0 * t])
    foot = 0.01 * np.maximum(np.sin(2 * np.pi * 2 * t), 0)
    return {"t": t, "base_quat": quat, "base_linvel": v, "foot_z": np.column_stack([foot, foot])}


def test_gate_has_all_fixed_conditions_and_paired_seeds():
    trials = protocol_trials()
    assert len(trials) == len({r.name for r in trials}) == 115
    assert sum(r.group == "push" for r in trials) == 40
    assert sum(r.group == "transition" for r in trials) == 5
    assert all({r.seed for r in trials if r.group == g} == set(range(5)) for g in {r.group for r in trials})


def test_sideways_motion_cannot_pass_forward_or_hide_standing():
    result = segment_metrics(trace(forward=0.0, lateral=0.15), 5, 15, "forward")
    assert not result["direction_ok"] and not result["cross_ok"]
    assert result["stationary_fraction"] == 1.0
    assert not result["stationary_ok"]


def test_backward_tracking_preserves_sign():
    good = segment_metrics(trace(forward=-0.12), 5, 15, "backward")
    wrong = segment_metrics(trace(forward=0.12), 5, 15, "backward")
    assert good["direction_ok"] and good["tracking_ok"] and good["cross_ok"]
    assert good["signed_axis_mean"] == pytest.approx(0.12)
    assert not wrong["direction_ok"] and not wrong["tracking_ok"]


def test_yaw_unwrap_and_stop_metrics():
    turn = segment_metrics(trace(forward=0.0, yaw_rate=0.6), 5, 15, "turn_left")
    assert turn["mean_velocity"][2] == pytest.approx(0.6)
    assert turn["direction_ok"] and turn["tracking_ok"] and turn["cross_ok"]
    stopped = segment_metrics(trace(forward=0), 5, 15, "stop")
    drifting = segment_metrics(trace(forward=0.03), 5, 15, "stop")
    assert stopped["stop_ok"] and not drifting["stop_ok"]


def test_recovery_requires_full_window_and_later_fall_disqualifies_it():
    arrays = trace()
    arrays["base_linvel"][(arrays["t"] > 8) & (arrays["t"] <= 10), 0] = 0.5
    result = recovery_metrics(arrays, "forward")
    # First qualifying mean can include some disturbed samples; criterion is on window mean.
    assert 2 < result["recovery_s"] <= 3.0
    assert result["recovery_ok"]
    arrays["base_quat"][-1] = [np.cos(np.pi / 6), 0, np.sin(np.pi / 6), 0]
    assert not recovery_metrics(arrays, "forward")["recovery_ok"]


def test_recovery_cannot_be_credited_with_one_calm_sample():
    arrays = trace()
    bad = arrays["t"] > 8
    arrays["base_quat"][bad] = [np.cos(np.pi / 6), 0, np.sin(np.pi / 6), 0]
    arrays["base_quat"][-1] = [1, 0, 0, 0]
    assert recovery_metrics(arrays, "forward")["recovery_s"] is None


def test_missing_trials_cannot_pass_by_empty_all():
    result = summarise([])
    assert not result["criteria"]["complete"] and not result["all_pass"]
