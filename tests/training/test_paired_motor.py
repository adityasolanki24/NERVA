import numpy as np

from experiments.locomotion_curriculum.paired_motor import motor_metrics, summarize


def trace(command):
    t = (np.arange(1000) + 1) * .02
    yaw = command[2] * t
    quaternion = np.column_stack([np.cos(yaw / 2), np.zeros((1000, 2)), np.sin(yaw / 2)])
    return {"t": t, "base_quat": quaternion,
            "base_linvel": np.tile(np.array([command[0], command[1], 0], dtype=float), (1000, 1)),
            "base_pos": np.column_stack([command[0] * t, command[1] * t, np.ones(1000) * .15]),
            "foot_z": np.zeros((1000, 2))}


def test_complete_matching_translation_turn_and_rest_pass():
    for command in ((.074, 0, 0), (0, -.074, 0), (0, 0, -.6), (0, 0, 0)):
        assert motor_metrics(trace(command), command, True)["all_pass"]


def test_incomplete_survival_never_scores_and_turn_translation_and_rest_drift_fail():
    command = (0, 0, .6)
    assert motor_metrics(trace(command), command, False) == {"all_pass": False, "tracking_available": False}
    arrays = trace(command)
    arrays["base_linvel"][:, 0] = .08
    assert not motor_metrics(arrays, command, True)["criteria"]["cross"]
    arrays = trace((0, 0, 0))
    arrays["base_pos"][-1, 0] = .11
    assert not motor_metrics(arrays, (0, 0, 0), True)["criteria"]["rest_displacement"]
    arrays = trace((0, 0, 0))
    assert not motor_metrics(arrays, (.074, 0, 0), True)["all_pass"]


def test_empty_or_incomplete_comparison_cannot_pass():
    result = summarize([])
    assert not result["pilot_improvement"]
    assert not result["criteria"]["complete"]
    assert result["candidate_to_untrained_error_ratio"] is None
