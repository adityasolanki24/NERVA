"""Native-MuJoCo motor metrics for neutral command tracking (moved unchanged, 2026-10-10).

velocities/smooth come from the B2 gate (docs/b2_robustness_gate.md); motor_metrics and the deployment
contract metadata from the paired neutral evaluation (docs/neutral_learning_pilot.md).
"""
import numpy as np

from nerva.analysis import gait_metrics as gm
from nerva.training.neutral_reference import COMMANDS, CONTRACT

DT = 0.02
WINDOW = 50


def smooth(values):
    values = np.asarray(values)
    return np.stack([np.convolve(values[:, i], np.ones(WINDOW) / WINDOW, mode="valid")
                     for i in range(values.shape[1])], axis=1)


def velocities(arrays):
    rpy = gm.quat_to_rpy(arrays["base_quat"])
    linear = gm.heading_frame_velocity(arrays["base_linvel"], rpy[:, 2])
    yaw = np.gradient(np.unwrap(rpy[:, 2]), DT)
    return np.column_stack([linear, yaw])


def deployment_metadata(policy_hash):
    return {"contract": CONTRACT, "observation_size": 101, "period_steps": 27,
            "head_commands_zero": True, "commands": COMMANDS, "policy_sha256": policy_hash}


def motor_metrics(arrays, command, completed):
    """Never give incomplete or failed rollouts a successful tracking score."""
    if not completed:
        return {"all_pass": False, "tracking_available": False}
    mask = (arrays["t"] >= 5) & (arrays["t"] < 20)
    raw = velocities(arrays)[mask]
    filtered = smooth(raw)
    command = np.asarray(command)
    moving = bool(np.any(command))
    axis = int(np.argmax(np.abs(command)))
    horizontal = float(np.sqrt(np.mean(np.sum(filtered[:, :2] ** 2, axis=1))))
    yaw = float(gm.rms(filtered[:, 2]))
    displacement = float(np.linalg.norm(arrays["base_pos"][-1, :2] - arrays["base_pos"][0, :2]))
    criteria = {"completed": True}
    result = {"tracking_available": True, "mean_velocity": raw.mean(axis=0).tolist(),
              "horizontal_rms": horizontal, "yaw_rms": yaw, "displacement_m": displacement,
              "lift_mm": [float(1000 * gm.lift_height(arrays["foot_z"][mask, i])) for i in range(2)]}
    if moving:
        signed = float(raw[:, axis].mean() * np.sign(command[axis]))
        error = float(gm.rms(filtered[:, axis] - command[axis]))
        stationary = float(np.mean(np.abs(filtered[:, axis]) < (.1 if axis == 2 else .01)))
        cross = float(gm.rms(filtered[:, 1 - axis])) if axis < 2 else horizontal
        criteria.update(direction=signed >= .5 * abs(command[axis]), tracking=error <= .6 * abs(command[axis]),
                        stationary=stationary <= .1, cross=(cross <= .05 and yaw <= .2) if axis < 2 else cross <= .03)
        result.update(signed_axis_mean=signed, axis_tracking_rmse=error,
                      normalized_axis_rmse=error / abs(command[axis]), stationary_fraction=stationary,
                      orthogonal_rms=cross)
    else:
        criteria.update(rest_speed=horizontal <= .02, rest_yaw=yaw <= .15, rest_displacement=displacement <= .10)
    return {**result, "criteria": {key: bool(value) for key, value in criteria.items()}, "all_pass": all(criteria.values())}
