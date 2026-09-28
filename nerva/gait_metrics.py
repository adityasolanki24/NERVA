"""Gait measurements from logged simulation data. Pure NumPy, no simulator.

All functions take plain arrays sampled at a fixed step `dt` (50 Hz control
rate in practice) and should be applied to a steady-state window.

Measurement choices (see development_log.md for why):
  - Cadence comes from the dominant frequency of foot height, NOT from counting
    contact on/off switches. Contact chatters and miscounts steps.
  - Velocities are expressed in the robot's heading frame (yaw only), so a
    slowly turning robot still reports forward speed correctly.
"""

from __future__ import annotations

import numpy as np

G = 9.81
FALL_TILT_DEG = 45.0


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """(N, 4) w-x-y-z quaternions → (N, 3) roll, pitch, yaw [rad], ZYX convention."""
    w, x, y, z = np.asarray(quat, dtype=float).T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=1)


def tilt_deg(quat: np.ndarray) -> np.ndarray:
    """Angle between the base z-axis and world vertical [deg]."""
    _, x, y, _ = np.asarray(quat, dtype=float).T
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0)))


def heading_frame_velocity(world_linvel: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """(N, 3) world-frame base velocity → (N, 2) [forward, left] in the heading frame."""
    vx, vy = world_linvel[:, 0], world_linvel[:, 1]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * vx + s * vy, -s * vx + c * vy], axis=1)


def dominant_frequency(signal: np.ndarray, dt: float, fmin: float = 0.5, pad: int = 8) -> float:
    """Frequency [Hz] of the largest spectral peak above fmin (Hann window, zero-padded)."""
    x = np.asarray(signal, dtype=float)
    x = (x - x.mean()) * np.hanning(len(x))
    n = pad * len(x)
    spectrum = np.abs(np.fft.rfft(x, n))
    freqs = np.fft.rfftfreq(n, dt)
    spectrum[freqs < fmin] = 0.0
    return float(freqs[np.argmax(spectrum)])


def lift_height(foot_z: np.ndarray) -> float:
    """Robust foot lift: 95th minus 5th percentile of foot height [m]."""
    return float(np.percentile(foot_z, 95) - np.percentile(foot_z, 5))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def summarise(arrays: dict[str, np.ndarray], dt: float, mass: float) -> dict[str, float]:
    """All RQ1 metrics for one steady-state window.

    `arrays` keys: base_pos (N,3), base_quat (N,4), base_linvel (N,3),
    base_angvel (N,3), foot_z (N,2), joint_pos (N,14), action (N,14),
    tau_sq (N,), power (N,).
    """
    rpy = quat_to_rpy(arrays["base_quat"])
    v = heading_frame_velocity(arrays["base_linvel"], rpy[:, 2])
    v_fwd = float(v[:, 0].mean())
    gait_hz = dominant_frequency(arrays["foot_z"][:, 0], dt)
    tilt = tilt_deg(arrays["base_quat"])
    power = float(np.mean(arrays["power"]))
    return {
        "v_fwd": v_fwd,
        "v_lat": float(v[:, 1].mean()),
        "yaw_rate": float(np.mean(np.gradient(np.unwrap(rpy[:, 2]), dt))),
        "gait_hz": gait_hz,
        "cadence_steps_per_s": 2.0 * gait_hz,
        "stride_m": v_fwd / gait_hz if gait_hz > 0 else float("nan"),
        "lift_left_mm": 1000 * lift_height(arrays["foot_z"][:, 0]),
        "lift_right_mm": 1000 * lift_height(arrays["foot_z"][:, 1]),
        "base_height_m": float(arrays["base_pos"][:, 2].mean()),
        "pitch_mean_deg": float(np.degrees(rpy[:, 1].mean())),
        "pitch_std_deg": float(np.degrees(rpy[:, 1].std())),
        "roll_std_deg": float(np.degrees(rpy[:, 0].std())),
        "sway_rate_rms": rms(arrays["base_angvel"][:, :2]),  # roll/pitch rate, rad/s
        "action_rate_rms": rms(np.diff(arrays["action"], axis=0)),  # per control step
        "joint_range_rms_rad": rms(np.ptp(arrays["joint_pos"], axis=0)),
        "tau_sq": float(np.mean(arrays["tau_sq"])),
        "power_w": power,
        "cost_of_transport": power / (mass * G * abs(v_fwd)) if abs(v_fwd) > 1e-3 else float("nan"),
        "max_tilt_deg": float(tilt.max()),
        "fell": float(tilt.max() > FALL_TILT_DEG),
    }
