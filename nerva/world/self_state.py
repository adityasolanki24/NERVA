"""Self-state estimate: the robot's own condition as appraisal and safety see it (refactor stage E).

Only quantities a real robot measures or can estimate are used:
  tilt_deg        from the base orientation (IMU orientation estimate on hardware)
  angular_speed   |base angular velocity| (gyro)
  speed           |horizontal base velocity| (state estimator / odometry on hardware; MuJoCo's base
                  velocity here, which a hardware estimator would only approximate)
  feet_contact    None: Open Duck has foot switches, but NERVA does not read contacts in simulation yet
  stability_risk  DERIVED heuristic, not a sensor:
                    max( clip((tilt − RISK_TILT_START) / (RISK_TILT_FULL − RISK_TILT_START)),
                         clip((angular_speed − RISK_GYRO_START) / RISK_GYRO_SPAN) )
  escape_room     None: not computed yet
Thresholds are NERVA design choices (falls in these scenes happen beyond ~45°; normal B2 walking tilts
up to ~16°, measured 2026-10-01/02).
"""

from __future__ import annotations

import numpy as np

from nerva.analysis import gait_metrics as gm
from nerva.interfaces import SelfState

RISK_TILT_START, RISK_TILT_FULL = 10.0, 30.0  # deg
RISK_GYRO_START, RISK_GYRO_SPAN = 2.0, 4.0  # rad/s


def tilt_from_quat(quat_wxyz) -> float:
    """Angle between the body z axis and world vertical, degrees (same formula as the gait metrics)."""
    return float(gm.tilt_deg(np.asarray(quat_wxyz, dtype=float)[None])[0])


def stability_risk(tilt_deg: float, angular_speed: float) -> float:
    tilt_term = (tilt_deg - RISK_TILT_START) / (RISK_TILT_FULL - RISK_TILT_START)
    gyro_term = (angular_speed - RISK_GYRO_START) / RISK_GYRO_SPAN
    return float(min(1.0, max(0.0, tilt_term, gyro_term)))


def estimate_self_state(t: float, qpos, qvel, locomotion_mode: str = "") -> SelfState:
    """From the floating base: qpos[3:7] orientation (w, x, y, z), qvel[0:3] linear, qvel[3:6] angular."""
    tilt = tilt_from_quat(qpos[3:7])
    ang = float(np.linalg.norm(qvel[3:6]))
    return SelfState(time_s=t, tilt_deg=tilt, angular_speed=ang, speed=float(np.linalg.norm(qvel[0:2])),
                     feet_contact=None, stability_risk=stability_risk(tilt, ang), escape_room=None,
                     locomotion_mode=locomotion_mode)
