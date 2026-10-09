"""Explicit body-frame derivatives and stable periodic neutral reference schema.

Independent of historical polynomial pickles and their runtime lookup.
"""
from __future__ import annotations

import numpy as np

HARMONICS = 5


def pose_velocities(t, xyz, quat_xyzw, joints):
    """Backward differences; angular derivative in world then current body axes.

    Output excludes the first sample, whose derivative is unknown.
    """
    from scipy.spatial.transform import Rotation
    t, xyz, q, joints = (np.asarray(v, dtype=float) for v in (t, xyz, quat_xyzw, joints))
    if (t.ndim != 1 or len(t) < 2 or xyz.shape != (len(t), 3) or q.shape != (len(t), 4)
            or joints.ndim != 2 or len(joints) != len(t) or np.any(np.diff(t) <= 0)
            or not all(np.isfinite(v).all() for v in (t, xyz, q, joints))
            or np.any(np.linalg.norm(q, axis=1) < 1e-12)):
        raise ValueError("malformed timestamped poses")
    rotations = Rotation.from_quat(q)
    dt = np.diff(t)[:, None]
    world_linear = np.diff(xyz, axis=0) / dt
    world_angular = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt
    inverse = rotations[1:].inv()
    return {"linear_body": inverse.apply(world_linear), "angular_body": inverse.apply(world_angular),
            "linear_world": world_linear, "angular_world": world_angular,
            "joint_velocity": np.diff(joints, axis=0) / dt}


def basis(t, period, derivative=False):
    t = np.asarray(t)
    if period <= 0:
        raise ValueError("period must be positive")
    columns = [np.zeros_like(t) if derivative else np.ones_like(t)]
    for k in range(1, HARMONICS + 1):
        omega = 2 * np.pi * k / period
        angle = omega * t
        columns.extend((-omega * np.sin(angle), omega * np.cos(angle)) if derivative
                       else (np.cos(angle), np.sin(angle)))
    return np.column_stack(columns)


def fit_reference(t, joints, contact, linear, angular, period, static=False):
    values = {"joint_position": joints, "contacts": contact, "linear_body": linear, "angular_body": angular}
    design = basis(t, period)
    coefficients = {}
    for name, y in values.items():
        if static:
            coeff = np.zeros((design.shape[1], y.shape[1]))
            coeff[0] = y.mean(axis=0) if name in ("joint_position", "contacts") else 0
        else:
            coeff, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
            if rank != design.shape[1]:
                raise ValueError("deficient phase coverage")
        coefficients[name] = coeff.tolist()
    return {"version": 1, "basis": "periodic_fourier", "harmonics": HARMONICS,
            "period_s": float(period), "velocity_frame": "current_body", "pose_quaternion": "xyzw",
            "static": static, "coefficients": coefficients}


def sample_reference(reference, t):
    if (reference.get("version") != 1 or reference.get("basis") != "periodic_fourier"
            or reference.get("harmonics") != HARMONICS or reference.get("velocity_frame") != "current_body"):
        raise ValueError("unsupported reference schema")
    period = reference["period_s"]
    design, derivative = basis(t, period), basis(t, period, derivative=True)
    c = {name: np.asarray(value) for name, value in reference["coefficients"].items()}
    result = {name: design @ value for name, value in c.items()}
    result["joint_velocity"] = derivative @ c["joint_position"]
    return result


def exact_reference(subset, command):
    matches = [r for r in subset if np.allclose(r["command"], command, atol=1e-8, rtol=0)]
    if len(matches) != 1:
        raise ValueError("subset has no unique exact reference for command")
    return matches[0]["reference"]
