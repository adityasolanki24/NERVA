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


def basis(t, period, derivative=False, harmonics=HARMONICS):
    t = np.asarray(t)
    if not np.isfinite(period) or period <= 0 or harmonics not in (5, 12):
        raise ValueError("period must be positive")
    columns = [np.zeros_like(t) if derivative else np.ones_like(t)]
    for k in range(1, harmonics + 1):
        omega = 2 * np.pi * k / period
        angle = omega * t
        columns.extend((-omega * np.sin(angle), omega * np.cos(angle)) if derivative
                       else (np.cos(angle), np.sin(angle)))
    return np.column_stack(columns)


def fit_reference(t, joints, contact, linear, angular, period, static=False, *, interval_start=None,
                  joint_velocity=None):
    t = np.asarray(t, dtype=float)
    joints, contact, linear, angular = (np.asarray(v, dtype=float) for v in (joints, contact, linear, angular))
    if (t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or any(y.shape != (len(t), width) or not np.isfinite(y).all()
                   for y, width in ((joints, 16), (contact, 2), (linear, 3), (angular, 3)))):
        raise ValueError("malformed reference samples")
    version = 2 if interval_start is not None else 1
    harmonics = 12 if version == 2 else HARMONICS
    values = {"joint_position": joints, "contacts": contact, "linear_body": linear, "angular_body": angular}
    design = basis(t, period, harmonics=harmonics)
    if version == 2:
        start = np.asarray(interval_start)
        if (start.shape != np.asarray(t).shape or np.any(np.asarray(t) <= start)
                or not np.isfinite(start).all() or joint_velocity is None
                or np.asarray(joint_velocity).shape != joints.shape or not np.isfinite(joint_velocity).all()):
            raise ValueError("malformed joint intervals")
        intervals = (design - basis(start, period, harmonics=harmonics)) / (np.asarray(t) - start)[:, None]
    coefficients = {}
    for name, y in values.items():
        if static:
            coeff = np.zeros((design.shape[1], y.shape[1]))
            coeff[0] = y.mean(axis=0) if name in ("joint_position", "contacts") else 0
        else:
            x, target = design, y
            if version == 2 and name == "joint_position":
                x = np.vstack([design, .02 * intervals])
                target = np.vstack([y, .02 * np.asarray(joint_velocity)])
            coeff, _, rank, _ = np.linalg.lstsq(x, target, rcond=None)
            if rank != design.shape[1]:
                raise ValueError("deficient phase coverage")
        coefficients[name] = coeff.tolist()
    return {"version": version, "basis": "periodic_fourier", "harmonics": harmonics,
            "period_s": float(period), "velocity_frame": "current_body", "pose_quaternion": "xyzw",
            "static": static, "coefficients": coefficients}


def sample_reference(reference, t):
    harmonics = {1: 5, 2: 12}.get(reference.get("version"))
    if (harmonics is None or reference.get("basis") != "periodic_fourier"
            or reference.get("harmonics") != harmonics or reference.get("velocity_frame") != "current_body"):
        raise ValueError("unsupported reference schema")
    period = reference["period_s"]
    design = basis(t, period, harmonics=harmonics)
    derivative = basis(t, period, derivative=True, harmonics=harmonics)
    c = {name: np.asarray(value) for name, value in reference["coefficients"].items()}
    expected = {"joint_position": 16, "contacts": 2, "linear_body": 3, "angular_body": 3}
    if (set(c) != set(expected) or any(c[name].shape != (2 * harmonics + 1, width)
                                     or not np.isfinite(c[name]).all() for name, width in expected.items())
            or not np.isfinite(t).all()):
        raise ValueError("malformed reference coefficients or times")
    result = {name: design @ value for name, value in c.items()}
    result["joint_velocity"] = derivative @ c["joint_position"]
    return result


def exact_reference(subset, command):
    matches = [r for r in subset if np.allclose(r["command"], command, atol=1e-8, rtol=0)]
    if len(matches) != 1:
        raise ValueError("subset has no unique exact reference for command")
    return matches[0]["reference"]
