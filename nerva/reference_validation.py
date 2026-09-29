"""Physical plausibility checks for fitted Open Duck reference-motion pickles.

Why (docs/development_log.md, 2026-09-29): regenerating the neutral set (R0) gave 41/240
gaits whose RIGHT KNEE had the opposite sign to the shipped references (a backward-bent
knee: mirror-image inverse-kinematics solution). The home pose has both knees at about
+1.37 rad. Training a policy to imitate such gaits would be wrong, so every generated
reference set must pass this check before it is used for training.

Layout (upstream fit_poly.py): per gait 40 fitted signals; the first 16 are joint
positions in the generator's joint order, which includes two antennas (indices 9, 10)
that the 14-actuator MuJoCo model does not have.
"""

from __future__ import annotations

import numpy as np

REFERENCE_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "left_antenna", "right_antenna",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
KNEES = {"left_knee": 3, "right_knee": 14}
KNEE_LIMIT = 1.5708  # MJCF knee range ±π/2 (open_duck_mini_v2.xml); reported, not enforced


def evaluate_gait(value: dict, samples: int = 100) -> np.ndarray:
    """Evaluate one fitted gait's polynomials over one cycle: (samples, n_signals)."""
    t = np.linspace(0.0, 1.0, samples)
    coeffs = value["coefficients"]
    keys = sorted(coeffs, key=lambda k: int(k.split("_")[1]))
    # fit_poly stores coefficients lowest order first
    return np.stack([np.polyval(np.flip(np.asarray(coeffs[k], dtype=float)), t) for k in keys], axis=1)


def validate_reference(reference: dict) -> dict:
    """Report backward knees, non-finite gaits and knee-limit exceedances for a fitted pickle."""
    backward, nonfinite, over_limit = [], [], []
    for key, value in reference.items():
        joints = evaluate_gait(value)[:, :16]
        if not np.all(np.isfinite(joints)):
            nonfinite.append(key)
            continue
        for name, idx in KNEES.items():
            if joints[:, idx].min() <= 0.0:
                backward.append({"gait": key, "joint": name,
                                 "min": float(joints[:, idx].min()), "max": float(joints[:, idx].max())})
        if np.abs(joints[:, list(KNEES.values())]).max() > KNEE_LIMIT:
            over_limit.append(key)
    return {
        "gaits": len(reference),
        "backward_knee": backward,
        "backward_knee_gaits": sorted({b["gait"] for b in backward}),
        "nonfinite_gaits": nonfinite,
        "knee_over_limit_gaits": len(over_limit),  # informational: shipped references do this too
        "valid": not backward and not nonfinite,
    }
