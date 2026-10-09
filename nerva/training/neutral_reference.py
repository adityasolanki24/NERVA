"""Hash-verified discrete references for the opt-in neutral motor candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from nerva.training.reference_kinematics import sample_reference
from nerva.training.reference_validation import REFERENCE_JOINTS

CONTRACT = "neutral_motor_v1"
COMMANDS = ((0., 0., 0.), (.074, 0., 0.), (-.074, 0., 0.),
            (0., .074, 0.), (0., -.074, 0.), (0., 0., .60), (0., 0., -.60))
REQUIRED_CRITERIA = {"positive_knees", "joint_position_fit", "joint_velocity_fit", "linear_fit",
                     "angular_fit", "contact_fit", "command_tracking", "joint_limits"}


def verified_references(root):
    """Admit five repair targets and two corrected turns; verify reports and artifacts."""
    root = Path(root)
    records, provenance = [], []
    for suffix in ("repair", "pivot_retry"):
        directory = root / f"experiments/locomotion_curriculum/results_reference_{suffix}"
        report = directory / "trials.json"
        rows = json.loads(report.read_text(encoding="utf-8"))
        for row in rows:
            turn = row["condition"].startswith("turn")
            if (suffix == "repair" and turn) or (suffix == "pivot_retry" and not turn):
                continue
            if (not row["all_pass"] or set(row["criteria"]) != REQUIRED_CRITERIA
                    or not all(row["criteria"].values())):
                raise ValueError("reference admission failed")
            raw_suffix = "pivot-retry" if turn else "repair"
            raw = root / f"experiments/cloud_runs/neutral-reference-{raw_suffix}" / row["condition"]
            recording = [p for p in raw.glob("*.json") if p.name not in ("preset.json", "reference.json")]
            if len(recording) != 1:
                raise ValueError("missing unique raw recording")
            reference_path = raw / "reference.json"
            for path, key in ((recording[0], "recording_sha256"), (reference_path, "reference_sha256")):
                if hashlib.sha256(path.read_bytes()).hexdigest() != row[key]:
                    raise ValueError("reference artifact hash mismatch")
            ref = json.loads(reference_path.read_text(encoding="utf-8"))
            if (ref.get("version") != 2 or ref.get("velocity_frame") != "current_body"
                    or tuple(ref.get("joint_names", ())) != REFERENCE_JOINTS
                    or bool(ref.get("static")) != (tuple(row["command"]) == (0., 0., 0.))
                    or ref.get("joint_velocity_semantics") != "analytic_instantaneous"
                    or ref.get("contact_semantics") != ("static_geometric_support" if ref["static"] else "planned_support")):
                raise ValueError("reference contract mismatch")
            sample_reference(ref, np.arange(1000) * .54 / 1000)
            records.append({"command": row["command"], "reference": ref})
            provenance.append({"condition": row["condition"], "command": row["command"],
                               "report": report.relative_to(root).as_posix(),
                               "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                               "recording_sha256": row["recording_sha256"],
                               "reference_sha256": row["reference_sha256"]})
    if len(records) != 7 or {tuple(r["command"]) for r in records} != set(COMMANDS):
        raise ValueError("incomplete exact command coverage")
    return records, {"contract": CONTRACT, "mixed_provenance": True, "references": provenance,
                     "learned_motor_readiness": False}


class NeutralReference:
    """JAX sampler with exact discrete lookup; unsupported commands emit nonfinite targets."""
    def __init__(self, records):
        import jax.numpy as jp
        if len(records) != 7 or {tuple(r["command"]) for r in records} != set(COMMANDS):
            raise ValueError("need all seven unique commands")
        arrays = []
        for record in records:
            ref = record["reference"]
            if ref["version"] != 2 or abs(ref["period_s"] - .54) > 1e-6:
                raise ValueError("candidate requires version-2, 0.54-second references")
            sample_reference(ref, np.arange(27) * .02)
            c = {name: np.asarray(value) for name, value in ref["coefficients"].items()}
            derivative = np.zeros_like(c["joint_position"])
            for k in range(1, 13):
                omega = 2 * np.pi * k / .54
                derivative[2 * k - 1] = omega * c["joint_position"][2 * k]
                derivative[2 * k] = -omega * c["joint_position"][2 * k - 1]
            arrays.append(np.hstack([c["joint_position"], derivative, c["contacts"],
                                     c["linear_body"], c["angular_body"]]))
        self.coefficients = jp.asarray(np.stack(arrays), dtype=jp.float32)
        self.commands = jp.asarray([r["command"] for r in records], dtype=jp.float32)
        self.n_styles, self.style_dim = 1, 0
        self.styles = jp.zeros((1, 0))
        self.periods = [.54]
        self.records = records

    def nb_steps_in_period(self, style):
        del style
        return 27

    def get_reference_motion(self, dx, dy, dtheta, index, style=0):
        import jax.numpy as jp
        del style
        command = jp.asarray([dx, dy, dtheta])
        matches = jp.all(jp.abs(self.commands - command) <= 1e-7, axis=1)
        theta = (index % 27) * (2 * jp.pi / 27)
        angles = theta * jp.arange(1, 13)
        design = jp.concatenate([jp.ones(1), jp.stack([jp.cos(angles), jp.sin(angles)], axis=1).reshape(-1)])
        result = design @ self.coefficients[jp.argmax(matches)]
        return jp.where(jp.sum(matches) == 1, result, jp.full(40, jp.nan))
