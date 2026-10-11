"""E1′ styled references (docs/expressive_posture_e1prime.md): hash-verified grid and bilinear JAX sampler.

Grid e_pitch ∈ {−1, 0, 1} × e_crouch ∈ {0, 1}, seven neutral commands each. (0, 0) is exactly the admitted
neutral set; the other five styles come from E1 Phase A (halved-height run), mapped as
(p, crouch 0) ← (p, h 0) and (p, crouch 1) ← (p, h −1).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from nerva.training.neutral_reference import COMMANDS, REQUIRED_CRITERIA, NeutralReference, verified_references

PHASE_A_RUN = "experiments/locomotion_curriculum/results_e1_references_p1.0_h0.5"
PHASE_A_RAW = "experiments/cloud_runs/e1-styled-references_p1.0_h0.5"
PITCH_NODES, CROUCH_NODES = (-1, 0, 1), (0, 1)
CHECKPOINTS = ((-.5, .5), (.5, .5))
NAMES = {(0., 0., 0.): "stand", (.074, 0., 0.): "forward", (-.074, 0., 0.): "backward", (0., .074, 0.): "left",
         (0., -.074, 0.): "right", (0., 0., .6): "turn_left", (0., 0., -.6): "turn_right"}


def phase_a_style(e):
    """E1′ (e_pitch, e_crouch) → the E1 Phase A (e_pitch, e_height) whose recordings it uses."""
    return (e[0], -e[1])


def _tag(e):
    return f"p{e[0]:+.1f}_h{e[1]:+.1f}"


def styled_grid(root):
    """{(e_pitch, e_crouch): seven reference records}, every artifact hash- and criteria-verified."""
    root = Path(root)
    rows = json.loads((root / PHASE_A_RUN / "trials.json").read_text(encoding="utf-8"))
    by = {(tuple(r["e"]), r["condition"]): r for r in rows}
    grid = {}
    for p in PITCH_NODES:
        for c in CROUCH_NODES:
            if (p, c) == (0, 0):
                grid[(p, c)] = verified_references(root)[0]
                continue
            records = []
            for command in COMMANDS:
                row = by[(phase_a_style((p, c)), NAMES[tuple(command)])]
                if (not row["all_pass"] or set(row["criteria"]) != REQUIRED_CRITERIA
                        or not all(row["criteria"].values())):
                    raise ValueError("styled reference admission failed")
                records.append({"command": list(command), "reference": _verified(root, row)})
            grid[(p, c)] = records
    return grid


def _verified(root, row):
    directory = root / PHASE_A_RAW / _tag(row["e"]) / row["condition"]
    recording = [p for p in directory.glob("*.json") if p.name not in ("preset.json", "reference.json")]
    if len(recording) != 1:
        raise ValueError("missing unique raw recording")
    for path, key in ((recording[0], "recording_sha256"), (directory / "reference.json", "reference_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != row[key]:
            raise ValueError("styled reference artifact hash mismatch")
    return json.loads((directory / "reference.json").read_text(encoding="utf-8"))


def style_weights(e, xp=np):
    """Piecewise-bilinear weights over the 3 × 2 grid; exact at the nodes."""
    p, c = e[0], e[1]
    wp = xp.stack([xp.maximum(0., -p), 1. - xp.abs(p), xp.maximum(0., p)])
    wc = xp.stack([1. - c, c])
    return wp[:, None] * wc[None, :]


class StyledNeutralReference(NeutralReference):
    """NeutralReference whose coefficients are bilinearly interpolated in e = (e_pitch, e_crouch)."""

    def __init__(self, grid):
        import jax.numpy as jp
        if set(grid) != {(p, c) for p in PITCH_NODES for c in CROUCH_NODES}:
            raise ValueError("need the full 3 × 2 style grid")
        tables = [[NeutralReference(grid[(p, c)]) for c in CROUCH_NODES] for p in PITCH_NODES]
        commands = np.asarray(tables[1][0].commands)
        for row in tables:
            for table in row:
                if not np.array_equal(np.asarray(table.commands), commands):
                    raise ValueError("styles must share the command order")
        super().__init__(grid[(0, 0)])  # neutral coefficients, command order, validation
        self.style_coefficients = jp.stack([jp.stack([t.coefficients for t in row]) for row in tables])
        self.style_dim = 2

    def coefficients_at(self, e):
        import jax.numpy as jp
        w = style_weights(jp.asarray(e, dtype=jp.float32), jp)
        return jp.einsum("pc,pcmkf->mkf", w, self.style_coefficients)

    def get_styled_motion(self, dx, dy, dtheta, index, e):
        import jax.numpy as jp
        command = jp.asarray([dx, dy, dtheta])
        matches = jp.all(jp.abs(self.commands - command) <= 1e-7, axis=1)
        theta = (index % 27) * (2 * jp.pi / 27)
        angles = theta * jp.arange(1, 13)
        design = jp.concatenate([jp.ones(1), jp.stack([jp.cos(angles), jp.sin(angles)], axis=1).reshape(-1)])
        result = design @ self.coefficients_at(e)[jp.argmax(matches)]
        return jp.where(jp.sum(matches) == 1, result, jp.full(40, jp.nan))
