"""E1′ Phase A′: admit the styled reference grid (docs/expressive_posture_e1prime.md §3, preregistration 0c221cf).

No regeneration. Verifies every grid recording/reference hash and its eight admission criteria (via
`styled_grid`), the two interpolation checkpoints' recorded hashes and interpolation errors, and the recorded
reproducibility of the neutral set; records the reference feature changes that define the range criteria.

Usage: python -m experiments.locomotion_curriculum.e1prime_references [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from nerva.training.motor_artifacts import write_json
from nerva.training.styled_reference import (
    CHECKPOINTS, NAMES, PHASE_A_RUN, StyledNeutralReference, _tag, _verified, phase_a_style, styled_grid,
)

PREREGISTRATION = "0c221cf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_e1prime_references"))
    args = ap.parse_args()
    root = Path.cwd().resolve()
    if args.out.exists():
        raise FileExistsError("never overwrite prior outputs")
    grid = styled_grid(root)  # raises on any hash or criterion failure
    sampler = StyledNeutralReference(grid)
    rows = json.loads((root / PHASE_A_RUN / "trials.json").read_text(encoding="utf-8"))
    summary = json.loads((root / PHASE_A_RUN / "summary.json").read_text(encoding="utf-8"))
    by = {(tuple(r["e"]), r["condition"]): r for r in rows}
    checkpoints = {}
    for e in CHECKPOINTS:
        for name in NAMES.values():
            row = by[(phase_a_style(e), name)]
            _verified(root, row)  # recording and fitted reference hashes
            measured = summary["interpolation"][f"{_tag(phase_a_style(e))} {name}"]
            checkpoints[f"{e} {name}"] = {**measured, "ok": measured["leg_rmse_rad"] <= .05
                                          and measured["contact_agreement"] >= .90}
    features = {}
    for name in NAMES.values():
        f = {k: by[(phase_a_style(k), name)]["features"] for k in ((-1, 0), (1, 0), (0, 0), (0, 1))}
        features[name] = {"delta_pitch_rad": f[(1, 0)]["base_pitch_rad"] - f[(-1, 0)]["base_pitch_rad"],
                          "delta_height_m": f[(0, 1)]["base_height_m"] - f[(0, 0)]["base_height_m"]}
    neutral = np.asarray(sampler.coefficients_at([0., 0.]))
    gate = {"grid_hashes_and_criteria": True, "styles": len(grid) * 7,
            "checkpoint_interpolation_ok": all(v["ok"] for v in checkpoints.values()),
            "neutral_reproducibility_ok": bool(summary["repro_ok"]),
            "neutral_anchor_exact": bool(np.array_equal(neutral, np.asarray(sampler.coefficients)))}
    gate["phase_a_prime_passes"] = all(v for k, v in gate.items() if k != "styles")
    args.out.mkdir(parents=True)
    write_json(args.out / "summary.json", {
        "preregistration_commit": PREREGISTRATION,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        **gate, "checkpoints": checkpoints, "reference_feature_changes": features})
    print(json.dumps(gate, indent=1))
    print({k: (round(v["delta_pitch_rad"], 4), round(v["delta_height_m"], 4)) for k, v in features.items()})


if __name__ == "__main__":
    main()
