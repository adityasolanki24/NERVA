"""Compare a regenerated reference pickle with the shipped upstream control.

Both inputs must be trusted local files: Python pickle is not a safe interchange
format. The JSON report focuses on grid overlap, periods, shapes and numerical
differences for keys shared by both files.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _periods(reference: dict) -> list[float]:
    return [float(value["period"]) for value in reference.values()]


def _coefficient_vector(value: dict) -> np.ndarray:
    coefficients = value["coefficients"]
    return np.concatenate([
        np.asarray(coefficients[key], dtype=float).ravel()
        for key in sorted(coefficients, key=lambda item: int(item.split("_")[-1]))
    ])


def compare_references(shipped: dict, generated: dict) -> dict[str, object]:
    shipped_keys, generated_keys = set(shipped), set(generated)
    common = sorted(shipped_keys & generated_keys)
    deltas: list[float] = []
    shape_mismatches: list[str] = []
    nonfinite: list[str] = []
    for key in common:
        left, right = _coefficient_vector(shipped[key]), _coefficient_vector(generated[key])
        if left.shape != right.shape:
            shape_mismatches.append(key)
            continue
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            nonfinite.append(key)
            continue
        deltas.append(float(np.max(np.abs(left - right))))

    shipped_periods = _periods(shipped)
    generated_periods = _periods(generated)
    return {
        "shipped_keys": len(shipped_keys),
        "generated_keys": len(generated_keys),
        "common_keys": len(common),
        "only_shipped_keys": len(shipped_keys - generated_keys),
        "only_generated_keys": len(generated_keys - shipped_keys),
        "common_key_examples": common[:10],
        "only_shipped_examples": sorted(shipped_keys - generated_keys)[:10],
        "only_generated_examples": sorted(generated_keys - shipped_keys)[:10],
        "shipped_period_range": [min(shipped_periods), max(shipped_periods)] if shipped_periods else None,
        "generated_period_range": [min(generated_periods), max(generated_periods)] if generated_periods else None,
        "shape_mismatches": shape_mismatches,
        "nonfinite_common_keys": nonfinite,
        "max_abs_coefficient_delta_on_common_keys": max(deltas) if deltas else None,
        "mean_max_abs_coefficient_delta_on_common_keys": float(np.mean(deltas)) if deltas else None,
    }


def load_trusted(path: Path) -> dict:
    with path.open("rb") as stream:
        value = pickle.load(stream)  # trusted local research artifact
    if not isinstance(value, dict):
        raise TypeError(f"{path} did not contain a dictionary")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shipped", type=Path)
    parser.add_argument("generated", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = compare_references(load_trusted(args.shipped), load_trusted(args.generated))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
