"""PAD dynamic-range metrics on scenario logs (Model B saturation evaluation, development log 2026-10-02)."""

from __future__ import annotations

import numpy as np

DIMS = ("valence", "arousal", "dominance")
NEAR_SATURATION = 0.9


def pad_stats(rows) -> dict:
    """Per dimension: min, max, mean, std and the fraction of samples with |value| > NEAR_SATURATION."""
    out = {}
    for d in DIMS:
        v = np.array([r[d] for r in rows], dtype=float)
        out[d] = {"min": round(float(v.min()), 3), "max": round(float(v.max()), 3), "mean": round(float(v.mean()), 4),
                  "std": round(float(v.std()), 4), "frac_sat": round(float(np.mean(np.abs(v) > NEAR_SATURATION)), 4)}
    return out


def pooled(stats_list) -> dict:
    """Pool per-run stats of equal-length runs: overall min/max, mean saturation fraction, pooled std."""
    out = {}
    for d in DIMS:
        s = [st[d] for st in stats_list]
        means = np.array([x["mean"] for x in s])
        var = np.mean([x["std"] ** 2 + x["mean"] ** 2 for x in s]) - means.mean() ** 2
        out[d] = {"min": min(x["min"] for x in s), "max": max(x["max"] for x in s),
                  "frac_sat": round(float(np.mean([x["frac_sat"] for x in s])), 4),
                  "std": round(float(np.sqrt(max(var, 0.0))), 4)}
    return out
