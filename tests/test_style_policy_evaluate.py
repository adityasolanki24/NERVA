import numpy as np

from experiments.style_policy.evaluate import FEATURES, LEVELS, analyse, style_vec


def _rows(effect=1.0, crosstalk=0.0, seeds=range(10)):
    """Synthetic trials: dimension k moves its own feature by effect·level (+ small seed noise)."""
    rows, rng = [], np.random.default_rng(0)
    base = {"gait_hz": 1.8, "lift_mm": 12.0, "pitch_mean_deg": 3.0, "v_fwd": 0.12}
    for s in seeds:
        noise = {f: rng.normal(0, 0.01) for f in base}
        for k, feature in FEATURES.items():
            for lv in LEVELS:
                m = {f: base[f] + noise[f] for f in base}
                m[feature] += effect * lv
                m["v_fwd"] += crosstalk * lv
                rows.append({"policy": "S1", "style": list(style_vec(k, lv)), "seed": s,
                             "fell": 0.0, "tracking_err": abs(m["v_fwd"] - 0.15), **m})
        for p, st in (("B0", None), ("B1", [0.0, 0.0, 0.0])):
            rows.append({"policy": p, "style": st, "seed": s, "fell": 0.0, "tracking_err": 0.03,
                         **{f: base[f] for f in base}})
    return rows


def test_clean_style_effect_passes():
    report = analyse(_rows(), list(range(10)))
    assert all(report["criteria"]["monotonic_10_of_10"].values())
    assert all(report["criteria"]["crosstalk_smaller"].values())
    assert report["criteria"]["no_falls"] and report["dimensions"]["e1"]["direction"] == "increasing"


def test_crosstalk_larger_than_intended_effect_fails():
    report = analyse(_rows(effect=0.02, crosstalk=0.2), list(range(10)))
    assert not report["overall_pass"]
    assert not any(report["criteria"]["crosstalk_smaller"].values())
