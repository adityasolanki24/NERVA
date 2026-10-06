"""Apply the preregistered scenario criteria of the Model B saturation fix to results_scenarios/ (2026-10-02).

Per model (A, B, Bv2), over the default (vision), two-person and together scenarios, 5 seeds each:
  behaviour   all existing criteria pass; no falls
  saturation  time with |x| > 0.9 ≤ 5% per dimension (sample-weighted over all runs)
  range       pooled std ≥ 0.05 for valence and arousal (dominance reported)
  response    max |V| ≥ 0.2 and max |A| ≥ 0.2 in every scenario

With --root results_controllability --models Bv2 it also applies the 2026-10-06 dominance criteria:
pooled min D ≤ −0.10; mean D over the lunge (46.5–49.5 s) < 0 in ≥ 4/5 default seeds; mean D while A returns
< while B returns in ≥ 4/5 two-person seeds; pooled std of D ≥ 0.05.

Usage: python experiments/affect_models/summarise_scenarios.py [--root DIR] [--models A B Bv2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "results_scenarios"
DIMS = ("valence", "arousal", "dominance")
DURATION = {"default": 100.0, "memory": 135.0, "together": 125.0}


def load(model: str) -> dict:
    out = {}
    for sc in DURATION:
        f = ROOT / f"{sc}_{model}" / ("evaluation_together.json" if sc == "together" else "evaluation.json")
        d = json.loads(f.read_text(encoding="utf-8"))
        out[sc] = d.get("seeds") or d["runs"]
    return out


def behaviour(sc: str, runs) -> tuple[bool, str]:
    keys = {"default": ("curiosity", "fear", "habituation", "safe"),
            "memory": ("b_not_blamed", "a_remembered", "b_welcomed", "touch_to_b"),
            "together": ("avoids_a", "engages_b")}[sc]
    runs = [r for r in runs if r.get("memory", True)]
    counts = {k: sum(bool(r[k]) for r in runs) for k in keys}
    falls = sum(bool(r.get("fell", False)) for r in runs)
    ok = all(v == len(runs) for v in counts.values()) and falls == 0
    return ok, ", ".join(f"{k} {v}/{len(runs)}" for k, v in counts.items()) + f", falls {falls}"


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_scenarios")
    ap.add_argument("--models", nargs="+", default=["A", "B", "Bv2"])
    args = ap.parse_args()
    ROOT = Path(__file__).resolve().parent / args.root
    report = {}
    for model in args.models:
        data = load(model)
        weights, stats = [], []
        per_sc = {}
        beh_ok = True
        for sc, runs in data.items():
            runs = [r for r in runs if r.get("memory", True)]
            ok, text = behaviour(sc, runs)
            beh_ok &= ok
            per_sc[sc] = {"behaviour": text,
                          "max_abs_V": max(max(abs(r["pad"]["valence"]["min"]), abs(r["pad"]["valence"]["max"])) for r in runs),
                          "max_abs_A": max(max(abs(r["pad"]["arousal"]["min"]), abs(r["pad"]["arousal"]["max"])) for r in runs)}
            for r in runs:
                weights.append(DURATION[sc])
                stats.append(r["pad"])
        w = np.array(weights) / sum(weights)
        pooled = {}
        for d in DIMS:
            means = np.array([s[d]["mean"] for s in stats])
            ex2 = np.array([s[d]["std"] ** 2 + s[d]["mean"] ** 2 for s in stats])
            pooled[d] = {"frac_sat": round(float(w @ np.array([s[d]["frac_sat"] for s in stats])), 4),
                         "std": round(float(np.sqrt(max(w @ ex2 - (w @ means) ** 2, 0.0))), 4),
                         "min": min(s[d]["min"] for s in stats), "max": max(s[d]["max"] for s in stats)}
        crit = {
            "behaviour": beh_ok,
            "saturation_le_5pct": all(pooled[d]["frac_sat"] <= 0.05 for d in DIMS),
            "range_std_V_A_ge_0_05": pooled["valence"]["std"] >= 0.05 and pooled["arousal"]["std"] >= 0.05,
            "response_every_scenario": all(v["max_abs_V"] >= 0.2 and v["max_abs_A"] >= 0.2 for v in per_sc.values()),
        }
        if "d_lunge_mean" in data["default"][0]:
            lunge = [r["d_lunge_mean"] for r in data["default"]]
            mem = [r for r in data["memory"] if r.get("memory", True)]
            crit["dominance_two_sided"] = pooled["dominance"]["min"] <= -0.10
            crit["dominance_low_in_lunge"] = sum(x < 0 for x in lunge) >= 4
            crit["dominance_A_below_B"] = sum(r["d_a_return"] < r["d_b_return"] for r in mem) >= 4
            crit["dominance_std"] = pooled["dominance"]["std"] >= 0.05
            per_sc["default"]["d_lunge_mean"] = [round(x, 3) for x in lunge]
            if data["default"][0].get("d_min_after_rapid") is not None:
                after = [r["d_min_after_rapid"] for r in data["default"]]
                crit["dominance_low_after_rapid_event"] = sum(x < 0 for x in after) >= 4
                crit.pop("dominance_low_in_lunge")  # replaced by the event-anchored criterion (2026-10-06)
                per_sc["default"]["d_min_after_rapid"] = [round(x, 3) for x in after]
            per_sc["memory"]["d_a_vs_b"] = [(round(r["d_a_return"], 3), round(r["d_b_return"], 3)) for r in mem]
        report[model] = {"criteria": crit, "all_pass": all(crit.values()), "pooled_pad": pooled, "scenarios": per_sc}
    (ROOT / "summary.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for model, r in report.items():
        print(f"== {model}: all pass {r['all_pass']}  {r['criteria']}")
        for d, v in r["pooled_pad"].items():
            print(f"   {d:9s} min {v['min']:+.2f} max {v['max']:+.2f} std {v['std']:.3f} |x|>0.9 {100 * v['frac_sat']:.1f}%")
        for sc, v in r["scenarios"].items():
            print(f"   {sc:8s} {v['behaviour']}  max|V| {v['max_abs_V']:.2f} max|A| {v['max_abs_A']:.2f}"
                  + (f"  D lunge {v['d_lunge_mean']}" if "d_lunge_mean" in v else "")
                  + (f"  min D after rapid {v['d_min_after_rapid']}" if "d_min_after_rapid" in v else "")
                  + (f"  D A vs B {v['d_a_vs_b']}" if "d_a_vs_b" in v else ""))


if __name__ == "__main__":
    main()
