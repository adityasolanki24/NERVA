"""Model B saturation fix: fixed-trace and convergence criteria for Model A, old Model B and Model B v2.

Preregistered in docs/development_log.md (2026-10-02, "Model B saturation: diagnosis ..."). Light (< 5 s CPU).

1. Fixed trace (affect-prototype timeline, v0 appraisal table):
     bounded; recovery to ||PAD|| < 0.05 within 60 s after the last event;
     event response direction: effect of each event = PAD(with all events) − PAD(without that event) at
     +2 s; the valence effect must have the sign of the event's desirability (every event with d ≠ 0), and
     the arousal effect must be > 0 for events with expectedness < 0.3.
2. Convergence: one identical persistent appraisal (r 0.6, d 0, l 0.5, e 0.3, c 0.8) from one source:
     every 2 s for 120 s: |mean(110–120 s) − mean(50–60 s)| < 0.02 per dimension, and max |x| < 0.9;
     rate invariance: mean(110–120 s) at a 1 s period within 0.05 of the 2 s value, per dimension.

Usage: python experiments/affect_models/saturation.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "affect_prototype"))
from run import TIMELINE  # noqa: E402

from nerva.affect.appraisal import EVENT_APPRAISALS, appraise  # noqa: E402
from nerva.affect.emotions import CategoricalAffectModel  # noqa: E402
from nerva.affect.model_b import AttractorAffectModel, DimensionalAffectModel, FacetAttractorAffectModel, OnsetAttractorAffectModel  # noqa: E402
from nerva.interfaces import AppraisalFrame, Event, OutcomeHypothesis  # noqa: E402

DT = 0.1
MODELS = {"A": CategoricalAffectModel, "B": DimensionalAffectModel, "Bv2": AttractorAffectModel,
          "Bv3": OnsetAttractorAffectModel, "Bv4": FacetAttractorAffectModel}
PERSISTENT = AppraisalFrame(OutcomeHypothesis("novel_stimulus", subject="ball-0", probability=0.5), relevance=0.6,
                            desirability=0.0, expectedness=0.3, controllability=0.8, persistent=True)


def trace(make, timeline, end_s):
    m, pending, out = make(), sorted(timeline), []
    for k in range(int(round(end_s / DT))):
        t = k * DT
        while pending and pending[0][0] <= t + 1e-9:
            m.add(appraise(Event(pending.pop(0)[1])))
        p = m.step(DT)
        out.append((p.valence, p.arousal, p.dominance))
    return np.array(out)


def fixed_trace(make) -> dict:
    last = max(t for t, _ in TIMELINE)
    end = last + 60.0
    full = trace(make, TIMELINE, end)
    t = (np.arange(len(full)) + 1) * DT
    after = np.where(t > last)[0]
    rec = next((float(t[i] - last) for i in after if np.linalg.norm(full[i]) < 0.05), None)
    events = []
    for i, (te, kind) in enumerate(sorted(TIMELINE)):
        without = trace(make, [ev for j, ev in enumerate(sorted(TIMELINE)) if j != i], end)
        k2 = int(round((te + 2.0) / DT)) - 1
        dv, da, dd = (full[k2] - without[k2]).tolist()
        a = EVENT_APPRAISALS[kind]
        ok_v = a.desirability == 0 or np.sign(dv) == np.sign(a.desirability)
        ok_a = a.expectedness >= 0.3 or da > 0
        events.append({"t": te, "event": kind, "dV": round(dv, 4), "dA": round(da, 4), "dD": round(dd, 4),
                       "valence_direction_ok": bool(ok_v), "arousal_ok": bool(ok_a)})
    return {"bounded": bool(np.all(np.abs(full) <= 1.0)), "recovery_s": rec,
            "recovers_within_60s": rec is not None,
            "direction_ok": f"{sum(e['valence_direction_ok'] and e['arousal_ok'] for e in events)}/{len(events)}",
            "events": events}


def repeated(make, period_s: float, end_s: float = 120.0) -> np.ndarray:
    m, out = make(), []
    every = int(round(period_s / DT))
    for k in range(int(round(end_s / DT))):
        if k % every == 0:
            m.add(PERSISTENT)
        p = m.step(DT)
        out.append((p.valence, p.arousal, p.dominance))
    return np.array(out)


def convergence(make) -> dict:
    two, one = repeated(make, 2.0), repeated(make, 1.0)

    def window(x, a, b):
        return x[int(a / DT):int(b / DT)].mean(0)

    drift = np.abs(window(two, 110, 120) - window(two, 50, 60))
    rate_gap = np.abs(window(one, 110, 120) - window(two, 110, 120))
    return {"steady_2s": np.round(window(two, 110, 120), 3).tolist(),
            "steady_1s": np.round(window(one, 110, 120), 3).tolist(),
            "drift_60_to_120": np.round(drift, 4).tolist(), "converges": bool(np.all(drift < 0.02)),
            "max_abs": round(float(np.abs(two).max()), 3), "below_0_9": bool(np.abs(two).max() < 0.9),
            "rate_gap": np.round(rate_gap, 4).tolist(), "rate_invariant": bool(np.all(rate_gap < 0.05))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()
    result = {name: {"fixed_trace": fixed_trace(make), "convergence": convergence(make)} for name, make in MODELS.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "saturation.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for name, r in result.items():
        ft, cv = r["fixed_trace"], r["convergence"]
        print(f"{name:4s} bounded {ft['bounded']}  recovery {ft['recovery_s']} s  direction {ft['direction_ok']}  | "
              f"converges {cv['converges']} max {cv['max_abs']}  rate-invariant {cv['rate_invariant']}  "
              f"steady 2s {cv['steady_2s']} 1s {cv['steady_1s']}")
        for e in ft["events"]:
            if not (e["valence_direction_ok"] and e["arousal_ok"]):
                print("      direction fail:", e)


if __name__ == "__main__":
    main()
