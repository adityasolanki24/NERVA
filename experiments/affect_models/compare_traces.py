"""RQ8, part 1: Model A vs Model B on a fixed appraisal trace (refactor stage G). Light (< 1 s CPU).

The scripted event timeline of experiments/affect_prototype (v0 appraisal table) is fed to both models.
Criteria fixed before the first run (development log, 2026-10-02):
  bounded      every PAD component stays within [−1, 1]                         (each model)
  recovers     ‖PAD − baseline‖ < 0.05 within 60 s after the last event          (each model)
  agreement    for every event, the sign of the valence change 2 s after it is the same in A and B
               (events with |ΔV| < 0.01 in either model count as "no change" and must agree on that)
Also reported, not judged: Pearson correlation of each PAD dimension between A and B, and the peak
tendencies per event. No winner is chosen from this trace.

Usage: python experiments/affect_models/compare_traces.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "affect_prototype"))
from run import TIMELINE  # noqa: E402

from nerva.affect.appraisal import appraise  # noqa: E402
from nerva.affect.emotions import CategoricalAffectModel  # noqa: E402
from nerva.affect.model_b import DimensionalAffectModel  # noqa: E402
from nerva.interfaces import TENDENCIES, Event  # noqa: E402

DT = 0.1
END_S = max(t for t, _ in TIMELINE) + 60.0


def simulate(model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pending = sorted(TIMELINE)
    ts, pad, tend = [], [], []
    for k in range(int(round(END_S / DT))):
        t = k * DT
        while pending and pending[0][0] <= t + 1e-9:
            model.add(appraise(Event(pending.pop(0)[1])))
        p = model.step(DT)
        ts.append(t + DT)
        pad.append((p.valence, p.arousal, p.dominance))
        tend.append([getattr(model.tendencies, n) for n in TENDENCIES])
    return np.array(ts), np.array(pad), np.array(tend)


def sign(x: float) -> int:
    return 0 if abs(x) < 0.01 else (1 if x > 0 else -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()
    t, pa, ta = simulate(CategoricalAffectModel())
    _, pb, tb = simulate(DimensionalAffectModel())
    last = max(te for te, _ in TIMELINE)

    def recovery(p):
        after = np.where(t > last)[0]
        ok = [i for i in after if np.linalg.norm(p[i]) < 0.05]
        return round(float(t[ok[0]] - last), 1) if ok else None

    events = []
    for te, kind in sorted(TIMELINE):
        i0 = int(round(te / DT)) - 1
        i2 = min(int(round((te + 2.0) / DT)), len(t) - 1)
        da, db = pa[i2, 0] - pa[i0, 0], pb[i2, 0] - pb[i0, 0]
        window = slice(i0 + 1, min(i0 + 1 + int(4.0 / DT), len(t)))
        events.append({"t": te, "event": kind, "dV_A": round(float(da), 3), "dV_B": round(float(db), 3),
                       "agree": sign(da) == sign(db),
                       "peak_tendencies_A": dict(zip(TENDENCIES, np.round(ta[window].max(0), 3).tolist())),
                       "peak_tendencies_B": dict(zip(TENDENCIES, np.round(tb[window].max(0), 3).tolist()))})
    rec_a, rec_b = recovery(pa), recovery(pb)
    result = {
        "bounded": {"A": bool(np.all(np.abs(pa) <= 1)), "B": bool(np.all(np.abs(pb) <= 1))},
        "recovery_s_after_last_event": {"A": rec_a, "B": rec_b},
        "recovers_within_60s": {"A": rec_a is not None, "B": rec_b is not None},
        "valence_sign_agreement": f"{sum(e['agree'] for e in events)}/{len(events)}",
        "pearson": {d: round(float(np.corrcoef(pa[:, i], pb[:, i])[0, 1]), 3)
                    for i, d in enumerate(("valence", "arousal", "dominance"))},
        "peak_abs": {"A": np.round(np.abs(pa).max(0), 3).tolist(), "B": np.round(np.abs(pb).max(0), 3).tolist()},
        "events": events,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "traces.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=1))
    for e in events:
        print(e["t"], e["event"], "dV A", e["dV_A"], "B", e["dV_B"], "agree", e["agree"])


if __name__ == "__main__":
    main()
