"""RQ9: do learned event prototypes predict grounded outcomes as well as the hand-coded event labels?

Refactor stage H (docs/architecture.md; docs/research_questions.md RQ9). Simulation only, light (CPU,
about 2 minutes). Protocol fixed before the first run (development log, 2026-10-02):

Data   the reactive scenarios (default, memory, together) × seeds 0..5, interleaved by seed, policy B2
       (neutral style, backlash scene), simulated detector. One feature vector per perception frame (10 Hz):
         x = [ d / 2 m, closing speed / 0.5 m/s, |bearing| / 1 rad, visible, touched this frame, tilt / 10° ]
       for the nearest person track (none: d = 5 m, other entries 0).
Learned BoundaryDetector (prediction-error boundaries) → embedding [x_after, x_after − x_before]
       → PrototypeMemory (online, ≤ 12 prototypes). Prediction model and prototypes persist across runs.
Targets per frame: an adverse event (collision_risk estimate or stability_loss) within the next 3 s; a benign_contact
       within the next 3 s (OutcomeMonitor, nerva/world/outcomes.py).
Predictors per frame: context = the most recent prototype (learned) or hand-coded appraised event kind
       (baseline) within the last 2 s, else "none"; P(target | context) from a Beta(0.5, 4.5)-smoothed table.
       Base rate: P(target) from the same prior and the frames seen so far.
Evaluation: run-level prequential. Each run is predicted with tables learned from earlier runs only, then
       the tables are updated with that run. Metric: mean Brier score over all frames (lower is better).
Reading (fixed in advance): RQ9 is supported if the learned prototypes' Brier score is ≤ the hand-coded
       labels' for both targets, falsified if worse on either. Both are also compared with the base rate.
       Prototype ↔ label correspondence (normalised mutual information at boundaries) is reported, not judged.

Usage: python experiments/event_learning/run.py --policy B2.onnx [--seeds 6] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reactive"))
import scenario  # noqa: E402

from nerva.events.prototypes import PrototypeMemory  # noqa: E402
from nerva.events.segmentation import BoundaryDetector  # noqa: E402

HORIZON_S, CONTEXT_S = 3.0, 2.0
PRIOR_A, PRIOR_B = 0.5, 4.5
FRAME_DT = 1.0 / scenario.FRAME_HZ
SCENARIOS = {"default": (scenario.default_scenario, 100.0), "memory": (scenario.memory_scenario, 135.0),
             "together": (scenario.together_scenario, 125.0)}
# "near_collision" was renamed "collision_risk" (a RiskEstimate) on 2026-10-02; same detector and threshold
TARGETS = {"adverse": ("collision_risk", "contact_impact", "stability_loss"), "benign": ("benign_contact",)}


def features(tracks, touched: bool, tilt: float) -> np.ndarray:
    people = [tr for tr in tracks if tr.kind == "person"]
    if not people:
        return np.array([2.5, 0.0, 0.0, 0.0, float(touched), tilt / 10.0])
    p = min(people, key=lambda tr: tr.distance)
    return np.array([p.distance / 2.0, p.approach_speed / 0.5, abs(p.bearing), float(p.visible), float(touched),
                     tilt / 10.0])


def collect(policy: str, name: str, seed: int) -> dict:
    make, duration = SCENARIOS[name]
    sim, rows, frames, fired = scenario.run(policy, seed=seed, duration=duration, agents=make(),
                                            record_every=scenario.PERCEIVE_EVERY, backlash_scene=True,
                                            neutral_style=True)
    touches = [t for t, kind, *_ in fired if kind == "touch_gentle"]
    xs, ts = [], []
    for i, fr in enumerate(frames):
        t = i * scenario.PERCEIVE_EVERY * scenario.CTRL_DT
        tilt = rows[min(i * scenario.PERCEIVE_EVERY, len(rows) - 1)]["tilt_deg"]
        touched = any(abs(t - tt) < FRAME_DT / 2 for tt in touches)
        xs.append(features(fr[3], touched, tilt))
        ts.append(t)
    return {"name": name, "seed": seed, "t": np.array(ts), "x": np.array(xs),
            "events": [(t, kind) for t, kind, *_ in fired],
            "outcomes": [(o.time_s, o.kind) for o in sim.outcomes + sim.risks]}


def targets_for(run: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, kinds in TARGETS.items():
        times = [t for t, k in run["outcomes"] if k in kinds]
        out[name] = np.array([any(t < to <= t + HORIZON_S for to in times) for t in run["t"]], dtype=float)
    return out


def context_series(times, stamped) -> list[str]:
    """For each frame time, the most recent (time, label) within CONTEXT_S before or at it, else 'none'."""
    ctx, j, last = [], 0, None
    stamped = sorted(stamped)
    for t in times:
        while j < len(stamped) and stamped[j][0] <= t + 1e-9:
            last = stamped[j]
            j += 1
        ctx.append(last[1] if last is not None and t - last[0] <= CONTEXT_S else "none")
    return ctx


class RateTable:
    def __init__(self):
        self.n, self.k = defaultdict(float), defaultdict(float)

    def p(self, ctx: str) -> float:
        return (self.k[ctx] + PRIOR_A) / (self.n[ctx] + PRIOR_A + PRIOR_B)

    def update(self, ctxs, ys) -> None:
        for c, y in zip(ctxs, ys):
            self.n[c] += 1
            self.k[c] += y


def nmi(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    pa, pb, pab = Counter(a), Counter(b), Counter(zip(a, b))
    mi = sum(c / n * math.log((c / n) / (pa[x] / n * pb[y] / n)) for (x, y), c in pab.items())
    ha = -sum(c / n * math.log(c / n) for c in pa.values())
    hb = -sum(c / n * math.log(c / n) for c in pb.values())
    return float(mi / math.sqrt(ha * hb)) if ha > 0 and hb > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()
    t0 = time.time()
    runs = [collect(args.policy, name, s) for s in range(args.seeds) for name in SCENARIOS]
    scale = np.ones(6)  # features are pre-scaled to O(1) in features()
    detector, protos = BoundaryDetector(6), PrototypeMemory()
    tables = {m: {tg: RateTable() for tg in TARGETS} for m in ("learned", "handcoded", "base")}
    sq = {m: {tg: [] for tg in TARGETS} for m in tables}
    pairs, n_boundaries = [], 0
    for run in runs:
        detector._prev = None  # a new episode: no prediction across the reset
        stamped, prev = [], None
        for t, x in zip(run["t"], run["x"] / scale):
            if detector.step(x) and prev is not None:
                p = protos.assign(np.concatenate([x, x - prev]))
                stamped.append((t, f"P{p.pid}"))
                near = [k for te, k in run["events"] if abs(te - t) <= 0.5]
                pairs.append((f"P{p.pid}", near[0] if near else "none"))
                n_boundaries += 1
            prev = x
        ctx = {"learned": context_series(run["t"], stamped),
               "handcoded": context_series(run["t"], run["events"]),
               "base": ["all"] * len(run["t"])}
        ys = targets_for(run)
        for m in tables:
            for tg in TARGETS:
                preds = np.array([tables[m][tg].p(c) for c in ctx[m]])
                sq[m][tg].extend(((preds - ys[tg]) ** 2).tolist())
        for m in tables:  # learn from this run only after predicting it
            for tg in TARGETS:
                tables[m][tg].update(ctx[m], ys[tg])
        # prototype outcome tables, for inspection
        for (t, lab) in stamped:
            pid = int(lab[1:])
            followed = [k for to, k in run["outcomes"] if t < to <= t + HORIZON_S]
            protos.resolve(protos.prototypes[pid], followed)
    brier = {m: {tg: float(np.mean(v)) for tg, v in d.items()} for m, d in sq.items()}
    verdict = all(brier["learned"][tg] <= brier["handcoded"][tg] for tg in TARGETS)
    summary = {
        "runs": len(runs), "frames": len(sq["base"]["adverse"]), "boundaries": n_boundaries,
        "prototypes": len(protos.prototypes), "brier": brier,
        "rq9_supported": verdict,
        "nmi_prototype_vs_label_at_boundaries": nmi([a for a, _ in pairs], [b for _, b in pairs]),
        "prototype_table": [{"pid": p.pid, "count": p.count, "centre": np.round(p.centre, 2).tolist(),
                             "p_adverse": round(max(p.p(k) for k in TARGETS["adverse"]), 3),
                             "p_benign": round(p.p("benign_contact"), 3)} for p in protos.prototypes],
        "label_counts_at_boundaries": Counter(b for _, b in pairs).most_common(),
        "seconds": round(time.time() - t0, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "prototype_table"}, indent=1))
    for row in summary["prototype_table"]:
        print(row)


if __name__ == "__main__":
    main()
