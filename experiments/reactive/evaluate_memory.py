"""Evaluate entity memory (docs/memory_design.md §5, phase M1) on the two-person scenario.

memory_scenario(): A lunges early; B (a new person) approaches, later pets the robot; A returns 70 s after
the lunge; then B returns. Perception is image-based (colour + depth vision). Criteria stated before
running, each compared with memory ON and OFF (ablation):
  b_not_blamed   while B first approaches (46-58 s) the robot is not wary of B (no watch/retreat/freeze)
  a_remembered   while A returns (94-106 s) the robot is wary (watch/retreat/freeze ≥ 50 % of the time)
  b_welcomed     after B returns (120-130 s) the robot approaches or inspects B
  touch_to_b     (memory only) B's warmth > 0 > A's warmth after the petting, and A ≠ B as identities
Refactor stage D adds (criteria above unchanged):
  --learning legacy|grounded   entity/place memory learns from emotions (legacy) or measured outcomes
  --backlash --neutral-style   evaluate a neutral-style policy (B2) in its training scene
  a_threat_40 / a_threat_116   person#0's threat when A has left (40 s) and at the end of A's return (116 s);
                               a_outcomes_between = adverse outcomes attributed to A in between. Recorded to
                               check self-reinforcement (RQ7): no growth expected without a new adverse outcome.
Usage: python experiments/reactive/evaluate_memory.py --policy S1.onnx [--seeds 5] [--learning grounded]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario  # noqa: E402

from nerva.analysis.pad_metrics import pad_stats, pooled  # noqa: E402

WARY = {"watch", "retreat", "freeze"}


def modes(rows, t0, t1):
    return [r["mode"] for r in rows if t0 <= r["t"] < t1]


def _threat_at(sim, eid: str, t: float):
    best = None
    for tt, recs in getattr(sim, "memory_trace", []):
        if tt <= t and eid in recs:
            best = recs[eid][0]
    return best


def evaluate(policy: str, seed: int, use_memory: bool, learning: str = "legacy", backlash: bool = False,
             neutral: bool = False, appraisal_mode: str = "legacy", affect_model: str | None = None,
             head_pitch_down=None, head_yaw_max=None, profile: str = "v2", margin_controllability: bool = False) -> dict:
    sim, rows, _, _ = scenario.run(policy, seed=seed, duration=135.0, agents=scenario.memory_scenario(),
                                   perception_mode="vision", use_memory=use_memory, memory_learning=learning,
                                   backlash_scene=backlash, neutral_style=neutral, appraisal_mode=appraisal_mode,
                                   affect_model=affect_model, head_pitch_down=head_pitch_down,
                                   head_yaw_max=head_yaw_max, profile=profile, margin_controllability=margin_controllability)
    first_b, return_a, return_b = modes(rows, 46, 58), modes(rows, 94, 106), modes(rows, 120, 130)
    result = {
        "seed": seed, "memory": use_memory,
        "b_not_blamed": not (set(first_b) & WARY),
        "a_remembered": sum(m in WARY for m in return_a) >= 0.5 * len(return_a),
        "b_welcomed": bool(set(return_b) & {"approach", "inspect"}),
        "max_tilt_deg": max(r["tilt_deg"] for r in rows),
        "fell": max(r["tilt_deg"] for r in rows) > 45.0,
        "pad": pad_stats(rows),
        "d_a_return": sum(r["dominance"] for r in rows if 94 <= r["t"] < 106) / max(1, len(return_a)),
        "d_b_return": sum(r["dominance"] for r in rows if 120 <= r["t"] < 130) / max(1, len(return_b)),
    }
    if use_memory:
        recs = sorted(sim.memory.records.values(), key=lambda r: r.eid)
        people = [r for r in recs if r.kind == "person"]
        result["identities"] = {r.eid: {"threat": round(r.threat, 3), "warmth": round(r.warmth, 3),
                                        "trust": round(r.trust, 3)} for r in recs}
        result["touch_to_b"] = (len(people) == 2 and people[1].warmth > 0 > people[0].warmth)
        result["learning"] = learning
        result["a_threat_40"] = _threat_at(sim, "person#0", 40.0)
        result["a_threat_116"] = _threat_at(sim, "person#0", 116.0)
        result["outcomes"] = [(round(o.time_s, 1), o.kind, o.source) for o in sim.outcomes + sim.risks]
    return result


def evaluate_together(policy: str, seed: int, use_memory: bool, learning: str = "legacy", backlash: bool = False,
                      neutral: bool = False, appraisal_mode: str = "legacy", affect_model: str | None = None,
                      profile: str = "v2", margin_controllability: bool = False) -> dict:
    """A (feared) and B (liked) return together at 95 s. Criteria, stated in advance, over 100-125 s:
    avoids_a: A never closer than 1.0 m, and watch/retreat/freeze occurs; engages_b: approach/inspect
    occurs and B comes closer than A on average."""
    _, rows, _, _ = scenario.run(policy, seed=seed, duration=125.0, agents=scenario.together_scenario(),
                                 perception_mode="vision", use_memory=use_memory, memory_learning=learning,
                                 backlash_scene=backlash, neutral_style=neutral, appraisal_mode=appraisal_mode,
                                 affect_model=affect_model, profile=profile,
                                 margin_controllability=margin_controllability)
    tilt = max(r["tilt_deg"] for r in rows)
    w = [r for r in rows if 100 <= r["t"] < 125]
    ms = {r["mode"] for r in w}
    mean_a = sum(r["dist_person"] for r in w) / len(w)
    mean_b = sum(r["dist_person_b"] for r in w) / len(w)
    return {"seed": seed, "memory": use_memory,
            "avoids_a": min(r["dist_person"] for r in w) >= 1.0 and bool(ms & WARY),
            "engages_b": bool(ms & {"approach", "inspect"}) and mean_b < mean_a,
            "mean_dist_a": round(mean_a, 2), "mean_dist_b": round(mean_b, 2), "fell": tilt > 45.0,
            "pad": pad_stats(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--together", action="store_true", help="only the two-people-at-once scenario")
    ap.add_argument("--learning", choices=("legacy", "grounded"), default="legacy")
    ap.add_argument("--backlash", action="store_true", help="evaluate in the scene the policies were trained in")
    ap.add_argument("--neutral-style", action="store_true", help="policy trained on the neutral style only (B1/B2)")
    ap.add_argument("--no-ablation", action="store_true", help="skip the memory-OFF runs")
    ap.add_argument("--profile", choices=("legacy", "v2"), default="v2")
    ap.add_argument("--appraisal", choices=("legacy", "frames"), default="legacy")
    ap.add_argument("--margin-controllability", action="store_true",
                    help="in-view controllability of people = reaction margin (2026-10-06)")
    ap.add_argument("--affect", choices=("A", "B", "Bv2", "Bv3"), default=None, help="default: Bv2 for v2, A for legacy")
    ap.add_argument("--head-pitch-down", type=float, default=None, help="downward head limit (B2: -0.2)")
    ap.add_argument("--head-yaw-max", type=float, default=None, help="head yaw limit (B2: 0.4)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results_memory")
    args = ap.parse_args()
    if args.together:
        mems = (True,) if args.no_ablation else (True, False)
        results = [evaluate_together(args.policy, s, m, args.learning, args.backlash, args.neutral_style,
                                     args.appraisal, args.affect, args.profile, args.margin_controllability)
                   for m in mems for s in range(args.seeds)]
        summary = {("memory" if m else "no_memory"): {k: f"{sum(r[k] for r in results if r['memory'] == m)}/{args.seeds}"
                                                     for k in ("avoids_a", "engages_b")} for m in mems}
        summary["pad_pooled_memory"] = pooled([r["pad"] for r in results if r["memory"]])
        for r in results:
            print(r)
        print("SUMMARY", json.dumps(summary))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "evaluation_together.json").write_text(
            json.dumps({"summary": summary, "runs": results}, indent=1) + "\n", encoding="utf-8")
        return
    mems = (True,) if args.no_ablation else (True, False)
    results = [evaluate(args.policy, s, m, args.learning, args.backlash, args.neutral_style, args.appraisal, args.affect,
                        args.head_pitch_down, args.head_yaw_max, args.profile, args.margin_controllability)
               for m in mems for s in range(args.seeds)]
    summary = {}
    for m in mems:
        rs = [r for r in results if r["memory"] == m]
        keys = ["b_not_blamed", "a_remembered", "b_welcomed"] + (["touch_to_b"] if m else [])
        summary["memory" if m else "no_memory"] = {k: f"{sum(r[k] for r in rs)}/{len(rs)}" for k in keys}
    summary["pad_pooled_memory"] = pooled([r["pad"] for r in results if r["memory"]])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(json.dumps({"summary": summary, "runs": results}, indent=1) + "\n",
                                              encoding="utf-8")
    for r in results:
        print({k: v for k, v in r.items() if k != "identities"}, r.get("identities", ""))
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
