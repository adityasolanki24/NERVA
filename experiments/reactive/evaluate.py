"""Evaluate the reactive scenario over several seeds (docs/reactive_behaviour_design.md §5).

Criteria stated before running:
  curiosity   after the ball appears (4 s), the distance to it decreases and the robot stops within 0.3-1.0 m
  fear        after the lunge (onset 46.5 s), the distance to the person increases within 3 s of its end
              and a freeze or retreat mode occurs
  habituation peak interest in the 5 s after the person's second appearance is lower than after the first
  safety      no falls (tilt > 45 deg)
Seeds vary the perception noise and the robot's initial joint noise. Light: ~3 s of CPU per seed.

Usage: python experiments/reactive/evaluate.py --policy S1.onnx [--seeds 5] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario  # noqa: E402

BALL_T, LUNGE_T, LUNGE_END_T = 4.0, 46.5, 47.5
PERSON_APPEARS = (30.0, 72.0)


def window(rows, key, t0, t1):
    return np.array([r[key] for r in rows if t0 <= r["t"] < t1])


def evaluate_seed(policy: str, seed: int, selector: str = "utility", head_limit=None,
                  perception_mode: str = "simulated", backlash: bool = False, neutral: bool = False,
                  appraisal_mode: str = "legacy", affect_model: str = "A", head_pitch_down=None, head_yaw_max=None) -> dict:
    _, rows, _, fired = scenario.run(policy, seed=seed, selector=selector, walking_head_limit=head_limit,
                                     perception_mode=perception_mode, backlash_scene=backlash,
                                     neutral_style=neutral, appraisal_mode=appraisal_mode, affect_model=affect_model,
                                     head_pitch_down=head_pitch_down, head_yaw_max=head_yaw_max)
    ball = window(rows, "dist_ball", BALL_T + 0.1, 30.0)
    person_after = window(rows, "dist_person", LUNGE_END_T, LUNGE_END_T + 3.0)
    modes_after = {r["mode"] for r in rows if LUNGE_T <= r["t"] < LUNGE_T + 5.0}
    # "interest" is Model A's label; for a model without labels the same criterion uses the exploration
    # tendency (identical to interest under Model A, nerva/affect/tendencies.py)
    key = "interest" if affect_model == "A" else "tend_explore"
    interest = [window(rows, key, t, t + 5.0).max() for t in PERSON_APPEARS]
    result = {
        "seed": seed,
        "ball_start_m": float(ball[0]), "ball_min_m": float(ball.min()),
        "curiosity": bool(ball.min() < ball[0] - 0.2 and 0.3 <= ball.min() <= 1.0),
        "person_at_lunge_end_m": float(person_after[0]), "person_3s_later_m": float(person_after[-1]),
        "modes_after_lunge": sorted(modes_after),
        "fear": bool(person_after[-1] > person_after[0] and modes_after & {"freeze", "retreat"}),
        "peak_interest_first_second": [float(x) for x in interest],
        "habituation": bool(interest[1] < interest[0]),
        "max_tilt_deg": float(max(r["tilt_deg"] for r in rows)),
        "events": [(round(te, 1), kind) for te, kind, *_ in fired],
    }
    result["safe"] = result["max_tilt_deg"] < 45.0
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--selector", choices=("utility", "rules"), default="utility")
    ap.add_argument("--perception", choices=("simulated", "vision"), default="simulated")
    ap.add_argument("--backlash", action="store_true", help="evaluate in the scene the policies were trained in")
    ap.add_argument("--neutral-style", action="store_true", help="policy trained on the neutral style only (B1/B2)")
    ap.add_argument("--head-limit", choices=("s1", "s3"), default="s1",
                    help="walking head-offset limit for the policy (S3 tolerates more head motion)")
    ap.add_argument("--appraisal", choices=("legacy", "frames"), default="legacy")
    ap.add_argument("--affect", choices=("A", "B"), default="A")
    ap.add_argument("--head-pitch-down", type=float, default=None, help="downward head limit (B2: -0.2)")
    ap.add_argument("--head-yaw-max", type=float, default=None, help="head yaw limit (B2: 0.4)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()
    from nerva.behaviour.modes import S1_WALKING_HEAD_LIMIT, S3_WALKING_HEAD_LIMIT
    limit = {"s1": S1_WALKING_HEAD_LIMIT, "s3": S3_WALKING_HEAD_LIMIT}[args.head_limit]
    results = [evaluate_seed(args.policy, s, args.selector, limit, args.perception, args.backlash, args.neutral_style,
                             args.appraisal, args.affect, args.head_pitch_down, args.head_yaw_max)
               for s in range(args.seeds)]
    summary = {c: f"{sum(r[c] for r in results)}/{len(results)}" for c in ("curiosity", "fear", "habituation", "safe")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(json.dumps({"summary": summary, "seeds": results}, indent=1) + "\n",
                                              encoding="utf-8")
    for r in results:
        print(f"seed {r['seed']}: curiosity {r['curiosity']} (ball {r['ball_start_m']:.2f}->{r['ball_min_m']:.2f} m)  "
              f"fear {r['fear']} ({r['person_at_lunge_end_m']:.2f}->{r['person_3s_later_m']:.2f} m, {r['modes_after_lunge']})  "
              f"habituation {r['habituation']} {np.round(r['peak_interest_first_second'], 2).tolist()}  "
              f"tilt {r['max_tilt_deg']:.1f}")
    print("SUMMARY", summary)


if __name__ == "__main__":
    main()
