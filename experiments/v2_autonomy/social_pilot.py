"""Run a local outcome-grounded social-attention pilot and write auditable traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reactive.scenario import memory_scenario, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="existing neutral NERVA walking policy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--duration", type=float, default=135.0)
    args = ap.parse_args()
    if args.duration < 120.0:
        raise ValueError("social pilot needs the later re-encounter; duration must be at least 120 s")

    sim, rows, _, _ = run(args.policy, seed=args.seed, duration=args.duration, agents=memory_scenario(),
                          perception_mode="vision", use_memory=True, attention=True)
    greetings = [row for row in rows if row["mode"] == "greet"]
    summary = {
        "seed": args.seed,
        "duration_s": args.duration,
        "perception": "rendered_rgbd",
        "memory_learning": "grounded",
        "attention": True,
        "greeting_frames": len(greetings),
        "greeting_targets": sorted({row["target"] for row in greetings}),
        "greeting_first_s": min((row["t"] for row in greetings), default=None),
        "max_social_affinity": max(row["social_affinity"] for row in rows),
        "max_tilt_deg": max(row["tilt_deg"] for row in rows),
        "safety_interventions": sim.safety_interventions,
        "modes": dict(Counter(row["mode"] for row in rows)),
        "attention_targets": dict(Counter(row["attention_target"] for row in rows if row["attention_target"])),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.out / "timeline.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
