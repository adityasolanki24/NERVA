"""Local matched-seed comparisons for the v2 attention, memory and expression pilots.

The world schedules and MuJoCo seed are identical for every condition in a
seed.  This is an engineering comparison, not a statistical validation or a
claim of social generalisation.  Every run saves its complete control trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reactive import scenario  # noqa: E402
from experiments.v2_autonomy.scenarios import curiosity_repeat_scenario, social_history_scenario  # noqa: E402


CONDITIONS = {
    "full": {"selector": "utility", "use_memory": True, "attention": True},
    "no_memory": {"selector": "utility", "use_memory": False, "attention": True},
    "no_attention": {"selector": "utility", "use_memory": True, "attention": False},
    "fixed_rules": {"selector": "rules", "use_memory": False, "attention": False},
}


def _safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    return value


def _write_trace(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_safe(row), separators=(",", ":")) + "\n")


def _duration(rows, predicate) -> float:
    return round(sum(predicate(row) for row in rows) * scenario.CTRL_DT, 2)


def _gaze(rows, start: float, end: float) -> dict:
    errors = [row["gaze_command_error_rad"] for row in rows
              if start <= row["t"] < end and row["gaze_target_visible"]
              and np.isfinite(row["gaze_command_error_rad"])]
    return {"tracked_s": round(len(errors) * scenario.CTRL_DT, 2),
            "mean_command_error_rad": round(float(np.mean(errors)), 3) if errors else None}


def _motion_ranges(rows, start: float, end: float) -> dict:
    focus = [row for row in rows if start <= row["t"] < end]
    return {key: round(float(np.ptp([row[key] for row in focus])), 3) if focus else 0.0
            for key in ("neck_pitch_actual", "head_pitch_actual", "head_yaw_actual", "head_roll_actual")}


def _common(rows, sim, start: float, end: float) -> dict:
    close = [row["dist_person"] for row in rows if start <= row["t"] < end and row["dist_person"] < 20.0]
    return {
        "attention_target_s": _duration(rows, lambda row: start <= row["t"] < end and bool(row["attention_target"])),
        "approach_s": _duration(rows, lambda row: start <= row["t"] < end and row["mode"] == "approach"),
        "retreat_s": _duration(rows, lambda row: start <= row["t"] < end and row["mode"] == "retreat"),
        "inspect_s": _duration(rows, lambda row: start <= row["t"] < end and row["mode"] == "inspect"),
        "greeting_s": _duration(rows, lambda row: start <= row["t"] < end and row["mode"] == "greet"),
        "min_person_distance_m": round(float(min(close)), 3) if close else None,
        "max_tilt_deg": round(max(row["tilt_deg"] for row in rows if start <= row["t"] < end), 3),
        "safety_interventions": sim.safety_interventions,
        "risk_estimates": len(sim.risks),
        "gaze": _gaze(rows, start, end),
        "actual_head_range_rad": _motion_ranges(rows, start, end),
    }


def _curiosity_metrics(rows, sim) -> dict:
    metrics = _common(rows, sim, 4.0, 56.0)
    metrics.update({
        "first_inspection_s": _duration(rows, lambda row: 4.0 <= row["t"] < 36.0 and row["mode"] == "inspect"),
        "repeat_inspection_s": _duration(rows, lambda row: 42.0 <= row["t"] < 56.0 and row["mode"] == "inspect"),
        "ball_attention_first_s": _duration(rows, lambda row: 4.0 <= row["t"] < 36.0 and row["attention_target"].startswith("ball")),
        "ball_attention_repeat_s": _duration(rows, lambda row: 42.0 <= row["t"] < 56.0 and row["attention_target"].startswith("ball")),
        "mode_seconds": {mode: _duration(rows, lambda row, mode=mode: row["mode"] == mode)
                         for mode in sorted({row["mode"] for row in rows})},
    })
    return metrics


def _social_metrics(rows, sim) -> dict:
    metrics = _common(rows, sim, 38.0, 56.0)
    metrics.update({
        "watch_s": _duration(rows, lambda row: 38.0 <= row["t"] < 56.0 and row["mode"] == "watch"),
        "max_affinity": round(max(row["social_affinity"] for row in rows), 3),
        "max_memory_warmth": round(max((row["mem_warmth"] for row in rows if np.isfinite(row["mem_warmth"])), default=0.0), 3),
        "mode_seconds_test": {mode: _duration(rows, lambda row, mode=mode: 38.0 <= row["t"] < 56.0 and row["mode"] == mode)
                              for mode in sorted({row["mode"] for row in rows})},
    })
    return metrics


def _policy_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="62,63", help="comma-separated MuJoCo seeds")
    parser.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated condition names")
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    conditions = [item for item in args.conditions.split(",") if item]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out / "traces"
    traces_dir.mkdir(exist_ok=True)
    results = []
    scenario_specs = (("curiosity", curiosity_repeat_scenario, _curiosity_metrics),
                      ("positive", lambda: social_history_scenario("positive"), _social_metrics),
                      ("cautious", lambda: social_history_scenario("cautious"), _social_metrics))
    for seed in seeds:
        for condition in conditions:
            for name, make_agents, metric_fn in scenario_specs:
                sim, rows, _, _ = scenario.run(args.policy, seed=seed, duration=56.0, agents=make_agents(),
                                                perception_mode="vision", **CONDITIONS[condition])
                _write_trace(traces_dir / f"{condition}_{name}_seed{seed}.jsonl", rows)
                results.append({"seed": seed, "condition": condition, "scenario": name,
                                "metrics": metric_fn(rows, sim)})
                print(f"finished seed={seed} condition={condition} scenario={name}", flush=True)
    metadata = {
        "purpose": "local matched-seed engineering comparison; not statistical evidence",
        "policy": {"path": str(Path(args.policy).resolve()), "sha256": _policy_hash(Path(args.policy))},
        "seeds": seeds, "duration_s": 56.0, "perception": "robot_eye_rgbd", "profile": "v2",
        "conditions": CONDITIONS, "scenarios": [name for name, _, _ in scenario_specs],
        "metrics": "commanded gaze error is camera-track bearing versus head command; head ranges are simulated encoder positions",
    }
    (args.out / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.out / "matched_seed_results.json").write_text(json.dumps(_safe(results), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_safe(results), indent=2))


if __name__ == "__main__":
    main()
