"""Missing rapid detections must not substitute the retired lunge criterion."""

import importlib.util
import json
import sys
from pathlib import Path


def test_missing_rapid_event_fails_current_criterion(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "experiments/affect_models/summarise_scenarios.py"
    spec = importlib.util.spec_from_file_location("facet_summary", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pad = {d: {"mean": 0.0, "std": 0.2, "min": -0.3, "max": 0.3, "frac_sat": 0.0}
           for d in module.DIMS}
    root = tmp_path / "results"
    for sc, keys in {"default": ("curiosity", "fear", "habituation", "safe"),
                     "memory": ("b_not_blamed", "a_remembered", "b_welcomed", "touch_to_b"),
                     "together": ("avoids_a", "engages_b")}.items():
        runs = [{"seed": s, "pad": pad, **dict.fromkeys(keys, True), "d_lunge_mean": -0.2,
                 "d_min_after_rapid": None, "d_a_return": -0.1, "d_b_return": 0.2} for s in range(5)]
        folder = root / f"{sc}_Bv4"
        folder.mkdir(parents=True)
        name = "evaluation_together.json" if sc == "together" else "evaluation.json"
        (folder / name).write_text(json.dumps({"runs": runs}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(script), "--root", str(root), "--models", "Bv4"])
    module.main()
    report = json.loads((root / "summary.json").read_text(encoding="utf-8"))["Bv4"]
    assert report["criteria"]["dominance_low_after_rapid_event"] is False
    assert "dominance_low_in_lunge" not in report["criteria"]
    assert report["all_pass"] is False
