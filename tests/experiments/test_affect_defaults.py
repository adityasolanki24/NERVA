"""Adopted defaults and explicit historical paths resolve consistently."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_adopted_and_historical_affect_options():
    path = Path(__file__).resolve().parents[2] / "experiments/reactive/scenario.py"
    spec = importlib.util.spec_from_file_location("scenario", path)
    module = importlib.util.module_from_spec(spec)
    # Dataclass processing queries the defining module.
    import sys
    prior = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.affect_options("v2") == ("Bv4", True, True)
        assert module.affect_options("legacy") == ("A", False, False)
        for model in ("A", "B", "Bv2", "Bv3"):
            assert module.affect_options("v2", model) == (model, False, False)
        assert module.affect_options("v2", "Bv4", False, False) == ("Bv4", False, False)
        assert module.affect_options("v2", "Bv3", True, True) == ("Bv3", True, True)
        evaluator_path = path.with_name("evaluate_memory.py")
        evaluator_spec = importlib.util.spec_from_file_location("memory_evaluator", evaluator_path)
        evaluator = importlib.util.module_from_spec(evaluator_spec)
        evaluator_spec.loader.exec_module(evaluator)
        sim = SimpleNamespace(memory=SimpleNamespace(learning="grounded", records={}), outcomes=[], risks=[])
        rows = [{"t": 0.1, "mode": "explore", "tilt_deg": 0.0,
                 "valence": 0.0, "arousal": 0.0, "dominance": 0.0}]
        module.run = lambda *args, **kwargs: (sim, rows, [], [])
        assert evaluator.evaluate("unused", 0, True)["learning"] == "grounded"
    finally:
        if prior is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = prior
