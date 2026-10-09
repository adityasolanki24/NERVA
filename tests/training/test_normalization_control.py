"""Prospective decision boundaries must keep mechanism and optimizer outcomes separate."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from experiments.locomotion_curriculum.normalization_control import assess, validate_baseline


def stages():
    return [{"stage": name, "all_pass": True, "criteria": {"finite": True},
             "normalizer_count": count, "callback_steps": [0, 16],
             "metrics": {"training/v_loss": .1, "training/kl_mean": .1},
             "normalization_diagnostics": {
                 "integrity_pass": True, "variance_eps": .0001, "peak_normalized": 100.,
                 "std": {"state": {"min": .01, "max_inverse_std": 100.}}}}
            for name, count in (("fresh", 16), ("warm", 32))]


def control():
    rows = stages()
    rows[0]["metrics"] = {"training/v_loss": 884221440., "training/kl_mean": 155226423296.}
    return rows, [{"peak_normalized": 449070.}] * 2


def test_amplification_pass_cannot_hide_optimizer_failure():
    baseline, diagnostics = control()
    actual = stages()
    assert assess(actual, baseline, diagnostics)["all_required_pass"]
    actual[0]["metrics"]["training/kl_mean"] = 1.00001
    result = assess(actual, baseline, diagnostics)
    assert not result["all_required_pass"]
    assert result["stages"][0]["criteria"]["probe_magnitude_bound"]
    assert not result["stages"][0]["criteria"]["kl_screen"]


def test_warm_stage_and_declared_probe_boundary_are_required():
    baseline, diagnostics = control()
    actual = stages()
    actual[1]["normalization_diagnostics"]["peak_normalized"] = 100.00001
    result = assess(actual, baseline, diagnostics)
    assert result["stages"][0]["all_pass"]
    assert not result["stages"][1]["criteria"]["probe_magnitude_bound"]
    assert not result["all_required_pass"]
    with pytest.raises(ValueError, match="complete ordered"):
        assess(actual[:1], baseline, diagnostics)


def test_frozen_control_rejects_changed_config_source_or_unsuccessful_pair():
    root = Path(__file__).resolve().parents[2]
    folder = root / "experiments/locomotion_curriculum/results_neutral_ppo_bytes"
    protocol, rows, manifest = [json.loads((folder / name).read_text(encoding="utf-8"))
                                for name in ("protocol.json", "stages.json", "reference_manifest.json")]
    # The captured training runtime is pinned separately from the lightweight test runner.
    protocol["python_version"] = sys.version.split()[0]
    arguments = [protocol, rows, manifest, protocol["source_hashes"], protocol["versions"], manifest]
    validate_baseline(*arguments)
    bad = deepcopy(protocol)
    bad["config"]["discounting"] = .99
    with pytest.raises(ValueError, match="mismatch"):
        validate_baseline(bad, *arguments[1:])
    with pytest.raises(ValueError, match="mismatch"):
        validate_baseline(protocol, rows, manifest, {}, protocol["versions"], manifest)
    bad_rows = deepcopy(rows)
    bad_rows[1]["all_pass"] = False
    with pytest.raises(ValueError, match="successful strict pair"):
        validate_baseline(protocol, bad_rows, *arguments[2:])
