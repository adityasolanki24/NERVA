"""Neutral long motor gate (docs/neutral_motor_gate.md): trial set and all-required decision."""
import pytest

pytest.importorskip("mujoco")
from experiments.locomotion_curriculum.neutral_gate import NAMES, protocol_trials, summarise  # noqa: E402


def rows(fail=None):
    out = []
    for condition in ("no_latency", "latency"):
        for t in protocol_trials():
            r = {"condition": condition, "trial": t.name, "group": t.group, "seed": t.seed, "command": t.command,
                 "push_deg": t.push_deg, "completed": True, "fell": False, "ok": True, "shadow_interventions": 0}
            if t.group == "transition":
                r["phases"] = [{"command": "rest", "ok": True}]
            if fail and fail(condition, t):
                r["ok"] = False
            out.append(r)
    return out


def test_trial_set_matches_the_preregistration():
    trials = protocol_trials()
    assert len(trials) == 80 and len({t.name for t in trials}) == 80
    assert {t.command for t in trials if t.group == "steady"} == set(NAMES)


def test_gate_requires_both_latency_conditions():
    assert summarise(rows())["gate_passes"]
    one = summarise(rows(lambda c, t: c == "latency" and t.name == "steady_turn_left_3"))
    assert not one["gate_passes"] and one["conditions"]["no_latency"]["all_pass"]


def test_push_recovery_allows_one_miss_per_direction_only():
    assert summarise(rows(lambda c, t: t.name == "push_forward_90_0"))["gate_passes"]
    assert not summarise(rows(lambda c, t: t.name in ("push_forward_90_0", "push_forward_90_1")))["gate_passes"]
