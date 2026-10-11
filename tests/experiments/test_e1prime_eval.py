"""E1′ decision rule (docs/expressive_posture_e1prime.md §6) on synthetic rows."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
from experiments.locomotion_curriculum.e1prime_eval import (  # noqa: E402
    COMMANDS, CROUCH_SWEEP, HELD_OUT_POINTS, NAMES, PITCH_SWEEP, SEEDS, TRAINED_POINTS, decide,
)

REF = {n: {"delta_pitch_rad": .2, "delta_height_m": -.007} for n in
       ("stand", "forward", "backward", "left", "right", "turn_left", "turn_right")}


def rows(pitch_gain=.15, crouch_gain=-.006, leak=0., speed=1., fall=None):
    out = []
    points = list(dict.fromkeys(PITCH_SWEEP + CROUCH_SWEEP + TRAINED_POINTS + HELD_OUT_POINTS))
    for e in points:
        for latency in ((False, True) if e in TRAINED_POINTS + HELD_OUT_POINTS else (False,)):
            for name in NAMES:
                for seed in SEEDS:
                    v = np.array(COMMANDS[name]) * (speed if e != (0., 0.) else 1.)
                    out.append({"e": list(e), "command": name, "seed": seed, "latency": latency,
                                "fell": fall == (e, name), "shadow_interventions": 0,
                                "metrics": {"all_pass": fall != (e, name), "mean_velocity": v.tolist()},
                                "features": {"base_pitch_rad": pitch_gain * e[0] + leak * e[1],
                                             "base_height_m": .2 + crouch_gain * e[1] + leak * e[0]}})
    return out


def ramps(ok=True):
    out = []
    for axis in ("pitch", "crouch"):
        for seed in SEEDS:
            x = np.linspace(-1, 1, 20) if axis == "pitch" else np.linspace(0, 1, 20)
            y = x * (.15 if axis == "pitch" else -.006)
            if not ok:
                y = np.where(x > x.mean(), y[-1], y[0])  # a mode switch
            out.append({"axis": axis, "seed": seed, "shadow_interventions": 0, "windows": [
                {"e": [v, 0.] if axis == "pitch" else [0., v], "base_pitch_rad": w if axis == "pitch" else 0.,
                 "base_height_m": .2 + (w if axis == "crouch" else 0.)} for v, w in zip(x, y)]})
    return out


def test_all_criteria_pass_on_clean_synthetic_control():
    d = decide(rows(), ramps(), REF, {"gate_passes": True})
    assert d["e1_prime_passes"], d["criteria"]


def test_small_range_crosstalk_speed_and_mode_switch_each_fail():
    assert not decide(rows(pitch_gain=.04), ramps(), REF, {"gate_passes": True})["criteria"]["3_direction_and_range"]
    assert not decide(rows(leak=.003), ramps(), REF, {"gate_passes": True})["criteria"]["5_crosstalk"]
    assert not decide(rows(speed=.7), ramps(), REF, {"gate_passes": True})["criteria"]["7_task_interference"]
    assert not decide(rows(), ramps(ok=False), REF, {"gate_passes": True})["criteria"]["8_continuity"]
    d = decide(rows(fall=((1., 1.), "forward")), ramps(), REF, {"gate_passes": True})
    assert not d["criteria"]["6_task_across_style_space"] and d["hypothesis_supported"] and not d["e1_prime_passes"]
    assert not decide(rows(), ramps(), REF, {"gate_passes": False})["e1_prime_passes"]
