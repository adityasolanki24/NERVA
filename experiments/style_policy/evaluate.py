"""S1 evaluation: does the style input e change the intended gait feature, and only that?

Implements docs/style_policy_design.md §5, fixed before training:
  - sweep each e_k over {-1, -0.5, 0, 0.5, 1} with the others at 0
  - same command and paired seeds as RQ1: vx = 0.15 m/s, flat ground, raw accelerometer,
    training observation noise, ±0.02 rad initial joint noise, 20 s, metrics over 5-20 s
  - intended features: e1 → cadence (gait_hz), e2 → foot lift, e3 → torso pitch
  - baselines B0 (upstream training, our cloud) and B1 (NERVA env, neutral refs), same seeds

Success criteria (design §5):
  1. each intended feature changes monotonically over the 5 values, in 10/10 paired seeds
  2. cross-talk: for each other feature, |change between e_k = ±1| < |intended change|,
     both in "normalised units". The design did not define these. S1 used the across-seed
     std at neutral, which is 0 for the FFT-quantised gait frequency (2026-09-30). Fixed
     before S2: the within-condition std pooled over all S1-policy conditions, floored at
     the metric's resolution (RESOLUTION).
  3. no walking falls
  4. forward-tracking error |v_fwd − 0.15| no worse than B0's by more than 25%

Usage (heavy: run in the cloud, cloud/jobs/s1_eval.sh):
    python experiments/style_policy/evaluate.py --s1 S1.onnx --b0 B0.onnx --b1 B1.onnx --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

COMMAND_VX = 0.15
CTRL_DT = 0.02
SECONDS, WINDOW_START_S, INIT_NOISE = 20.0, 5.0, 0.02
LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
FEATURES = {0: "gait_hz", 1: "lift_mm", 2: "pitch_mean_deg"}  # intended feature per dimension
TRACKING_TOLERANCE = 1.25
# gait_hz: FFT bin of gait_metrics.dominant_frequency over the 15 s window (pad 8).
RESOLUTION = {"gait_hz": 1.0 / ((SECONDS - WINDOW_START_S) * 8), "lift_mm": 0.1,
              "pitch_mean_deg": 0.01, "v_fwd": 1e-4}


def trial(job: tuple[str, str, tuple[float, float, float] | None, int]) -> dict:
    from nerva.analysis import gait_metrics as gm
    from nerva.interfaces import BehaviourCommand
    from nerva.sim.open_duck import OpenDuckSim, to_arrays

    policy_name, policy_path, style, seed = job
    sim = OpenDuckSim(raw_accel=True, init_joint_noise=INIT_NOISE, obs_noise=True, seed=seed,
                      policy_path=policy_path)
    sim.set_behaviour(BehaviourCommand(vx=COMMAND_VX))
    if style is not None:
        sim.set_style_vector(style)
    log = sim.run(SECONDS)
    full = to_arrays(log)
    m = gm.summarise(to_arrays(log, int(WINDOW_START_S / CTRL_DT)), CTRL_DT,
                     mass=float(sim.model.body_subtreemass[1]))
    m["fell"] = float(gm.tilt_deg(full["base_quat"]).max() > gm.FALL_TILT_DEG)
    m["lift_mm"] = 0.5 * (m["lift_left_mm"] + m["lift_right_mm"])
    m["tracking_err"] = abs(m["v_fwd"] - COMMAND_VX)
    return {"policy": policy_name, "style": list(style) if style else None, "seed": seed, **m}


def style_vec(k: int, level: float) -> tuple[float, float, float]:
    e = [0.0, 0.0, 0.0]
    e[k] = level
    return tuple(e)


def analyse(rows: list[dict], seeds: list[int]) -> dict:
    def val(policy, style, seed, f):
        for r in rows:
            if r["policy"] == policy and r["seed"] == seed and (r["style"] == (list(style) if style else None)):
                return r[f]
        raise KeyError((policy, style, seed))

    report: dict = {"dimensions": {}}
    all_features = list(FEATURES.values()) + ["v_fwd"]
    conditions = sorted({tuple(r["style"]) for r in rows if r["policy"] == "S1"})
    scale = {f: max(float(np.sqrt(np.mean([np.var([val("S1", c, s, f) for s in seeds]) for c in conditions]))),
                    RESOLUTION[f]) for f in all_features}
    report["normaliser"] = scale
    for k, feature in FEATURES.items():
        series = np.array([[val("S1", style_vec(k, lv), s, feature) for lv in LEVELS] for s in seeds])
        d = np.diff(series, axis=1)
        inc, dec = int(np.sum(np.all(d > 0, axis=1))), int(np.sum(np.all(d < 0, axis=1)))
        monotonic = max(inc, dec)
        delta = {f: np.array([val("S1", style_vec(k, 1.0), s, f) - val("S1", style_vec(k, -1.0), s, f)
                              for s in seeds]) for f in all_features}
        norm = {f: float(abs(delta[f].mean()) / scale[f]) if scale[f] > 0 else float("inf")
                for f in all_features}
        crosstalk_ok = all(norm[f] < norm[feature] for f in all_features if f != feature)
        report["dimensions"][f"e{k + 1}"] = {
            "intended_feature": feature,
            "mean_by_level": dict(zip(map(str, LEVELS), series.mean(axis=0).round(4).tolist())),
            "monotonic_seeds": monotonic, "direction": "increasing" if inc >= dec else "decreasing",
            "change_minus1_to_plus1": {f: float(delta[f].mean()) for f in all_features},
            "change_normalised": norm,
            "crosstalk_ok": crosstalk_ok,
            "pass_monotonic": monotonic == len(seeds),
        }
    from nerva.behaviour.style import S1_NEUTRAL_PERIOD_S, S1_TEMPO_GAIN, s1_foot_height
    report["dimensions"]["e1"]["reference_target"] = {
        str(lv): 1.0 / (S1_NEUTRAL_PERIOD_S * (1 - S1_TEMPO_GAIN * lv)) for lv in LEVELS}
    report["dimensions"]["e2"]["reference_target"] = {str(lv): 1000 * s1_foot_height(lv) for lv in LEVELS}
    s1_rows = [r for r in rows if r["policy"] == "S1"]
    report["walking_falls"] = {p: int(sum(r["fell"] for r in rows if r["policy"] == p))
                               for p in sorted({r["policy"] for r in rows})}
    report["tracking_err_mean"] = {p: float(np.mean([r["tracking_err"] for r in rows if r["policy"] == p]))
                                   for p in sorted({r["policy"] for r in rows})}
    conditions = {tuple(r["style"]) for r in s1_rows}
    report["s1_tracking_err_worst_condition"] = max(
        float(np.mean([r["tracking_err"] for r in s1_rows if tuple(r["style"]) == c])) for c in conditions)
    b0 = report["tracking_err_mean"]["B0"]
    report["criteria"] = {
        "monotonic_10_of_10": {d: v["pass_monotonic"] for d, v in report["dimensions"].items()},
        "crosstalk_smaller": {d: v["crosstalk_ok"] for d, v in report["dimensions"].items()},
        "no_falls": report["walking_falls"]["S1"] == 0,
        "tracking_within_25pct_of_B0": report["s1_tracking_err_worst_condition"] <= TRACKING_TOLERANCE * b0,
    }
    c = report["criteria"]
    report["overall_pass"] = (all(c["monotonic_10_of_10"].values()) and all(c["crosstalk_smaller"].values())
                              and c["no_falls"] and c["tracking_within_25pct_of_B0"])
    return report


def main() -> None:
    global SECONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--b0", required=True)
    ap.add_argument("--b1", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true", help="2 seeds, 6 s: a plumbing check, not a result")
    args = ap.parse_args()
    seeds = list(range(2 if args.quick else args.seeds))
    if args.quick:
        SECONDS = 6.0

    # B1 was trained in StyleJoystick with the neutral style only, so it observes e = 0.
    neutral = style_vec(0, 0.0)
    jobs = [("B0", args.b0, None, s) for s in seeds] + [("B1", args.b1, neutral, s) for s in seeds]
    check_inputs({args.b0: 101, args.b1: 104, args.s1: 104})
    styles = sorted({style_vec(k, lv) for k in FEATURES for lv in LEVELS})
    jobs += [("S1", args.s1, st, s) for st in styles for s in seeds]
    started = time.time()
    # Import once in the parent: mujoco_playground clones its model menagerie on first
    # import, and parallel workers doing that at once collide (cloud s1_eval, 2026-09-29).
    import playground.open_duck_mini_v2.mujoco_infer  # noqa: F401
    with Pool(args.workers, initializer=_set_seconds, initargs=(SECONDS,)) as pool:
        rows = pool.map(trial, jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "trials.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    report = analyse(rows, seeds)
    report["meta"] = {"trials": len(rows), "seconds": time.time() - started, "sim_seconds": SECONDS,
                      "seeds": len(seeds), "quick": args.quick,
                      "policies": {"S1": args.s1, "B0": args.b0, "B1": args.b1}}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def check_inputs(expected: dict[str, int]) -> None:
    """Fail before the pool starts if a policy's observation size is not what we will feed it."""
    import onnxruntime

    for path, size in expected.items():
        got = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"]).get_inputs()[0].shape[1]
        if got != size:
            raise ValueError(f"{path}: policy expects obs size {got}, evaluation feeds {size}")


def _set_seconds(seconds: float) -> None:
    global SECONDS
    SECONDS = seconds


if __name__ == "__main__":
    main()
