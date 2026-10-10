"""Descriptive left-turn report for docs/turn_translation_pilot.md (not a criterion; the decision is the gate).

For B2, the current (base-origin) candidate and the trained (turn-translation) candidate, from saved native
rollouts of the steady turn-left command:
- paired evaluation rollouts (seeds 0–2, no latency) for all three arms;
- neutral-gate rollouts (seeds 0–4, without and with latency) for the two neutral candidates.
Per trial over 5–20 s: commanded and achieved yaw rate, heading-frame vx and vy, horizontal translation RMS
(the gate metric), pivot centre, net displacement, and foot contacts (stance fraction per foot, double-support
fraction, touchdowns per second). Also writes a top-down trajectory figure for seed 0.

Usage: python -m experiments.locomotion_curriculum.turn_left_report --run RUN --out DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nerva.analysis.motor_eval import smooth, velocities
from nerva.training.motor_artifacts import write_json

BASE_RUN = Path("experiments/cloud_runs/base_origin_velocity-20261010-201858")
BASE_GATE = Path("experiments/cloud_runs/neutral-gate-local")
COMMANDED_YAW = 0.6
START = 250


def describe(path):
    a = dict(np.load(path))
    t = (np.arange(len(a["base_quat"])) + 1) * .02
    mask = (t >= 5) & (t < 20)
    vel = velocities(a)[mask]
    f = smooth(vel)
    mean, yaw = vel[:, :2].mean(axis=0), vel[:, 2].mean()
    contacts = np.asarray(a["contacts"], dtype=bool)[mask]
    touchdowns = np.sum(np.diff(contacts.astype(int), axis=0) == 1, axis=0)
    return {"commanded_yaw": COMMANDED_YAW, "achieved_yaw": float(yaw), "vx": float(mean[0]), "vy": float(mean[1]),
            "horizontal_rms": float(np.sqrt(np.mean(np.sum(f[:, :2] ** 2, axis=1)))),
            "pivot_forward_left_m": [float(-mean[1] / yaw), float(mean[0] / yaw)],
            "displacement_m": float(np.linalg.norm(a["base_pos"][-1, :2] - a["base_pos"][0, :2])),
            "stance_fraction_left_right": contacts.mean(axis=0).round(3).tolist(),
            "double_support_fraction": float(np.mean(contacts.all(axis=1))),
            "touchdowns_per_s_left_right": (touchdowns / 15.).round(2).tolist()}, a["base_pos"][:, :2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sources = {
        ("b2", "paired_no_latency"): [BASE_RUN / "eval" / f"turn_left_{s}_b2.npz" for s in range(3)],
        ("current", "paired_no_latency"): [BASE_RUN / "eval" / f"turn_left_{s}_base_candidate.npz" for s in range(3)],
        ("trained", "paired_no_latency"): [args.run / "eval" / f"turn_left_{s}_turn_candidate.npz" for s in range(3)],
    }
    for condition in ("no_latency", "latency"):
        sources[("current", f"gate_{condition}")] = [BASE_GATE / f"{condition}_steady_turn_left_{s}.npz" for s in range(5)]
        sources[("trained", f"gate_{condition}")] = [args.run / "gate" / f"{condition}_steady_turn_left_{s}.npz"
                                                     for s in range(5)]
    report, paths = {}, {}
    for (arm, source), files in sources.items():
        rows = []
        for i, f in enumerate(files):
            row, xy = describe(f)
            rows.append(row)
            if i == 0 and source == "paired_no_latency":
                paths[arm] = xy
        report.setdefault(arm, {})[source] = {
            "trials": rows, "mean": {k: float(np.mean([r[k] for r in rows])) for k in
                                     ("achieved_yaw", "vx", "vy", "horizontal_rms", "displacement_m",
                                      "double_support_fraction")}}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "turn_left_report.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    for arm, xy in paths.items():
        ax.plot(xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1], label=arm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Steady turn left, seed 0, 20 s: base-origin path")
    ax.legend()
    fig.savefig(args.out / "turn_left_trajectories.png", dpi=120, bbox_inches="tight")
    for arm, sources_ in report.items():
        for source, entry in sources_.items():
            m = entry["mean"]
            print(f"{arm:8s} {source:20s} yaw {m['achieved_yaw']:+.3f} vx {m['vx']:+.4f} vy {m['vy']:+.4f} "
                  f"hrms {m['horizontal_rms']:.4f} disp {m['displacement_m']:.3f} double {m['double_support_fraction']:.2f}")


if __name__ == "__main__":
    main()
