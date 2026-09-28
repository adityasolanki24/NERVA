"""RQ1: does one style variable produce measurably different, still-stable walking?

Method A (docs/development_log.md): style → gait-phase clock rate, 1 + 0.3·style.
Same task in every condition: walk forward, commanded 0.15 m/s, flat ground,
raw accelerometer, and the observation noise the policy was trained with
(resampled each step, seeded). The noise is the source of trial-to-trial
variation; without it every seed converges to the same limit cycle.

Two parts:
  gait  : 10 trials per style, seeded obs noise + initial joint noise (±0.02 rad), 20 s,
          metrics over t = 5-20 s. Seeds are shared across styles → paired design.
  push  : per style, a base-velocity kick at t = 8 s, 3 magnitudes × 8 directions,
          15 s total; outcome = fall (tilt > 45°) or not.

Usage:
    <venv>/Scripts/python experiments/expressive_locomotion/run.py [--quick]
Writes experiments/expressive_locomotion/results/<timestamp>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from nerva import gait_metrics as gm
from nerva.interfaces import BehaviourCommand, ExpressiveStyle
from nerva.open_duck_sim import OPEN_DUCK_ROOT, POLICY, OpenDuckSim, to_arrays
from nerva.style import PHASE_GAIN, style_to_phase_factor

STYLES = (-1.0, 0.0, 1.0)
COMMAND_VX = 0.15
CTRL_DT = 0.02

GAIT_SECONDS, GAIT_WINDOW_START_S, N_GAIT_TRIALS, INIT_NOISE = 20.0, 5.0, 10, 0.02
PUSH_SECONDS, PUSH_AT_S = 15.0, 8.0
PUSH_MAGNITUDES = (0.3, 0.6, 0.9)  # m/s; training pushes were 0.1-1.0 m/s
PUSH_DIRECTIONS_DEG = tuple(range(0, 360, 45))  # world frame; robot starts facing +x

# Metrics shown in the summary, in order.
REPORT = ["v_fwd", "v_lat", "gait_hz", "cadence_steps_per_s", "stride_m", "lift_left_mm",
          "lift_right_mm", "base_height_m", "pitch_mean_deg", "pitch_std_deg", "roll_std_deg",
          "sway_rate_rms", "action_rate_rms", "joint_range_rms_rad", "tau_sq", "power_w",
          "cost_of_transport", "max_tilt_deg", "fell"]


def git_rev(path: Path) -> str:
    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_sim(style: float, seed: int) -> OpenDuckSim:
    sim = OpenDuckSim(raw_accel=True, init_joint_noise=INIT_NOISE, obs_noise=True, seed=seed)
    sim.set_behaviour(BehaviourCommand(vx=COMMAND_VX, style=ExpressiveStyle(style)))
    return sim


def gait_trial(style: float, seed: int) -> dict:
    sim = make_sim(style, seed)
    log = sim.run(GAIT_SECONDS)
    full = to_arrays(log)
    metrics = gm.summarise(to_arrays(log, int(GAIT_WINDOW_START_S / CTRL_DT)), CTRL_DT,
                           mass=float(sim.model.body_subtreemass[1]))
    metrics["fell"] = float(gm.tilt_deg(full["base_quat"]).max() > gm.FALL_TILT_DEG)  # whole run
    return {"style": style, "phase_factor": sim.phase_factor, "seed": seed, **metrics}


def push_trial(style: float, magnitude: float, direction_deg: float, seed: int) -> dict:
    sim = make_sim(style, seed=seed)
    d = np.radians(direction_deg)
    log = sim.run(PUSH_SECONDS, pushes={int(PUSH_AT_S / CTRL_DT): (magnitude * np.cos(d), magnitude * np.sin(d))})
    tilt = gm.tilt_deg(to_arrays(log)["base_quat"])
    return {"style": style, "magnitude": magnitude, "direction_deg": direction_deg,
            "fell": float(tilt.max() > gm.FALL_TILT_DEG), "max_tilt_deg": float(tilt.max())}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def paired_consistency(rows: list[dict], metric: str, style: float) -> str:
    """How many seeds show the same sign of (style − neutral). n/n = consistent."""
    by = {(r["style"], r["seed"]): r[metric] for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    diffs = np.array([by[(style, s)] - by[(0.0, s)] for s in seeds])
    if np.all(diffs == 0):
        return "0"
    pos, neg = int(np.sum(diffs > 0)), int(np.sum(diffs < 0))
    return f"{'+' if pos >= neg else '−'}{max(pos, neg)}/{len(seeds)}"


def summary_markdown(gait: list[dict], push: list[dict], meta: dict) -> str:
    lines = [f"# RQ1 results — {meta['timestamp']}", "",
             f"Style → phase factor 1 + {PHASE_GAIN}·style. Command vx = {COMMAND_VX} m/s. "
             f"{N_GAIT_TRIALS} seeded trials per style, metrics over t = {GAIT_WINDOW_START_S:g}–{GAIT_SECONDS:g} s.",
             "", "Values: mean ± std across trials. `n/N`: seeds where the paired difference from "
             "Neutral has the same sign (N/N = consistent in every trial).", "",
             "| metric | Style −1 | Neutral | Style +1 | −1 vs 0 | +1 vs 0 |", "|---|---|---|---|---|---|"]
    for k in REPORT:
        cells = []
        for s in STYLES:
            vals = np.array([r[k] for r in gait if r["style"] == s])
            cells.append(f"{vals.mean():.4g} ± {vals.std():.2g}")
        lines.append(f"| {k} | {' | '.join(cells)} | {paired_consistency(gait, k, -1.0)} | "
                     f"{paired_consistency(gait, k, 1.0)} |")
    lines += ["", "## Push robustness (falls / trials)", "",
              "| magnitude m/s | Style −1 | Neutral | Style +1 |", "|---|---|---|---|"]
    for mag in PUSH_MAGNITUDES:
        cells = []
        for s in STYLES:
            rs = [r for r in push if r["style"] == s and r["magnitude"] == mag]
            cells.append(f"{int(sum(r['fell'] for r in rs))}/{len(rs)}")
        lines.append(f"| {mag} | {' | '.join(cells)} |")
    lines += ["", "## Run metadata", "", "```json", json.dumps(meta, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> None:
    global N_GAIT_TRIALS, PUSH_MAGNITUDES, PUSH_DIRECTIONS_DEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 trials, 1 push magnitude, 2 directions (smoke test)")
    args = ap.parse_args()
    if args.quick:
        N_GAIT_TRIALS, PUSH_MAGNITUDES, PUSH_DIRECTIONS_DEG = 2, (0.6,), (0, 180)

    here = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + ("-quick" if args.quick else "")
    out = here / "results" / stamp
    out.mkdir(parents=True)
    t0 = time.time()

    gait = []
    for s in STYLES:
        for seed in range(N_GAIT_TRIALS):
            gait.append(gait_trial(s, seed))
            print(f"gait style={s:+.0f} seed={seed} v_fwd={gait[-1]['v_fwd']:+.3f} fell={gait[-1]['fell']:.0f}", flush=True)

    push = []
    for s in STYLES:
        for mag in PUSH_MAGNITUDES:
            for j, deg in enumerate(PUSH_DIRECTIONS_DEG):
                push.append(push_trial(s, mag, deg, seed=1000 + j))  # same seed per direction across styles
        print(f"push style={s:+.0f} falls={int(sum(r['fell'] for r in push if r['style'] == s))}", flush=True)

    meta = {
        "timestamp": stamp,
        "wall_time_s": round(time.time() - t0, 1),
        "styles": {str(s): style_to_phase_factor(ExpressiveStyle(s)) for s in STYLES},
        "command_vx": COMMAND_VX, "raw_accel": True, "init_joint_noise_rad": INIT_NOISE,
        "obs_noise": "training noise_config (joystick.py), seeded",
        "gait": {"trials": N_GAIT_TRIALS, "seconds": GAIT_SECONDS, "window_start_s": GAIT_WINDOW_START_S},
        "push": {"at_s": PUSH_AT_S, "seconds": PUSH_SECONDS, "magnitudes": list(PUSH_MAGNITUDES),
                 "directions_deg": list(PUSH_DIRECTIONS_DEG)},
        "policy": POLICY.name,
        "nerva_rev": git_rev(here),
        "open_duck_playground_rev": git_rev(OPEN_DUCK_ROOT / "Open_Duck_Playground"),
        "open_duck_mini_rev": git_rev(OPEN_DUCK_ROOT / "Open_Duck_Mini"),
        "python": platform.python_version(),
    }
    write_csv(out / "gait_trials.csv", gait)
    write_csv(out / "push_trials.csv", push)
    (out / "summary.md").write_text(summary_markdown(gait, push, meta), encoding="utf-8")
    print(f"\nwrote {out}  ({meta['wall_time_s']} s)")


if __name__ == "__main__":
    main()
