"""RQ1b: does the gait-clock style change HOW the robot walks once walking SPEED is matched?

The first RQ1 run (run.py) gave every style the same *command*. Measured speed
then differed by 2x, so posture, sway and effort differences may simply follow
from speed. Here every style gets the command that makes it reach the same
*measured* speed.

Procedure (all parameters below, recorded in the results):
  1. Calibration: per style, sweep commanded vx over CAL_GRID with CAL_SEEDS
     (disjoint from evaluation seeds), measure steady-state forward speed.
  2. Command selection: linear interpolation on the first bracket where mean
     speed crosses the target, then secant refinement on CAL_SEEDS until the
     mean is within TOL/2 (max REFINE_ITERS).
  3. Evaluation: EVAL_SEEDS (same as run.py, paired across styles) at the
     selected commands. Matched if |mean speed − target| ≤ TOL.
  4. Speed-sensitivity control: from the calibration sweep, the local slope of
     each metric vs measured speed (Neutral) predicts how much of an observed
     difference a residual speed mismatch could explain.
  5. Push robustness at the primary target (same protocol as run.py).

Only commands inside the trained range (vx ≤ 0.15) are used; Style −1 cannot
go faster than ≈0.055 m/s there, which is why targets are ≤ 0.045 m/s.

Usage:  <venv>/Scripts/python experiments/expressive_locomotion/speed_matched.py [--quick]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as base  # noqa: E402  (shared trial functions; run.py behaviour unchanged)

STYLES = base.STYLES
TARGETS = (0.045, 0.025)  # m/s; first is primary (push tests run there)
TOL = 0.005  # m/s, allowed |mean achieved − target| on evaluation seeds
CAL_GRID = tuple(round(v, 3) for v in np.arange(0.0, 0.1501, 0.015))
CAL_SEEDS = (100, 101, 102)
EVAL_SEEDS = tuple(range(base.N_GAIT_TRIALS))
REFINE_ITERS = 4
SLOPE_WINDOW = 0.03  # m/s either side of the target for the local metric-vs-speed fit

METRICS = [m for m in base.REPORT if m not in ("fell",)] + ["fell"]


def mean_speed(style: float, vx: float, seeds) -> tuple[float, list[dict]]:
    rows = [base.gait_trial(style, s, vx) for s in seeds]
    return float(np.mean([r["v_fwd"] for r in rows])), rows


def select_command(style, target, curve):
    """curve: list of (vx, mean_speed) on CAL_GRID. Returns (vx, achieved, history)."""
    vxs, vs = np.array([c[0] for c in curve]), np.array([c[1] for c in curve])
    idx = next((i for i in range(len(vs) - 1) if (vs[i] - target) * (vs[i + 1] - target) <= 0), None)
    if idx is None:
        return None, None, [("unreachable", float(vs.min()), float(vs.max()))]
    lo, hi = (vxs[idx], vs[idx]), (vxs[idx + 1], vs[idx + 1])
    vx = lo[0] + (target - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1]) if hi[1] != lo[1] else lo[0]
    history = []
    for _ in range(REFINE_ITERS):
        v, _ = mean_speed(style, vx, CAL_SEEDS)
        history.append((round(float(vx), 5), round(v, 5)))
        if abs(v - target) <= TOL / 2:
            break
        # secant step using the closer bracket end, clipped to the bracket
        if (v - target) * (lo[1] - target) <= 0:
            hi = (vx, v)
        else:
            lo = (vx, v)
        vx = lo[0] + (target - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1]) if hi[1] != lo[1] else vx
    return float(vx), v, history


def local_slopes(cal_rows, style, target):
    """d(metric)/d(speed) from calibration trials of `style` whose speed is near the target."""
    rows = [r for r in cal_rows if r["style"] == style and abs(r["v_fwd"] - target) <= SLOPE_WINDOW]
    v = np.array([r["v_fwd"] for r in rows])
    out = {}
    for m in METRICS:
        y = np.array([r[m] for r in rows], dtype=float)
        ok = np.isfinite(y)
        out[m] = float(np.polyfit(v[ok], y[ok], 1)[0]) if ok.sum() >= 3 and np.ptp(v[ok]) > 1e-3 else float("nan")
    return out, len(rows)


def paired(rows, metric, style):
    by = {(r["style"], r["seed"]): r[metric] for r in rows}
    d = np.array([by[(style, s)] - by[(0.0, s)] for s in EVAL_SEEDS], dtype=float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    sign = f"{'+' if pos >= neg else '−'}{max(pos, neg)}/{len(d)}" if (pos or neg) else "0"
    return float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0, sign


def fmt(x, digits=4):
    return "nan" if not np.isfinite(x) else f"{x:.{digits}g}"


def target_section(target, commands, eval_rows, slopes, n_slope):
    lines = [f"## Target {target} m/s", "",
             "| style | command vx | achieved speed (mean ± sd) | within ±" + f"{TOL} |", "|---|---|---|---|"]
    for s in STYLES:
        v = np.array([r["v_fwd"] for r in eval_rows if r["style"] == s])
        ok = abs(v.mean() - target) <= TOL
        lines.append(f"| {s:+g} | {commands[s]:.4f} | {v.mean():.4f} ± {v.std(ddof=1):.4f} | {'yes' if ok else '**NO**'} |")
    lines += ["", f"Speed-sensitivity slopes from {n_slope} Neutral calibration trials within ±{SLOPE_WINDOW} m/s.",
              "`Δ` = paired mean difference from Neutral ± sd over seeds; `n/N` = seeds with that sign; "
              "`speed-explained` = slope × residual speed difference (what the speed mismatch alone would predict).", "",
              "| metric | Style −1 | Neutral | Style +1 | Δ(−1) | n/N | speed-explained | Δ(+1) | n/N | speed-explained |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    v_neu = np.mean([r["v_fwd"] for r in eval_rows if r["style"] == 0.0])
    for m in METRICS:
        cells = [f"{fmt(np.mean([r[m] for r in eval_rows if r['style'] == s]))} ± "
                 f"{fmt(np.std([r[m] for r in eval_rows if r['style'] == s], ddof=1), 2)}" for s in STYLES]
        extra = []
        for s in (-1.0, 1.0):
            dm, dsd, sign = paired(eval_rows, m, s)
            dv = np.mean([r["v_fwd"] for r in eval_rows if r["style"] == s]) - v_neu
            extra += [f"{fmt(dm, 3)} ± {fmt(dsd, 2)}", sign, fmt(slopes[m] * dv, 2)]
        lines.append(f"| {m} | {' | '.join(cells)} | {' | '.join(extra)} |")
    return lines


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    global CAL_GRID, CAL_SEEDS, EVAL_SEEDS, TARGETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse grid, 1 cal seed, 2 eval seeds, no pushes")
    args = ap.parse_args()
    if args.quick:
        CAL_GRID, CAL_SEEDS, EVAL_SEEDS, TARGETS = (0.0, 0.05, 0.1, 0.15), (100,), (0, 1), (0.045,)

    here = Path(__file__).resolve().parent
    stamp = "speed_matched-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ("-quick" if args.quick else "")
    out = here / "results" / stamp
    out.mkdir(parents=True)
    t0 = time.time()

    # 1. calibration sweep
    cal_rows, curves = [], {}
    for s in STYLES:
        curves[s] = []
        for vx in CAL_GRID:
            v, rows = mean_speed(s, vx, CAL_SEEDS)
            cal_rows += [{"command_vx": vx, **r} for r in rows]
            curves[s].append((vx, v))
        print(f"calibration style={s:+.0f}: " + " ".join(f"{vx:.3f}->{v:.3f}" for vx, v in curves[s]), flush=True)

    # 2-4. per target: select commands, evaluate, slopes
    commands, refine, sections, eval_all = {}, {}, [], []
    for target in TARGETS:
        commands[target], refine[target] = {}, {}
        for s in STYLES:
            vx, v, hist = select_command(s, target, curves[s])
            refine[target][s] = hist
            if vx is None:
                raise SystemExit(f"target {target} unreachable for style {s}: {hist}")
            commands[target][s] = vx
            print(f"target {target} style={s:+.0f}: command {vx:.4f} (cal speed {v:.4f}) via {hist}", flush=True)
        rows = [{"target": target, "command_vx": commands[target][s], **base.gait_trial(s, seed, commands[target][s])}
                for s in STYLES for seed in EVAL_SEEDS]
        eval_all += rows
        slopes, n_slope = local_slopes(cal_rows, 0.0, target)
        sections += target_section(target, commands[target], rows, slopes, n_slope) + [""]

    # 5. push robustness at the primary target
    push = []
    if not args.quick:
        t = TARGETS[0]
        for s in STYLES:
            for mag in base.PUSH_MAGNITUDES:
                for j, deg in enumerate(base.PUSH_DIRECTIONS_DEG):
                    push.append({"target": t, "command_vx": commands[t][s],
                                 **base.push_trial(s, mag, deg, seed=1000 + j, vx=commands[t][s])})
        sections += [f"## Push robustness at {TARGETS[0]} m/s (falls / {len(base.PUSH_DIRECTIONS_DEG)} directions)", "",
                     "| magnitude m/s | Style −1 | Neutral | Style +1 |", "|---|---|---|---|"]
        for mag in base.PUSH_MAGNITUDES:
            cells = [f"{int(sum(r['fell'] for r in push if r['style'] == s and r['magnitude'] == mag))}"
                     for s in STYLES]
            sections.append(f"| {mag} | {' | '.join(cells)} |")
        sections.append("")

    meta = {"timestamp": stamp, "wall_time_s": round(time.time() - t0, 1), "targets": list(TARGETS),
            "tolerance_m_s": TOL, "cal_grid": list(CAL_GRID), "cal_seeds": list(CAL_SEEDS),
            "eval_seeds": list(EVAL_SEEDS), "refine_iters_max": REFINE_ITERS, "slope_window": SLOPE_WINDOW,
            "commands": {str(t): {str(s): commands[t][s] for s in STYLES} for t in TARGETS},
            "refinement": {str(t): {str(s): refine[t][s] for s in STYLES} for t in TARGETS},
            "calibration_curves": {str(s): curves[s] for s in STYLES},
            "sim": "raw accel, training obs noise, init joint noise 0.02 rad (as run.py)",
            "nerva_rev": base.git_rev(here), "open_duck_playground_rev": base.git_rev(base.OPEN_DUCK_ROOT / "Open_Duck_Playground")}
    write_csv(out / "calibration_trials.csv", cal_rows)
    write_csv(out / "gait_trials.csv", eval_all)
    if push:
        write_csv(out / "push_trials.csv", push)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "summary.md").write_text("\n".join([f"# RQ1b speed-matched results — {stamp}", ""] + sections
                                              + ["## Metadata", "", "```json", json.dumps(meta, indent=2), "```"]),
                                    encoding="utf-8")
    print(f"\nwrote {out}  ({meta['wall_time_s']} s)")


if __name__ == "__main__":
    main()
