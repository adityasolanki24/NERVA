"""RQ1c feasibility: is head posture a second expressive channel, independent of the gait clock?

Head posture h ∈ [-1, 1] (+1 = head up) is applied as a head_pitch offset on top of
the walking policy, the way the Open Duck hardware runtime applies gamepad head
commands (nerva/open_duck_sim.py:set_head_offset). No retraining.

    head_pitch offset = -0.5·h  (h ≥ 0, head up)     = -0.3·h  (h < 0, head down)
    (negative head_pitch raises the head; verified by rendering. Range stays inside
     the hardware gamepad range [-0.78, 0.30] rad.)

Protocol: as run.py (forward command 0.15 m/s, raw accel, training obs noise,
paired seeds 0-9), for h ∈ HEADS × style ∈ {-1, 0, +1}. Pushes at style 0 for
h ∈ {-1, 0, +1}.

Questions:
  1. Does h change the achieved head angle (the intended effect)?
  2. Does h leave the walk unchanged (speed, cadence, torso pitch, sway, stability)?
  3. Does h combine with the tempo style without interacting?

Usage:  <venv>/Scripts/python experiments/expressive_locomotion/head_posture.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as base  # noqa: E402

from nerva import gait_metrics as gm  # noqa: E402
from nerva.open_duck_sim import to_arrays  # noqa: E402

HEADS = (-1.0, -0.5, 0.0, 0.5, 1.0)
STYLES = base.STYLES
SEEDS = tuple(range(base.N_GAIT_TRIALS))
PUSH_HEADS = (-1.0, 0.0, 1.0)
PUSH_MAGNITUDES = (0.6, 0.9)
HEAD_PITCH_JOINT = 6  # index in the 14-actuator order (constants / open_duck_baseline.md §2)
METRICS = ["v_fwd", "cadence_steps_per_s", "stride_m", "lift_left_mm", "pitch_mean_deg", "roll_std_deg",
           "sway_rate_rms", "power_w", "max_tilt_deg", "fell"]


def head_pitch_offset(h: float) -> float:
    return -0.5 * h if h >= 0 else -0.3 * h


def trial(style: float, h: float, seed: int, pushes=None, seconds=base.GAIT_SECONDS) -> dict:
    sim = base.make_sim(style, seed)
    sim.set_head_offset(head_pitch=head_pitch_offset(h))
    log = sim.run(seconds, pushes=pushes)
    full = to_arrays(log)
    win = to_arrays(log, int(base.GAIT_WINDOW_START_S / base.CTRL_DT))
    m = gm.summarise(win, base.CTRL_DT, mass=float(sim.model.body_subtreemass[1]))
    m["fell"] = float(gm.tilt_deg(full["base_quat"]).max() > gm.FALL_TILT_DEG)
    m["max_tilt_deg"] = float(gm.tilt_deg(full["base_quat"]).max())
    m["head_joint_rad"] = float(win["joint_pos"][:, HEAD_PITCH_JOINT].mean())
    return {"style": style, "head": h, "head_pitch_offset": head_pitch_offset(h), "seed": seed, **m}


def paired_delta(rows, metric, style, h):
    by = {(r["style"], r["head"], r["seed"]): r[metric] for r in rows}
    d = np.array([by[(style, h, s)] - by[(style, 0.0, s)] for s in SEEDS])
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return d.mean(), d.std(ddof=1), f"{'+' if pos >= neg else '−'}{max(pos, neg)}/{len(d)}" if pos or neg else "0"


def main():
    here = Path(__file__).resolve().parent
    stamp = "head_posture-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    out = here / "results" / stamp
    out.mkdir(parents=True)
    t0 = time.time()

    rows = []
    for s in STYLES:
        for h in HEADS:
            rows += [trial(s, h, seed) for seed in SEEDS]
            print(f"style={s:+.0f} head={h:+.1f} done", flush=True)

    push = []
    for h in PUSH_HEADS:
        for mag in PUSH_MAGNITUDES:
            for j, deg in enumerate(base.PUSH_DIRECTIONS_DEG):
                d = np.radians(deg)
                r = trial(0.0, h, 1000 + j, pushes={int(base.PUSH_AT_S / base.CTRL_DT): (mag * np.cos(d), mag * np.sin(d))},
                          seconds=base.PUSH_SECONDS)
                push.append({"head": h, "magnitude": mag, "direction_deg": deg, "fell": r["fell"],
                             "max_tilt_deg": r["max_tilt_deg"]})

    lines = [f"# RQ1c head-posture feasibility — {stamp}", "",
             "Mean over 10 paired seeds. Δ = paired difference vs head 0 at the same style and seed "
             "(mean, and n/N seeds with that sign).", ""]
    lines += ["## Achieved head joint angle (rad; offset added to the policy's own head target)", "",
              "| style \\ head | " + " | ".join(f"{h:+g}" for h in HEADS) + " |", "|---" * (len(HEADS) + 1) + "|"]
    for s in STYLES:
        lines.append(f"| {s:+g} | " + " | ".join(
            f"{np.mean([r['head_joint_rad'] for r in rows if r['style'] == s and r['head'] == h]):+.3f}" for h in HEADS) + " |")
    for m in METRICS:
        lines += ["", f"## {m}", "", "| style \\ head | " + " | ".join(f"{h:+g}" for h in HEADS) + " |",
                  "|---" * (len(HEADS) + 1) + "|"]
        for s in STYLES:
            cells = []
            for h in HEADS:
                v = np.mean([r[m] for r in rows if r["style"] == s and r["head"] == h])
                if h == 0.0:
                    cells.append(f"{v:.4g}")
                else:
                    dm, _, sign = paired_delta(rows, m, s, h)
                    cells.append(f"{v:.4g} (Δ{dm:+.3g}, {sign})")
            lines.append(f"| {s:+g} | " + " | ".join(cells) + " |")
    lines += ["", "## Push robustness, style 0 (falls / 8 directions)", "",
              "| magnitude | " + " | ".join(f"head {h:+g}" for h in PUSH_HEADS) + " |", "|---" * (len(PUSH_HEADS) + 1) + "|"]
    for mag in PUSH_MAGNITUDES:
        lines.append(f"| {mag} | " + " | ".join(
            str(int(sum(r["fell"] for r in push if r["head"] == h and r["magnitude"] == mag))) for h in PUSH_HEADS) + " |")

    meta = {"timestamp": stamp, "wall_time_s": round(time.time() - t0, 1), "heads": list(HEADS),
            "head_pitch_offsets": {str(h): head_pitch_offset(h) for h in HEADS}, "styles": list(STYLES),
            "seeds": list(SEEDS), "command_vx": base.COMMAND_VX, "push_heads": list(PUSH_HEADS),
            "push_magnitudes": list(PUSH_MAGNITUDES), "nerva_rev": base.git_rev(here)}
    lines += ["", "## Metadata", "", "```json", json.dumps(meta, indent=2), "```", ""]
    for name, data in (("trials.csv", rows), ("push_trials.csv", push)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}  ({meta['wall_time_s']} s)")


if __name__ == "__main__":
    main()
