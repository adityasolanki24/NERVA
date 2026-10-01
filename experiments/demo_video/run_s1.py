"""DEMO video, S1: scripted events → appraisal → affect (PAD) → behaviour → style-conditioned policy S1.

THIS IS A DEMONSTRATION, NOT AN EXPERIMENT. Same scripted scenario, affect model and
renderer as run.py (method A); the difference is HOW affect reaches movement:
  run.py     PAD → gait-clock rate + head offset, unmodified upstream policy
  run_s1.py  PAD → style vector e (nerva.behaviour.pad_style.s1_behaviour) → the S1 policy observes e
             and was trained to imitate that style's reference motion (tempo, torso pitch).
The PAD → e mapping is a hand-designed, unvalidated NERVA design choice. S1 never saw
combined or intermediate styles in training; the video shows what it does with them.

Usage (renders offscreen; heavy, run in the cloud: cloud/jobs/demo_s1.sh):
    python experiments/demo_video/run_s1.py --policy S1.onnx [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as demo  # noqa: E402  (same directory: scenario, chart and renderer)

from nerva.affect.emotions import CategoricalAffectModel  # noqa: E402
from nerva.affect.appraisal import appraise  # noqa: E402
from nerva.behaviour.pad_style import s1_behaviour  # noqa: E402
from nerva.interfaces import Event, PADState  # noqa: E402
from nerva.sim.open_duck import OpenDuckSim  # noqa: E402

TITLE = "NERVA affect demo (S1) — PAD → style vector e → style-conditioned policy; mapping hand-designed, not validated"
STYLE_LINES = (("e_tempo", "e1 tempo", "#9c6644"), ("e_torso_pitch", "e3 torso pitch (+ = lean fwd)", "#5e60ce"))


def status(r, rows, t):
    return (f"PAD  V {r['valence']:+.2f}  A {r['arousal']:+.2f}  D {r['dominance']:+.2f}     "
            f"behaviour: {r['behaviour']} (cmd {r['cmd_vx']:.2f} m/s)   e = (tempo {r['e_tempo']:+.2f}, "
            f"step 0, torso {r['e_torso_pitch']:+.2f})   measured speed {demo.smooth_speed(rows, t):.3f} m/s")


def simulate(policy: str):
    sim = OpenDuckSim(raw_accel=True, obs_noise=True, seed=demo.SEED, policy_path=policy)
    affect = CategoricalAffectModel()
    pending = list(demo.EVENTS)
    blocked_until = -1.0
    dec = s1_behaviour(PADState(), False)
    sim.set_behaviour(dec.command)
    rows, frames_qpos, fired = [], [], []
    for k in range(int(round(demo.DURATION / demo.CTRL_DT))):
        t = k * demo.CTRL_DT
        while pending and pending[0][0] <= t + 1e-9:
            te, kind, push = pending.pop(0)
            appraisal = appraise(Event(kind))
            new = affect.add(appraisal)
            fired.append((te, kind, appraisal, [(e.label, e.intensity) for e in new]))
            if kind == "obstacle_blocking_goal":
                blocked_until = te + 6.0
            if push is not None:
                sim.push(*push)
        if k % demo.AFFECT_EVERY == 0:
            pad = affect.step(demo.AFFECT_EVERY * demo.CTRL_DT)
            dec = s1_behaviour(pad, t < blocked_until)
            sim.set_behaviour(dec.command)
        sim.step_physics(10)
        d = sim.data
        yaw = np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]), 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
        w, x, y, z = d.qpos[3:7]
        pad, e = affect.pad, dec.command.style_vector
        rows.append({"t": round(t + demo.CTRL_DT, 3), "valence": pad.valence, "arousal": pad.arousal,
                     "dominance": pad.dominance,
                     **{lbl: sum(em.intensity for em in affect.emotions if em.label == lbl) for lbl in demo.LABELS},
                     "cmd_vx": dec.command.vx, "e_tempo": e.tempo, "e_torso_pitch": e.torso_pitch,
                     "behaviour": dec.reason,
                     "v_fwd": float(np.cos(yaw) * d.qvel[0] + np.sin(yaw) * d.qvel[1]),
                     "pitch_deg": float(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))),
                     "tilt_deg": float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))})
        if k % demo.RENDER_EVERY == 0:
            frames_qpos.append(d.qpos.copy())
    return sim, rows, frames_qpos, fired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="S1 ONNX policy (104-value observation)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results_s1")
    ap.add_argument("--no-video", action="store_true", help="simulate and write timeline.csv only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sim, rows, frames_qpos, fired = simulate(args.policy)
    print(f"simulated {demo.DURATION} s: max tilt {max(r['tilt_deg'] for r in rows):.1f} deg, "
          f"{len(frames_qpos)} frames", flush=True)
    for te, kind, a, emos in fired:
        print(f"  {te:5.1f} s  {kind:28s} -> {', '.join(f'{lbl} {i:.2f}' for lbl, i in emos)}")
    with (args.out / "timeline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if args.no_video:
        return
    keyframes = demo.render(sim, rows, frames_qpos, fired, args.out / "nerva_affect_demo_s1.mp4",
                            title=TITLE, status_fn=status, style_lines=STYLE_LINES)
    for kind, arr in keyframes.items():
        Image.fromarray(arr).save(args.out / f"keyframe_{kind}.png")
    print(f"wrote {args.out / 'nerva_affect_demo_s1.mp4'}")


if __name__ == "__main__":
    main()
