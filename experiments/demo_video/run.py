"""DEMO video: scripted events → appraisal → affect (PAD) → behaviour → Open Duck walking policy.

THIS IS A DEMONSTRATION, NOT AN EXPERIMENT. The PAD → behaviour mapping below is
hand-designed for visualisation and has not been validated; it is NOT the
research result that decides how affect should drive movement (see
docs/research_questions.md RQ1c). It exists to show the closed loop running.

Loop (all in one process, simulation only):
  events (scripted times)  → nerva.appraisal.appraise → CategoricalAffectModel (affect v0.1)
  affect updated at 10 Hz  → demo_behaviour(PAD, context) → BehaviourCommand + head posture
  OpenDuckSim at 50 Hz policy / 500 Hz physics, unmodified Open Duck policy.
  The "near_fall" event coincides with a real sideways push, so the stumble is physical.

Demo behaviour mapping (hand-designed; keeps WHAT separate from HOW; never PAD → joints):
  WHAT  walk forward at 0.13 m/s; stop while the goal is blocked (obstacle event, 6 s)
        or while arousal > 0.2 (pause under threat).
  HOW   tempo style = clip(4·arousal, -1, 1)               (gait clock, RQ1)
        head posture h = clip(1.25·(valence + dominance), -0.5, 0.5)   (+ = head up)
        |h| is limited to 0.5 because head offsets slow the walk (RQ1c).

Usage:  <venv>/Scripts/python experiments/demo_video/run.py
Writes experiments/demo_video/results/{nerva_affect_demo.mp4, timeline.csv, keyframe_*.png}.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from nerva.affect import CategoricalAffectModel  # noqa: E402
from nerva.appraisal import appraise  # noqa: E402
from nerva.interfaces import BehaviourCommand, Event, ExpressiveStyle  # noqa: E402
from nerva.open_duck_sim import OpenDuckSim  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
DURATION = 66.0  # s
CTRL_DT = 0.02  # 50 Hz policy
AFFECT_EVERY = 5  # control steps → 10 Hz affect/behaviour update
FPS = 25
RENDER_EVERY = int(round(1 / (FPS * CTRL_DT)))  # control steps per video frame
SEED = 0
SIM_W, SIM_H, CAP_H = 640, 540, 110

EVENTS = [  # (time s, event kind, physical push (dvx, dvy) or None)
    (2.0, "successful_walking", None),
    (8.0, "successful_walking", None),
    (15.0, "person_approaching_slowly", None),
    (25.0, "person_approaching_rapidly", None),
    (27.0, "near_fall", (0.0, 0.6)),  # 0.6 m/s sideways: visible stumble (~13 deg tilt), recovers; 0.75 falls
    (40.0, "obstacle_blocking_goal", None),
    (52.0, "successful_walking", None),
]
LABELS = ["joy", "hope", "fear", "distress", "surprise"]
COLOURS = {"joy": "#2a9d8f", "hope": "#8ab17d", "fear": "#e76f51", "distress": "#6d597a", "surprise": "#f4a261",
           "valence": "#2a7ab9", "arousal": "#d1492e", "dominance": "#3b9a57"}


@dataclass
class Behaviour:
    vx: float
    style: float
    head: float
    reason: str


def demo_behaviour(v: float, a: float, d: float, blocked: bool) -> Behaviour:
    """Hand-designed DEMO mapping (see module docstring). Not a research result."""
    if blocked:
        vx, reason = 0.0, "stop: goal blocked"
    elif a > 0.2:
        vx, reason = 0.0, "pause: high arousal"
    else:
        vx, reason = 0.13, "walk forward"
    return Behaviour(vx, float(np.clip(4.0 * a, -1, 1)), float(np.clip(1.25 * (v + d), -0.5, 0.5)), reason)


def head_pitch_offset(h: float) -> float:  # same convention as experiments/expressive_locomotion/head_posture.py
    return -0.5 * h if h >= 0 else -0.3 * h


def simulate():
    sim = OpenDuckSim(raw_accel=True, obs_noise=True, seed=SEED)
    affect = CategoricalAffectModel()
    pending = list(EVENTS)
    blocked_until = -1.0
    beh = demo_behaviour(0, 0, 0, False)
    rows, frames_qpos, fired = [], [], []
    n = int(round(DURATION / CTRL_DT))
    for k in range(n):
        t = k * CTRL_DT
        while pending and pending[0][0] <= t + 1e-9:
            te, kind, push = pending.pop(0)
            appraisal = appraise(Event(kind))
            new = affect.add(appraisal)
            fired.append((te, kind, appraisal, [(e.label, e.intensity) for e in new]))
            if kind == "obstacle_blocking_goal":
                blocked_until = te + 6.0
            if push is not None:
                sim.push(*push)
        if k % AFFECT_EVERY == 0:
            pad = affect.step(AFFECT_EVERY * CTRL_DT)
            beh = demo_behaviour(pad.valence, pad.arousal, pad.dominance, t < blocked_until)
            sim.set_behaviour(BehaviourCommand(vx=beh.vx, style=ExpressiveStyle(beh.style)))
            sim.set_head_offset(head_pitch=head_pitch_offset(beh.head))
        sim.step_physics(10)
        d = sim.data
        yaw = np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]), 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
        pad = affect.pad
        w, x, y, _ = d.qpos[3:7]
        rows.append({"t": round(t + CTRL_DT, 3), "valence": pad.valence, "arousal": pad.arousal,
                     "dominance": pad.dominance,
                     **{lbl: sum(e.intensity for e in affect.emotions if e.label == lbl) for lbl in LABELS},
                     "cmd_vx": beh.vx, "tempo_style": beh.style, "head_posture": beh.head, "behaviour": beh.reason,
                     "v_fwd": float(np.cos(yaw) * d.qvel[0] + np.sin(yaw) * d.qvel[1]),
                     "tilt_deg": float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))})
        if k % RENDER_EVERY == 0:
            frames_qpos.append(d.qpos.copy())
    return sim, rows, frames_qpos, fired


def smooth(x, n=25):
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


METHOD_A_STYLE_LINES = (("tempo_style", "tempo style", "#9c6644"), ("head_posture", "head posture", "#5e60ce"))


def chart(rows, fired, style_lines=METHOD_A_STYLE_LINES):
    """Full chart, background-only chart (axes, legends, event markers; no data), and the
    pixel x-range / y-range of the data area, for a progressive left-to-right reveal."""
    t = np.array([r["t"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(SIM_W / 100, SIM_H / 100), dpi=100, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2, 1]})
    ax1, ax2, ax3 = axes
    for lbl in LABELS:
        ax1.plot(t, [r[lbl] for r in rows], color=COLOURS[lbl], lw=1.4, label=lbl)
    ax1.set_ylabel("emotion\nintensity", fontsize=8)
    ax1.set_ylim(-0.02, 1.0)
    ax1.legend(fontsize=6.5, ncol=5, loc="upper left", frameon=False)
    for key in ("valence", "arousal", "dominance"):
        ax2.plot(t, [r[key] for r in rows], color=COLOURS[key], lw=1.8, label=key)
    ax2.axhline(0, color="0.7", lw=0.6)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_ylabel("PAD", fontsize=8)
    ax2.legend(fontsize=6.5, ncol=3, loc="upper left", frameon=False)
    data_lines = [ln for ax in (ax1, ax2) for ln in ax.get_lines()]
    ax3.plot(t, [r["cmd_vx"] for r in rows], color="0.4", lw=1.2, ls="--", label="commanded vx")
    ax3.plot(t, smooth(np.array([r["v_fwd"] for r in rows])), color="k", lw=1.4, label="measured speed (1 s avg)")
    ax3b = ax3.twinx()
    for key, label, colour in style_lines:
        ax3b.plot(t, [r[key] for r in rows], color=colour, lw=1.2, label=label)
    ax3b.set_ylim(-1.05, 1.05)
    ax3b.tick_params(labelsize=7)
    ax3.set_ylim(-0.05, 0.2)
    ax3.set_ylabel("m/s", fontsize=8)
    ax3.set_xlabel("time (s)", fontsize=8)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, fontsize=6, ncol=2, loc="upper left", frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=7)
        for te, *_ in fired:
            ax.axvline(te, color="0.75", lw=0.7, ls=":")
    ax3.set_xlim(0, DURATION)
    data_lines += list(ax3.get_lines()) + list(ax3b.get_lines())
    data_lines = [ln for ln in data_lines if len(ln.get_xdata()) == len(t)]  # curves only, not markers
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    for ln in data_lines:
        ln.set_alpha(0.0)
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    x0 = ax3.transData.transform((0, 0))[0]
    x1 = ax3.transData.transform((DURATION, 0))[0]
    top = fig.bbox.height - ax1.get_window_extent().y1
    bottom = fig.bbox.height - ax3.get_window_extent().y0
    plt.close(fig)
    return img, bg, x0, x1, int(top), int(bottom)


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def caption_text(te, kind, a, emos):
    em = ", ".join(f"{lbl} {i:.2f}" for lbl, i in emos) or "no emotion"
    return (f"t = {te:.0f} s   EVENT: {kind.replace('_', ' ')}" + ("  (+ real sideways push)" if kind == "near_fall" else ""),
            f"appraisal: relevance {a.relevance:.1f}, desirability {a.desirability:+.1f}, likelihood {a.likelihood:.1f}, "
            f"expectedness {a.expectedness:.1f}, controllability {a.controllability:.1f}   →   {em}")


TITLE = "NERVA affect demo — simulation, hand-designed PAD→behaviour mapping (not validated)"


def method_a_status(r, rows, t):
    return (f"PAD  V {r['valence']:+.2f}  A {r['arousal']:+.2f}  D {r['dominance']:+.2f}     "
            f"behaviour: {r['behaviour']} (cmd {r['cmd_vx']:.2f} m/s)   tempo style {r['tempo_style']:+.2f}   "
            f"head {r['head_posture']:+.2f}   measured speed {smooth_speed(rows, t):.3f} m/s")


def render(sim, rows, frames_qpos, fired, path, title=TITLE, status_fn=method_a_status,
           style_lines=METHOD_A_STYLE_LINES):
    model = sim.model
    # The loaded MJCF's default offscreen buffer is 640x480; upstream base.py raises it the same way.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, SIM_W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, SIM_H)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, SIM_H, SIM_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.95, 145.0, -15.0
    chart_img, chart_bg, x0, x1, top, bottom = chart(rows, fired, style_lines)
    f_big, f_small, f_status = font(19), font(14), font(14)
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    look = None
    keyframes = {}
    try:
        for i, qpos in enumerate(frames_qpos):
            t = i * RENDER_EVERY * CTRL_DT
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            target = qpos[0:3] + np.array([0.0, 0.0, 0.08])
            look = target if look is None else 0.9 * look + 0.1 * target  # smooth camera follow
            cam.lookat[:] = look
            renderer.update_scene(data, camera=cam)
            sim_img = renderer.render()

            ch = chart_bg.copy()
            cx = int(x0 + (x1 - x0) * t / DURATION)
            ch[:, :cx + 1] = chart_img[:, :cx + 1]  # data revealed up to the current time
            ch[top:bottom, max(cx - 1, 0):cx + 1] = (220, 30, 30)

            frame = Image.new("RGB", (SIM_W * 2, SIM_H + CAP_H), "white")
            frame.paste(Image.fromarray(sim_img), (0, 0))
            frame.paste(Image.fromarray(ch), (SIM_W, 0))
            draw = ImageDraw.Draw(frame)
            r = rows[min(int(round(t / CTRL_DT)), len(rows) - 1)]
            draw.text((10, 8), title, fill=(40, 40, 40), font=f_small)
            draw.text((10, 28), f"t = {t:5.1f} s", fill=(40, 40, 40), font=f_small)
            recent = [f for f in fired if f[0] <= t < f[0] + 4.0]
            y = SIM_H + 8
            if recent:
                l1, l2 = caption_text(*recent[-1])
                draw.rectangle([0, SIM_H, SIM_W * 2, SIM_H + 58], fill=(255, 243, 205))
                draw.text((12, y), l1, fill=(120, 60, 0), font=f_big)
                draw.text((12, y + 28), l2, fill=(60, 60, 60), font=f_small)
            status = status_fn(r, rows, t)
            draw.text((12, SIM_H + 70), status, fill=(20, 20, 20), font=f_status)
            arr = np.asarray(frame)
            writer.append_data(arr)
            for te, kind, *_ in fired:
                if abs(t - (te + 1.5)) < 0.5 * RENDER_EVERY * CTRL_DT:
                    keyframes[kind] = arr
    finally:
        writer.close()
        renderer.close()
    return keyframes


def smooth_speed(rows, t, window=1.0):
    k = int(round(t / CTRL_DT))
    lo = max(0, k - int(window / CTRL_DT))
    seg = [r["v_fwd"] for r in rows[lo:k + 1]] or [0.0]
    return float(np.mean(seg))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim, rows, frames_qpos, fired = simulate()
    print(f"simulated {DURATION} s: max tilt {max(r['tilt_deg'] for r in rows):.1f} deg, "
          f"{len(frames_qpos)} frames", flush=True)
    for te, kind, a, emos in fired:
        print(f"  {te:5.1f} s  {kind:28s} -> {', '.join(f'{l} {i:.2f}' for l, i in emos)}")
    with (OUT / "timeline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keyframes = render(sim, rows, frames_qpos, fired, OUT / "nerva_affect_demo.mp4")
    for kind, arr in keyframes.items():
        Image.fromarray(arr).save(OUT / f"keyframe_{kind}.png")
    print(f"wrote {OUT / 'nerva_affect_demo.mp4'}")


if __name__ == "__main__":
    main()
