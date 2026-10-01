"""Render the reactive-behaviour scenario: scene view + robot's-eye inset + live affect charts.

DEMONSTRATION, NOT AN EXPERIMENT (docs/reactive_behaviour_design.md). Panels:
  left   third-person view following the robot and the person; inset: the robot's head camera
         with the simulated detector's boxes (green = detected this frame, grey = remembered)
  right  emotion intensities, PAD, and the behaviour mode with the distance to the person
  bottom the latest perception event, its appraisal and elicited emotions; the current behaviour

Usage (renders offscreen; heavy, run in the cloud: cloud/jobs/reactive_demo.sh):
    python experiments/reactive/render.py --policy S1.onnx [--out DIR] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario  # noqa: E402

from nerva.sim.world import BALL_RADIUS, PERSON_HEIGHT  # noqa: E402

FPS = 25
EVERY = int(round(1 / (FPS * scenario.CTRL_DT)))
W, H, CAP_H = 720, 560, 110
EYE_W, EYE_H = 256, 192
MODE_COLOURS = {"explore": "#bbbbbb", "orient": "#8ecae6", "approach": "#90be6d", "inspect": "#43aa8b",
                "watch": "#f9c74f", "freeze": "#9d4edd", "retreat": "#f94144", "withdraw": "#577590"}
EMO_COLOURS = {"joy": "#2a9d8f", "hope": "#8ab17d", "fear": "#e76f51", "distress": "#6d597a",
               "surprise": "#f4a261", "interest": "#3a86ff"}
PAD_COLOURS = {"valence": "#2a7ab9", "arousal": "#d1492e", "dominance": "#3b9a57"}


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def chart(rows, fired, duration):
    t = np.array([r["t"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(W / 100, H / 100), dpi=100, sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1, 0.9]})
    ax1, ax2, ax3 = axes
    for lbl, c in EMO_COLOURS.items():
        ax1.plot(t, np.minimum([r[lbl] for r in rows], 1.5), color=c, lw=1.3, label=lbl)
    ax1.set_ylabel("emotion\nintensity", fontsize=8)
    ax1.set_ylim(-0.02, 1.5)
    ax1.legend(fontsize=6.5, ncol=6, loc="upper left", frameon=False)
    for key, c in PAD_COLOURS.items():
        ax2.plot(t, [r[key] for r in rows], color=c, lw=1.7, label=key)
    ax2.axhline(0, color="0.7", lw=0.6)
    ax2.set_ylim(-0.7, 0.7)
    ax2.set_ylabel("PAD", fontsize=8)
    ax2.legend(fontsize=6.5, ncol=3, loc="upper left", frameon=False)
    modes = [r["mode"] for r in rows]
    start = 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            ax3.axvspan(t[start], t[i - 1], ymin=0.0, ymax=0.25, color=MODE_COLOURS[modes[start]], lw=0)
            start = i
    dist = np.array([r["dist_person"] if r["dist_person"] < 10 else np.nan for r in rows])
    ax3.plot(t, dist, color="k", lw=1.3, label="distance to person (m)")
    dist_b = np.array([r.get("dist_person_b", 99) if r.get("dist_person_b", 99) < 10 else np.nan for r in rows])
    if np.isfinite(dist_b).any():
        ax3.plot(t, dist_b, color="#2d6a4f", lw=1.3, ls="--", label="distance to person B (m)")
    ax3.set_ylim(-1.5, 5)
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_xlabel("time (s)", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODE_COLOURS.values()]
    ax3.legend(list(ax3.get_lines()) + handles, [ln.get_label() for ln in ax3.get_lines()] + list(MODE_COLOURS),
               fontsize=5.8, ncol=5, loc="upper left", frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=7)
        for te, *_ in fired:
            ax.axvline(te, color="0.8", lw=0.6, ls=":")
    ax3.set_xlim(0, duration)
    lines = [ln for ax in axes for ln in ax.get_lines() if len(ln.get_xdata()) == len(t)]
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    for ln in lines:
        ln.set_alpha(0.0)
    for p in ax3.patches:
        p.set_alpha(0.0)
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    x0, x1 = ax3.transData.transform((0, 0))[0], ax3.transData.transform((duration, 0))[0]
    top = int(fig.bbox.height - ax1.get_window_extent().y1)
    bottom = int(fig.bbox.height - ax3.get_window_extent().y0)
    plt.close(fig)
    return img, bg, x0, x1, top, bottom


def project(model, data, cam_id, p):
    """World point → pixel in the robot-eye image, or None if behind the camera."""
    rel = p - data.cam_xpos[cam_id]
    r = data.cam_xmat[cam_id].reshape(3, 3)
    x, y, z = rel @ r[:, 0], rel @ r[:, 1], rel @ r[:, 2]
    if z > -0.02:
        return None
    f = (EYE_H / 2) / np.tan(np.radians(model.cam_fovy[cam_id]) / 2)
    return EYE_W / 2 + f * x / -z, EYE_H / 2 - f * y / -z


def boxes(model, data, cam_id, tracks, positions):
    out = []
    for tr in tracks:
        c = positions.get(tr.kind)
        if c is None:
            continue
        if tr.kind == "person":
            pts = [c + np.array([0, dy, dz]) for dy in (-0.25, 0.25) for dz in (0.0, PERSON_HEIGHT)]
        else:
            pts = [c + np.array([0, dy, dz]) for dy in (-BALL_RADIUS, BALL_RADIUS) for dz in (-BALL_RADIUS, BALL_RADIUS)]
        px = [q for q in (project(model, data, cam_id, p) for p in pts) if q is not None]
        if px:
            us, vs = zip(*px)
            u0, v0 = max(0, min(us)), max(0, min(vs))
            u1, v1 = min(EYE_W - 1, max(us)), min(EYE_H - 1, max(vs))
            if u1 > u0 and v1 > v0:  # skip boxes entirely outside the image
                out.append((tr, u0, v0, u1, v1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=100.0)
    ap.add_argument("--perception", choices=("simulated", "vision"), default="vision")
    ap.add_argument("--scenario", choices=("default", "memory"), default="default")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    memory = args.scenario == "memory"
    if memory:
        args.duration = max(args.duration, 135.0)
    sim, rows, frames, fired = scenario.run(args.policy, seed=args.seed, duration=args.duration, record_every=EVERY,
                                            perception_mode=args.perception, use_memory=memory,
                                            agents=scenario.memory_scenario() if memory else None)
    print(f"simulated {args.duration} s, {len(frames)} frames, max tilt {max(r['tilt_deg'] for r in rows):.1f} deg",
          flush=True)
    for te, kind, a, emos in fired:
        print(f"  {te:5.1f} s  {kind:28s} -> {', '.join(f'{lbl} {i:.2f}' for lbl, i in emos)}")
    with (args.out / "timeline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    model = sim.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    data = mujoco.MjData(model)
    scene_r, eye_r = mujoco.Renderer(model, H, W), mujoco.Renderer(model, EYE_H, EYE_W)
    eye_id = model.camera("robot_eye").id
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = 135.0, -22.0
    ch_img, ch_bg, x0, x1, top, bottom = chart(rows, fired, args.duration)
    f_big, f_small, f_tiny = font(19), font(14), font(12)
    mocap = {n: model.body_mocapid[model.body(n).id] for n in ("person", "ball", "person_b")}
    writer = imageio.get_writer(args.out / "nerva_reactive_demo.mp4", fps=FPS, codec="libx264", quality=8,
                                macro_block_size=1)
    look, dist_cam = None, 1.6
    try:
        for i, (qpos, mpos, mquat, tracks, vision_boxes) in enumerate(frames):
            t = i * EVERY * scenario.CTRL_DT
            data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = qpos, mpos, mquat
            mujoco.mj_forward(model, data)
            robot = qpos[0:3]
            person = mpos[mocap["person"]]
            if np.linalg.norm(mpos[mocap["person_b"]][:2] - qpos[:2]) < np.linalg.norm(person[:2] - qpos[:2]):
                person = mpos[mocap["person_b"]]  # frame whichever person is nearer
            target = robot + np.array([0, 0, 0.15])
            want = 1.6
            ball = mpos[mocap["ball"]]
            near = [(np.linalg.norm(e[:2] - robot[:2]), e, lift) for e, lift in ((person, 0.35), (ball, 0.05))]
            near = [n for n in near if n[0] < 3.5]
            if near:  # frame the robot together with the closest thing it can react to
                d_e, e, lift = min(near, key=lambda n: n[0])
                target = 0.5 * (robot + np.array([e[0], e[1], 0.0])) + np.array([0, 0, 0.15 + lift])
                want = 1.4 + 0.7 * d_e
            look = target if look is None else 0.93 * look + 0.07 * target
            dist_cam = 0.95 * dist_cam + 0.05 * want
            cam.lookat[:], cam.distance = look, dist_cam
            scene_r.update_scene(data, camera=cam)
            img = Image.fromarray(scene_r.render())
            eye_r.update_scene(data, camera=eye_id)
            eye = Image.fromarray(eye_r.render())
            ed = ImageDraw.Draw(eye)
            positions = {"person": np.array([person[0], person[1], 0.0]),
                         "ball": mpos[mocap["ball"]].copy()}
            row_now = rows[min(int(round(t / scenario.CTRL_DT)), len(rows) - 1)]
            if vision_boxes is not None:  # the blobs the colour/depth vision found this frame
                sx, sy = EYE_W / scenario.EYE_W, EYE_H / scenario.EYE_H
                dist = {tr.kind: tr.distance for tr in tracks}
                for kind, r0, c0, r1, c1 in vision_boxes:
                    ed.rectangle([c0 * sx, r0 * sy, (c1 + 1) * sx, (r1 + 1) * sy], outline=(40, 220, 60), width=2)
                    name = kind
                    if kind == "person" and row_now["identity"]:
                        name = f"person {row_now['identity']}"
                    label = f"{name} {dist[kind]:.1f} m" if kind in dist else name
                    ed.text((c0 * sx + 3, max(0, r0 * sy - 14)), label, fill=(40, 220, 60), font=f_tiny)
            else:
                for tr, u0, v0, u1, v1 in boxes(model, data, eye_id, tracks, positions):
                    colour = (40, 220, 60) if tr.visible else (160, 160, 160)
                    ed.rectangle([u0, v0, u1, v1], outline=colour, width=2)
                    ed.text((u0 + 3, max(0, v0 - 14)), f"{tr.kind} {tr.distance:.1f} m", fill=colour, font=f_tiny)
            ed.rectangle([0, 0, EYE_W - 1, EYE_H - 1], outline=(255, 255, 255), width=2)
            ed.text((6, EYE_H - 16), "robot's eye: colour + depth vision" if vision_boxes is not None
                    else "robot's eye (simulated detector)", fill=(255, 255, 255), font=f_tiny)
            img.paste(eye, (W - EYE_W - 8, 52))

            ch = ch_bg.copy()
            cx = int(x0 + (x1 - x0) * t / args.duration)
            ch[:, :cx + 1] = ch_img[:, :cx + 1]
            ch[top:bottom, max(cx - 1, 0):cx + 1] = (220, 30, 30)

            frame = Image.new("RGB", (W * 2, H + CAP_H), "white")
            frame.paste(img, (0, 0))
            frame.paste(Image.fromarray(ch), (W, 0))
            draw = ImageDraw.Draw(frame)
            r = rows[min(int(round(t / scenario.CTRL_DT)), len(rows) - 1)]
            draw.text((10, 8), ("NERVA memory demo — two people, colour+depth vision, entity memory; robot NOT scripted"
                                if memory else "NERVA reactive demo — simulation; scripted person/ball, robot NOT scripted"),
                      fill=(30, 30, 30), font=f_small)
            draw.text((10, 28), f"t = {t:5.1f} s   mode: {r['mode'].upper()}", fill=(30, 30, 30), font=f_big)
            recent = [f for f in fired if f[0] <= t < f[0] + 4.0]
            if recent:
                te, kind, a, emos = recent[-1]
                em = ", ".join(f"{lbl} {v:.2f}" for lbl, v in emos) or "no emotion"
                draw.rectangle([0, H, W * 2, H + 58], fill=(255, 243, 205))
                draw.text((12, H + 6), f"t = {te:.1f} s   PERCEIVED: {kind.replace('_', ' ')}", fill=(120, 60, 0),
                          font=f_big)
                draw.text((12, H + 34), f"appraisal: relevance {a.relevance:.2f}, desirability {a.desirability:+.2f}, "
                          f"expectedness {a.expectedness:.2f}, controllability {a.controllability:.2f}   →   {em}",
                          fill=(60, 60, 60), font=f_small)
            draw.text((12, H + 70), f"behaviour: {r['reason']}   (vx {r['cmd_vx']:+.2f}, turn {r['cmd_yaw']:+.2f})   "
                      f"PAD V {r['valence']:+.2f} A {r['arousal']:+.2f} D {r['dominance']:+.2f}   "
                      f"style tempo {r['e_tempo']:+.2f} lean {r['e_torso_pitch']:+.2f}   "
                      f"head pitch {r['head_pitch']:+.2f} yaw {r['head_yaw']:+.2f}"
                      + (f"   MEMORY {r['identity']}: threat {r['mem_threat']:.2f} warmth {r['mem_warmth']:+.2f}"
                         if r.get("identity") else ""),
                      fill=(20, 20, 20), font=f_tiny)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        scene_r.close()
        eye_r.close()
    print(f"wrote {args.out / 'nerva_reactive_demo.mp4'}")


if __name__ == "__main__":
    main()
