"""Compose the final NERVA demo video from Isaac Sim renders plus NERVA's own data.

Layout (same as experiments/reactive/render.py, with Isaac images):
  left   Isaac scene view; inset: Isaac render from the robot's eye, with the boxes NERVA's colour/depth
         vision found in the same view (MuJoCo, same camera pose and field of view)
  right  emotion intensities, PAD, behaviour mode and distances to the people (live cursor)
  bottom the latest perception event, its appraisal and the emotions it elicited; behaviour and memory status

Inputs come from experiments/isaac/export_replay.py (the .json sidecar) and the Isaac job's frames/ and
frames_eye/ folders. Runs locally (light: image compositing and encoding).

Usage: python experiments/isaac/compose.py --render DIR --sidecar replay.json --out demo.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reactive"))
import render as panels  # noqa: E402  (chart and fonts of the MuJoCo demo)

MAIN_W, MAIN_H = 960, 540
EYE_W, EYE_H = 288, 216
TOP, CAP_H = 30, 110


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=Path, required=True, help="Isaac output folder with frames/ and frames_eye/")
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="NERVA in Isaac Sim — behaviour, affect and memory computed in simulation; "
                                       "robot not scripted")
    args = ap.parse_args()
    side = json.loads(args.sidecar.read_text(encoding="utf-8"))
    rows = [{k: (math.nan if v is None else v) for k, v in r.items()} for r in side["rows"]]
    fps = side["fps"]
    duration = len(rows) / fps
    events = side["events"]
    fired = [(e["t"], e["kind"]) for e in events]
    chart_img, chart_bg, x0, x1, top, bottom = panels.chart(rows, fired, duration)
    chart_w = chart_img.shape[1]
    vis_h, vis_w = side["vision_size"]
    f_big, f_small, f_tiny = panels.font(19), panels.font(14), panels.font(12)
    main_frames = sorted((args.render / "frames").glob("rgb_*.png"))
    eye_frames = sorted((args.render / "frames_eye").glob("rgb_*.png"))
    n = min(len(main_frames), len(rows))
    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    try:
        for i in range(n):
            r = rows[i]
            t = i / fps
            frame = Image.new("RGB", (MAIN_W + chart_w, TOP + MAIN_H + CAP_H), "white")
            frame.paste(Image.open(main_frames[i]).convert("RGB").resize((MAIN_W, MAIN_H)), (0, TOP))
            if i < len(eye_frames):
                eye = Image.open(eye_frames[i]).convert("RGB").resize((EYE_W, EYE_H))
                ed = ImageDraw.Draw(eye)
                sx, sy = EYE_W / vis_w, EYE_H / vis_h
                for kind, r0, c0, r1, c1 in side["boxes"][i] if i < len(side["boxes"]) else []:
                    ed.rectangle([c0 * sx, r0 * sy, (c1 + 1) * sx, (r1 + 1) * sy], outline=(40, 220, 60), width=2)
                    label = str(r["identity"]) if kind == "person" and r.get("identity") else kind
                    ed.text((c0 * sx + 3, max(0, r0 * sy - 14)), label, fill=(40, 220, 60), font=f_tiny)
                ed.rectangle([0, 0, EYE_W - 1, EYE_H - 1], outline=(255, 255, 255), width=2)
                ed.text((6, EYE_H - 16), "robot's eye: vision detections", fill=(255, 255, 255), font=f_tiny)
                frame.paste(eye, (MAIN_W - EYE_W - 10, TOP + 10))
            ch = chart_bg.copy()
            cx = int(x0 + (x1 - x0) * t / duration)
            ch[:, :cx + 1] = chart_img[:, :cx + 1]
            ch[top:bottom, max(cx - 1, 0):cx + 1] = (220, 30, 30)
            ch_img = Image.fromarray(ch).resize((chart_w, int(chart_img.shape[0] * 1.0)))
            frame.paste(ch_img.crop((0, 0, chart_w, min(ch_img.height, MAIN_H + TOP))), (MAIN_W, 0))
            draw = ImageDraw.Draw(frame)
            draw.text((10, 6), args.title, fill=(30, 30, 30), font=f_small)
            draw.text((14, TOP + 10), f"t = {t:5.1f} s   {str(r['mode']).upper()}", fill=(255, 255, 255), font=f_big)
            recent = [e for e in events if e["t"] <= t < e["t"] + 4.0]
            y = TOP + MAIN_H
            if recent:
                e = recent[-1]
                em = ", ".join(f"{lbl} {v:.2f}" for lbl, v in e["emotions"]) or "(no emotion labels: affect Model B)"
                draw.rectangle([0, y, frame.width, y + 58], fill=(255, 243, 205))
                draw.text((12, y + 6), f"t = {e['t']:.1f} s   PERCEIVED: {e['kind'].replace('_', ' ')}",
                          fill=(120, 60, 0), font=f_big)
                draw.text((12, y + 34), f"appraisal: relevance {e['relevance']:.2f}, desirability {e['desirability']:+.2f}, "
                          f"expectedness {e['expectedness']:.2f}, controllability {e['controllability']:.2f}   →   {em}",
                          fill=(60, 60, 60), font=f_small)
            memory = (f"   MEMORY {r['identity']}: threat {r['mem_threat']:.2f} warmth {r['mem_warmth']:+.2f}"
                      if r.get("identity") and not math.isnan(r.get("mem_threat", math.nan)) else "")
            draw.text((12, y + 72), f"behaviour: {r['reason']}   (vx {r['cmd_vx']:+.2f}, turn {r['cmd_yaw']:+.2f})   "
                      f"PAD V {r['valence']:+.2f} A {r['arousal']:+.2f} D {r['dominance']:+.2f}" + memory,
                      fill=(20, 20, 20), font=f_tiny)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    print(f"wrote {args.out}: {n} frames")


if __name__ == "__main__":
    main()
