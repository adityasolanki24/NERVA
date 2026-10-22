"""Render the first two v2 local-physics milestone videos.

The only inputs are frames returned by the closed-loop MuJoCo scenario.  Each
panel has an observer view and a raw robot-eye inset.  The final cuts have no
history, target or emotion labels so their motion can be judged directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reactive import scenario  # noqa: E402
from experiments.v2_autonomy.scenarios import curiosity_repeat_scenario, social_history_scenario  # noqa: E402

FPS = 25
EVERY = int(round(1 / (FPS * scenario.CTRL_DT)))
W, H = 720, 480
EYE_W, EYE_H = 240, 180


def _render_frames(sim, frames):
    """Render an observer view with an unannotated robot-eye inset."""
    model = sim.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    data = mujoco.MjData(model)
    scene_r, eye_r = mujoco.Renderer(model, H, W), mujoco.Renderer(model, EYE_H, EYE_W)
    eye_id = model.camera("robot_eye").id
    mocap = {name: model.body_mocapid[model.body(name).id] for name in ("person", "person_b", "ball")}
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 135.0, -22.0, 2.5
    out, look, distance = [], None, 3.8
    try:
        for qpos, mpos, mquat, _tracks, _boxes in frames:
            data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = qpos, mpos, mquat
            mujoco.mj_forward(model, data)
            robot = qpos[:3]
            candidates = [mpos[mocap[name]] for name in mocap if np.linalg.norm(mpos[mocap[name]][:2] - robot[:2]) < 4.0]
            entity = min(candidates, key=lambda p: np.linalg.norm(p[:2] - robot[:2]), default=None)
            # Keep the small robot large enough to judge its physical head and
            # body motion.  A full human silhouette is less useful here than
            # an inspectable NERVA; the raw eye inset still shows its view.
            target = robot + np.array([0.0, 0.0, 0.22]) if entity is None else 0.78 * robot + 0.22 * entity + [0, 0, 0.28]
            desired_distance = (2.0 if entity is None else 2.15)
            look = target if look is None else 0.90 * look + 0.10 * target
            distance = 0.92 * distance + 0.08 * desired_distance
            cam.lookat[:], cam.distance = look, distance
            scene_r.update_scene(data, camera=cam)
            panel = Image.fromarray(scene_r.render())
            eye_r.update_scene(data, camera=eye_id)
            eye = Image.fromarray(eye_r.render())
            draw = ImageDraw.Draw(panel)
            x, y = W - EYE_W - 12, 12
            panel.paste(eye, (x, y))
            draw.rectangle((x - 2, y - 2, x + EYE_W + 1, y + EYE_H + 1), outline="white", width=2)
            out.append(np.asarray(panel))
    finally:
        scene_r.close()
        eye_r.close()
    return out


def _window(frames, start_s: float, end_s: float):
    start, end = int(round(start_s * FPS)), int(round(end_s * FPS))
    return frames[start:end]


def _write_pair(left, right, output: Path, labels: tuple[str, str] | None = None) -> None:
    n = min(len(left), len(right))
    if n == 0:
        raise ValueError("requested render window has no frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    try:
        for a, b in zip(left[:n], right[:n]):
            if labels:
                a, b = _label(a, labels[0]), _label(b, labels[1])
            writer.append_data(np.concatenate((a, b), axis=1))
    finally:
        writer.close()


def _label(frame: np.ndarray, label: str) -> np.ndarray:
    """Condition labels are only used in the separate debugging renders."""
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), label)
    draw.rectangle((8, 8, 20 + bbox[2], 20 + bbox[3]), fill=(0, 0, 0))
    draw.text((14, 14), label, fill="white")
    return np.asarray(image)


def _safe(value):
    """JSON-safe trace values; unavailable values are explicit nulls, never NaN."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    return value


def _write_trace(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_safe(row), separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _scenario_steps(agents):
    return {name: agent.steps for name, agent in agents.items()}


def _run(policy: str, seed: int, duration: float, agents):
    return scenario.run(policy, seed=seed, duration=duration, agents=agents, record_every=EVERY,
                        perception_mode="vision", use_memory=True, attention=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=62)
    args = ap.parse_args()

    # One continuous curiosity lifetime.  The two windows expose its first and
    # later ball encounter without inserting or retiming any frames.
    curiosity_agents = curiosity_repeat_scenario()
    curiosity_sim, curiosity_rows, curiosity_frames, curiosity_events = _run(args.policy, args.seed, 56.0, curiosity_agents)
    curiosity = _render_frames(curiosity_sim, curiosity_frames)
    _write_pair(_window(curiosity, 3.5, 17.5), _window(curiosity, 41.5, 55.5), args.out / "01_curiosity.mp4")
    _write_pair(_window(curiosity, 3.5, 17.5), _window(curiosity, 41.5, 55.5),
                args.out / "debug" / "01_curiosity_labelled.mp4", ("first ball encounter", "later ball encounter"))

    # Separate lifetimes differ only before 38 s; their later meeting follows
    # the identical script.  The paired clip itself carries no condition label.
    positive_agents, cautious_agents = social_history_scenario("positive"), social_history_scenario("cautious")
    positive_sim, positive_rows, positive_frames, positive_events = _run(args.policy, args.seed, 56.0, positive_agents)
    cautious_sim, cautious_rows, cautious_frames, cautious_events = _run(args.policy, args.seed, 56.0, cautious_agents)
    positive, cautious = _render_frames(positive_sim, positive_frames), _render_frames(cautious_sim, cautious_frames)
    _write_pair(_window(positive, 37.5, 55.5), _window(cautious, 37.5, 55.5), args.out / "02_social_histories_unlabeled.mp4")
    _write_pair(_window(positive, 37.5, 55.5), _window(cautious, 37.5, 55.5),
                args.out / "debug" / "02_social_histories_labelled.mp4",
                ("positive measured-contact history", "camera-observed near-miss history"))

    summary = {
        "seed": args.seed,
        "perception": "rendered_rgbd",
        "attention": True,
        "curiosity_first_inspect_s": round(sum(r["mode"] == "inspect" and 4 <= r["t"] < 36 for r in curiosity_rows) * scenario.CTRL_DT, 2),
        "curiosity_repeat_inspect_s": round(sum(r["mode"] == "inspect" and 42 <= r["t"] < 56 for r in curiosity_rows) * scenario.CTRL_DT, 2),
        "positive_greeting_s": round(sum(r["mode"] == "greet" and r["t"] >= 38 for r in positive_rows) * scenario.CTRL_DT, 2),
        "cautious_greeting_s": round(sum(r["mode"] == "greet" and r["t"] >= 38 for r in cautious_rows) * scenario.CTRL_DT, 2),
        "max_tilt_deg": {"curiosity": max(r["tilt_deg"] for r in curiosity_rows),
                           "positive": max(r["tilt_deg"] for r in positive_rows),
                           "cautious": max(r["tilt_deg"] for r in cautious_rows)},
        "safety_interventions": {"curiosity": curiosity_sim.safety_interventions,
                                  "positive": positive_sim.safety_interventions,
                                  "cautious": cautious_sim.safety_interventions},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    traces = {
        "curiosity_timeline.jsonl": curiosity_rows,
        "positive_timeline.jsonl": positive_rows,
        "cautious_timeline.jsonl": cautious_rows,
    }
    for name, rows in traces.items():
        _write_trace(args.out / name, rows)
    settings = {
        "git_revision": _git_revision(), "seed": args.seed, "duration_s": 56.0,
        "controller": {"path": str(Path(args.policy).resolve()), "sha256": _sha256(Path(args.policy))},
        "perception": "robot_eye_rgbd", "attention": True, "memory": "grounded_outcomes",
        "selector": "utility", "profile": "v2", "affect_model": "Bv4", "control_dt_s": scenario.CTRL_DT,
        "recording": {"fps": FPS, "every_control_steps": EVERY, "observer_size": [W, H], "eye_size": [EYE_W, EYE_H]},
        "scenarios": {"curiosity": _scenario_steps(curiosity_agents), "positive": _scenario_steps(positive_agents),
                      "cautious": _scenario_steps(cautious_agents)},
        "video_windows_s": {"curiosity": [[3.5, 17.5], [41.5, 55.5]], "social": [[37.5, 55.5], [37.5, 55.5]]},
        # Appraisal objects are intentionally not serialised: the complete
        # robot-observable state is in each timeline, while this is an audit
        # index of when perception/appraisal events fired.
        "fired_events": {"curiosity": [[round(t, 3), kind] for t, kind, *_ in curiosity_events],
                         "positive": [[round(t, 3), kind] for t, kind, *_ in positive_events],
                         "cautious": [[round(t, 3), kind] for t, kind, *_ in cautious_events]},
    }
    (args.out / "settings.json").write_text(json.dumps(_safe(settings), indent=2) + "\n", encoding="utf-8")
    videos = ("01_curiosity.mp4", "02_social_histories_unlabeled.mp4",
              "debug/01_curiosity_labelled.mp4", "debug/02_social_histories_labelled.mp4")
    manifest = {
        "unlabeled_panel_key": {"01_curiosity.mp4": ["first encounter", "later reappearance"],
                                  "02_social_histories_unlabeled.mp4": ["positive history", "cautious history"]},
        "files": {name: {"sha256": _sha256(args.out / name), "bytes": (args.out / name).stat().st_size}
                  for name in videos},
        "traces": {name: {"sha256": _sha256(args.out / name), "rows": len(rows)} for name, rows in traces.items()},
    }
    (args.out / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
