"""Render a deterministic Open Duck ONNX policy rollout to MP4.

This is an inspection aid, not an evaluation.  A short smoke-trained policy is
not expected to walk well; scientific comparisons belong in the experiment
scripts with repeated trials and explicit metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from nerva.interfaces import BehaviourCommand
from nerva.sim.open_duck import OpenDuckSim


def render(policy: Path, output: Path, seconds: float, vx: float) -> dict[str, float | bool | str]:
    sim = OpenDuckSim(raw_accel=True, policy_path=policy)
    sim.set_behaviour(BehaviourCommand(vx=vx))
    fps = 25
    control_dt = sim.inf.sim_dt * sim.inf.decimation
    render_every = round(1 / (fps * control_dt))
    qpos, speed, tilt, height = [], [], [], []
    for step in range(round(seconds / control_dt)):
        sim.step_physics(sim.inf.decimation)
        d = sim.data
        w, x, y, _ = d.qpos[3:7]
        yaw = np.arctan2(2 * (w * d.qpos[6] + x * y), 1 - 2 * (y * y + d.qpos[6] ** 2))
        speed.append(float(np.cos(yaw) * d.qvel[0] + np.sin(yaw) * d.qvel[1]))
        tilt.append(float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))))
        height.append(float(d.qpos[2]))
        if step % render_every == 0:
            qpos.append(d.qpos.copy())

    width, frame_height = 720, 540
    sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
    sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, frame_height)
    data = mujoco.MjData(sim.model)
    renderer = mujoco.Renderer(sim.model, frame_height, width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 1.0, 145.0, -15.0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    look = None
    try:
        for i, state in enumerate(qpos):
            data.qpos[:] = state
            mujoco.mj_forward(sim.model, data)
            target = state[:3] + np.array([0.0, 0.0, 0.08])
            look = target if look is None else 0.9 * look + 0.1 * target
            camera.lookat[:] = look
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            t = i / fps
            draw.rectangle((8, 8, 455, 58), fill=(255, 255, 255, 220))
            draw.text((18, 15), "NERVA cloud smoke checkpoint (inspection only)", fill=(20, 20, 20))
            draw.text((18, 35), f"t={t:4.1f}s  command vx={vx:.2f} m/s  step=327,680", fill=(20, 20, 20))
            writer.append_data(np.asarray(image))
    finally:
        writer.close()
        renderer.close()

    warmup = min(len(speed) - 1, round(2 / control_dt))
    metrics: dict[str, float | bool | str] = {
        "policy": str(policy),
        "duration_s": seconds,
        "command_vx_m_s": vx,
        "mean_forward_speed_after_2s_m_s": float(np.mean(speed[warmup:])),
        "max_tilt_deg": float(np.max(tilt)),
        "minimum_base_height_m": float(np.min(height)),
        "fell": bool(np.min(height) < 0.15 or np.max(tilt) > 60),
    }
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--vx", type=float, default=0.10)
    args = parser.parse_args()
    print(json.dumps(render(args.policy, args.output, args.seconds, args.vx), indent=2))


if __name__ == "__main__":
    main()
