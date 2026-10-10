"""Maintained paired native-MuJoCo motor comparison for neutral policies (any number of arms) + videos.

Protocol (unchanged from docs/neutral_learning_pilot.md, reused by docs/gpu_neutral_pilot.md):
backlash scene, raw accelerometer, observation noise, initial joint noise ±0.02 rad, seeds 0/1/2, zero head
offsets, the seven neutral commands, 20 s trials scored over 5–20 s, no safety override; a trial aborts on a
nonfinite state or tilt > 45°. Metrics and pass rules: nerva.analysis.motor_eval.motor_metrics.

GPU pilot evaluation:
  python -m experiments.locomotion_curriculum.motor_compare --run experiments/cloud_runs/<run> \\
         --out experiments/locomotion_curriculum/results_gpu_pilot
  python -m experiments.locomotion_curriculum.motor_compare --run ... --out ... --render

Gait-averaged tracking evaluation (docs/gait_averaged_tracking_pilot.md): add --protocol gait_averaged_tracking.
Base-origin velocity evaluation (docs/base_origin_velocity_pilot.md): add --protocol base_origin_velocity.
Turn-translation comparison clips (docs/turn_translation_pilot.md): add --protocol turn_translation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from nerva.analysis import gait_metrics as gm
from nerva.analysis.motor_eval import deployment_metadata, motor_metrics
from nerva.interfaces import BehaviourCommand
from nerva.sim.open_duck import OpenDuckSim, SCENE_BACKLASH, to_arrays
from nerva.training.b2_warm_start import B2_SHA256, b2_location
from nerva.training.motor_artifacts import write_json
from nerva.training.neutral_reference import COMMANDS

NAMES = ("rest", "forward", "backward", "left", "right", "turn_left", "turn_right")
PILOT = "experiments/cloud_runs/neutral-learning-pilot-corrected"
SEEDS = (0, 1, 2)
GPU_PILOT = "experiments/cloud_runs/neutral_gpu_pilot-20261010-152812"
GAIT_RUN = "experiments/cloud_runs/gait_averaged_tracking-20261010-170130"
BASE_RUN = "experiments/cloud_runs/base_origin_velocity-20261010-201858"
LABELS = {"b2": "Historical B2", "untrained_neutral": "Untrained neutral clock",
          "pilot_candidate": "Local pilot candidate (start)", "gpu_candidate": "GPU candidate",
          "gpu_pilot_candidate": "GPU pilot candidate (start)", "gait_candidate": "Gait-averaged candidate",
          "gait_start": "Gait-averaged candidate (start)", "base_candidate": "Base-origin candidate",
          "base_start": "Base-origin candidate (current)", "turn_candidate": "Turn-translation candidate"}


def gpu_pilot_arms(root: Path, run: Path) -> dict[str, Path]:
    return {"b2": b2_location(root).with_suffix(".onnx"), "untrained_neutral": root / PILOT / "initial.onnx",
            "pilot_candidate": root / PILOT / "candidate.onnx", "gpu_candidate": run / "candidate.onnx"}


def gait_averaged_arms(root: Path, run: Path) -> dict[str, Path]:
    return {"b2": b2_location(root).with_suffix(".onnx"), "untrained_neutral": root / PILOT / "initial.onnx",
            "gpu_pilot_candidate": root / GPU_PILOT / "candidate.onnx", "gait_candidate": run / "candidate.onnx"}


def base_origin_arms(root: Path, run: Path) -> dict[str, Path]:
    return {"b2": b2_location(root).with_suffix(".onnx"), "untrained_neutral": root / PILOT / "initial.onnx",
            "gait_start": root / GAIT_RUN / "candidate.onnx", "base_candidate": run / "candidate.onnx"}


def turn_translation_arms(root: Path, run: Path) -> dict[str, Path]:
    return {"b2": b2_location(root).with_suffix(".onnx"), "untrained_neutral": root / PILOT / "initial.onnx",
            "base_start": root / BASE_RUN / "candidate.onnx", "turn_candidate": run / "candidate.onnx"}


def turn_translation_summary(arms: dict) -> dict:
    """Descriptive only: the decision for docs/turn_translation_pilot.md is the neutral gate, not this table."""
    new, start = arms["turn_candidate"], arms["base_start"]
    return {"descriptive_only": True, "decision_source": "neutral_gate.py --candidate turn_translation",
            "turn_candidate_commands_passing": new["commands_passing"],
            "base_start_commands_passing": start["commands_passing"],
            "turn_left_horizontal_rms": {"base_start": start["commands"]["turn_left"]["horizontal_rms"],
                                         "turn_candidate": new["commands"]["turn_left"]["horizontal_rms"]}}


def export_parity(run: Path) -> dict:
    """The exported GPU candidate must reproduce its final checkpoint (100 probes, ≤ 1e-5)."""
    import jax
    import jax.numpy as jp
    import onnxruntime as ort
    from brax.training.agents.ppo import checkpoint
    summary = json.loads((run / "report" / "training_summary.json").read_text(encoding="utf-8"))
    folder = run / "checkpoints" / summary["final_checkpoint"]
    params = checkpoint.load(folder)
    policy = jax.jit(checkpoint.load_policy(folder, deterministic=True))
    session = ort.InferenceSession(str(run / "candidate.onnx"), providers=["CPUExecutionProvider"])
    mean, std = np.asarray(params[0].mean["state"]), np.asarray(params[0].std["state"])
    worst = 0.
    for obs in mean + np.random.default_rng(31).normal(size=(100, 101)).astype(np.float32) * std:
        a = policy({"state": jp.asarray(obs), "privileged_state": jp.zeros(212)}, jax.random.PRNGKey(0))[0]
        b = session.run(None, {"obs": obs[None]})[0][0]
        worst = max(worst, float(np.max(np.abs(np.asarray(a) - b))))
    return {"probes": 100, "max_action_error": worst, "all_pass": worst <= 1e-5, "checkpoint": folder.name}


def run_trial(policy: Path, policy_hash: str, historical_b2: bool, command, seed: int):
    with contextlib.redirect_stdout(io.StringIO()):
        sim = OpenDuckSim(policy_path=policy, scene=SCENE_BACKLASH, raw_accel=True, obs_noise=True,
                          init_joint_noise=.02, seed=seed)
    if historical_b2:
        sim.set_style_vector((0, 0, 0))
    else:
        sim.set_neutral_motor_contract(deployment_metadata(policy_hash))
    sim.set_behaviour(BehaviourCommand(vx=command[0], vy=command[1], yaw_rate=command[2]))
    log, qpos, qvel, fell, reason, max_tilt = [], [], [], False, None, 0.
    for _ in range(1000):
        log.extend(sim.run(.02))
        qpos.append(sim.data.qpos.copy())
        qvel.append(sim.data.qvel.copy())
        if not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
            fell, reason = True, "nonfinite"
            break
        tilt = float(gm.tilt_deg(sim.data.qpos[3:7][None])[0])
        max_tilt = max(max_tilt, tilt)
        if tilt > 45:
            fell, reason = True, "tilt_above_45"
            break
    arrays = to_arrays(log)
    arrays["t"] = (np.arange(len(log)) + 1) * .02
    completed = not fell and len(log) == 1000
    row = {"completed": completed, "fell": fell, "stop_reason": reason, "simulated_seconds": len(log) * .02,
           "max_tilt_deg": max_tilt, "metrics": motor_metrics(arrays, command, completed)}
    return row, {**arrays, "qpos": np.asarray(qpos), "qvel": np.asarray(qvel)}


def summarize(rows, arms) -> dict:
    out = {}
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        cmds = {}
        for name in NAMES:
            rs = [r for r in sel if r["command"] == name]
            m = [r["metrics"] for r in rs]
            cmds[name] = {"passes": sum(x["all_pass"] for x in m), "falls": sum(r["fell"] for r in rs),
                          "all_pass": len(rs) == len(SEEDS) and all(x["all_pass"] for x in m),
                          "mean_velocity": [x.get("mean_velocity") for x in m],
                          "signed_axis_mean": [x.get("signed_axis_mean") for x in m],
                          "axis_rmse": [x.get("axis_tracking_rmse") for x in m],
                          "orthogonal_or_turn_translation_rms": [x.get("orthogonal_rms") for x in m],
                          "horizontal_rms": [x.get("horizontal_rms") for x in m],
                          "displacement_m": [x.get("displacement_m") for x in m],
                          "lift_mm": [x.get("lift_mm") for x in m],
                          "criteria": [x.get("criteria") for x in m]}
        errors = [r["metrics"]["normalized_axis_rmse"] for r in sel
                  if r["command"] != "rest" and r["metrics"]["tracking_available"]]
        out[arm] = {"trials": len(sel), "falls": sum(r["fell"] for r in sel), "commands": cmds,
                    "commands_passing": sum(c["all_pass"] for c in cmds.values()),
                    "normalized_tracking_rmse": float(np.mean(errors)) if len(errors) == 6 * len(SEEDS) else None}
    return out


def gpu_pilot_decision(arms: dict) -> dict:
    """Success criteria fixed in docs/gpu_neutral_pilot.md (preregistration a3ba5ca)."""
    gpu, control, start = arms["gpu_candidate"], arms["untrained_neutral"], arms["pilot_candidate"]

    def speeds_ok(name):
        values = gpu["commands"][name]["signed_axis_mean"]
        return all(v is not None and v >= .5 * .074 for v in values)

    def turn_ok(name):
        values = gpu["commands"][name]["horizontal_rms"]
        return all(v is not None and v <= .03 for v in values)

    hypothesis = all(speeds_ok(n) for n in ("forward", "backward", "left")) and all(turn_ok(n) for n in ("turn_left", "turn_right"))
    no_added_falls = gpu["falls"] <= min(control["falls"], start["falls"])
    ratio = (gpu["normalized_tracking_rmse"] / control["normalized_tracking_rmse"]
             if gpu["normalized_tracking_rmse"] is not None and control["normalized_tracking_rmse"] else None)
    improvement = gpu["commands_passing"] == 7 and ratio is not None and ratio <= .9
    return {"criteria": {"hypothesis_speed_and_turn_translation": hypothesis, "no_added_falls": no_added_falls,
                         "seven_commands_and_10pct_tracking": improvement},
            "gpu_to_untrained_error_ratio": ratio,
            "gpu_to_pilot_error_ratio": (gpu["normalized_tracking_rmse"] / start["normalized_tracking_rmse"]
                                         if ratio is not None and start["normalized_tracking_rmse"] else None),
            "hypothesis_supported": hypothesis and no_added_falls,
            "pilot_passes": hypothesis and no_added_falls and improvement, "long_gate_passed": False}


def gait_averaged_decision(arms: dict) -> dict:
    """Criteria fixed in docs/gait_averaged_tracking_pilot.md (preregistration a143ae4)."""
    new, control, start = arms["gait_candidate"], arms["untrained_neutral"], arms["gpu_pilot_candidate"]

    def speeds_ok(name):
        values = new["commands"][name]["signed_axis_mean"]
        return all(v is not None and v >= .5 * .074 for v in values)

    hypothesis = all(speeds_ok(n) for n in ("forward", "backward", "left", "right"))
    no_added_falls = new["falls"] <= min(control["falls"], start["falls"]) and new["commands"]["rest"]["passes"] == 3
    ratio = (new["normalized_tracking_rmse"] / control["normalized_tracking_rmse"]
             if new["normalized_tracking_rmse"] is not None and control["normalized_tracking_rmse"] else None)
    overall = new["commands_passing"] == 7 and ratio is not None and ratio <= .9
    return {"criteria": {"hypothesis_four_translation_speeds": hypothesis, "no_added_falls_and_rest": no_added_falls,
                         "seven_commands_and_10pct_tracking": overall},
            "gait_to_untrained_error_ratio": ratio,
            "gait_to_gpu_pilot_error_ratio": (new["normalized_tracking_rmse"] / start["normalized_tracking_rmse"]
                                              if ratio is not None and start["normalized_tracking_rmse"] else None),
            "hypothesis_supported": hypothesis and no_added_falls,
            "pilot_passes": hypothesis and no_added_falls and overall, "long_gate_passed": False}


def base_origin_decision(arms: dict) -> dict:
    """Criteria fixed in docs/base_origin_velocity_pilot.md (preregistration 69a6f0f)."""
    new, control, start = arms["base_candidate"], arms["untrained_neutral"], arms["gait_start"]

    def turn_ok(name):
        values = new["commands"][name]["horizontal_rms"]
        return all(v is not None and v <= .03 for v in values)

    hypothesis = turn_ok("turn_left") and turn_ok("turn_right")
    kept = all(new["commands"][n]["passes"] == 3 for n in ("rest", "forward", "backward", "left", "right"))
    no_regression = kept and new["falls"] <= min(control["falls"], start["falls"])
    ratio = (new["normalized_tracking_rmse"] / control["normalized_tracking_rmse"]
             if new["normalized_tracking_rmse"] is not None and control["normalized_tracking_rmse"] else None)
    overall = new["commands_passing"] == 7 and ratio is not None and ratio <= .9
    return {"criteria": {"hypothesis_turn_translation_at_base": hypothesis, "no_regression": no_regression,
                         "seven_commands_and_10pct_tracking": overall},
            "base_to_untrained_error_ratio": ratio,
            "base_to_gait_start_error_ratio": (new["normalized_tracking_rmse"] / start["normalized_tracking_rmse"]
                                               if ratio is not None and start["normalized_tracking_rmse"] else None),
            "hypothesis_supported": hypothesis and no_regression,
            "pilot_passes": hypothesis and no_regression and overall, "long_gate_passed": False}


PROTOCOLS = {
    "gpu_neutral_pilot": {"preregistration": "a3ba5ca", "arms": gpu_pilot_arms, "decision": gpu_pilot_decision,
                          "columns": ("b2", "pilot_candidate", "gpu_candidate")},
    "gait_averaged_tracking": {"preregistration": "a143ae4", "arms": gait_averaged_arms,
                               "decision": gait_averaged_decision,
                               "columns": ("b2", "gpu_pilot_candidate", "gait_candidate")},
    "base_origin_velocity": {"preregistration": "69a6f0f", "arms": base_origin_arms,
                             "decision": base_origin_decision, "columns": ("b2", "gait_start", "base_candidate")},
    "turn_translation": {"preregistration": "PENDING", "arms": turn_translation_arms,
                         "decision": turn_translation_summary, "columns": ("b2", "base_start", "turn_candidate")},
}


def evaluate(root: Path, run: Path, out: Path, protocol: str = "gpu_neutral_pilot") -> None:
    start = time.monotonic()
    raw = run / "eval"
    raw.mkdir(exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    spec = PROTOCOLS[protocol]
    arms = spec["arms"](root, run)
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in arms.items()}
    if hashes["b2"] != B2_SHA256:
        raise ValueError("B2 source hash mismatch")
    parity = export_parity(run)
    write_json(out / "export_parity.json", parity)
    if not parity["all_pass"]:
        raise ValueError("GPU candidate export parity failed")
    write_json(out / "evaluation_protocol.json", {"protocol": protocol, "preregistration_commit": spec["preregistration"],
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "policy_hashes": hashes, "commands": dict(zip(NAMES, COMMANDS)), "seeds": list(SEEDS), "seconds": 20,
        "scoring_seconds": [5, 20], "scene": "flat_terrain_backlash", "raw_accel": True, "observation_noise": True,
        "joint_initial_noise": .02, "safety_override": False, "checkpoint_selection": "final accepted only"})
    rows = []
    for name, command in zip(NAMES, COMMANDS):
        for seed in SEEDS:
            for arm, path in arms.items():
                row, arrays = run_trial(path, hashes[arm], arm == "b2", command, seed)
                np.savez_compressed(raw / f"{name}_{seed}_{arm}.npz", **arrays)
                rows.append({"arm": arm, "command": name, "seed": seed, **row})
                write_json(out / "evaluation.json", rows)
        print(name, {a: sum(r["metrics"]["all_pass"] for r in rows if r["command"] == name and r["arm"] == a)
                     for a in arms}, flush=True)
    summary = summarize(rows, arms)
    write_json(out / "motor_summary.json", {"arms": summary, **spec["decision"](summary),
                                            "wall_seconds": round(time.monotonic() - start, 1)})


def render(run: Path, out: Path, columns=("b2", "pilot_candidate", "gpu_candidate")) -> None:
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import mujoco
    from PIL import Image, ImageDraw
    model = mujoco.MjModel.from_xml_path(str(SCENE_BACKLASH))
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 360, 480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = .95, 145., -20.
    rows = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    videos, width = [], 480 * len(columns)
    try:
        for name, target in zip(NAMES, COMMANDS):
            path = run / f"comparison_{name}.mp4"
            if path.exists():
                raise FileExistsError("never overwrite videos")
            trials = [next(r for r in rows if r["arm"] == a and r["command"] == name and r["seed"] == 0) for a in columns]
            traces = [np.load(run / "eval" / f"{name}_0_{a}.npz") for a in columns]
            with imageio.get_writer(path, fps=25, codec="libx264", quality=7, macro_block_size=1) as writer:
                for tick in range(0, 1000, 2):
                    frame = Image.new("RGB", (width, 440), "white")
                    draw = ImageDraw.Draw(frame)
                    for col, (trace, trial, arm) in enumerate(zip(traces, trials, columns)):
                        i = min(tick, len(trace["qpos"]) - 1)
                        data.qpos[:], data.qvel[:] = trace["qpos"][i], trace["qvel"][i]
                        mujoco.mj_forward(model, data)
                        cam.lookat[:] = data.qpos[:3] + [0, 0, .08]
                        renderer.update_scene(data, camera=cam)
                        frame.paste(Image.fromarray(renderer.render()), (480 * col, 0))
                        x = 480 * col + 8
                        ok = trial["metrics"]["all_pass"]
                        draw.text((x, 364), f"{LABELS[arm]} | {name} {target}", fill="black")
                        status = f"fell at {trial['simulated_seconds']:.1f}s" if trial["fell"] else "completed 20 s"
                        draw.text((x, 382), f"t={tick * .02:5.2f}s | {status} | {'PASS' if ok else 'FAIL'}",
                                  fill="black" if ok else "red")
                        v = trial["metrics"].get("mean_velocity")
                        if v is not None:
                            draw.text((x, 400), f"mean vx {v[0]:+.3f} vy {v[1]:+.3f} yaw {v[2]:+.3f}", fill="black")
                    writer.append_data(np.asarray(frame))
                    if tick == 250:
                        frame.save(run / f"comparison_{name}.png")
            for trace in traces:
                trace.close()
            videos.append({"command": name, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "bytes": path.stat().st_size, "frames": 500, "fps": 25, "columns": list(columns)})
            write_json(out / "videos.json", videos)
            print("video", name, flush=True)
        combined = run / "comparison_all_commands.mp4"
        playlist = run / "video_playlist.txt"
        playlist.write_text("".join(f"file '{(run / v['file']).as_posix()}'\n" for v in videos), encoding="utf-8")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "concat", "-safe", "0", "-i",
                        str(playlist), "-c", "copy", str(combined)], check=True, timeout=120)
        write_json(out / "combined_video.json", {"file": combined.name, "seconds": 140, "bytes": combined.stat().st_size,
                                                 "sha256": hashlib.sha256(combined.read_bytes()).hexdigest()})
    finally:
        renderer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="fetched GPU pilot run directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="gpu_neutral_pilot")
    a = ap.parse_args()
    root = Path.cwd().resolve()
    if a.render:
        render(a.run.resolve(), a.out, PROTOCOLS[a.protocol]["columns"])
    else:
        if (a.out / "evaluation.json").exists():
            raise FileExistsError("never overwrite an evaluation")
        evaluate(root, a.run.resolve(), a.out, a.protocol)


if __name__ == "__main__":
    main()
