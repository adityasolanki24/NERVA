"""Preregistered 63 paired native-MuJoCo motor trials and visible seven-command clips."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import smooth, velocities, write_json
from experiments.locomotion_curriculum.learning_support import B2_SHA256, b2_location
from nerva.analysis import gait_metrics as gm
from nerva.interfaces import BehaviourCommand
from nerva.sim.open_duck import OpenDuckSim, SCENE_BACKLASH, to_arrays
from nerva.training.neutral_reference import COMMANDS, CONTRACT

NAMES = ("rest", "forward", "backward", "left", "right", "turn_left", "turn_right")
ARMS = ("b2", "untrained_neutral", "candidate")


def deployment_metadata(policy_hash):
    return {"contract": CONTRACT, "observation_size": 101, "period_steps": 27,
            "head_commands_zero": True, "commands": COMMANDS, "policy_sha256": policy_hash}


def motor_metrics(arrays, command, completed):
    """Never give incomplete or failed rollouts a successful tracking score."""
    if not completed:
        return {"all_pass": False, "tracking_available": False}
    mask = (arrays["t"] >= 5) & (arrays["t"] < 20)
    raw = velocities(arrays)[mask]
    filtered = smooth(raw)
    command = np.asarray(command)
    moving = bool(np.any(command))
    axis = int(np.argmax(np.abs(command)))
    horizontal = float(np.sqrt(np.mean(np.sum(filtered[:, :2] ** 2, axis=1))))
    yaw = float(gm.rms(filtered[:, 2]))
    displacement = float(np.linalg.norm(arrays["base_pos"][-1, :2] - arrays["base_pos"][0, :2]))
    criteria = {"completed": True}
    result = {"tracking_available": True, "mean_velocity": raw.mean(axis=0).tolist(),
              "horizontal_rms": horizontal, "yaw_rms": yaw, "displacement_m": displacement,
              "lift_mm": [float(1000 * gm.lift_height(arrays["foot_z"][mask, i])) for i in range(2)]}
    if moving:
        signed = float(raw[:, axis].mean() * np.sign(command[axis]))
        error = float(gm.rms(filtered[:, axis] - command[axis]))
        stationary = float(np.mean(np.abs(filtered[:, axis]) < (.1 if axis == 2 else .01)))
        cross = float(gm.rms(filtered[:, 1 - axis])) if axis < 2 else horizontal
        criteria.update(direction=signed >= .5 * abs(command[axis]), tracking=error <= .6 * abs(command[axis]),
                        stationary=stationary <= .1, cross=(cross <= .05 and yaw <= .2) if axis < 2 else cross <= .03)
        result.update(signed_axis_mean=signed, axis_tracking_rmse=error,
                      normalized_axis_rmse=error / abs(command[axis]), stationary_fraction=stationary,
                      orthogonal_rms=cross)
    else:
        criteria.update(rest_speed=horizontal <= .02, rest_yaw=yaw <= .15, rest_displacement=displacement <= .10)
    return {**result, "criteria": {key: bool(value) for key, value in criteria.items()}, "all_pass": all(criteria.values())}


def summarize(rows):
    arms = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        commands = {name: {"completed": sum(row["completed"] for row in selected if row["command"] == name),
                           "falls": sum(row["fell"] for row in selected if row["command"] == name),
                           "all_pass": len([row for row in selected if row["command"] == name]) == 3
                           and all(row["metrics"]["all_pass"] for row in selected if row["command"] == name),
                           "mean_velocity": [row["metrics"].get("mean_velocity") for row in selected if row["command"] == name],
                           "axis_rmse": [row["metrics"].get("axis_tracking_rmse") for row in selected if row["command"] == name]}
                    for name in NAMES}
        errors = [row["metrics"]["normalized_axis_rmse"] for row in selected
                  if row["command"] != "rest" and row["metrics"]["tracking_available"]]
        arms[arm] = {"trials": len(selected), "falls": sum(row["fell"] for row in selected), "commands": commands,
                     "all_commands_pass": all(command["all_pass"] for command in commands.values()),
                     "normalized_tracking_rmse": float(np.mean(errors)) if len(errors) == 18 else None}
    candidate, control = arms["candidate"], arms["untrained_neutral"]
    ratio = (candidate["normalized_tracking_rmse"] / control["normalized_tracking_rmse"]
             if candidate["normalized_tracking_rmse"] is not None and control["normalized_tracking_rmse"] else None)
    criteria = {"complete": len(rows) == 63 and all(value["trials"] == 21 for value in arms.values()),
                "seven_commands": candidate["all_commands_pass"],
                "no_added_falls": candidate["falls"] <= min(arms["b2"]["falls"], control["falls"]),
                "tracking_improvement": ratio is not None and ratio <= .9}
    return {"arms": arms, "candidate_to_untrained_error_ratio": ratio, "criteria": criteria,
            "pilot_improvement": all(criteria.values()), "long_gate_passed": False, "cloud_work": False}


def export_parity(raw):
    import jax
    import jax.numpy as jp
    import onnxruntime as ort
    from brax.training.agents.ppo import checkpoint
    folder = sorted((raw / "checkpoints").iterdir())[-1]
    params = checkpoint.load(folder)
    policy = jax.jit(checkpoint.load_policy(folder, deterministic=True))
    session = ort.InferenceSession(str(raw / "candidate.onnx"), providers=["CPUExecutionProvider"])
    mean, std = np.asarray(params[0].mean["state"]), np.asarray(params[0].std["state"])
    max_error = 0.
    for obs in mean + np.random.default_rng(31).normal(size=(100, 101)).astype(np.float32) * std:
        old = policy({"state": jp.asarray(obs), "privileged_state": jp.zeros(212)}, jax.random.PRNGKey(0))[0]
        new = session.run(None, {"obs": obs[None]})[0][0]
        max_error = max(max_error, float(np.max(np.abs(old - new))))
    return {"probes": 100, "max_action_error": max_error, "all_pass": max_error <= 1e-5}


def evaluate(root, raw, output, resume=False):
    start = time.monotonic()
    policy_paths = {"b2": b2_location(root).with_suffix(".onnx"),
                    "untrained_neutral": raw / "initial.onnx", "candidate": raw / "candidate.onnx"}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in policy_paths.items()}
    if hashes["b2"] != B2_SHA256:
        raise ValueError("B2 source hash mismatch")
    parity = export_parity(raw)
    if not (output / "export_parity.json").exists():
        write_json(output / "export_parity.json", parity)
    if not parity["all_pass"]:
        raise ValueError("candidate ONNX export parity failed")
    protocol = {"preregistration_commit": "8fd4fd3",
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "policy_hashes": hashes, "commands": dict(zip(NAMES, COMMANDS)), "seeds": [0, 1, 2],
        "seconds": 20, "scoring_seconds": [5, 20], "scene": "flat_terrain_backlash", "raw_accel": True,
        "observation_noise": True, "joint_initial_noise": .02, "safety_override": False, "wall_cap_s": 600}
    if resume:
        original = json.loads((output / "evaluation_protocol.json").read_text(encoding="utf-8"))
        admitted = json.loads(json.dumps(protocol))
        if any(original[key] != value for key, value in admitted.items() if key != "implementation_commit"):
            raise ValueError("resume policies or evaluation protocol changed")
        write_json(output / "evaluation_resume.json", {"repair_commit": protocol["implementation_commit"],
                   "original_commit": original["implementation_commit"], "criteria_changed": False,
                   "reason": "repair candidate metadata field commands; preserve completed trial"})
        rows = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    else:
        write_json(output / "evaluation_protocol.json", protocol)
        rows = []
    for command_name, command in zip(NAMES, COMMANDS):
        for seed in range(3):
            for arm in ARMS:
                if any(row["command"] == command_name and row["seed"] == seed and row["arm"] == arm for row in rows):
                    if not (raw / f"{command_name}_{seed}_{arm}.npz").is_file():
                        raise ValueError("completed trial artifact missing")
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    sim = OpenDuckSim(policy_path=policy_paths[arm], scene=SCENE_BACKLASH, raw_accel=True,
                                      obs_noise=True, init_joint_noise=.02, seed=seed)
                if arm == "b2":
                    sim.set_style_vector((0, 0, 0))
                else:
                    sim.set_neutral_motor_contract(deployment_metadata(hashes[arm]))
                sim.set_behaviour(BehaviourCommand(vx=command[0], vy=command[1], yaw_rate=command[2]))
                log, qpos, qvel, fell, reason = [], [], [], False, None
                max_tilt = 0.
                for _ in range(1000):
                    log.extend(sim.run(.02))
                    qpos.append(sim.data.qpos.copy())
                    qvel.append(sim.data.qvel.copy())
                    if not np.isfinite((sim.data.qpos,)).all() or not np.isfinite(sim.data.qvel).all():
                        fell, reason = True, "nonfinite"
                        break
                    tilt = float(gm.tilt_deg(sim.data.qpos[3:7][None])[0])
                    max_tilt = max(max_tilt, tilt)
                    if tilt > 45:
                        fell, reason = True, "tilt_above_45"
                        break
                arrays = to_arrays(log)
                arrays["t"] = (np.arange(len(log)) + 1) * .02
                name = f"{command_name}_{seed}_{arm}"
                np.savez_compressed(raw / f"{name}.npz", **arrays, qpos=qpos, qvel=qvel)
                completed = not fell and len(log) == 1000
                row = {"arm": arm, "command": command_name, "seed": seed, "completed": completed,
                       "fell": fell, "stop_reason": reason, "simulated_seconds": len(log) * .02,
                       "max_tilt_deg": max_tilt, "metrics": motor_metrics(arrays, command, completed)}
                rows.append(row)
                write_json(output / "evaluation.json", rows)
            print(command_name, "seed", seed, "candidate passes", rows[-1]["metrics"]["all_pass"], flush=True)
    write_json(output / "motor_summary.json", {**summarize(rows), "wall_seconds": round(time.monotonic() - start, 2)})


def render(raw, output):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import mujoco
    from PIL import Image, ImageDraw
    model = mujoco.MjModel.from_xml_path(str(SCENE_BACKLASH))
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = .95, 145., -20.
    rows = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    videos = []
    try:
        for command, target in zip(NAMES, COMMANDS):
            path = raw / f"comparison_{command}.mp4"
            if path.exists():
                raise FileExistsError("never overwrite videos")
            trials = [next(row for row in rows if row["arm"] == arm and row["command"] == command and row["seed"] == 0)
                      for arm in ("b2", "candidate")]
            traces = [np.load(raw / f"{command}_0_{arm}.npz", allow_pickle=False) for arm in ("b2", "candidate")]
            with imageio.get_writer(path, fps=25, codec="libx264", quality=7, macro_block_size=1) as writer:
                for tick in range(0, 1000, 2):
                    frame = Image.new("RGB", (1280, 540), "white")
                    for col, (trace, trial) in enumerate(zip(traces, trials)):
                        index = min(tick, len(trace["qpos"]) - 1)
                        data.qpos[:], data.qvel[:] = trace["qpos"][index], trace["qvel"][index]
                        mujoco.mj_forward(model, data)
                        cam.lookat[:] = data.qpos[:3] + [0, 0, .08]
                        renderer.update_scene(data, camera=cam)
                        frame.paste(Image.fromarray(renderer.render()), (640 * col, 0))
                        draw = ImageDraw.Draw(frame)
                        label = "Historical B2" if col == 0 else "Trained neutral candidate"
                        draw.text((640 * col + 12, 484), f"{label} | {command} {target} | t={tick * .02:.2f}s", fill="black")
                        status = f"fell at {trial['simulated_seconds']:.2f}s; final pose held" if trial["fell"] else "completed 20s"
                        decision = "PASS" if trial["metrics"]["all_pass"] else "FAIL"
                        draw.text((640 * col + 12, 508), f"{status} | motor criteria {decision}",
                                  fill="red" if not trial["metrics"]["all_pass"] else "black")
                        velocity = trial["metrics"].get("mean_velocity")
                        if velocity is not None:
                            draw.text((640 * col + 12, 524),
                                      f"Mean vx/vy/yaw: {velocity[0]:+.3f} / {velocity[1]:+.3f} / {velocity[2]:+.3f}", fill="black")
                    writer.append_data(np.asarray(frame))
                    if tick == 250:
                        frame.save(raw / f"comparison_{command}.png")
            for trace in traces:
                trace.close()
            videos.append({"command": command, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "bytes": path.stat().st_size, "frames": 500, "fps": 25})
            write_json(output / "videos.json", videos)
            print("video ready", command, flush=True)
        combined = raw / "comparison_all_commands.mp4"
        if combined.exists():
            raise FileExistsError("never overwrite combined video")
        playlist = raw / "video_playlist.txt"
        playlist.write_text("".join(f"file '{(raw / entry['file']).as_posix()}'\n" for entry in videos), encoding="utf-8")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", str(playlist), "-c", "copy", str(combined)], check=True, timeout=60)
        write_json(output / "combined_video.json", {"file": combined.name, "commands": list(NAMES),
            "seconds": 140, "bytes": combined.stat().st_size,
            "sha256": hashlib.sha256(combined.read_bytes()).hexdigest()})
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--corrected", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue preserved partial evaluation with identical policies")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    suffix = "-corrected" if args.corrected else "-retry" if args.retry else ""
    raw = root / f"experiments/cloud_runs/neutral-learning-pilot{suffix}"
    output = root / ("experiments/locomotion_curriculum/results_neutral_learning" + suffix.replace("-", "_"))
    if args.worker:
        (render(raw, output) if args.render else evaluate(root, raw, output, resume=args.resume))
        return
    name = "render" if args.render else "evaluation"
    if ((output / f"{name}.json").exists() and not args.resume) or (args.render and (output / "videos.json").exists()):
        raise FileExistsError("never overwrite comparison artifacts")
    with (raw / f"{name}{'-resume' if args.resume else ''}.log").open("x", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.paired_motor", "--worker",
                        *(["--render"] if args.render else []), *(["--retry"] if args.retry else []),
                        *(["--corrected"] if args.corrected else []), *(["--resume"] if args.resume else [])],
                       stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=600, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    print(name, "complete", flush=True)


if __name__ == "__main__":
    main()
