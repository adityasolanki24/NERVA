"""Capped seven-motion recording and held-out reference validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from nerva.sim.open_duck import OPEN_DUCK_ROOT
from nerva.training.reference_kinematics import fit_reference, pose_velocities, sample_reference

CONDITIONS = {"stand": (0., 0., 0.), "forward": (.074, 0., 0.), "backward": (-.074, 0., 0.),
              "left": (0., .074, 0.), "right": (0., -.074, 0.),
              "turn_left": (0., 0., .60), "turn_right": (0., 0., -.60)}


def wsl_path(path):
    path = Path(path).resolve()
    return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]


def validate(recording, command, static=False):
    frames = np.asarray(recording["Frames"], dtype=float)
    t = np.asarray(recording["FrameTimes"], dtype=float)
    offsets = recording["Frame_offset"][0]
    period = float(recording["Placo"]["period"])
    if frames.ndim != 2 or len(frames) != 400 or len(t) != 400 or not np.isfinite(frames).all():
        raise ValueError("incomplete or nonfinite recording")
    j = offsets["joints_pos"]
    joints = frames[:, j:j + 16]
    velocities = pose_velocities(t, frames[:, :3], frames[:, 3:7], joints)
    contact = frames[1:, offsets["foot_contacts"]:offsets["foot_contacts"] + 2]
    joints, t = joints[1:], t[1:]
    train, held = (t >= 2) & (t < 4), (t >= 4) & (t < 6)
    if train.sum() < 80 or held.sum() < 80 or abs(period - .54) > 1e-6:
        raise ValueError("period or interval coverage mismatch")
    ref = fit_reference(t[train], joints[train], contact[train], velocities["linear_body"][train],
                        velocities["angular_body"][train], period, static)
    pred = sample_reference(ref, t[held])
    errors = {}
    for name, actual in (("joint_position", joints), ("joint_velocity", velocities["joint_velocity"]),
                         ("linear_body", velocities["linear_body"]), ("angular_body", velocities["angular_body"])):
        errors[name] = np.sqrt(np.mean((pred[name] - actual[held]) ** 2, axis=0)).tolist()
    contact_agree = float(np.mean((pred["contacts"] > .5) == (contact[held] > .5)))
    cycle = sample_reference(ref, np.arange(100) / 100 * period)["joint_position"]
    scored = (t >= 2) & (t < 6)
    knee_min = np.minimum(joints[scored][:, [3, 14]].min(0), cycle[:, [3, 14]].min(0))
    mean = np.r_[velocities["linear_body"][held].mean(0)[:2], velocities["angular_body"][held].mean(0)[2]]
    command = np.asarray(command)
    axis = int(np.argmax(np.abs(command)))
    if static:
        command_ok = (np.abs(velocities["linear_body"][scored]).max() <= 1e-6
                      and np.abs(velocities["angular_body"][scored]).max() <= 1e-6)
    else:
        command_ok = np.sign(command[axis]) * mean[axis] >= .5 * abs(command[axis])
        command_ok &= abs(mean[axis] - command[axis]) <= .3 * abs(command[axis])
        command_ok &= all(abs(mean[i]) <= (.10 if i == 2 else .02) for i in range(3) if i != axis)
    criteria = {"positive_knees": bool(np.all(knee_min > 0)),
                "joint_position_fit": max(errors["joint_position"]) <= .03,
                "joint_velocity_fit": max(errors["joint_velocity"]) <= .5,
                "linear_fit": max(errors["linear_body"]) <= .03,
                "angular_fit": max(errors["angular_body"]) <= .10,
                "contact_fit": contact_agree >= .90, "command_tracking": bool(command_ok)}
    return {"criteria": criteria, "all_pass": all(criteria.values()), "rmse_per_component": errors,
            "contact_agreement": contact_agree, "knee_min_rad": knee_min.tolist(),
            "knee_limit_exceeded": bool(np.abs(cycle[:, [3, 14]]).max() > 1.5708),
            "mean_vx_vy_wz": mean.tolist(), "timestamp_dt_range_s": [float(np.diff(t).min()), float(np.diff(t).max())],
            "train_samples": int(train.sum()), "heldout_samples": int(held.sum())}, ref


def known_pose_checks():
    from scipy.spatial.transform import Rotation
    rows = []
    for dt in (.01, .02, .04):
        for yaw in (0., 1.):
            for omega in ((0., 0., .6), (.2, -.3, .4)):
                t = np.arange(10) * dt
                initial = Rotation.from_euler("z", yaw)
                rotation = Rotation.from_rotvec(t[:, None] * omega) * initial
                xyz = t[:, None] * np.array([.1, -.2, .05])
                actual = pose_velocities(t, xyz, rotation.as_quat(), np.zeros((10, 16)))
                err = float(np.max(np.abs(actual["angular_world"] - omega)))
                back = rotation[1:].apply(actual["linear_body"])
                frame_err = float(np.max(np.abs(back - [.1, -.2, .05])))
                rows.append({"dt_s": dt, "yaw_rad": yaw, "omega_world": list(omega),
                             "angular_error": err, "frame_roundtrip_error": frame_err,
                             "pass": err <= 1e-6 and frame_err <= 1e-8})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator-python", required=True, help="existing Linux Python interpreter with Placo")
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_reference_subset"))
    ap.add_argument("--raw-dir", type=Path, default=Path("experiments/cloud_runs/neutral-reference-subset"))
    args = ap.parse_args()
    if args.out.exists() or args.raw_dir.exists():
        raise FileExistsError("never overwrite prior outputs")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit code before evaluation")
    root = OPEN_DUCK_ROOT / "Open_Duck_reference_motion_generator"
    source = root / "open_duck_reference_motion_generator/gait_generator.py"
    preset_path = source.parent / "robots/open_duck_mini_v2/placo_presets/medium.json"
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True)
    args.raw_dir.mkdir(parents=True)
    write_json(args.out / "protocol.json", {"preregistration_commit": "7800728",
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preset_sha256": hashlib.sha256(preset_path.read_bytes()).hexdigest(), "conditions": CONDITIONS,
        "workers": 2, "per_recording_cap_s": 180, "wall_cap_s": 900, "cloud_work": False})
    start = time.monotonic()

    def record(item):
        name, command = item
        directory = args.raw_dir / name
        directory.mkdir()
        effective = {**preset, **dict(zip(("dx", "dy", "dtheta"), np.array(command) * .54 / 2))}
        path = directory / "preset.json"
        write_json(path, effective)
        remaining = 900 - (time.monotonic() - start)
        if remaining <= 0:
            return {"condition": name, "command": list(command), "all_pass": False, "error_type": "WallCap"}
        cap = min(180, remaining)
        command_line = ["wsl", "--exec", "timeout", "--signal=TERM", "--kill-after=5s", f"{cap:.2f}s",
                        args.generator_python,
                        wsl_path(Path(__file__).with_name("record_reference.py")), "--source", wsl_path(source),
                        "--duck", "open_duck_mini_v2", "--preset", wsl_path(path), "--name", name,
                        "--output_dir", wsl_path(directory), "--length", "8"]
        if name == "stand":
            command_line.append("--stand")
        try:
            with (directory / "generate.log").open("w", encoding="utf-8") as stream:
                subprocess.run(command_line, stdout=stream, stderr=subprocess.STDOUT, check=True,
                               timeout=cap + 10)
            files = [p for p in directory.glob("*.json") if p.name != "preset.json"]
            if len(files) != 1:
                raise ValueError("missing recording")
            recording = json.loads(files[0].read_text(encoding="utf-8"))
            result, ref = validate(recording, command, name == "stand")
            write_json(directory / "reference.json", ref)
            return {"condition": name, "command": list(command), "recording_sha256":
                    hashlib.sha256(files[0].read_bytes()).hexdigest(), "reference_sha256":
                    hashlib.sha256((directory / "reference.json").read_bytes()).hexdigest(), **result}
        except (ValueError, KeyError, subprocess.SubprocessError) as error:
            # Exception messages may contain private subprocess paths; public report records only type.
            return {"condition": name, "command": list(command), "all_pass": False,
                    "error_type": type(error).__name__}

    checks = known_pose_checks()
    if not all(c["pass"] for c in checks):
        raise RuntimeError("known-rotation validation failed")
    rows = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for row in executor.map(record, CONDITIONS.items()):
            rows.append(row)
            write_json(args.out / "trials.json", rows)
            print(row["condition"], row["all_pass"], row.get("criteria", row.get("error_type")), flush=True)
    report = {"all_pass": len(rows) == 7 and all(r["all_pass"] for r in rows),
              "known_pose_checks": checks, "wall_seconds": round(time.monotonic() - start, 2),
              "usable_for_training": False, "reason": "subset only; dynamic motor readiness untested"}
    write_json(args.out / "summary.json", report)
    print(report["all_pass"], report["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
