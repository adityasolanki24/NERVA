"""Fixed B2 neutral gate; protocol: docs/b2_robustness_gate.md.

No training, affect or cloud calls. Raw primary motor trials and separate
deterministic safety diagnostics. Outputs contain no deployment/personal paths.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from nerva.analysis import gait_metrics as gm
from nerva.interfaces import BehaviourCommand, StyleVector
from nerva.safety import SafetySupervisor
from nerva.sim.capabilities import capabilities_for
from nerva.sim.open_duck import OpenDuckSim, SCENE_BACKLASH, to_arrays
from nerva.world.self_state import estimate_self_state
from nerva.analysis.motor_eval import smooth, velocities  # noqa: F401,E402
from nerva.training.motor_artifacts import write_json  # noqa: F401,E402

DT = 0.02
WINDOW = 50
SEEDS = range(5)
COMMANDS = {
    "stop": (0.0, 0.0, 0.0), "forward": (0.15, 0.0, 0.0), "backward": (-0.15, 0.0, 0.0),
    "left": (0.0, 0.10, 0.0), "right": (0.0, -0.10, 0.0),
    "turn_left": (0.0, 0.0, 0.60), "turn_right": (0.0, 0.0, -0.60),
}
PHASES = ("stop", "forward", "backward", "left", "right", "turn_left", "turn_right", "stop")


@dataclass(frozen=True)
class Trial:
    name: str
    group: str
    seed: int
    command: str = "forward"
    seconds: float = 20.0
    pitch: float = 0.0
    yaw: float = 0.0
    push_deg: int | None = None


def protocol_trials():
    out = []
    for command in list(COMMANDS)[1:]:
        out += [Trial(f"steady_{command}_{s}", "steady", s, command) for s in SEEDS]
    out += [Trial(f"transition_{s}", "transition", s, seconds=40.0) for s in SEEDS]
    for command in ("forward", "backward"):
        for deg in (0, 90, 180, 270):
            out += [Trial(f"push_{command}_{deg}_{s}", "push", s, command, 15.0, push_deg=deg) for s in SEEDS]
    for i, (pitch, yaw) in enumerate(((-0.2, -0.4), (-0.2, 0.4), (0.6, -0.4), (0.6, 0.4))):
        out += [Trial(f"standing_head_{i}_{s}", "standing_head", s, "stop", 15.0, pitch, yaw) for s in SEEDS]
    for i, (pitch, yaw) in enumerate(((-0.15, 0.0), (0.15, 0.0), (0.0, -0.08), (0.0, 0.08))):
        out += [Trial(f"walking_head_{i}_{s}", "walking_head", s, pitch=pitch, yaw=yaw) for s in SEEDS]
    return out


def segment_metrics(arrays, start, end, command):
    t = arrays["t"]
    mask = (t >= start - 1e-9) & (t < end - 1e-9)
    raw = velocities(arrays)[mask]
    if len(raw) < WINDOW:
        raise ValueError("measurement interval lacks a full tracking window")
    filtered = smooth(raw)
    cmd = np.array(COMMANDS[command])
    moving = bool(np.any(cmd))
    axis = int(np.argmax(np.abs(cmd)))
    signed_mean = float(raw[:, axis].mean() * np.sign(cmd[axis])) if moving else 0.0
    tracking = float(gm.rms(filtered[:, axis] - cmd[axis])) if moving else 0.0
    threshold = 0.10 if axis == 2 else 0.01
    stationary = float(np.mean(np.abs(filtered[:, axis]) < threshold)) if moving else None
    cross = float(gm.rms(filtered[:, 1 - axis])) if moving and axis < 2 else None
    horizontal = float(np.sqrt(np.mean(np.sum(filtered[:, :2] ** 2, axis=1))))
    yaw_rms = float(gm.rms(filtered[:, 2]))
    direction_ok = signed_mean >= 0.5 * abs(cmd[axis]) if moving else True
    tracking_ok = tracking <= 0.6 * abs(cmd[axis]) if moving else True
    cross_ok = (cross <= 0.05 and yaw_rms <= 0.20) if moving and axis < 2 else horizontal <= 0.03
    return {
        "command": command, "interval_s": [start, end], "mean_velocity": raw.mean(0).tolist(),
        "signed_axis_mean": signed_mean, "axis_tracking_rmse": tracking,
        "stationary_fraction": stationary, "orthogonal_rms": cross,
        "horizontal_rms": horizontal, "yaw_rms": yaw_rms,
        "lift_mm": [1000 * gm.lift_height(arrays["foot_z"][mask, i]) for i in range(2)],
        "direction_ok": bool(direction_ok), "tracking_ok": bool(tracking_ok),
        "cross_ok": bool(cross_ok), "stationary_ok": stationary <= 0.10 if moving else True,
        "stop_ok": horizontal <= 0.02 and yaw_rms <= 0.15 if not moving else True,
    }


def recovery_metrics(arrays, command):
    t = arrays["t"]
    axis = int(np.argmax(np.abs(COMMANDS[command])))
    v = velocities(arrays)
    before = (t >= 6.0 - 1e-9) & (t < 8.0 - 1e-9)
    baseline = float(v[before, axis].mean())
    tolerance = max(0.03, 0.25 * abs(baseline))
    after = t > 8.0 + 1e-9
    filtered = smooth(v[after])
    tilt = gm.tilt_deg(arrays["base_quat"])
    stable = np.convolve((tilt[after] < 20.0).astype(int), np.ones(WINDOW, dtype=int), mode="valid") == WINDOW
    recovered = stable & (np.abs(filtered[:, axis] - baseline) <= tolerance) & (np.abs(filtered[:, 2]) <= 0.20)
    candidates = np.flatnonzero(recovered)
    recovery_s = float(t[after][WINDOW - 1 + candidates[0]] - 8.0) if len(candidates) else None
    return {"pre_push_axis_mean": baseline, "velocity_tolerance": tolerance, "recovery_s": recovery_s,
            "recovery_ok": recovery_s is not None and recovery_s <= 5.0 and not np.any(tilt > 45.0)}


def request(trial, t):
    name = PHASES[min(int(round(t / DT)) // 250, 7)] if trial.group == "transition" else trial.command
    cmd = BehaviourCommand(*COMMANDS[name], style_vector=StyleVector())
    ramp = min(t / 0.5, 1.0)
    return cmd, (0.0, ramp * trial.pitch, ramp * trial.yaw, 0.0)


def rollout(policy, trial, raw_dir, safety_on=False):
    # Upstream constructor prints local paths; keep those diagnostics private.
    with contextlib.redirect_stdout(io.StringIO()):
        sim = OpenDuckSim(policy_path=policy, scene=SCENE_BACKLASH, raw_accel=True,
                          obs_noise=True, init_joint_noise=0.02, seed=trial.seed)
    supervisor = SafetySupervisor()
    log, stops = [], []
    completed = True
    for k in range(int(round(trial.seconds / DT))):
        t = k * DT
        if trial.push_deg is not None and k == 400:
            yaw = gm.quat_to_rpy(sim.data.qpos[3:7][None])[0, 2]
            angle = yaw + np.radians(trial.push_deg)
            sim.push(0.3 * np.cos(angle), 0.3 * np.sin(angle))
        cmd, head = request(trial, t)
        filtered_cmd, filtered_head, reason = supervisor.filter(
            cmd, head, estimate_self_state(t, sim.data.qpos, sim.data.qvel))
        sim.set_head_offset(* (filtered_head if safety_on else head))
        sim.set_behaviour(filtered_cmd if safety_on else cmd)
        log.extend(sim.run(DT))
        stops.append(bool(reason))
        if not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
            completed = False
            break
    arrays = to_arrays(log)
    arrays["t"] = (np.arange(len(log)) + 1) * DT
    suffix = "safety" if safety_on else "primary"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir / f"{trial.name}_{suffix}.npz", **arrays, safety_stopped=stops)
    result = {"trial": trial.name, "group": trial.group, "seed": trial.seed, "command": trial.command,
              "head_pitch": trial.pitch, "head_yaw": trial.yaw, "push_deg": trial.push_deg,
              "safety_applied": safety_on, "completed": completed,
              "safety_interventions": supervisor.interventions, "safety_stop_fraction": float(np.mean(stops))}
    if not completed:
        return {**result, "fell": True, "segments": [], "recovery_ok": False}
    tilt = gm.tilt_deg(arrays["base_quat"])
    result.update(max_tilt_deg=float(tilt.max()), fell=bool(np.any(tilt > 45.0)))
    if trial.group == "transition":
        result["segments"] = [segment_metrics(arrays, 5 * i + 2, 5 * i + 5, name)
                              for i, name in enumerate(PHASES)]
    else:
        result["segments"] = [segment_metrics(arrays, 5, trial.seconds, trial.command)]
    if trial.group == "push":
        result.update(recovery_metrics(arrays, trial.command))
    return result


def summarise(rows):
    groups = {g: [r for r in rows if r["group"] == g]
              for g in ("steady", "transition", "push", "standing_head", "walking_head")}
    expected = {"steady": 30, "transition": 5, "push": 40, "standing_head": 20, "walking_head": 20}
    complete = len(rows) == 115 and len({r["trial"] for r in rows}) == 115
    complete &= all(len(groups[g]) == n for g, n in expected.items()) and all(r["completed"] for r in rows)
    stable = all(not r["fell"] for r in rows if r["group"] != "push")
    steady = groups["steady"]
    transitions = groups["transition"]
    moving_phases = [s for r in transitions for s in r["segments"] if s["command"] != "stop"]
    stationary_ok = all(s["stationary_ok"] for r in steady for s in r["segments"])
    stationary_ok &= all(s["stationary_ok"] for s in moving_phases)
    track_keys = ("direction_ok", "tracking_ok", "cross_ok")
    tracking_ok = all(all(s[k] for k in track_keys) for r in steady for s in r["segments"])
    transition_ok = all(s["direction_ok"] and s["tracking_ok"] and s["stop_ok"]
                        for r in transitions for s in r["segments"])
    recovery = {}
    for cmd in ("forward", "backward"):
        for deg in (0, 90, 180, 270):
            rs = [r for r in groups["push"] if r["command"] == cmd and r["push_deg"] == deg]
            recovery[f"{cmd}_{deg}"] = sum(r.get("recovery_ok", False) for r in rs)
    push_ok = all(not r["fell"] for r in groups["push"]) and all(n >= 4 for n in recovery.values())
    forward = {r["seed"]: r["segments"][0]["signed_axis_mean"] for r in steady
               if r["command"] == "forward" and r["segments"]}
    head_ok = all(not r["fell"] for r in groups["standing_head"])
    for r in groups["walking_head"]:
        s = r["segments"][0] if r["segments"] else None
        head_ok &= (s is not None and all(s[k] for k in (*track_keys, "stationary_ok"))
                    and s["signed_axis_mean"] >= 0.75 * forward.get(r["seed"], float("inf")))
    criteria = {"complete": bool(complete), "unperturbed_stability": stable,
                "steady_tracking": tracking_ok, "no_standing_solution": stationary_ok,
                "transitions_stops": transition_ok, "push_recovery": push_ok, "head_tolerance": bool(head_ok)}
    return {"criteria": criteria, "all_pass": all(criteria.values()), "push_recovered_counts_of_5": recovery,
            "groups": {g: {"trials": len(rs), "falls": sum(r["fell"] for r in rs),
                           "shadow_interventions": sum(r["safety_interventions"] for r in rs),
                           "mean_shadow_stop_fraction": float(np.mean([r["safety_stop_fraction"] for r in rs])) if rs else None}
                       for g, rs in groups.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_b2"))
    ap.add_argument("--raw-dir", type=Path, default=Path("experiments/cloud_runs/b2-gate-local"))
    args = ap.parse_args()
    cap = capabilities_for(str(args.policy))
    if (cap.name != "B2" or cap.style_input != "neutral_only" or not args.policy.is_file()
            or not args.policy.name.endswith("_300482560.onnx")):
        raise ValueError("gate requires the B2 checkpoint in its identified run folder")
    if args.out.exists():
        raise FileExistsError("use a new output directory; never overwrite a previous gate")
    args.out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    # out is already created but empty; git does not regard an empty directory as dirty.
    if dirty:
        raise RuntimeError("commit the implementation before evaluating")
    write_json(args.out / "protocol.json", {"preregistration_commit": "14a139c", "implementation_commit": rev,
               "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(), "seeds": list(SEEDS),
               "dt_s": DT, "scene": "flat_terrain_backlash", "style": [0, 0, 0],
               "primary_safety": "shadow only", "cloud_work": False})
    t0 = time.monotonic()
    rows, diagnostics = [], []
    for trial in protocol_trials():
        result = rollout(args.policy, trial, args.raw_dir)
        rows.append(result)
        print(f"{len(rows)}/115 {trial.name}: fell={result['fell']} shadow_stops={result['safety_interventions']}", flush=True)
        write_json(args.out / "trials.json", rows)
        if result["safety_interventions"] and result["completed"]:
            diagnostics.append(rollout(args.policy, trial, args.raw_dir, safety_on=True))
            write_json(args.out / "safety_diagnostics.json", diagnostics)
    report = summarise(rows)
    report.update(wall_seconds=round(time.monotonic() - t0, 2), safety_diagnostic_replays=len(diagnostics))
    write_json(args.out / "summary.json", report)
    write_json(args.out / "safety_diagnostics.json", diagnostics)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
