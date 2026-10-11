"""Neutral long motor gate, with and without latency; protocol: docs/neutral_motor_gate.md (b6ec4c5).

No training, affect or cloud calls. Every trial runs with latency off and on (same seed). Primary motor
trials have no safety override; the deterministic SafetySupervisor runs in shadow only. Steady trials use
the pilot evaluation's per-trial rules (motor_metrics); transitions and pushes reuse the B2 gate's
definitions at the neutral command magnitudes.

Usage: python -m experiments.locomotion_curriculum.neutral_gate [--candidate NAME] --out DIR [--raw-dir DIR]
Candidates are fixed here with their preregistration and policy hash; the protocol is identical for all.
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
from nerva.analysis.motor_eval import deployment_metadata, motor_metrics, push_recovery, smooth, velocities
from nerva.interfaces import BehaviourCommand
from nerva.safety import SafetySupervisor
from nerva.sim.open_duck import OpenDuckSim, SCENE_BACKLASH, to_arrays
from nerva.training.motor_artifacts import write_json
from nerva.training.neutral_reference import COMMANDS as NEUTRAL_COMMANDS
from nerva.world.self_state import estimate_self_state

CANDIDATES = {
    "base_origin_velocity": {  # docs/neutral_motor_gate.md
        "preregistration": "b6ec4c5",
        "policy": "experiments/cloud_runs/base_origin_velocity-20261010-201858/candidate.onnx",
        "sha256": "9394fa5db7fc613f7ee311ec329e39a693f790ba219e26913db7a7464a11c5a2"},
    "turn_translation": {  # docs/turn_translation_pilot.md; the run's final candidate, hash from its summary
        "preregistration": "366aec9", "policy": None, "sha256": None},
    "e1_prime": {  # docs/expressive_posture_e1prime.md: neutral retention at e = (0, 0), styled 103-input contract
        "preregistration": "0c221cf", "policy": None, "sha256": None, "styled": True},
}
PREREGISTRATION = CANDIDATES["base_origin_velocity"]["preregistration"]
POLICY = CANDIDATES["base_origin_velocity"]["policy"]
POLICY_SHA256 = CANDIDATES["base_origin_velocity"]["sha256"]
DT, WINDOW = 0.02, 50
SEEDS = range(5)
NAMES = ("rest", "forward", "backward", "left", "right", "turn_left", "turn_right")
COMMANDS = dict(zip(NAMES, (tuple(c) for c in NEUTRAL_COMMANDS)))
PHASES = ("rest", "forward", "backward", "left", "right", "turn_left", "turn_right", "rest")
PUSH_STEP, PUSH_SPEED = 400, 0.30


@dataclass(frozen=True)
class Trial:
    name: str
    group: str
    seed: int
    command: str = "forward"
    seconds: float = 20.0
    push_deg: int | None = None


def protocol_trials():
    out = []
    for command in NAMES:
        out += [Trial(f"steady_{command}_{s}", "steady", s, command) for s in SEEDS]
    out += [Trial(f"transition_{s}", "transition", s, "rest", 40.0) for s in SEEDS]
    for command in ("forward", "backward"):
        for deg in (0, 90, 180, 270):
            out += [Trial(f"push_{command}_{deg}_{s}", "push", s, command, 15.0, deg) for s in SEEDS]
    return out


def phase_metrics(arrays, start, end, command):
    """B2 gate transition-phase metrics at the neutral magnitudes."""
    t = arrays["t"]
    mask = (t >= start - 1e-9) & (t < end - 1e-9)
    raw = velocities(arrays)[mask]
    if len(raw) < WINDOW:
        raise ValueError("measurement interval lacks a full tracking window")
    filtered = smooth(raw)
    cmd = np.array(COMMANDS[command])
    moving = bool(np.any(cmd))
    axis = int(np.argmax(np.abs(cmd)))
    horizontal = float(np.sqrt(np.mean(np.sum(filtered[:, :2] ** 2, axis=1))))
    yaw_rms = float(gm.rms(filtered[:, 2]))
    out = {"command": command, "interval_s": [start, end], "mean_velocity": raw.mean(0).tolist(),
           "horizontal_rms": horizontal, "yaw_rms": yaw_rms}
    if moving:
        signed = float(raw[:, axis].mean() * np.sign(cmd[axis]))
        tracking = float(gm.rms(filtered[:, axis] - cmd[axis]))
        stationary = float(np.mean(np.abs(filtered[:, axis]) < (0.10 if axis == 2 else 0.01)))
        out.update(signed_axis_mean=signed, axis_tracking_rmse=tracking, stationary_fraction=stationary,
                   ok=bool(signed >= .5 * abs(cmd[axis]) and tracking <= .6 * abs(cmd[axis]) and stationary <= .10))
    else:
        out.update(ok=bool(horizontal <= .02 and yaw_rms <= .15))
    return out


def recovery_metrics(arrays, command):
    """Push recovery for this gate's command magnitudes (nerva.analysis.motor_eval.push_recovery)."""
    return push_recovery(arrays, int(np.argmax(np.abs(COMMANDS[command]))))


def command_at(trial, k):
    name = PHASES[min(k // 250, 7)] if trial.group == "transition" else trial.command
    return BehaviourCommand(*COMMANDS[name])


def rollout(policy, policy_hash, trial, latency, raw_dir, styled=False, style=(0., 0.)):
    with contextlib.redirect_stdout(io.StringIO()):  # upstream constructor prints local paths
        sim = OpenDuckSim(policy_path=policy, scene=SCENE_BACKLASH, raw_accel=True, obs_noise=True,
                          init_joint_noise=0.02, seed=trial.seed, action_delay=latency)
    sim.set_neutral_motor_contract(deployment_metadata(policy_hash, styled))
    if styled:
        sim.set_neutral_style(*style)
    supervisor = SafetySupervisor()
    log, stops, completed = [], [], True
    for k in range(int(round(trial.seconds / DT))):
        if trial.push_deg is not None and k == PUSH_STEP:
            yaw = gm.quat_to_rpy(sim.data.qpos[3:7][None])[0, 2]
            angle = yaw + np.radians(trial.push_deg)
            sim.push(PUSH_SPEED * np.cos(angle), PUSH_SPEED * np.sin(angle))
        command = command_at(trial, k)
        _, _, reason = supervisor.filter(command, (0., 0., 0., 0.),
                                         estimate_self_state(k * DT, sim.data.qpos, sim.data.qvel))
        sim.set_behaviour(command)
        log.extend(sim.run(DT))
        stops.append(bool(reason))
        if not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
            completed = False
            break
    arrays = to_arrays(log)
    arrays["t"] = (np.arange(len(log)) + 1) * DT
    condition = "latency" if latency else "no_latency"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir / f"{condition}_{trial.name}.npz", **arrays, safety_stopped=stops)
    tilt = gm.tilt_deg(arrays["base_quat"]) if len(log) else np.array([np.inf])
    completed = completed and len(log) == int(round(trial.seconds / DT))
    row = {"condition": condition, "trial": trial.name, "group": trial.group, "seed": trial.seed,
           "command": trial.command, "push_deg": trial.push_deg, "completed": completed,
           "fell": bool(not completed or np.any(tilt > 45.0)), "max_tilt_deg": float(np.max(tilt)),
           "shadow_interventions": supervisor.interventions, "shadow_stop_fraction": float(np.mean(stops))}
    if not completed:
        return {**row, "ok": False}
    if trial.group == "steady":
        row["metrics"] = motor_metrics(arrays, COMMANDS[trial.command], not row["fell"])
        row["ok"] = bool(row["metrics"]["all_pass"])
    elif trial.group == "transition":
        row["phases"] = [phase_metrics(arrays, 5 * i + 2, 5 * i + 5, name) for i, name in enumerate(PHASES)]
        row["ok"] = bool(not row["fell"] and all(p["ok"] for p in row["phases"]))
    else:
        row.update(recovery_metrics(arrays, trial.command))
        row["ok"] = bool(not row["fell"] and row["recovery_ok"])
    return row


def summarise(rows):
    out = {}
    for condition in ("no_latency", "latency"):
        rs = [r for r in rows if r["condition"] == condition]
        steady = [r for r in rs if r["group"] == "steady"]
        transitions = [r for r in rs if r["group"] == "transition"]
        pushes = [r for r in rs if r["group"] == "push"]
        steady_passes = {n: sum(r["ok"] for r in steady if r["command"] == n) for n in NAMES}
        phase_fail = [f"seed {r['seed']} phase {i} {p['command']}" for r in transitions
                      for i, p in enumerate(r.get("phases", [])) if not p["ok"]]
        recovered = {f"{c}_{d}": sum(r["ok"] for r in pushes if r["command"] == c and r["push_deg"] == d)
                     for c in ("forward", "backward") for d in (0, 90, 180, 270)}
        criteria = {
            "stability": len(rs) == 80 and all(r["completed"] for r in rs)
            and not any(r["fell"] for r in steady + transitions),
            "steady_5_of_5": all(n == 5 for n in steady_passes.values()),
            "transitions": len(transitions) == 5 and all(r["ok"] for r in transitions),
            "pushes": not any(r["fell"] for r in pushes) and all(n >= 4 for n in recovered.values())}
        out[condition] = {"criteria": criteria, "all_pass": all(criteria.values()),
                          "steady_passes_of_5": steady_passes, "transition_phase_failures": phase_fail,
                          "push_recovered_of_5": recovered,
                          "falls": sum(r["fell"] for r in rs),
                          "shadow_interventions": sum(r["shadow_interventions"] for r in rs)}
    return {"conditions": out, "gate_passes": all(c["all_pass"] for c in out.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=sorted(CANDIDATES), default="base_origin_velocity")
    ap.add_argument("--run", type=Path, help="fetched run directory (turn_translation)")
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_neutral_gate"))
    ap.add_argument("--raw-dir", type=Path, default=Path("experiments/cloud_runs/neutral-gate-local"))
    args = ap.parse_args()
    root = Path.cwd().resolve()
    spec = dict(CANDIDATES[args.candidate])
    if spec["policy"] is None:  # a run's final candidate: its hash comes from the run's own training summary
        summary = json.loads((args.run / "report" / "training_summary.json").read_text(encoding="utf-8"))
        spec["policy"] = (args.run / "candidate.onnx").resolve().relative_to(root).as_posix()
        spec["sha256"] = summary["candidate_onnx_sha256"]
    if spec["preregistration"] == "PENDING":
        raise SystemExit("candidate is not preregistered")
    policy = root / spec["policy"]
    policy_hash = hashlib.sha256(policy.read_bytes()).hexdigest()
    if policy_hash != spec["sha256"]:
        raise ValueError("policy hash differs from the preregistration")
    if args.out.exists():
        raise FileExistsError("use a new output directory; never overwrite a previous gate")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit the implementation before evaluating")
    args.out.mkdir(parents=True)
    write_json(args.out / "protocol.json", {
        "candidate": args.candidate, "gate_preregistration": "b6ec4c5",
        "style": [0., 0.] if spec.get("styled") else None,
        "preregistration_commit": spec["preregistration"],
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "policy": spec["policy"], "policy_sha256": policy_hash, "seeds": list(SEEDS), "dt_s": DT,
        "scene": "flat_terrain_backlash", "conditions": ["no_latency", "latency"],
        "latency_model": "applied action from 0/1/2 control steps ago, uniform, seeded stream (seed, 1)",
        "push": {"step": PUSH_STEP, "speed_m_s": PUSH_SPEED}, "primary_safety": "shadow only", "cloud_work": False})
    start, rows = time.monotonic(), []
    trials = protocol_trials()
    for latency in (False, True):
        for trial in trials:
            rows.append(rollout(policy, policy_hash, trial, latency, args.raw_dir, spec.get("styled", False)))
            write_json(args.out / "trials.json", rows)
        print("latency" if latency else "no latency", "done", flush=True)
    report = summarise(rows)
    report["wall_seconds"] = round(time.monotonic() - start, 1)
    write_json(args.out / "summary.json", report)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
