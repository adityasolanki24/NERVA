"""Local, capped support-region measurement; docs/b2_long_turn_experiment.md."""
import argparse
import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.locomotion_curriculum.gate import DT, COMMANDS, segment_metrics, write_json
from nerva.analysis import gait_metrics as gm
from nerva.interfaces import BehaviourCommand, StyleVector
from nerva.safety import SafetySupervisor
from nerva.sim.capabilities import capabilities_for
from nerva.sim.open_duck import OPEN_DUCK_ROOT, OpenDuckSim, SCENE_BACKLASH, to_arrays
from nerva.world.self_state import estimate_self_state

DURATION = 65
WALL_CAP = 900
POINTS = ("base", "com", "foot_midpoint", "contact_centroid")


class KinematicRecorder:
    """Forward kinematics on isolated data, excluding non-robot bodies."""
    def __init__(self, sim):
        import mujoco
        self.mj = mujoco
        self.model = sim.model
        self.data = mujoco.MjData(self.model)
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("expected one free robot joint")
        self.root = int(self.model.jnt_bodyid[free[0]])
        self.bodies = np.flatnonzero(self.model.body_rootid == self.root)
        self.mass = self.model.body_mass[self.bodies]
        if self.mass.sum() <= 0:
            raise ValueError("robot subtree has no mass")
        self.sites = sim._foot_sites

    def sample(self, sim):
        d = self.data
        d.qpos[:] = sim.data.qpos
        d.qvel[:] = sim.data.qvel
        d.time = sim.data.time
        self.mj.mj_forward(self.model, d)
        feet = d.site_xpos[self.sites].copy()
        contacts = np.asarray(sim.inf.get_feet_contacts(d), dtype=bool)
        com = np.average(d.xipos[self.bodies], weights=self.mass, axis=0)
        if not np.isfinite(com).all() or not np.isfinite(feet).all():
            raise ValueError("nonfinite kinematic measurement")
        centroid = feet[contacts].mean(axis=0) if contacts.any() else np.full(3, np.nan)
        return com, feet, contacts, centroid


def centre_metrics(centres, times):
    centres = np.asarray(centres)
    delta = centres[-1] - centres[0]
    distance = float(np.linalg.norm(delta))
    pairwise = centres[:, None] - centres[None, :]
    return {"max_separation_m": float(np.linalg.norm(pairwise, axis=2).max()),
            "endpoint_delta_xy_m": delta.tolist(), "endpoint_distance_m": distance,
            "endpoint_speed_m_s": distance / float(times[-1] - times[0])}


def analyse(arrays, command, completed, interventions):
    t = arrays["t"]
    yaw = np.unwrap(gm.quat_to_rpy(arrays["base_quat"])[:, 2])
    tilt = gm.tilt_deg(arrays["base_quat"])
    points = {"base": arrays["base_pos"][:, :2], "com": arrays["com"][:, :2],
              "foot_midpoint": arrays["foot_xyz"].mean(axis=1)[:, :2],
              "contact_centroid": arrays["contact_centroid"][:, :2]}
    contacts = arrays["measured_contacts"].any(axis=1)
    start = np.searchsorted(t, 5)
    blocks, reversal = [], False
    if len(t) > start:
        if command == "stop":
            blocks = [np.flatnonzero((t >= a) & ((t < a + 10) if a < 55 else (t <= 65)))
                      for a in range(5, 65, 10)]
            blocks = [idx for idx in blocks if len(idx)]
        else:
            sign = 1 if command == "turn_left" else -1
            phase = sign * (yaw[start:] - yaw[start])
            reversal = bool(np.any(np.diff(phase) < -.10))
            boundaries = [start]
            for k in range(1, int(np.floor(phase.max() / (2 * np.pi))) + 1):
                boundaries.append(start + int(np.flatnonzero(phase >= 2 * np.pi * k)[0]))
            blocks = [np.arange(a, b) for a, b in zip(boundaries, boundaries[1:])]
    times = [float(t[idx].mean()) for idx in blocks]
    coverage = [float(contacts[idx].mean()) for idx in blocks]
    centres = {name: [p[idx][contacts[idx]].mean(axis=0).tolist() if name == "contact_centroid"
                      and contacts[idx].any() else p[idx].mean(axis=0).tolist()
                      for idx in blocks] for name, p in points.items()}
    # Missing contact centres are explicit nulls in public reports.
    centres["contact_centroid"] = [c if np.isfinite(c).all() else None for c in centres["contact_centroid"]]
    metrics = {name: centre_metrics(c, times) for name, c in centres.items()
               if len(c) >= 2 and all(x is not None for x in c)}
    valid = (completed and len(t) == 3250 and np.allclose(t, np.arange(1, 3251) * DT)
             and len(blocks) >= 4 and not reversal and interventions == 0
             and all(c >= .90 for c in coverage) and not np.any(tilt > 45))
    classification = "inconclusive"
    if valid:
        com, foot = metrics["com"], metrics["foot_midpoint"]
        local = all(m["max_separation_m"] <= .05 and m["endpoint_speed_m_s"] <= .001 for m in (com, foot))
        a, b = np.array(com["endpoint_delta_xy_m"]), np.array(foot["endpoint_delta_xy_m"])
        coherent = (com["endpoint_distance_m"] >= .10 and foot["endpoint_distance_m"] >= .10
                    and np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) >= np.cos(np.pi / 6))
        classification = "local_region" if local else "coherent_migration" if coherent else "inconclusive"
    return {"classification": classification, "description_eligible": bool(valid), "completed": completed,
            "fell": bool(np.any(tilt > 45)), "max_tilt_deg": float(tilt.max()),
            "shadow_interventions": interventions, "large_yaw_reversal": reversal,
            "blocks": len(blocks), "block_mean_times_s": times, "block_contact_coverage": coverage,
            "block_centres_xy_m": centres, "metrics": metrics,
            "base_velocity_gate": segment_metrics(arrays, 5, 65, command) if len(t) >= 300 else None}


def rollout(policy, command, seed, raw_dir, deadline):
    with contextlib.redirect_stdout(io.StringIO()):
        sim = OpenDuckSim(policy_path=policy, scene=SCENE_BACKLASH, raw_accel=True,
                          obs_noise=True, init_joint_noise=.02, seed=seed)
    recorder, safety = KinematicRecorder(sim), SafetySupervisor()
    logs, measurements = [], []
    stop_reason = None
    for k in range(3250):
        if time.monotonic() >= deadline:
            stop_reason = "wall_cap"
            break
        cmd = BehaviourCommand(*COMMANDS[command], style_vector=StyleVector())
        safety.filter(cmd, (0., 0., 0., 0.), estimate_self_state(k * DT, sim.data.qpos, sim.data.qvel))
        sim.set_behaviour(cmd)
        logs.extend(sim.run(DT))
        if not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
            stop_reason = "nonfinite_state"
            break
        measurements.append(recorder.sample(sim))
        if gm.tilt_deg(sim.data.qpos[3:7][None])[0] > 45:
            stop_reason = "fall"
            break
    # Preserve usable measurements on termination; a nonfinite row is not analysed.
    arrays = to_arrays(logs[:len(measurements)])
    arrays["t"] = np.arange(1, len(measurements) + 1) * DT
    for index, name in enumerate(("com", "foot_xyz", "measured_contacts", "contact_centroid")):
        arrays[name] = np.array([m[index] for m in measurements])
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{command}_{seed}.npz"
    np.savez_compressed(path, **arrays)
    if not measurements:
        result = {"classification": "inconclusive", "completed": False, "blocks": 0,
                  "shadow_interventions": safety.interventions}
    else:
        result = analyse(arrays, command, stop_reason is None, safety.interventions)
    return {"command": command, "seed": seed, "stop_reason": stop_reason,
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **result}


def summarise(rows):
    counts = {cmd: {c: sum(r["command"] == cmd and r["classification"] == c for r in rows)
                   for c in ("local_region", "coherent_migration", "inconclusive")}
              for cmd in ("turn_left", "turn_right", "stop")}
    expected = {(cmd, seed) for cmd in counts for seed in range(5)}
    complete = len(rows) == 15 and {(r["command"], r["seed"]) for r in rows} == expected
    complete &= all(r["completed"] for r in rows)
    supported = [c for c in ("local_region", "coherent_migration")
                 if complete and counts["stop"]["local_region"] >= 4
                 and all(counts[d][c] >= 4 for d in ("turn_left", "turn_right"))]
    return {"complete": bool(complete), "counts": counts,
            "aggregate": supported[0] if supported else "inconclusive", "original_gate": "failed; unchanged"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_long_turn"))
    ap.add_argument("--raw-dir", type=Path, default=Path("experiments/cloud_runs/b2-long-turn-local"))
    args = ap.parse_args()
    if args.out.exists() or args.raw_dir.exists():
        raise FileExistsError("never overwrite previous outputs")
    if (capabilities_for(str(args.policy)).name != "B2" or not args.policy.is_file()
            or not args.policy.name.endswith("_300482560.onnx")):
        raise ValueError("requires final B2 checkpoint")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before evaluating")
    args.out.mkdir(parents=True)
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(args.out / "protocol.json", {"preregistration_commit": "dec8fba", "implementation_commit": rev,
               "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
               "scene_sha256": hashlib.sha256(SCENE_BACKLASH.read_bytes()).hexdigest(),
               "upstream_commit": subprocess.check_output(["git", "-C", str(OPEN_DUCK_ROOT / "Open_Duck_Playground"),
                                                           "rev-parse", "HEAD"], text=True).strip(),
               "seeds": list(range(5)), "duration_s": DURATION, "dt_s": DT, "wall_cap_s": WALL_CAP,
               "cloud_work": False, "safety": "shadow only", "style": [0, 0, 0],
               "raw_accel": True, "obs_noise": True, "init_joint_noise_rad": .02,
               "head_offsets": [0, 0, 0, 0], "scene": "flat_terrain_backlash"})
    start = time.monotonic()
    rows = []
    for cmd in ("turn_left", "turn_right", "stop"):
        for seed in range(5):
            row = rollout(args.policy, cmd, seed, args.raw_dir, start + WALL_CAP)
            rows.append(row)
            write_json(args.out / "trials.json", rows)
            print(f"{len(rows)}/15 {cmd} {seed}: {row['classification']} stop={row['stop_reason']}", flush=True)
            if row["stop_reason"] in ("wall_cap", "nonfinite_state"):
                break
        if time.monotonic() >= start + WALL_CAP or rows[-1]["stop_reason"] == "nonfinite_state":
            break
    report = summarise(rows)
    report["wall_seconds"] = round(time.monotonic() - start, 2)
    write_json(args.out / "summary.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
