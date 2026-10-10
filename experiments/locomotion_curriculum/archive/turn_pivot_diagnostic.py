"""Diagnostic: where do the trained pure turns pivot, and which point do the training rewards measure?
(After the gait-averaged tracking run, docs/gait_averaged_tracking_pilot.md; development log 2026-10-10.)

Post-hoc diagnostic, found while exploring the saved native rollouts: no criteria, no training, CPU only.
From the paired native evaluation rollouts (seeds 0/1/2, 5–20 s) of every arm evaluated so far it reports,
for both pure turns and two translations:
- horizontal RMS of the 1 s-smoothed heading-frame velocity (the evaluation's turn metric) at the base
  origin (what the evaluation measures) and at the IMU site (what upstream's training rewards measure);
- the pivot centre (mean heading-frame velocity / mean yaw rate) relative to the base origin.
It also reports the model geometry (IMU, feet, centre of mass in the base frame) and the gait-averaged
tracking score of the verified reference turns when their velocity is measured at each point.

Usage: python -m experiments.locomotion_curriculum.turn_pivot_diagnostic [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nerva.analysis import gait_metrics as gm
from nerva.analysis.motor_eval import smooth, velocities
from nerva.training.motor_artifacts import write_json

RUNS = {"b2": "neutral_gpu_pilot-20261010-152812", "untrained_neutral": "neutral_gpu_pilot-20261010-152812",
        "pilot_candidate": "neutral_gpu_pilot-20261010-152812", "gpu_candidate": "neutral_gpu_pilot-20261010-152812",
        "gait_candidate": "gait_averaged_tracking-20261010-170130"}
COMMANDS = ("turn_left", "turn_right", "forward", "left")
START = 250  # 5 s


def horizontal_rms(v):
    f = smooth(v)
    return float(np.sqrt(np.mean(np.sum(f ** 2, axis=1))))


def geometry(model, data):
    import mujoco
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    base = model.body("base").id
    rotation, origin = data.xmat[base].reshape(3, 3), data.xpos[base]
    out = {name: (rotation.T @ (data.site(name).xpos - origin)).round(4).tolist()
           for name in ("imu", "left_foot", "right_foot")}
    out["centre_of_mass"] = (rotation.T @ (data.subtree_com[base] - origin)).round(4).tolist()
    return out


def trial(model, data, path):
    import mujoco
    arrays = dict(np.load(path))
    qpos, qvel = arrays["qpos"][START:], arrays["qvel"][START:]
    imu_site = model.site("imu").id
    at_base, at_imu = [], []
    for q, v in zip(qpos, qvel):
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_forward(model, data)
        site = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, imu_site, site, 0)
        yaw = gm.quat_to_rpy(q[3:7][None])[0, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        heading = np.array([[c, s], [-s, c]])
        at_base.append(heading @ v[:2])
        at_imu.append(heading @ site[3:5])
    t = (np.arange(len(arrays["base_quat"])) + 1) * .02
    vel = velocities(arrays)[(t >= 5) & (t < 20)]
    mean, yaw_rate = vel[:, :2].mean(axis=0), vel[:, 2].mean()
    return {"base_horizontal_rms": horizontal_rms(np.array(at_base)), "imu_horizontal_rms": horizontal_rms(np.array(at_imu)),
            "evaluation_horizontal_rms": horizontal_rms(vel[:, :2]), "mean_yaw_rate": float(yaw_rate),
            "pivot_forward_left_m": [float(-mean[1] / yaw_rate), float(mean[0] / yaw_rate)] if abs(yaw_rate) > .1 else None}


def reference_scores(root):
    from nerva.motor_contract import planar_tracking
    from nerva.training.neutral_reference import verified_references
    from nerva.training.reference_kinematics import sample_reference
    records, _ = verified_references(root)
    t = np.arange(27 * 40) * .02
    offset = np.array([-.08, 0., .05])
    out = {}
    for record in records:
        command = np.asarray(record["command"])
        if not command[2]:
            continue
        sample = sample_reference(record["reference"], t % .54)
        base = np.asarray(sample["linear_body"])
        imu = base + np.cross(np.asarray(sample["angular_body"]), offset)
        cmd = np.r_[command, np.zeros(4)]

        def averaged(v):
            return np.array([v[max(0, i - 26):i + 1, :2].mean(0) for i in range(len(v))])[27:]
        out[str(command.tolist())] = {
            "mean_velocity_at_base": base[:, :2].mean(axis=0).round(4).tolist(),
            "mean_velocity_at_imu": imu[:, :2].mean(axis=0).round(4).tolist(),
            "gait_averaged_tracking_at_base": float(np.mean(planar_tracking(cmd, averaged(base)))),
            "gait_averaged_tracking_at_imu": float(np.mean(planar_tracking(cmd, averaged(imu))))}
    return out


def main():
    import mujoco

    from nerva.sim.open_duck import SCENE_BACKLASH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_turn_pivot"))
    args = ap.parse_args()
    root = Path.cwd().resolve()
    model = mujoco.MjModel.from_xml_path(str(SCENE_BACKLASH))
    data = mujoco.MjData(model)
    rows = []
    for arm, run in RUNS.items():
        for command in COMMANDS:
            for seed in (0, 1, 2):
                path = root / "experiments/cloud_runs" / run / "eval" / f"{command}_{seed}_{arm}.npz"
                rows.append({"arm": arm, "command": command, "seed": seed, **trial(model, data, path)})
            sel = [r for r in rows if r["arm"] == arm and r["command"] == command]
            print(arm, command, "base", round(np.mean([r["base_horizontal_rms"] for r in sel]), 3),
                  "imu", round(np.mean([r["imu_horizontal_rms"] for r in sel]), 3), flush=True)
    summary = {"geometry_base_frame_m": geometry(model, data), "reference_turns": reference_scores(root), "arms": {}}
    for arm in RUNS:
        summary["arms"][arm] = {}
        for command in COMMANDS:
            sel = [r for r in rows if r["arm"] == arm and r["command"] == command]
            entry = {k: float(np.mean([r[k] for r in sel])) for k in
                     ("base_horizontal_rms", "imu_horizontal_rms", "evaluation_horizontal_rms", "mean_yaw_rate")}
            if sel[0]["pivot_forward_left_m"] is not None:
                entry["pivot_forward_left_m"] = np.mean([r["pivot_forward_left_m"] for r in sel], axis=0).round(4).tolist()
            summary["arms"][arm][command] = entry
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "trials.json", rows)
    write_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
