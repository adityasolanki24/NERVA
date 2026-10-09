"""Timing diagnostic on fixed saved fits; no regeneration or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from nerva.training.reference_kinematics import fit_reference, sample_reference


def compare(reference, t, joints):
    t, joints = np.asarray(t), np.asarray(joints)
    if (t.ndim != 1 or joints.shape != (len(t), 16) or np.any(np.diff(t) <= 0)
            or not np.isfinite(t).all() or not np.isfinite(joints).all()):
        raise ValueError("malformed timestamped joints")
    endpoints = (t[1:] >= 4) & (t[1:] < 6)
    if endpoints.sum() < 1:
        raise ValueError("no held-out samples")
    end, start = t[1:][endpoints], t[:-1][endpoints]
    target = (np.diff(joints, axis=0) / np.diff(t)[:, None])[endpoints]
    predictions = {"endpoint": sample_reference(reference, end)["joint_velocity"],
                   "midpoint": sample_reference(reference, (start + end) / 2)["joint_velocity"],
                   "interval_average": (sample_reference(reference, end)["joint_position"] -
                                        sample_reference(reference, start)["joint_position"]) / (end - start)[:, None]}
    rmse = {name: np.sqrt(np.mean((value - target) ** 2, axis=0)).tolist() for name, value in predictions.items()}
    maxima = {name: max(value) for name, value in rmse.items()}
    base = maxima["endpoint"]
    supported = all(maxima[name] <= .5 and maxima[name] <= .5 * base for name in ("midpoint", "interval_average"))
    return {"rmse_per_joint": rmse, "max_component_rmse": maxima,
            "ratios_to_endpoint": {name: value / base if base > 1e-12 else None for name, value in maxima.items()},
            "timing_accounted": bool(supported), "heldout_intervals": int(endpoints.sum())}


def known_checks():
    rows = []
    for dt in (.01, .02, .04):
        t = np.arange(0, 8, dt)
        joints = np.sin(2 * np.pi * t[:, None] / .54) * np.ones((1, 16))
        ref = fit_reference(t, joints, np.ones((len(t), 2)), np.zeros((len(t), 3)), np.zeros((len(t), 3)), .54)
        result = compare(ref, t, joints)["max_component_rmse"]
        rows.append({"dt_s": dt, **result, "pass": result["interval_average"] <= 1e-8
                     and result["midpoint"] < result["endpoint"]})
    t = np.arange(0, 8, .02)
    ref = fit_reference(t, np.ones((len(t), 16)), np.ones((len(t), 2)), np.zeros((len(t), 3)),
                        np.zeros((len(t), 3)), .54, static=True)
    static = compare(ref, t, np.ones((len(t), 16)))["max_component_rmse"]
    rows.append({"static": True, **static, "pass": max(static.values()) <= 1e-8})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("experiments/cloud_runs/neutral-reference-subset"))
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_derivative_alignment"))
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("never overwrite diagnostic results")
    if not args.worker:
        try:
            subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.derivative_alignment", "--worker",
                            "--raw-dir", str(args.raw_dir), "--out", str(args.out)], check=True, timeout=120)
        except subprocess.TimeoutExpired:
            args.out.mkdir(parents=True, exist_ok=True)
            write_json(args.out / "timeout.json", {"all_pass": False, "reason": "120-second cap"})
            raise
        return
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit code before evaluation")
    original_path = Path("experiments/locomotion_curriculum/results_reference_subset/trials.json")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    checks = known_checks()
    if not all(r["pass"] for r in checks):
        raise RuntimeError("known signal checks failed")
    rows = []
    for prior in original:
        directory = args.raw_dir / prior["condition"]
        recordings = [p for p in directory.glob("*.json") if p.name not in ("reference.json", "preset.json")]
        if len(recordings) != 1:
            raise ValueError("missing unique original recording")
        reference_path = directory / "reference.json"
        if (hashlib.sha256(recordings[0].read_bytes()).hexdigest() != prior["recording_sha256"]
                or hashlib.sha256(reference_path.read_bytes()).hexdigest() != prior["reference_sha256"]):
            raise ValueError("original artifact hash mismatch")
        recording = json.loads(recordings[0].read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        frames = np.array(recording["Frames"])
        offset = recording["Frame_offset"][0]["joints_pos"]
        result = compare(reference, recording["FrameTimes"], frames[:, offset:offset + 16])
        if not np.allclose(result["rmse_per_joint"]["endpoint"], prior["rmse_per_component"]["joint_velocity"],
                           atol=1e-8, rtol=0):
            raise ValueError("original metric reproduction failed")
        rows.append({"condition": prior["condition"], **result, "original_reference_pass": prior["all_pass"]})
    moving = [r for r in rows if r["condition"] != "stand"]
    report = {"aggregate": "timing_accounted" if len(moving) == 6 and all(r["timing_accounted"] for r in moving)
              else "timing_alone_insufficient", "known_signal_checks": checks, "trials": rows,
              "original_subset": "0/7 pass; unchanged", "usable_for_training": False}
    args.out.mkdir(parents=True)
    write_json(args.out / "report.json", report)
    write_json(args.out / "protocol.json", {"preregistration_commit": "b24aed2",
               "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "original_report_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
               "input_hashes_verified": True, "wall_cap_s": 120, "cloud_work": False})
    print(report["aggregate"], [(r["condition"], r["max_component_rmse"]) for r in rows], flush=True)


if __name__ == "__main__":
    main()
