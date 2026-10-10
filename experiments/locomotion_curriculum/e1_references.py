"""E1 Phase A: styled references and their gate (docs/expressive_posture_experiment.md §3, preregistration 630610f).

Records the seven neutral commands for each style e = (e_pitch, e_height) with the recorder, repair settings and
geometric-turn correction that produced the admitted neutral set (`reference_subset.py` repair/pivot modes; only
`walk_trunk_pitch` and `walk_com_height` change), validates each recording with the unchanged admission function,
and evaluates the Phase A gate:
1. every grid recording passes all eight admission criteria;
2. the regenerated (0, 0) set matches the admitted neutral references (leg-joint position RMSE ≤ 0.05 rad);
3. bilinear interpolation of the four surrounding grid references matches each (±0.5, ±0.5) checkpoint
   (leg-joint position RMSE ≤ 0.05 rad, contact agreement ≥ 90%);
4. range rule: a failing axis may be halved once (--pitch-scale / --height-scale 0.5), else stop.

Usage: python -m experiments.locomotion_curriculum.e1_references --generator-python LINUX_PYTHON
       [--pitch-scale 1|0.5] [--height-scale 1|0.5]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

from experiments.locomotion_curriculum.reference_subset import CONDITIONS, known_pose_checks, validate, wsl_path
from nerva.analysis import gait_metrics as gm
from nerva.sim.open_duck import OPEN_DUCK_ROOT
from nerva.training.motor_artifacts import write_json
from nerva.training.neutral_reference import verified_references
from nerva.training.reference_kinematics import sample_reference

PREREGISTRATION = "630610f"
GRID = [(p, h) for p in (-1, 0, 1) for h in (-1, 0, 1) if (p, h) != (0, 0)]
CHECKPOINTS = [(p, h) for p in (-.5, .5) for h in (-.5, .5)]
REPRO = [(0, 0)]
LEG = list(range(5)) + list(range(11, 16))
WORKERS, PER_RECORDING_CAP_S, WALL_CAP_S = 4, 180, 3600
SAMPLES = np.arange(1000) / 1000 * .54


def style_tag(e):
    return f"p{e[0]:+.1f}_h{e[1]:+.1f}"


def style_preset(base, e, pitch_scale, height_scale):
    return {**base, "walk_trunk_pitch": -4 + 6 * e[0] * pitch_scale,
            "walk_com_height": round(.215 + .012 * e[1] * height_scale, 6)}


def root_features(recording):
    frames = np.asarray(recording["Frames"], dtype=float)
    t = np.asarray(recording["FrameTimes"], dtype=float)
    scored = (t >= 2) & (t < 6)
    quat = frames[:, 3:7][:, [3, 0, 1, 2]]  # xyzw → wxyz
    pitch = gm.quat_to_rpy(quat)[:, 1]
    return {"base_pitch_rad": float(pitch[scored].mean()), "base_height_m": float(frames[scored, 2].mean())}


def leg_rmse(a, b):
    return float(np.sqrt(np.mean((sample_reference(a, SAMPLES)["joint_position"][:, LEG]
                                  - sample_reference(b, SAMPLES)["joint_position"][:, LEG]) ** 2)))


def interpolate(refs, weights):
    out = dict(refs[0])
    out["coefficients"] = {k: sum(w * np.asarray(r["coefficients"][k]) for r, w in zip(refs, weights)).tolist()
                           for k in refs[0]["coefficients"]}
    return out


def gate(rows, references, neutral):
    by = {(tuple(r["e"]), r["condition"]): r for r in rows}
    grid_ok = all(by[(e, c)]["all_pass"] for e in GRID for c in CONDITIONS)
    grid_fail = [f"{style_tag(e)} {c}" for e in GRID for c in CONDITIONS if not by[(e, c)]["all_pass"]]
    repro = {}
    for c in CONDITIONS:
        ref = references.get(((0, 0), c))
        repro[c] = leg_rmse(ref, neutral[c]) if ref is not None else None
    repro_ok = all(v is not None and v <= .05 for v in repro.values())
    interp = {}
    for e in CHECKPOINTS:
        sp, sh = int(np.sign(e[0])), int(np.sign(e[1]))
        corners = [(0, 0), (sp, 0), (0, sh), (sp, sh)]
        for c in CONDITIONS:
            refs = [neutral[c] if k == (0, 0) else references.get((k, c)) for k in corners]
            target = references.get((e, c))
            if any(r is None for r in refs) or target is None:
                interp[f"{style_tag(e)} {c}"] = None
                continue
            mix = interpolate(refs, [.25] * 4)
            a, b = sample_reference(mix, SAMPLES), sample_reference(target, SAMPLES)
            interp[f"{style_tag(e)} {c}"] = {
                "leg_rmse_rad": leg_rmse(mix, target),
                "contact_agreement": float(np.mean((a["contacts"] > .5) == (b["contacts"] > .5)))}
    interp_ok = all(v is not None and v["leg_rmse_rad"] <= .05 and v["contact_agreement"] >= .90
                    for v in interp.values())
    return {"grid_all_pass": grid_ok, "grid_failures": grid_fail, "repro_leg_rmse_rad": repro,
            "repro_ok": repro_ok, "interpolation": interp, "interpolation_ok": interp_ok,
            "phase_a_passes": bool(grid_ok and repro_ok and interp_ok)}


def feature_targets(rows):
    by = {(tuple(r["e"]), r["condition"]): r.get("features") for r in rows}
    out = {}
    for c in CONDITIONS:
        try:
            out[c] = {"delta_pitch_rad": by[((1, 0), c)]["base_pitch_rad"] - by[((-1, 0), c)]["base_pitch_rad"],
                      "delta_height_m": by[((0, 1), c)]["base_height_m"] - by[((0, -1), c)]["base_height_m"]}
        except (KeyError, TypeError):
            out[c] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator-python", required=True, help="existing Linux Python interpreter with Placo")
    ap.add_argument("--pitch-scale", type=float, choices=(1., .5), default=1.)
    ap.add_argument("--height-scale", type=float, choices=(1., .5), default=1.)
    args = ap.parse_args()
    suffix = "" if (args.pitch_scale, args.height_scale) == (1., 1.) else f"_p{args.pitch_scale}_h{args.height_scale}"
    out = Path(f"experiments/locomotion_curriculum/results_e1_references{suffix}")
    raw_dir = Path(f"experiments/cloud_runs/e1-styled-references{suffix}")
    if out.exists() or raw_dir.exists():
        raise FileExistsError("never overwrite prior outputs")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit code before generating")
    root = OPEN_DUCK_ROOT / "Open_Duck_reference_motion_generator"
    source = root / "open_duck_reference_motion_generator/gait_generator.py"
    preset_path = source.parent / "robots/open_duck_mini_v2/placo_presets/medium.json"
    base = json.loads(preset_path.read_text(encoding="utf-8"))
    base.update(walk_com_height=.215, walk_foot_height=.020, walk_foot_rise_ratio=.30)  # the repair preset
    urdf = source.parent / "robots/open_duck_mini_v2/open_duck_mini_v2.urdf"
    limits = {j.attrib["name"]: (float(j.find("limit").attrib["lower"]), float(j.find("limit").attrib["upper"]))
              for j in ET.parse(urdf).getroot().findall("joint") if j.find("limit") is not None}
    for name in ("left_knee", "right_knee"):
        limits[name] = (.01, np.pi / 2)
    styles = GRID + CHECKPOINTS + REPRO
    out.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    write_json(out / "protocol.json", {
        "preregistration_commit": PREREGISTRATION,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preset_sha256": hashlib.sha256(preset_path.read_bytes()).hexdigest(),
        "base_preset": base, "pitch_scale": args.pitch_scale, "height_scale": args.height_scale,
        "styles": {"grid": GRID, "interpolation_checkpoints": CHECKPOINTS, "reproducibility": REPRO},
        "conditions": CONDITIONS, "workers": WORKERS, "per_recording_cap_s": PER_RECORDING_CAP_S,
        "wall_cap_s": WALL_CAP_S, "cloud_work": False})
    if not all(c["pass"] for c in known_pose_checks()):
        raise RuntimeError("known-rotation validation failed")
    start, references = time.monotonic(), {}

    def record(item):
        e, (name, command) = item
        directory = raw_dir / style_tag(e) / name
        directory.mkdir(parents=True)
        preset = style_preset(base, e, args.pitch_scale, args.height_scale)
        effective = {**preset, **dict(zip(("dx", "dy", "dtheta"), np.array(command) * .54 / 2))}
        write_json(directory / "preset.json", effective)
        row = {"e": list(e), "condition": name, "command": list(command),
               "walk_trunk_pitch_deg": preset["walk_trunk_pitch"], "walk_com_height_m": preset["walk_com_height"]}
        remaining = WALL_CAP_S - (time.monotonic() - start)
        if remaining <= 0:
            return {**row, "all_pass": False, "error_type": "WallCap"}
        cap = min(PER_RECORDING_CAP_S, remaining)
        line = ["wsl", "--exec", "timeout", "--signal=TERM", "--kill-after=5s", f"{cap:.2f}s", args.generator_python,
                wsl_path(Path(__file__).with_name("record_reference.py")), "--source", wsl_path(source),
                "--duck", "open_duck_mini_v2", "--preset", wsl_path(directory / "preset.json"), "--name", name,
                "--output_dir", wsl_path(directory), "--length", "8", "--repair"]
        if name == "stand":
            line.append("--stand")
        if name.startswith("turn"):
            line.append("--pivot")
        try:
            with (directory / "generate.log").open("w", encoding="utf-8") as stream:
                subprocess.run(line, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=cap + 10)
            files = [p for p in directory.glob("*.json") if p.name not in ("preset.json", "reference.json")]
            if len(files) != 1:
                raise ValueError("missing recording")
            recording = json.loads(files[0].read_text(encoding="utf-8"))
            result, ref = validate(recording, command, name == "stand", joint_limits=limits)
            write_json(directory / "reference.json", ref)
            references[(e, name)] = ref
            return {**row, "pivot": recording.get("NeutralPivot"), "features": root_features(recording),
                    "recording_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
                    "reference_sha256": hashlib.sha256((directory / "reference.json").read_bytes()).hexdigest(),
                    **result}
        except (ValueError, KeyError, subprocess.SubprocessError) as error:
            return {**row, "all_pass": False, "error_type": type(error).__name__}  # messages may hold private paths

    rows = []
    items = [(e, c) for e in styles for c in CONDITIONS.items()]
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for row in executor.map(record, items):
            rows.append(row)
            write_json(out / "trials.json", rows)
            print(style_tag(row["e"]), row["condition"], row["all_pass"], flush=True)
    records, _ = verified_references(Path.cwd())
    names = {tuple(v): k for k, v in CONDITIONS.items()}
    neutral = {names[tuple(r["command"])]: r["reference"] for r in records}
    report = gate(rows, {(tuple(k[0]), k[1]): v for k, v in references.items()}, neutral)
    report.update(reference_feature_changes=feature_targets(rows), recordings=len(rows),
                  wall_seconds=round(time.monotonic() - start, 1))
    write_json(out / "summary.json", report)
    print(json.dumps({k: report[k] for k in ("grid_all_pass", "grid_failures", "repro_ok", "interpolation_ok",
                                             "phase_a_passes")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
