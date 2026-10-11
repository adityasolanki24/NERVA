"""E1′ evaluation and decision (docs/expressive_posture_e1prime.md §5–6, preregistration 0c221cf).

Committed before any E1′ result. Native MuJoCo, backlash scene, raw accelerometer, training observation noise,
initial joint noise ±0.02 rad, styled neutral contract (e appended), 20 s trials scored over 5–20 s.

- sweeps: e_pitch ∈ {−1, −0.5, 0, 0.5, 1} at e_crouch 0; e_crouch ∈ {0, 0.25, 0.5, 0.75, 1} at e_pitch 0; seven
  commands; seeds 0–2; no latency;
- style space: (−1, 0), (1, 0), (−1, 1), (0, 1) trained; (0.75, 0.75), (1, 1) held out; seven commands; seeds 0–2;
  without and with latency;
- ramps: forward command, 30 s, e ramped linearly across an axis over 5–25 s; seeds 0–2; 1 s windows.
The neutral retention gate is run separately: `neutral_gate.py --candidate e1_prime`.

Usage: python -m experiments.locomotion_curriculum.e1prime_eval --run RUN --out DIR [--gate DIR] [--render]
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
from nerva.safety import SafetySupervisor
from nerva.sim.open_duck import OpenDuckSim, SCENE_BACKLASH, to_arrays
from nerva.training.motor_artifacts import write_json
from nerva.training.neutral_reference import COMMANDS as NEUTRAL_COMMANDS
from nerva.world.self_state import estimate_self_state

PREREGISTRATION = "0c221cf"
DT = .02
SEEDS = (0, 1, 2)
NAMES = ("rest", "forward", "backward", "left", "right", "turn_left", "turn_right")
COMMANDS = dict(zip(NAMES, (tuple(c) for c in NEUTRAL_COMMANDS)))
PITCH_SWEEP = [(p, 0.) for p in (-1., -.5, 0., .5, 1.)]
CROUCH_SWEEP = [(0., c) for c in (0., .25, .5, .75, 1.)]
TRAINED_POINTS = [(-1., 0.), (1., 0.), (-1., 1.), (0., 1.)]
HELD_OUT_POINTS = [(.75, .75), (1., 1.)]
REFERENCE_SUMMARY = "experiments/locomotion_curriculum/results_e1prime_references/summary.json"
REFERENCE_NAMES = dict(zip(NAMES, ("stand", "forward", "backward", "left", "right", "turn_left", "turn_right")))


def tag(e):
    return f"p{e[0]:+.2f}_c{e[1]:+.2f}"


def features(arrays, start=5., end=20.):
    t = arrays["t"]
    mask = (t >= start - 1e-9) & (t < end - 1e-9)
    pitch = gm.quat_to_rpy(arrays["base_quat"])[:, 1]
    return {"base_pitch_rad": float(pitch[mask].mean()), "base_height_m": float(arrays["base_pos"][mask, 2].mean())}


def simulate(policy, digest, command, seed, latency, style_at, seconds):
    """style_at(t) → e for the control step starting at t."""
    with contextlib.redirect_stdout(io.StringIO()):
        sim = OpenDuckSim(policy_path=policy, scene=SCENE_BACKLASH, raw_accel=True, obs_noise=True,
                          init_joint_noise=.02, seed=seed, action_delay=latency)
    sim.set_neutral_motor_contract(deployment_metadata(digest, styled=True))
    supervisor = SafetySupervisor()
    behaviour = BehaviourCommand(*command)
    log, qpos, styles, completed, max_tilt = [], [], [], True, 0.
    for k in range(int(round(seconds / DT))):
        e = style_at(k * DT)
        sim.set_neutral_style(*e)
        styles.append(e)
        supervisor.filter(behaviour, (0., 0., 0., 0.), estimate_self_state(k * DT, sim.data.qpos, sim.data.qvel))
        sim.set_behaviour(behaviour)
        log.extend(sim.run(DT))
        qpos.append(sim.data.qpos.copy())
        if not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
            completed = False
            break
        max_tilt = max(max_tilt, float(gm.tilt_deg(sim.data.qpos[3:7][None])[0]))
        if max_tilt > 45:
            completed = False
            break
    arrays = to_arrays(log)
    arrays["t"] = (np.arange(len(log)) + 1) * DT
    arrays["qpos"], arrays["style"] = np.asarray(qpos), np.asarray(styles)
    completed = completed and len(log) == int(round(seconds / DT))
    return arrays, {"completed": completed, "fell": not completed, "max_tilt_deg": max_tilt,
                    "shadow_interventions": supervisor.interventions}


def steady(policy, digest, e, name, seed, latency, raw):
    arrays, row = simulate(policy, digest, COMMANDS[name], seed, latency, lambda t: e, 20.)
    np.savez_compressed(raw / f"{'latency' if latency else 'no_latency'}_{tag(e)}_{name}_{seed}.npz", **arrays)
    row.update(e=list(e), command=name, seed=seed, latency=latency,
               metrics=motor_metrics(arrays, COMMANDS[name], row["completed"]))
    if row["completed"]:
        row["features"] = features(arrays)
    return row


def ramp(policy, digest, axis, seed, raw):
    start, end = ((-1., 0.), (1., 0.)) if axis == "pitch" else ((0., 0.), (0., 1.))

    def style_at(t):
        u = min(1., max(0., (t - 5.) / 20.))
        return (start[0] + u * (end[0] - start[0]), start[1] + u * (end[1] - start[1]))
    arrays, row = simulate(policy, digest, COMMANDS["forward"], seed, False, style_at, 30.)
    np.savez_compressed(raw / f"ramp_{axis}_{seed}.npz", **arrays)
    windows = []
    if row["completed"]:
        pitch = gm.quat_to_rpy(arrays["base_quat"])[:, 1]
        for w in range(5, 25):
            m = (arrays["t"] > w + 1e-9) & (arrays["t"] <= w + 1 + 1e-9)
            windows.append({"t": [w, w + 1], "e": arrays["style"][m].mean(0).tolist(),
                            "base_pitch_rad": float(pitch[m].mean()), "base_height_m": float(arrays["base_pos"][m, 2].mean())})
    return {**row, "axis": axis, "seed": seed, "windows": windows}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def decide(rows, ramps, references, gate):
    """Criteria 2–8 of §6 (criterion 1, Phase A′, is recorded in results_e1prime_references/)."""
    def sel(e, name, latency=False):
        return [r for r in rows if tuple(r["e"]) == tuple(e) and r["command"] == name and r["latency"] == latency]

    def feat(e, name, key):
        return [r["features"][key] if r.get("features") else np.nan for r in sel(e, name)]

    keys = {"pitch": "base_pitch_rad", "crouch": "base_height_m"}
    sweeps = {"pitch": PITCH_SWEEP, "crouch": CROUCH_SWEEP}
    ref = {"pitch": {n: references[REFERENCE_NAMES[n]]["delta_pitch_rad"] for n in NAMES},
           "crouch": {n: references[REFERENCE_NAMES[n]]["delta_height_m"] for n in NAMES}}
    direction, monotonic, detail = True, True, {}
    for axis, points in sweeps.items():
        for name in NAMES:
            per_seed = np.array([feat(e, name, keys[axis]) for e in points])  # [5 points, 3 seeds]
            change = per_seed[-1] - per_seed[0]
            target = ref[axis][name]
            ok_dir = bool(np.all(np.sign(change) == np.sign(target)) and np.all(np.abs(change) >= .5 * abs(target)))
            means = np.nanmean(per_seed, axis=1) * np.sign(target)
            ok_mono = bool(np.all(np.diff(means) > 0))
            direction &= ok_dir
            monotonic &= ok_mono
            detail[f"{axis} {name}"] = {"seed_means": np.nanmean(per_seed, axis=1).tolist(),
                                        "end_to_end_per_seed": change.tolist(), "reference_change": target,
                                        "ratio_to_reference": (change / target).tolist(), "direction_range_ok": ok_dir,
                                        "monotonic_ok": ok_mono}
    crosstalk, cross_detail = True, {}
    for name in NAMES:
        own = {axis: np.nanmean(np.array(feat(sweeps[axis][-1], name, keys[axis]))
                                - np.array(feat(sweeps[axis][0], name, keys[axis]))) for axis in sweeps}
        for axis, other in (("pitch", "crouch"), ("crouch", "pitch")):
            leak = np.nanmean(np.array(feat(sweeps[axis][-1], name, keys[other]))
                              - np.array(feat(sweeps[axis][0], name, keys[other])))
            ratio = float(abs(leak) / abs(own[other])) if own[other] else float("inf")
            cross_detail[f"{axis}→{other} {name}"] = {"leak": float(leak), "own": float(own[other]), "ratio": ratio}
            crosstalk &= ratio <= .25
    space = TRAINED_POINTS + HELD_OUT_POINTS
    task_ok = all(sum(r["metrics"]["all_pass"] for r in sel(e, n)) == 3 for e in space for n in NAMES)
    no_falls = not any(r["fell"] for r in rows if tuple(r["e"]) in {tuple(p) for p in space})
    task_fail = [f"{tag(e)} {n}" for e in space for n in NAMES if sum(r["metrics"]["all_pass"] for r in sel(e, n)) < 3]
    interference, speed_detail = True, {}
    for e in {tuple(p) for p in PITCH_SWEEP + CROUCH_SWEEP + space}:
        for name in NAMES[1:]:
            axis = int(np.argmax(np.abs(COMMANDS[name])))
            sign = np.sign(COMMANDS[name][axis])

            def speed(point, name=name, axis=axis, sign=sign):
                v = [r["metrics"].get("mean_velocity") for r in sel(point, name)]
                return float(np.mean([x[axis] * sign for x in v])) if all(x is not None for x in v) else np.nan
            ratio = speed(e) / speed((0., 0.))
            speed_detail[f"{tag(e)} {name}"] = ratio
            interference &= bool(.75 <= ratio <= 1.33)
    continuity, ramp_detail = True, {}
    for r in ramps:
        key = keys[r["axis"]]
        target = ref[r["axis"]]["forward"]
        if not r["windows"]:
            continuity = False
            ramp_detail[f"{r['axis']} {r['seed']}"] = {"completed": False}
            continue
        x = np.array([w["e"][0 if r["axis"] == "pitch" else 1] for w in r["windows"]])
        y = np.array([w[key] for w in r["windows"]]) * np.sign(target)
        rho = spearman(x, y)
        total = y[-1] - y[0]
        jump = float(np.max(np.abs(np.diff(y))) / abs(total)) if total else float("inf")
        ok = rho >= .9 and jump <= .25
        continuity &= ok
        ramp_detail[f"{r['axis']} {r['seed']}"] = {"spearman": rho, "max_window_jump_fraction": jump, "ok": bool(ok)}
    criteria = {"2_neutral_retention": bool(gate.get("gate_passes", False)) if gate else None,
                "3_direction_and_range": direction, "4_monotonicity": monotonic, "5_crosstalk": crosstalk,
                "6_task_across_style_space": bool(task_ok and no_falls), "7_task_interference": interference,
                "8_continuity": continuity}
    hypothesis = all(criteria[k] for k in ("3_direction_and_range", "4_monotonicity", "5_crosstalk",
                                           "7_task_interference", "8_continuity"))
    return {"criteria": criteria, "hypothesis_supported": hypothesis,
            "e1_prime_passes": bool(hypothesis and criteria["2_neutral_retention"] and criteria["6_task_across_style_space"]),
            "sweeps": detail, "crosstalk": cross_detail, "task_failures": task_fail,
            "falls_in_style_space": int(sum(r["fell"] for r in rows if tuple(r["e"]) in {tuple(p) for p in space})),
            "speed_ratios": speed_detail, "ramps": ramp_detail,
            "shadow_interventions": int(sum(r["shadow_interventions"] for r in rows) + sum(r["shadow_interventions"] for r in ramps))}


def evaluate(root, run, out, gate_dir):
    summary = json.loads((run / "report" / "training_summary.json").read_text(encoding="utf-8"))
    policy = run / "candidate.onnx"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    if digest != summary["candidate_onnx_sha256"]:
        raise ValueError("candidate hash differs from the run's training summary")
    raw = run / "e1prime_eval"
    raw.mkdir(exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "protocol.json", {
        "preregistration_commit": PREREGISTRATION,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "policy_sha256": digest, "final_checkpoint": summary["final_checkpoint"], "seeds": list(SEEDS),
        "sweeps": {"pitch": PITCH_SWEEP, "crouch": CROUCH_SWEEP}, "trained_points": TRAINED_POINTS,
        "held_out_points": HELD_OUT_POINTS, "scene": "flat_terrain_backlash", "observation_noise": True,
        "joint_initial_noise": .02, "safety_override": False})
    start, rows = time.monotonic(), []
    points = list(dict.fromkeys(PITCH_SWEEP + CROUCH_SWEEP + TRAINED_POINTS + HELD_OUT_POINTS))
    for e in points:
        latencies = (False, True) if e in TRAINED_POINTS + HELD_OUT_POINTS else (False,)
        for latency in latencies:
            for name in NAMES:
                for seed in SEEDS:
                    rows.append(steady(policy, digest, e, name, seed, latency, raw))
        write_json(out / "trials.json", rows)
        print("style", tag(e), "done", flush=True)
    ramps = [ramp(policy, digest, axis, seed, raw) for axis in ("pitch", "crouch") for seed in SEEDS]
    write_json(out / "ramps.json", ramps)
    references = json.loads((root / REFERENCE_SUMMARY).read_text(encoding="utf-8"))["reference_feature_changes"]
    gate = json.loads((gate_dir / "summary.json").read_text(encoding="utf-8")) if gate_dir else None
    decision = decide(rows, ramps, references, gate)
    decision["wall_seconds"] = round(time.monotonic() - start, 1)
    write_json(out / "summary.json", decision)
    print(json.dumps({k: decision[k] for k in ("criteria", "hypothesis_supported", "e1_prime_passes")}, indent=1))


def render(run, out, neutral_policy):
    """Clips with one camera, 20 s (ramps 30 s), seed 0, forward command unless named: neutral candidate | E1′ e=0 |
    extremes; held-out; ramps. Failures are rendered like successes."""
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    model = mujoco.MjModel.from_xml_path(str(SCENE_BACKLASH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 360, 400)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = .9, 90., -10.  # side view: pitch and height visible
    raw = run / "e1prime_eval"
    neutral_raw = run / "e1prime_neutral_reference_clips"
    neutral_raw.mkdir(exist_ok=True)
    videos = []

    def neutral_trace(name):
        path = neutral_raw / f"{name}_0.npz"
        if not path.exists():
            with contextlib.redirect_stdout(io.StringIO()):
                sim = OpenDuckSim(policy_path=neutral_policy, scene=SCENE_BACKLASH, raw_accel=True, obs_noise=True,
                                  init_joint_noise=.02, seed=0)
            sim.set_neutral_motor_contract(deployment_metadata(hashlib.sha256(neutral_policy.read_bytes()).hexdigest()))
            sim.set_behaviour(BehaviourCommand(*COMMANDS[name]))
            q = []
            for _ in range(1000):
                sim.run(DT)
                q.append(sim.data.qpos.copy())
            np.savez_compressed(path, qpos=np.asarray(q))
        return np.load(path)["qpos"]

    def clip(file, columns, frames):
        path = run / file
        if path.exists():
            raise FileExistsError("never overwrite videos")
        width = 400 * len(columns)
        with imageio.get_writer(path, fps=25, codec="libx264", quality=7, macro_block_size=1) as writer:
            for tick in range(0, frames, 2):
                frame = Image.new("RGB", (width, 400), "white")
                draw = ImageDraw.Draw(frame)
                for col, (label, qpos, styles) in enumerate(columns):
                    i = min(tick, len(qpos) - 1)
                    data.qpos[:] = qpos[i]
                    mujoco.mj_forward(model, data)
                    cam.lookat[:] = data.qpos[:3] + [0, 0, .05]
                    renderer.update_scene(data, camera=cam)
                    frame.paste(Image.fromarray(renderer.render()), (400 * col, 0))
                    pitch = np.degrees(gm.quat_to_rpy(qpos[i][3:7][None])[0, 1])
                    e = "" if styles is None else f" e=({styles[min(i, len(styles) - 1)][0]:+.2f},{styles[min(i, len(styles) - 1)][1]:+.2f})"
                    draw.text((400 * col + 6, 362), label + e, fill="black")
                    draw.text((400 * col + 6, 380), f"t={tick * DT:5.2f}s pitch {pitch:+.1f}deg z {qpos[i][2]:.3f}m",
                              fill="black")
                writer.append_data(np.asarray(frame))
        videos.append({"file": file, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "columns": [c[0] for c in columns]})

    def trace(e, name, latency=False):
        a = np.load(raw / f"{'latency' if latency else 'no_latency'}_{tag(e)}_{name}_0.npz")
        return a["qpos"], a["style"]

    try:
        for name in ("forward", "rest", "turn_left"):
            cols = [("neutral candidate", neutral_trace(name), None), ("E1' e=0", *trace((0., 0.), name))]
            cols += [(f"E1' {lbl}", *trace(e, name)) for lbl, e in (("pitch-", (-1., 0.)), ("pitch+", (1., 0.)),
                                                                     ("crouch", (0., 1.)))]
            clip(f"e1prime_{name}_extremes.mp4", cols, 1000)
        clip("e1prime_held_out.mp4", [("E1' e=0", *trace((0., 0.), "forward")),
                                       ("held-out (0.75,0.75)", *trace((.75, .75), "forward")),
                                       ("held-out (1,1)", *trace((1., 1.), "forward"))], 1000)
        for axis in ("pitch", "crouch"):
            a = np.load(raw / f"ramp_{axis}_0.npz")
            clip(f"e1prime_ramp_{axis}.mp4", [(f"ramp {axis}", a["qpos"], a["style"])], 1500)
    finally:
        renderer.close()
    write_json(out / "videos.json", videos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gate", type=Path, help="neutral_gate.py --candidate e1_prime output directory")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    root = Path.cwd().resolve()
    if a.render:
        render(a.run.resolve(), a.out, root / "experiments/cloud_runs/turn_translation-20261011-000711/candidate.onnx")
        return
    if (a.out / "summary.json").exists():
        raise FileExistsError("never overwrite an evaluation")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit the evaluator before evaluating")
    evaluate(root, a.run.resolve(), a.out, a.gate)


if __name__ == "__main__":
    main()
