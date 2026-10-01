"""Export a NERVA MuJoCo scenario as a replay file for Isaac Sim rendering.

The behaviour, affect and memory loop runs in MuJoCo (where NERVA's walking policies work); Isaac Sim
re-renders the same motion with realistic lighting and human models (experiments/isaac/README.md).
The replay is KINEMATIC: the world pose of every robot body per frame, so Isaac needs no physics and no
articulation to reproduce the motion exactly.

Output .npz:
  t (F,), body_names (B,), body_pos (F, B, 3), body_quat (F, B, 4, w x y z)
  person_pos / person_quat, person_b_pos / person_b_quat, ball_pos (F, ...), modes (F,)

A sidecar <out>.json holds what the composed video shows beside the render: the per-frame affect,
behaviour and memory values (the reactive scenario's rows at 25 fps), the perception events with their
appraisals and emotions, and the boxes the colour/depth vision found in the robot's camera image.
The robot-eye camera's world pose per frame (eye_pos, eye_xmat) lets Isaac render the robot's point of view.

Usage: python experiments/isaac/export_replay.py --policy B2.onnx --out replay.npz [--scenario memory]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reactive"))
import scenario  # noqa: E402

FPS = 25
NOT_ROBOT = {"world", "floor", "person", "person_b", "ball"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenario", choices=("default", "memory", "together"), default="memory")
    ap.add_argument("--neutral-style", action="store_true", help="policy trained on the neutral style only (B1/B2)")
    args = ap.parse_args()
    agents = {"default": None, "memory": scenario.memory_scenario,
              "together": scenario.together_scenario}[args.scenario]
    duration = {"default": 100.0, "memory": 135.0, "together": 125.0}[args.scenario]
    every = int(round(1 / (FPS * scenario.CTRL_DT)))
    sim, rows, frames, fired = scenario.run(args.policy, duration=duration, record_every=every,
                                        agents=agents() if agents else None, perception_mode="vision",
                                        use_memory=args.scenario != "default", backlash_scene=True,
                                        neutral_style=args.neutral_style)
    model = sim.model
    data = mujoco.MjData(model)
    bodies = [i for i in range(model.nbody) if model.body(i).name not in NOT_ROBOT]
    names = [model.body(i).name for i in bodies]
    mocap = {n: model.body_mocapid[model.body(n).id] for n in ("person", "person_b", "ball")}
    eye = model.camera("robot_eye").id
    body_pos, body_quat, eye_pos, eye_xmat, boxes = [], [], [], [], []
    for qpos, mpos, mquat, _tracks, frame_boxes in frames:
        data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = qpos, mpos, mquat
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        body_pos.append(data.xpos[bodies].copy())
        body_quat.append(data.xquat[bodies].copy())
        eye_pos.append(data.cam_xpos[eye].copy())
        eye_xmat.append(data.cam_xmat[eye].reshape(3, 3).copy())
        boxes.append([[str(b[0]), *(int(v) for v in b[1:])] for b in (frame_boxes or [])])
    mpos = np.array([f[1] for f in frames])
    mquat = np.array([f[2] for f in frames])
    t = np.arange(len(frames)) / FPS
    modes = np.array([rows[min(int(round(ti / scenario.CTRL_DT)), len(rows) - 1)]["mode"] for ti in t])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, t=t, body_names=np.array(names), body_pos=np.array(body_pos, np.float32),
                        body_quat=np.array(body_quat, np.float32),
                        person_pos=mpos[:, mocap["person"]], person_quat=mquat[:, mocap["person"]],
                        person_b_pos=mpos[:, mocap["person_b"]], person_b_quat=mquat[:, mocap["person_b"]],
                        ball_pos=mpos[:, mocap["ball"]], modes=modes,
                        eye_pos=np.array(eye_pos, np.float32), eye_xmat=np.array(eye_xmat, np.float32),
                        eye_fovy=float(model.cam_fovy[eye]))
    keep = ("t", "valence", "arousal", "dominance", "joy", "hope", "fear", "distress", "surprise", "interest",
            "mode", "reason", "identity", "mem_threat", "mem_warmth", "dist_person", "dist_person_b",
            "cmd_vx", "cmd_yaw", "e_tempo", "e_torso_pitch", "head_pitch", "head_yaw", "tilt_deg",
            "tend_approach", "tend_explore", "tend_avoid", "tend_orient", "tend_freeze", "tend_withdraw")
    side = {
        "fps": FPS, "vision_size": [scenario.EYE_H, scenario.EYE_W],
        "rows": [{k: (None if isinstance(r.get(k), float) and r[k] != r[k] else r.get(k)) for k in keep}
                 for r in (rows[min(int(round(ti / scenario.CTRL_DT)), len(rows) - 1)] for ti in t)],
        "events": [{"t": round(te, 2), "kind": kind, "relevance": a.relevance, "desirability": a.desirability,
                    "expectedness": a.expectedness, "controllability": a.controllability,
                    "emotions": [[lbl, round(i, 3)] for lbl, i in emos]} for te, kind, a, emos in fired],
        "boxes": boxes,
        "memory": {k: {"threat": v.threat, "warmth": v.warmth, "trust": v.trust}
                   for k, v in getattr(sim, "memory", None).records.items()} if getattr(sim, "memory", None) else {},
    }
    args.out.with_suffix(".json").write_text(json.dumps(side), encoding="utf-8")
    print(f"wrote {args.out}: {len(t)} frames, {len(names)} bodies: {names}")


if __name__ == "__main__":
    main()
