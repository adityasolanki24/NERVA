"""Export a NERVA MuJoCo scenario as a replay file for Isaac Sim rendering.

The behaviour, affect and memory loop runs in MuJoCo (where NERVA's walking policies work); Isaac Sim
re-renders the same motion with realistic lighting and human models (experiments/isaac/README.md).
The replay is KINEMATIC: the world pose of every robot body per frame, so Isaac needs no physics and no
articulation to reproduce the motion exactly.

Output .npz:
  t (F,), body_names (B,), body_pos (F, B, 3), body_quat (F, B, 4, w x y z)
  person_pos / person_quat, person_b_pos / person_b_quat, ball_pos (F, ...), modes (F,)

Usage: python experiments/isaac/export_replay.py --policy S1.onnx --out replay.npz [--scenario memory]
"""

from __future__ import annotations

import argparse
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
    sim, rows, frames, _ = scenario.run(args.policy, duration=duration, record_every=every,
                                        agents=agents() if agents else None, perception_mode="vision",
                                        use_memory=args.scenario != "default", backlash_scene=True,
                                        neutral_style=args.neutral_style)
    model = sim.model
    data = mujoco.MjData(model)
    bodies = [i for i in range(model.nbody) if model.body(i).name not in NOT_ROBOT]
    names = [model.body(i).name for i in bodies]
    mocap = {n: model.body_mocapid[model.body(n).id] for n in ("person", "person_b", "ball")}
    body_pos, body_quat = [], []
    for qpos, mpos, mquat, *_ in frames:
        data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = qpos, mpos, mquat
        mujoco.mj_kinematics(model, data)
        body_pos.append(data.xpos[bodies].copy())
        body_quat.append(data.xquat[bodies].copy())
    mpos = np.array([f[1] for f in frames])
    mquat = np.array([f[2] for f in frames])
    t = np.arange(len(frames)) / FPS
    modes = np.array([rows[min(int(round(ti / scenario.CTRL_DT)), len(rows) - 1)]["mode"] for ti in t])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, t=t, body_names=np.array(names), body_pos=np.array(body_pos, np.float32),
                        body_quat=np.array(body_quat, np.float32),
                        person_pos=mpos[:, mocap["person"]], person_quat=mquat[:, mocap["person"]],
                        person_b_pos=mpos[:, mocap["person_b"]], person_b_quat=mquat[:, mocap["person_b"]],
                        ball_pos=mpos[:, mocap["ball"]], modes=modes)
    print(f"wrote {args.out}: {len(t)} frames, {len(names)} bodies: {names}")


if __name__ == "__main__":
    main()
