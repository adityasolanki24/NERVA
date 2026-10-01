"""Export a NERVA MuJoCo scenario as a replay file for Isaac Sim rendering.

The behaviour, affect and memory loop runs in MuJoCo (where NERVA's walking policies work); Isaac Sim
re-renders the same motion with realistic lighting and human models (experiments/isaac/README.md).
Output .npz: t, base_pos, base_quat (w x y z), joints (by name), joint_names, people positions/yaws,
ball position, and the behaviour mode per frame (for captions).

Usage: python experiments/isaac/export_replay.py --policy B2.onnx --out replay.npz [--scenario memory]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reactive"))
import scenario  # noqa: E402

FPS = 25


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
    names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0]  # hinge joints
    addr = [model.jnt_qposadr[model.joint(n).id] for n in names]
    mocap = {n: model.body_mocapid[model.body(n).id] for n in ("person", "person_b", "ball")}
    qpos = np.array([f[0] for f in frames])
    mpos = np.array([f[1] for f in frames])
    mquat = np.array([f[2] for f in frames])
    t = np.arange(len(frames)) / FPS
    modes = np.array([rows[min(int(round(ti / scenario.CTRL_DT)), len(rows) - 1)]["mode"] for ti in t])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, t=t, base_pos=qpos[:, 0:3], base_quat=qpos[:, 3:7], joints=qpos[:, addr],
                        joint_names=np.array(names), person_pos=mpos[:, mocap["person"]],
                        person_quat=mquat[:, mocap["person"]], person_b_pos=mpos[:, mocap["person_b"]],
                        person_b_quat=mquat[:, mocap["person_b"]], ball_pos=mpos[:, mocap["ball"]], modes=modes)
    print(f"wrote {args.out}: {len(t)} frames, joints {names}")


if __name__ == "__main__":
    main()
