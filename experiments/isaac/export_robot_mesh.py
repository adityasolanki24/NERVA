"""Export Open Duck's visual geometry from NERVA's MuJoCo model for Isaac Sim rendering.

Isaac Sim's MJCF importer creates no robot bodies from Open Duck's model, and its URDF import failed
(docs/development_log.md, 2026-10-01). Instead, the renderer builds the robot from this file: per visual
mesh geom, its body name, local pose in the body frame, colour and triangle mesh. Each MuJoCo body becomes
one top-level Xform in Isaac, so a replay sets world poses directly (render_replay.py).

Usage: python experiments/isaac/export_robot_mesh.py --out experiments/isaac/assets/open_duck_visual.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from nerva.sim.open_duck import SCENE_BACKLASH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(SCENE_BACKLASH))
    geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] == 0]
    out = {"body": [], "pos": [], "quat": [], "rgba": [], "vert_start": [], "vert_count": [],
           "face_start": [], "face_count": []}
    verts, faces = [], []
    for g in geoms:
        mesh = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        f0, nf = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
        out["body"].append(m.body(m.geom_bodyid[g]).name)
        out["pos"].append(m.geom_pos[g])
        out["quat"].append(m.geom_quat[g])  # w x y z, geom frame in body frame
        out["rgba"].append(m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]])
        out["vert_start"].append(sum(len(v) for v in verts))
        out["vert_count"].append(nv)
        out["face_start"].append(sum(len(f) for f in faces))
        out["face_count"].append(nf)
        verts.append(m.mesh_vert[v0:v0 + nv])  # in the mesh (= geom) frame
        faces.append(m.mesh_face[f0:f0 + nf])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, verts=np.concatenate(verts).astype(np.float32),
                        faces=np.concatenate(faces).astype(np.int32),
                        **{k: np.array(v) for k, v in out.items()})
    print(f"wrote {args.out}: {len(geoms)} geoms on {len(set(out['body']))} bodies, "
          f"{sum(len(v) for v in verts)} vertices")


if __name__ == "__main__":
    main()
