"""Render a NERVA MuJoCo replay in Isaac Sim (runs inside the Isaac Sim container on a cloud L4).

Kinematic replay (experiments/isaac/README.md): NERVA's robot is imported from its own MuJoCo model with
Isaac's MJCF importer, and every frame each robot body is placed at the world pose recorded in MuJoCo
(export_replay.py). Physics is never started, so nothing in Isaac changes the motion. People are Isaac
character assets moved along their recorded paths (rigid, not animated, in this version). A camera follows
the robot. Frames are written as PNGs; the caller encodes the video.

Learned in the spike (docs/development_log.md, 2026-10-01): Isaac 5.1 on Ubuntu + driver 570 renders;
the MJCF route imports the robot; a Y-up default stage and strong lights gave white frames.

Usage (in the container):
  /isaac-sim/python.sh render_replay.py --mjcf open_duck_mini_v2.xml --replay replay.npz --out OUT
"""

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mjcf", default="", help="unused fallback: Isaac's MJCF importer creates no bodies for Open Duck")
parser.add_argument("--robot-mesh", required=True, help="export_robot_mesh.py output (visual geometry per body)")
parser.add_argument("--replay", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--start", type=int, default=0, help="first replay frame")
parser.add_argument("--frames", type=int, default=750, help="number of rendered frames")
parser.add_argument("--stride", type=int, default=2, help="replay frames per rendered frame")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args, _ = parser.parse_known_args()
os.makedirs(args.out, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

CHARACTERS = {"person": "/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
              "person_b": "/Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd"}
log = {"warnings": []}
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def matrix(pos, quat_wxyz) -> Gf.Matrix4d:
    w, x, y, z = (float(v) for v in quat_wxyz)
    m = Gf.Matrix4d()
    m.SetTransform(Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))), Gf.Vec3d(*(float(v) for v in pos)))
    return m


def set_local(prim, local: Gf.Matrix4d) -> None:
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if len(ops) != 1 or ops[0].GetOpType() != UsdGeom.XformOp.TypeTransform:
        xf.ClearXformOpOrder()
        xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    xf.GetOrderedXformOps()[0].Set(local)


# ── scene ──
from isaacsim.core.api.objects import GroundPlane  # noqa: E402

GroundPlane("/World/Ground", size=40.0, color=np.array([0.42, 0.42, 0.40]))
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(400.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(900.0)
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun).AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(-50.0, 0.0, 35.0))

# Robot: one top-level Xform per MuJoCo body, holding that body's visual meshes at their local poses.
replay = np.load(args.replay)
names = [str(n) for n in replay["body_names"]]
geo = np.load(args.robot_mesh)
from pxr import Vt  # noqa: E402

prims = {}
UsdGeom.Xform.Define(stage, "/World/duck")
for i, body in enumerate(str(b) for b in geo["body"]):
    if body not in prims:
        prims[body] = UsdGeom.Xform.Define(stage, f"/World/duck/{body}").GetPrim()
    mesh = UsdGeom.Mesh.Define(stage, f"/World/duck/{body}/geom_{i}")
    v0, nv = int(geo["vert_start"][i]), int(geo["vert_count"][i])
    f0, nf = int(geo["face_start"][i]), int(geo["face_count"][i])
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(geo["verts"][v0:v0 + nv]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * nf))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(geo["faces"][f0:f0 + nf].reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    rgba = geo["rgba"][i]
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(c) for c in rgba[:3]))]))
    set_local(mesh.GetPrim(), matrix(geo["pos"][i], geo["quat"][i]))
log["bodies_found"] = sorted(prims)
log["bodies_missing"] = sorted(set(names) - set(prims))
log["bodies_without_geometry"] = sorted(set(names) - set(prims))

try:
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.core.utils.stage import add_reference_to_stage
    assets = get_assets_root_path()
    people = {}
    for key, rel in CHARACTERS.items():
        try:
            people[key] = add_reference_to_stage(assets + rel, f"/World/{key}")
        except Exception as e:
            log["warnings"].append(f"character {key}: {e!r}")
except Exception as e:
    people = {}
    log["warnings"].append(f"assets: {e!r}")

camera = UsdGeom.Camera.Define(stage, "/World/FollowCam")
camera.CreateFocalLengthAttr(18.0)
rp = rep.create.render_product("/World/FollowCam", (args.width, args.height))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=os.path.join(args.out, "frames"), rgb=True)
writer.attach([rp])

# ── replay ──
look = None
n_total = len(replay["t"])
for i in range(args.frames):
    k = min(args.start + i * args.stride, n_total - 1)
    for name, prim in prims.items():  # flat hierarchy: the body Xform's local pose is its world pose
        j = names.index(name)
        set_local(prim, matrix(replay["body_pos"][k, j], replay["body_quat"][k, j]))
    for key, prim in people.items():
        pos, quat = replay[f"{key}_pos"][k], replay[f"{key}_quat"][k]
        set_local(prim, matrix((pos[0], pos[1], 0.0), quat))
    base = replay["body_pos"][k, names.index("base")]
    target = np.array([base[0], base[1], 0.2])
    nearest = min((replay[f"{key}_pos"][k] for key in people), key=lambda p: np.linalg.norm(p[:2] - base[:2]),
                  default=None)
    if nearest is not None and np.linalg.norm(nearest[:2] - base[:2]) < 3.5:
        target = 0.6 * target + 0.4 * np.array([nearest[0], nearest[1], 0.5])
    look = target if look is None else 0.9 * look + 0.1 * target
    eye = look + np.array([-1.3, -1.6, 0.7])
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1))
    set_local(camera.GetPrim(), view.GetInverse())
    rep.orchestrator.step(rt_subframes=1)

rep.orchestrator.wait_until_complete()
log["frames"] = args.frames
log["replay_frames_used"] = [args.start, min(args.start + (args.frames - 1) * args.stride, n_total - 1)]
with open(os.path.join(args.out, "render_report.json"), "w") as f:
    json.dump(log, f, indent=1)
print("RENDER_DONE", json.dumps(log)[:500], flush=True)
app.close()
