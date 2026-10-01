"""Isaac Sim feasibility spike: can we import Open Duck, place a human, and render frames headless?

Runs INSIDE the Isaac Sim container (`/isaac-sim/python.sh isaac_spike.py --urdf ... --out ...`), on a
cloud L4 (cloud/jobs/isaac_spike.sh). It deliberately logs every step and keeps going where it can, so
one run answers as many questions as possible:
  1. Isaac Sim starts headless with RTX rendering
  2. the Open Duck URDF imports as an articulation (joint names match NERVA's MuJoCo model)
  3. the asset server is reachable and a human character can be referenced into the stage
  4. a camera renders PNG frames, and a replay file (if given) can drive the robot's pose
API names follow Isaac Sim 5.x (isaacsim.*), with fallbacks to 4.x (omni.isaac.*) where they differ.
Findings are written to <out>/spike_report.json.
"""

import argparse
import json
import os
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--replay", default="", help="optional .npz from experiments/isaac/export_replay.py")
parser.add_argument("--frames", type=int, default=60)
args, _ = parser.parse_known_args()
os.makedirs(args.out, exist_ok=True)
report = {"steps": {}}


def step(name):
    def wrap(fn):
        try:
            report["steps"][name] = {"ok": True, "result": fn()}
        except Exception as e:  # keep going: the spike's job is to find out what works
            report["steps"][name] = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-2000:]}
        print(f"[spike] {name}: {report['steps'][name]['ok']}", flush=True)
        with open(os.path.join(args.out, "spike_report.json"), "w") as f:
            json.dump(report, f, indent=1, default=str)
    return wrap


from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

stage = omni.usd.get_context().get_stage()
state = {}


@step("versions")
def _():
    import isaacsim
    return {"python": sys.version, "isaacsim": getattr(isaacsim, "__version__", "unknown")}


@step("ground_and_light")
def _():
    from isaacsim.core.api.objects import GroundPlane  # 5.x
    GroundPlane("/World/Ground", size=20.0, color=np.array([0.55, 0.55, 0.55]))
    light = UsdLux.DomeLight.Define(stage, "/World/Dome")
    light.CreateIntensityAttr(1500.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 30))
    return "ok"


@step("import_urdf")
def _():
    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.fix_base = False
    cfg.merge_fixed_joints = False
    cfg.make_default_prim = False
    cfg.import_inertia_tensor = True
    status, path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=args.urdf,
                                             import_config=cfg, get_articulation_root=True)
    state["robot"] = path
    joints = [p.GetName() for p in stage.Traverse() if p.GetTypeName().endswith("Joint")]
    return {"status": status, "prim": path, "joints": joints}


@step("assets_root")
def _():
    try:
        from isaacsim.storage.native import get_assets_root_path
    except ImportError:
        from omni.isaac.nucleus import get_assets_root_path
    root = get_assets_root_path()
    state["assets"] = root
    return root


@step("human_character")
def _():
    candidates = ["/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
                  "/Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd"]
    from isaacsim.core.utils.stage import add_reference_to_stage
    for c in candidates:
        try:
            prim = add_reference_to_stage(state["assets"] + c, "/World/Person")
            UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(1.5, 0.0, 0.0))
            UsdGeom.XformCommonAPI(prim).SetRotate(Gf.Vec3f(0, 0, 180))
            return {"loaded": c}
        except Exception as e:  # try the next one
            last = repr(e)
    raise RuntimeError(f"no character loaded: {last}")


@step("render_frames")
def _():
    from isaacsim.core.api import World
    import omni.replicator.core as rep
    world = World(stage_units_in_meters=1.0)
    world.reset()
    cam = rep.create.camera(position=(1.2, -1.6, 0.8), look_at=(0.5, 0.0, 0.3))
    rp = rep.create.render_product(cam, (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=os.path.join(args.out, "frames"), rgb=True)
    writer.attach([rp])
    replay = np.load(args.replay) if args.replay else None
    art = None
    if replay is not None:
        from isaacsim.core.prims import SingleArticulation
        art = SingleArticulation(state["robot"])
        art.initialize()
        names = list(art.dof_names)
        cols = [list(replay["joint_names"]).index(n) if n in list(replay["joint_names"]) else -1 for n in names]
    for i in range(args.frames):
        if art is not None:
            k = min(i, len(replay["t"]) - 1)
            q = np.array([replay["joints"][k, c] if c >= 0 else 0.0 for c in cols])
            art.set_joint_positions(q)
            art.set_world_pose(position=replay["base_pos"][k], orientation=replay["base_quat"][k])
        world.step(render=True)
        rep.orchestrator.step()
    return {"frames": args.frames, "replay": bool(args.replay)}


app.close()
