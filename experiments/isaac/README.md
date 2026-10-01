# Isaac Sim rendering (kinematic replay)

Goal: realistic demo videos (lighting, human models, later real person/face detection on rendered
frames) of NERVA's behaviour. **Isaac Sim is a renderer here, not the physics.** The perception →
appraisal → affect → memory → behaviour loop and the walking policy run in MuJoCo; Isaac re-renders the
recorded motion. Moving physics into Isaac would mean re-training locomotion there.

Pipeline:
1. `export_robot_mesh.py` → `assets/open_duck_visual.npz`: the robot's visual geometry from MuJoCo (44
   meshes on 15 bodies). Isaac's URDF and MJCF importers produced no robot bodies for Open Duck in the
   tested setup, so the robot is rebuilt from these meshes.
2. `export_replay.py` → `replays/<name>.npz` + `<name>.json`: runs a scenario in MuJoCo (vision perception,
   memory, backlash scene) and stores per-frame body poses, people poses and the robot-eye camera pose;
   the sidecar holds the affect, behaviour and memory values, events with appraisals, and vision boxes.
3. `render_replay.py` (inside the Isaac container, cloud L4): poses one Xform per robot body each frame,
   Isaac human characters for the people (static pose, not animated yet), a follow camera and a
   `RobotEye` camera at MuJoCo's eye pose → `frames/`, `frames_eye/`.
4. `compose.py` (local, light): scene view + robot-eye inset with vision boxes + affect charts +
   captions → MP4.

`isaac_spike.py` is the earlier feasibility check (headless start, import attempts, human character).

Cloud job: `cloud/jobs/isaac_spike.sh` (plain Ubuntu 22.04 + `nvidia-driver-570`, Isaac Sim 5.1
container; the Deep Learning VM image's driver had no Vulkan support). Running Isaac Sim requires
accepting the NVIDIA Omniverse License Agreement; the launcher passes it only with
`--accept-isaac-eula`, used only after the user accepted it. Privacy consent is N.
