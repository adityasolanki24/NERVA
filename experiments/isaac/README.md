# Isaac Sim rendering (stage 2, in progress)

Goal: realistic demo videos (lighting, human models, later real face/person detection) of NERVA's
behaviour. First approach: **kinematic replay**. The perception → appraisal → affect → memory →
behaviour loop and the walking policy keep running in MuJoCo, where they work. Isaac Sim re-renders the
recorded motion of the robot and the people. Moving physics into Isaac Sim would mean re-training
locomotion there; that is a later step.

- `isaac_spike.py`: feasibility check inside the Isaac Sim container: headless start, URDF import,
  human character from the asset server, rendered frames, optional replay → `spike_report.json`.
- `export_replay.py`: runs a NERVA scenario in MuJoCo and writes the motion as `.npz` for replay.
- Cloud job: `cloud/jobs/isaac_spike.sh` (L4).

Running Isaac Sim requires accepting the NVIDIA Omniverse License Agreement. The launcher passes it
only with `--accept-isaac-eula`, which is used only after the user has accepted it. Privacy consent
is set to N.
