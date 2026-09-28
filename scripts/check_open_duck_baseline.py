"""Headless sanity check of the unmodified Open Duck Mini v2 baseline policy.

Runs upstream `MjInfer.run()` (Open_Duck_Playground/playground/open_duck_mini_v2/
mujoco_infer.py) exactly as-is, but replaces the MuJoCo GUI viewer with a stub
that records the base pose after every physics step and stops after a fixed
simulated duration. Nothing in the upstream control loop is reimplemented here.

Usage (from anywhere):
    <OPEN_DUCK_ROOT>/.venv/Scripts/python scripts/check_open_duck_baseline.py
"""

import argparse
import os

import time
from pathlib import Path

import numpy as np

OPEN_DUCK_ROOT = Path(os.environ.get("OPEN_DUCK_ROOT", Path.home() / "dev" / "open_duck"))
PLAYGROUND = OPEN_DUCK_ROOT / "Open_Duck_Playground"

# `playground` is the Open Duck package, installed editable into the venv
# (pip install -e Open_Duck_Playground --no-deps); not DeepMind's `mujoco_playground`.
import mujoco.viewer  # noqa: E402
from playground.open_duck_mini_v2 import mujoco_infer  # noqa: E402

SCENE = PLAYGROUND / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
REFERENCE = PLAYGROUND / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
POLICY = OPEN_DUCK_ROOT / "Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"

# Commands in the same units the upstream keyboard handler uses:
# [lin_vel_x m/s, lin_vel_y m/s, ang_vel rad/s, neck_pitch, head_pitch, head_yaw, head_roll]
SCENARIOS = {
    "idle (zero command)": [0.0, 0.0, 0.0],
    "forward 0.15 m/s": [0.15, 0.0, 0.0],
    "backward 0.15 m/s": [-0.15, 0.0, 0.0],
    "lateral left 0.2 m/s": [0.0, 0.2, 0.0],
    "turn left 1.0 rad/s": [0.0, 0.0, 1.0],
}

FALL_TILT_DEG = 45.0  # base z-axis more than this from vertical counts as a fall


class RecordingViewer:
    """Stands in for mujoco.viewer.launch_passive(); records base pose per step."""

    def __init__(self, data, n_steps):
        self.data, self.n_steps, self.log = data, n_steps, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sync(self):
        q = self.data.qpos
        w, x, y, z = q[3:7]  # free-joint quaternion (MuJoCo order: w, x, y, z)
        up_z = 1.0 - 2.0 * (x * x + y * y)  # world-z component of the base z-axis
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.log.append((self.data.time, q[0], q[1], q[2], np.degrees(np.arccos(np.clip(up_z, -1, 1))), yaw))
        if len(self.log) >= self.n_steps:
            raise KeyboardInterrupt  # upstream run() catches this and exits cleanly


def run_scenario(command, seconds, raw_accel=False):
    m = mujoco_infer.MjInfer(str(SCENE), str(REFERENCE), str(POLICY), standing=False)
    if raw_accel:
        # Upstream get_obs() adds +1.3 to accelerometer x. Training (joystick.py) and the
        # hardware runtime do not, so pre-subtract it here to feed the policy the raw value.
        get_acc = m.get_accelerometer

        def raw(data):
            a = np.array(get_acc(data), copy=True)
            a[0] -= 1.3
            return a

        m.get_accelerometer = raw
    m.commands = list(command) + [0.0] * 4
    n_steps = int(seconds / m.sim_dt)
    holder = {}

    def fake_launch_passive(model, data, **kwargs):
        holder["v"] = RecordingViewer(data, n_steps)
        return holder["v"]

    mujoco.viewer.launch_passive = fake_launch_passive
    time.sleep = lambda s: None  # upstream paces to real time; skip that headless
    m.run()
    return np.array(holder["v"].log), m


def summarise(name, command, log, seconds):
    t, x, y, h, tilt, yaw = log.T
    fell = bool((tilt > FALL_TILT_DEG).any())
    half = len(t) // 2  # measure steady state over the second half only
    dt = t[-1] - t[half]
    vx_w, vy_w = (x[-1] - x[half]) / dt, (y[-1] - y[half]) / dt
    c, s = np.cos(yaw[half]), np.sin(yaw[half])  # rotate into the robot's heading frame
    v_fwd, v_lat = c * vx_w + s * vy_w, -s * vx_w + c * vy_w
    yaw_rate = np.unwrap(yaw)[-1] - np.unwrap(yaw)[half]
    print(
        f"{name:24s} cmd={command}  fell={fell}  max_tilt={tilt.max():5.1f}deg  "
        f"height={h.min():.3f}-{h.max():.3f}m  "
        f"v_fwd={v_fwd:+.3f}  v_lat={v_lat:+.3f} m/s  yaw_rate={yaw_rate / dt:+.2f} rad/s"
    )
    return fell


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--raw-accel", action="store_true",
                    help="cancel the +1.3 accelerometer-x offset that only mujoco_infer.py adds")
    args = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent)  # upstream run() writes mujoco_saved_obs.pkl to cwd
    offset = "none (raw)" if args.raw_accel else "+1.3 (as upstream mujoco_infer.py)"
    print(f"policy={POLICY.name}  scene={SCENE.name}  duration={args.seconds}s per scenario  "
          f"accel_x_offset={offset}\n")
    any_fell = False
    for name, cmd in SCENARIOS.items():
        log, _ = run_scenario(cmd, args.seconds, args.raw_accel)
        any_fell |= summarise(name, cmd, log, args.seconds)
    Path("mujoco_saved_obs.pkl").unlink(missing_ok=True)
    print("\nRESULT:", "at least one fall" if any_fell else "no falls in any scenario")
