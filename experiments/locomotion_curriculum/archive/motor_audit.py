"""Fixed synthetic probes of historical B2 targets; no training or rollouts."""
import argparse
import ast
import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from experiments.locomotion_curriculum.gate import COMMANDS, write_json
from nerva.sim.open_duck import OPEN_DUCK_ROOT, REFERENCE


def angular_probes(fn):
    from scipy.spatial.transform import Rotation
    rows = []
    for yaw in (0., 1.):
        previous = Rotation.from_euler("z", yaw).as_quat()
        current = Rotation.from_euler("z", yaw + .6 * .02).as_quat()
        actual = np.asarray(fn(current, previous, .02))
        rows.append({"initial_yaw_rad": yaw, "actual_xyz_rad_s": actual.tolist(),
                     "expected_xyz_rad_s": [0, 0, .6],
                     "matches_expected": bool(np.allclose(actual, [0, 0, .6], atol=1e-6, rtol=0))})
    return rows


def extract_angular_helper(path):
    from scipy.spatial.transform import Rotation
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_angular_velocity"]
    if len(functions) != 1 or [a.arg for a in functions[0].args.args] != ["quat", "prev_quat", "dt"]:
        raise ValueError("generator helper source mismatch")
    isolated = ast.Module(body=functions, type_ignores=[])
    namespace = {"np": np, "R": Rotation}
    exec(compile(isolated, "<audited-generator-helper>", "exec"), namespace)
    return namespace["compute_angular_velocity"]


def make_probe(reference):
    import jax.numpy as jp
    from nerva.training.style_joystick import StyleJoystick
    from playground.open_duck_mini_v2 import joystick

    class SensorProbe(StyleJoystick):
        """Synthetic sensor adapter: execute actual reward dispatch, no model."""
        def __init__(self):
            self._config = joystick.default_config()
            self._config.reward_config.scales.feet_height = -30.
            self.feet_height_scale, self.feet_air_time_scale = -30., 0.
            self.SREF = reference
            self._default_actuator = jp.zeros(14)

        def get_local_linvel(self, data):
            return data.local_linvel

        def get_gyro(self, data):
            return data.gyro

        def get_floating_base_qpos(self, qpos):
            return qpos[:7]

        def get_floating_base_qvel(self, qvel):
            return qvel[:6]

        def get_actuator_joints_qpos(self, qpos):
            return qpos[7:]

        def get_actuator_joints_qvel(self, qvel):
            return qvel[6:]

    return SensorProbe()


def synthetic_rewards(probe, frame, command, lift=.02, first_contact=(1, 0)):
    import jax.numpy as jp
    from nerva.training.style_joystick import STANCE_FOOT_SITE_Z
    data = SimpleNamespace(qpos=jp.array([0, 0, .15, 1, 0, 0, 0] + [.1] * 14),
                           qvel=jp.array([0] * 6 + [.1] * 14), actuator_force=jp.zeros(14),
                           local_linvel=jp.zeros(3), gyro=jp.zeros(3))
    info = {"command": jp.array(command, dtype=jp.float32), "current_reference_motion": frame,
            "last_act": jp.zeros(14), "style_idx": 0,
            "swing_peak": jp.full(2, STANCE_FOOT_SITE_Z + lift)}
    raw = probe._get_reward(data, jp.zeros(14), info, {}, jp.float32(0),
                            jp.array(first_contact), jp.array([True, True]))
    return {k: {"raw": float(v), "scale": float(probe._config.reward_config.scales[k]),
                "scaled": float(v * probe._config.reward_config.scales[k])} for k, v in raw.items()}


def tracking_probes():
    import jax.numpy as jp
    from playground.common.rewards import reward_tracking_ang_vel, reward_tracking_lin_vel
    rows = []
    for name in ("stop", "turn_left", "turn_right"):
        cmd = jp.array(COMMANDS[name])
        for vx in (0., .013, .10):
            for vy in (0., .05, .10, .11):
                for wz in (0., .033, .60):
                    lin = float(reward_tracking_lin_vel(cmd, jp.array([vx, vy, 0]), .01))
                    ang = float(reward_tracking_ang_vel(cmd, jp.array([0, 0, wz]), .01))
                    rows.append({"command": name, "velocity_vx_vy_wz": [vx, vy, wz],
                                 "linear_raw": lin, "angular_raw": ang,
                                 "linear_scaled": lin * 2.5, "angular_scaled": ang * 6})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_motor_audit"))
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("never overwrite audit outputs")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before evaluation")
    import jax.numpy as jp
    from nerva.training.style_joystick import StyledReference
    with contextlib.redirect_stdout(io.StringIO()):
        reference = StyledReference([str(REFERENCE)], [[0., 0., 0.]])
    n = int(reference.nb_steps[0])
    if n != 27:
        raise ValueError("unexpected training phase count")
    refs = {}
    for name, cmd in COMMANDS.items():
        frames = np.asarray([reference.get_reference_motion(*cmd, i, 0) for i in range(n)])
        if frames.shape != (27, 40) or not np.isfinite(frames).all():
            raise ValueError("malformed reference samples")
        grids = (reference.dxs[0], reference.dys[0], reference.dthetas[0])
        key = [float(g[int(jp.argmin(jp.abs(g - c)))]) for g, c in zip(grids, cmd)]
        refs[name] = {"command": list(cmd), "nearest_key": key, "phases": n,
                      "mean_linear_xyz": frames[:, 34:37].mean(0).tolist(),
                      "range_linear_xyz": [frames[:, 34:37].min(0).tolist(), frames[:, 34:37].max(0).tolist()],
                      "mean_angular_xyz": frames[:, 37:40].mean(0).tolist(),
                      "range_angular_xyz": [frames[:, 37:40].min(0).tolist(), frames[:, 37:40].max(0).tolist()],
                      "joint_position_range": [frames[:, :16].min(0).tolist(), frames[:, :16].max(0).tolist()],
                      "joint_velocity_range": [frames[:, 16:32].min(0).tolist(), frames[:, 16:32].max(0).tolist()],
                      "contact_range": [frames[:, 32:34].min(0).tolist(), frames[:, 32:34].max(0).tolist()]}
    probe = make_probe(reference)
    frame = reference.get_reference_motion(0., 0., 0., 0, 0)
    dispatch = []
    for vx, head in ((0., 0.), (.009, 0.), (.010, 0.), (.011, 0.), (0., .4)):
        command = [vx, 0, 0, 0, 0, head, 0]
        dispatch.append({"command": command, "rewards": synthetic_rewards(probe, frame, command)})
    height = []
    for vx in (0., .15):
        for lift in (0., .02, .04):
            for contact in ((0, 0), (1, 0), (1, 1)):
                reward = synthetic_rewards(probe, frame, [vx, 0, 0, 0, 0, 0, 0], lift, contact)["feet_height"]
                height.append({"command_vx": vx, "lift_m": lift, "first_contact": list(contact), **reward})
    generator = OPEN_DUCK_ROOT / "Open_Duck_reference_motion_generator"
    helper_path = generator / "open_duck_reference_motion_generator/gait_generator.py"
    angular = angular_probes(extract_angular_helper(helper_path))
    tracking = tracking_probes()
    stop_tracking = {tuple(r["velocity_vx_vy_wz"]): r for r in tracking if r["command"] == "stop"}
    lateral_equal = all(abs(stop_tracking[(vx, vy, 0.)]["linear_raw"] -
                            stop_tracking[(vx, 0., 0.)]["linear_raw"]) <= 1e-6
                        for vx in (0., .013, .10) for vy in (0., .05, .10))
    beyond_lower = all(stop_tracking[(vx, .11, 0.)]["linear_raw"] < stop_tracking[(vx, 0., 0.)]["linear_raw"]
                       for vx in (0., .013, .10))
    grids = {name: np.asarray(g[0]).tolist() for name, g in
             (("vx", reference.dxs), ("vy", reference.dys), ("wz", reference.dthetas))}
    report = {"grid": grids, "reference_targets": refs, "dispatch": dispatch,
              "height_probes": height, "tracking_probes": tracking, "generator_angular_probes": angular,
              "decisions": {"imitation_disabled_at_zero": all(abs(r["rewards"]["imitation"]["raw"]) <= 1e-6
                             for r in dispatch if r["command"][0] == 0),
                             "stationary_pose_cost_active": dispatch[0]["rewards"]["stand_still"]["raw"] > 0,
                             "lateral_tolerance_confirmed": lateral_equal and beyond_lower,
                             "height_no_contact_zero": all(abs(r["raw"]) <= 1e-6 for r in height
                                                            if r["first_contact"] == [0, 0]),
                             "height_target_touchdown_zero": all(abs(r["raw"]) <= 1e-6 for r in height
                                                                  if r["lift_m"] == .04),
                             "generator_yaw_matches_expected": all(r["matches_expected"] for r in angular)},
              "causation": "not established", "original_gate": "failed; unchanged"}
    upstream = OPEN_DUCK_ROOT / "Open_Duck_Playground"
    sources = {"reference": REFERENCE, "generator": helper_path, "fitter": generator / "scripts/fit_poly.py",
               "style_env": Path("nerva/training/style_joystick.py"),
               "training_runner": Path("nerva/training/train_style.py"), "b2_job": Path("cloud/jobs/b2_neutral.sh"),
               "b1_job": Path("cloud/jobs/b1_neutral.sh"),
               "rewards": upstream / "playground/common/rewards.py",
               "imitation": upstream / "playground/open_duck_mini_v2/custom_rewards.py",
               "joystick": upstream / "playground/open_duck_mini_v2/joystick.py",
               "base": upstream / "playground/open_duck_mini_v2/base.py",
               "reference_parser": upstream / "playground/common/poly_reference_motion.py"}
    provenance = {"preregistration_commit": "fa22d19",
                  "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": {label: hashlib.sha256(p.read_bytes()).hexdigest() for label, p in sources.items()},
                  "upstream_revisions": {label: subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"],
                                                                       text=True).strip()
                                         for label, p in (("playground", upstream), ("generator", generator))},
                  "synthetic_only": True, "cloud_work": False}
    args.out.mkdir(parents=True)
    write_json(args.out / "report.json", report)
    write_json(args.out / "provenance.json", provenance)
    print(report["decisions"], flush=True)


if __name__ == "__main__":
    main()
