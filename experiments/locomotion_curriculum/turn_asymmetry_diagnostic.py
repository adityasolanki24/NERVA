"""Diagnostic: why does the base-origin candidate's left turn keep a sideways pivot offset when its right turn
does not? (After docs/base_origin_velocity_pilot.md; development log 2026-10-10.) No criteria, no training.

1. MJX vs native. Deterministic rollouts of the base-origin candidate in its own MJX training environment
   (backlash scene, training noise, pushes off), seeds 0/1/2, 1,000 steps, both pure turns. Same metric as the
   native evaluation: horizontal RMS of the 1 s-smoothed heading-frame base-origin velocity over 5–20 s, plus
   the pivot centre. Compared with the saved native rollouts.
2. Reference symmetry. The fitted turn-left reference, mirrored (left/right legs swapped, hip yaw and roll
   negated), against the turn-right reference over every phase shift; contact and body-velocity mirror error.

3. Delays (added after parts 1–2, post-hoc): the training environment delays actions and IMU readings by a
   random 0–2 control steps; the native evaluation has neither. MJX rollouts with the delays as trained and
   disabled (maximum delay 1, so always 0), for the base-origin and the GPU pilot candidates.

Reading (fixed before running parts 1–2):
- If the MJX left turn translates ≤ 0.03 m/s while native does not, the offset is a MJX → native gap.
- If MJX also translates > 0.03 m/s with a similar sideways pivot, the policy learned it.
- A mirror joint error far above the fit tolerance would point at the reference itself.

Usage: python -m experiments.locomotion_curriculum.turn_asymmetry_diagnostic [--out DIR]
"""
from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path

import numpy as np

from nerva.analysis.motor_eval import smooth, velocities
from nerva.training.b2_warm_start import NETWORK, SHAPE, balanced_environment
from nerva.training.motor_artifacts import write_json
from nerva.training.neutral_reference import COMMANDS

RUN = "experiments/cloud_runs/base_origin_velocity-20261010-201858"
CHECKPOINT = RUN + "/checkpoints/000060480000"
DELAY_ARMS = {"base_candidate": CHECKPOINT,
              "gpu_candidate": "experiments/cloud_runs/neutral_gpu_pilot-20261010-152812/checkpoints/000060318720"}
TURNS = {"turn_left": 5, "turn_right": 6}
STEPS, START = 1000, 250
# Joint order of the reference rows 0:16 (and velocities 16:32).
LEFT, RIGHT = [0, 1, 2, 3, 4], [11, 12, 13, 14, 15]
# Hip yaw, roll and pitch flip sign under the sagittal mirror (the legs' pitch axes are mirrored in the model:
# standing hip pitch is −0.56 rad left, +0.68 right). The first run used [0, 1] and reported a spurious
# 0.56 rad mirror error; corrected 2026-10-10 before writing up (sign search: [0, 1, 2] is the only
# convention giving the forward walk's own left/right error, ≈ 0.04 rad).
NEGATE = [0, 1, 2]


def turn_metrics(quat, linvel):
    vel = velocities({"base_quat": quat, "base_linvel": linvel})[START:]
    f = smooth(vel)
    mean, yaw = vel[:, :2].mean(axis=0), vel[:, 2].mean()
    return {"horizontal_rms": float(np.sqrt(np.mean(np.sum(f[:, :2] ** 2, axis=1)))), "mean_yaw_rate": float(yaw),
            "pivot_forward_left_m": [float(-mean[1] / yaw), float(mean[0] / yaw)]}


def mjx_rollouts(root, reference, folder=CHECKPOINT, delays=True, source="mjx"):
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks
    params = checkpoint.load(root / folder)
    net = functools.partial(networks.make_ppo_networks, **NETWORK)(
        SHAPE, 14, preprocess_observations_fn=running_statistics.normalize)
    policy = networks.make_inference_fn(net)(params, deterministic=True)
    env = balanced_environment(reference, episode_length=STEPS + 10, replicas=1, persistent_command=True,
                               gait_averaged_tracking=True, base_origin_velocity=True)
    config = env.env.env.env._config
    config.push_config.enable = False
    if not delays:
        config.noise_config.action_max_delay = config.noise_config.imu_max_delay = 1  # randint(0, 1) == 0
    rows = []
    for seed in (0, 1, 2):
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(seed), 7))

        def step(carry, _):
            st, key = carry
            key, k = jax.random.split(key)
            action, _ = policy(st.obs, k)
            st = env.step(st, action)
            return (st, key), (st.data.qpos[:, 3:7], st.data.qvel[:, :3], st.done)

        _, (quat, linvel, done) = jax.jit(lambda s, k: jax.lax.scan(step, (s, k), None, STEPS))(
            state, jax.random.PRNGKey(100 + seed))
        quat, linvel, done = np.asarray(quat), np.asarray(linvel), np.asarray(done)
        for name, i in TURNS.items():
            rows.append({"source": source, "command": name, "seed": seed, "terminations": int(done[:, i].sum()),
                         **turn_metrics(quat[:, i], linvel[:, i])})
        print(source, "seed", seed, flush=True)
    return rows


def native_rows(root):
    rows = []
    for name in TURNS:
        for seed in (0, 1, 2):
            a = np.load(root / RUN / "eval" / f"{name}_{seed}_base_candidate.npz")
            rows.append({"source": "native", "command": name, "seed": seed,
                         **turn_metrics(a["base_quat"], a["base_linvel"])})
    return rows


def mirror(frames):
    out = frames.copy()
    for offset in (0, 16):
        left, right = frames[:, [offset + j for j in LEFT]], frames[:, [offset + j for j in RIGHT]]
        out[:, [offset + j for j in LEFT]], out[:, [offset + j for j in RIGHT]] = right, left
        for j in NEGATE:
            out[:, offset + LEFT[j]] *= -1
            out[:, offset + RIGHT[j]] *= -1
    out[:, [32, 33]] = frames[:, [33, 32]]  # contacts swap
    out[:, 35] *= -1  # lateral velocity
    out[:, 37] *= -1  # roll rate
    out[:, 39] *= -1  # yaw rate
    return out


def reference_symmetry(reference):
    left = np.array([np.asarray(reference.get_reference_motion(*COMMANDS[5], i)) for i in range(27)])
    right = np.array([np.asarray(reference.get_reference_motion(*COMMANDS[6], i)) for i in range(27)])
    mirrored = mirror(left)
    joints = LEFT + RIGHT
    errors = [float(np.sqrt(np.mean((np.roll(mirrored, k, axis=0)[:, joints] - right[:, joints]) ** 2)))
              for k in range(27)]
    best = int(np.argmin(errors))
    forward = np.array([np.asarray(reference.get_reference_motion(*COMMANDS[1], i)) for i in range(27)])
    self_mirror = mirror(forward)
    walk = min(float(np.sqrt(np.mean((np.roll(self_mirror, k, axis=0)[:, joints] - forward[:, joints]) ** 2)))
               for k in range(27))
    shifted = np.roll(mirrored, best, axis=0)
    return {"forward_walk_self_mirror_joint_rms_rad": walk, "best_phase_shift_steps": best, "joint_rms_rad_at_best_shift": errors[best],
            "joint_rms_rad_unshifted": errors[0],
            "contact_mismatch_fraction": float(np.mean((shifted[:, 32:34] > .5) != (right[:, 32:34] > .5))),
            "body_velocity_rms_at_best_shift": float(np.sqrt(np.mean((shifted[:, 34:40] - right[:, 34:40]) ** 2))),
            "turn_left_mean_body_velocity": left[:, 34:37].mean(axis=0).round(4).tolist(),
            "turn_right_mean_body_velocity": right[:, 34:37].mean(axis=0).round(4).tolist(),
            "turn_left_first_swing": "left" if left[0, 32] < .5 else "right" if left[0, 33] < .5 else "double",
            "turn_right_first_swing": "left" if right[0, 32] < .5 else "right" if right[0, 33] < .5 else "double"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_turn_asymmetry"))
    args = ap.parse_args()
    args.out = args.out.resolve()  # before chdir into the upstream checkout (never write there)
    root = Path.cwd().resolve()
    from nerva.sim.open_duck import OPEN_DUCK_ROOT
    from nerva.training.neutral_reference import NeutralReference, verified_references
    records, _ = verified_references(root)
    reference = NeutralReference(records)
    symmetry = reference_symmetry(reference)
    native = native_rows(root)
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    rows = native + mjx_rollouts(root, reference)
    for arm, folder in DELAY_ARMS.items():
        for delays in (True, False):
            if arm == "base_candidate" and delays:
                continue  # identical to part 1
            rows += mjx_rollouts(root, reference, folder, delays, f"mjx_{arm}_{'delays' if delays else 'no_delays'}")
    summary = {"reference_symmetry": symmetry, "turns": {}}
    for source in sorted({r["source"] for r in rows}):
        for name in TURNS:
            sel = [r for r in rows if r["source"] == source and r["command"] == name]
            summary["turns"][f"{source}_{name}"] = {
                "horizontal_rms": [round(r["horizontal_rms"], 4) for r in sel],
                "mean_yaw_rate": round(float(np.mean([r["mean_yaw_rate"] for r in sel])), 4),
                "pivot_forward_left_m": np.mean([r["pivot_forward_left_m"] for r in sel], axis=0).round(4).tolist()}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "rollouts.json", rows)
    write_json(args.out / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
