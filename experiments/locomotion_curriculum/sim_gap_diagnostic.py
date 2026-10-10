"""Diagnostic: does a neutral policy track the same commands in its MJX training environment as in the native
MuJoCo evaluation? (After the GPU pilot, docs/gpu_neutral_pilot.md; development log 2026-10-10.)

For each arm (local pilot candidate, GPU candidate) and each of the seven commands: deterministic actions,
seeds 0/1/2, 1,000 control steps in the training environment (PersistentNeutralJoystick, backlash scene,
training noise), with random pushes on (as trained) and off (as evaluated natively). Reports the mean
body-frame velocity (vx, vy) and gyro yaw rate over 5–20 s, plus true terminations, and compares with the
native evaluation's mean velocities from results_gpu_pilot/evaluation.json.

Reading (fixed before running): if MJX without pushes tracks a command that the native run misses
(e.g. right ≥ 0.037 m/s in MJX but 0.021 natively), the failure is a sim-to-sim gap, not missing learning;
if MJX also misses it, the policy itself does not track it. Diagnostic only: no criteria, no training.

Usage: python -m experiments.locomotion_curriculum.sim_gap_diagnostic [--out DIR]
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np

from nerva.training.b2_warm_start import NETWORK, SHAPE, balanced_environment
from nerva.training.motor_artifacts import write_json
from nerva.training.neutral_reference import COMMANDS

NAMES = ("rest", "forward", "backward", "left", "right", "turn_left", "turn_right")
ARMS = {"pilot_candidate": "experiments/cloud_runs/neutral-learning-pilot-corrected/checkpoints/000000028672",
        "gpu_candidate": "experiments/cloud_runs/neutral_gpu_pilot-20261010-152812/checkpoints/000060318720"}
STEPS, START = 1000, 250  # score 5–20 s


def rollout(env_factory, folder, seed, pushes):
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks
    params = checkpoint.load(folder)
    net = functools.partial(networks.make_ppo_networks, **NETWORK)(
        SHAPE, 14, preprocess_observations_fn=running_statistics.normalize)
    policy = networks.make_inference_fn(net)(params, deterministic=True)
    env = env_factory(pushes)
    inner = env.env.env.env  # AutoReset → Episode → BalancedVmap → PersistentNeutralJoystick
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(seed), 7))

    def step(carry, _):
        st, key = carry
        key, k = jax.random.split(key)
        action, _ = policy(st.obs, k)
        st = env.step(st, action)
        lin = jax.vmap(inner.get_local_linvel)(st.data)[:, :2]
        yaw = jax.vmap(inner.get_gyro)(st.data)[:, 2]
        return (st, key), (jp.concatenate([lin, yaw[:, None]], axis=1), st.done)

    (_, _), (vel, done) = jax.jit(lambda s: jax.lax.scan(step, (s, jax.random.PRNGKey(100 + seed)), None, STEPS))(state)
    vel, done = np.asarray(vel), np.asarray(done)
    return vel[START:].mean(axis=0), done.sum(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_sim_gap"))
    args = ap.parse_args()
    root = Path.cwd().resolve()
    from nerva.sim.open_duck import OPEN_DUCK_ROOT
    from nerva.training.neutral_reference import NeutralReference, verified_references
    records, _ = verified_references(root)
    reference = NeutralReference(records)
    native = json.loads((root / "experiments/locomotion_curriculum/results_gpu_pilot/evaluation.json").read_text())
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")

    def env_factory(pushes):
        env = balanced_environment(reference, episode_length=STEPS + 10, replicas=1, persistent_command=True)
        env.env.env.env._config.push_config.enable = bool(pushes)
        return env

    rows = []
    for arm, folder in ARMS.items():
        for pushes in (True, False):
            for seed in (0, 1, 2):
                vel, terminations = rollout(env_factory, root / folder, seed, pushes)
                for i, name in enumerate(NAMES):
                    rows.append({"arm": arm, "pushes": pushes, "seed": seed, "command": name,
                                 "mean_vx_vy_yaw": vel[i].tolist(), "terminations": int(terminations[i])})
                print(arm, "pushes" if pushes else "no pushes", "seed", seed, flush=True)
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for i, name in enumerate(NAMES):
            entry = {"command": COMMANDS[i]}
            for pushes in (True, False):
                vs = np.array([r["mean_vx_vy_yaw"] for r in rows if r["arm"] == arm and r["pushes"] == pushes
                               and r["command"] == name])
                entry["mjx_pushes" if pushes else "mjx_no_pushes"] = vs.mean(axis=0).round(4).tolist()
                entry["terminations_" + ("pushes" if pushes else "no_pushes")] = sum(
                    r["terminations"] for r in rows if r["arm"] == arm and r["pushes"] == pushes and r["command"] == name)
            nat = np.array([r["metrics"]["mean_velocity"] for r in native if r["arm"] == arm and r["command"] == name])
            entry["native_mujoco"] = nat.mean(axis=0).round(4).tolist()
            summary[arm][name] = entry
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "rollouts.json", rows)
    write_json(args.out / "summary.json", summary)
    for arm, cmds in summary.items():
        print("==", arm)
        for name, e in cmds.items():
            print(f"  {name:10s} cmd {e['command']}  MJX+push {e['mjx_pushes']}  MJX {e['mjx_no_pushes']}  native {e['native_mujoco']}"
                  f"  term {e['terminations_pushes']}/{e['terminations_no_pushes']}")


if __name__ == "__main__":
    main()
