"""Fixed 80-step CPU MJX candidate plumbing smoke; no optimisation or paid work."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from nerva.motor_contract import NeutralPhaseClock
from nerva.sim.open_duck import OPEN_DUCK_ROOT
from nerva.training.neutral_reference import NeutralReference, verified_references
from nerva.training.reference_kinematics import sample_reference


def reference_parity(reference):
    import jax
    get = jax.jit(reference.get_reference_motion)
    rows = []
    for record in reference.records:
        expected = sample_reference(record["reference"], np.arange(27) * .02)
        targets = np.hstack([expected["joint_position"], expected["joint_velocity"], expected["contacts"],
                             expected["linear_body"], expected["angular_body"]])
        actual = np.array([get(*record["command"], i) for i in range(27)])
        error = np.max(np.abs(actual - targets), axis=0)
        rows.append({"command": record["command"], "max_position_error": float(error[:16].max()),
                     "max_velocity_error": float(error[16:32].max()), "max_other_error": float(error[32:].max()),
                     "pass": bool(error[:16].max() <= 1e-5 and error[16:32].max() <= 1e-4
                                  and error[32:].max() <= 1e-5)})
    return rows


def run(root):
    import jax
    import jax.numpy as jp
    from nerva.training.neutral_joystick import NeutralJoystick
    records, manifest = verified_references(root)
    reference = NeutralReference(records)
    parity = reference_parity(reference)
    if not all(r["pass"] for r in parity):
        raise ValueError("target parity failed before dynamics")
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    env = NeutralJoystick(reference)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    # Reward components are sampled on actual post-step states; first-contact is unused.
    reward = jax.jit(lambda state: env._get_reward(state.data, jp.zeros(14), state.info, {}, state.done,
                                                 jp.zeros(2), state.info["last_contact"]))
    rows = []
    blocks = [[record["command"]] for record in records] + [[[0., 0., 0.], [.074, 0., 0.], [0., 0., 0.]]]
    for trial, commands in enumerate(blocks):
        state = reset(jax.random.PRNGKey(7))
        clock = NeutralPhaseClock(27)
        if state.obs["state"].shape != (101,) or state.obs["privileged_state"].shape != (212,):
            raise ValueError("candidate observation schema mismatch")
        for block, command in enumerate(commands):
            state.info["command"] = jp.asarray([*command, 0, 0, 0, 0], dtype=jp.float32)
            for tick in range(8):
                state = step(state, jp.zeros(14))
                raw_rewards = {k: float(v) for k, v in reward(state).items()}
                runtime = clock.tick(np.array([*command, 0, 0, 0, 0], dtype=np.float32))
                phase_error = float(np.max(np.abs(np.asarray(state.obs["state"][-2:]) - runtime["phase"])))
                target = reference.get_reference_motion(*command, runtime["index"])
                reference_error = float(np.max(np.abs(np.asarray(target) - np.asarray(state.info["current_reference_motion"]))))
                finite = all(np.isfinite(np.asarray(value)).all() for value in
                             (state.data.qpos, state.data.qvel, state.obs["state"], state.obs["privileged_state"],
                              state.reward, list(raw_rewards.values())))
                criteria = {"finite": finite, "phase_parity": phase_error <= 1e-6,
                            "index_parity": int(state.info["motor_phase_index"]) == runtime["index"],
                            "mode_parity": bool(state.info["motor_previous_rest"]) == runtime["rest"],
                            "reference_selection": reference_error <= 1e-5,
                            "command_observation": bool(np.allclose(state.obs["state"][6:13], runtime["command"], atol=1e-7))}
                row = {"trial": trial, "block": block, "tick": tick, "command": list(command),
                       "criteria": criteria, "all_pass": all(criteria.values()), "phase_error": phase_error,
                       "reference_error": reference_error, "terminated": bool(state.done), "reward": float(state.reward),
                       "raw_rewards": raw_rewards, "weighted_rewards": {k: v * env._config.reward_config.scales[k]
                                                                          for k, v in raw_rewards.items()}}
                rows.append(row)
                if not row["all_pass"]:
                    return {"all_pass": False, "rows": rows, "target_parity": parity, "manifest": manifest}
        print("trial", trial, "finite", True, flush=True)
    return {"all_pass": len(rows) == 80 and all(r["all_pass"] for r in rows), "rows": rows,
            "target_parity": parity, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_neutral_smoke"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, output = Path.cwd(), args.out.resolve()
    if output.exists():
        raise FileExistsError("never overwrite smoke results")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before evaluation")
    if not args.worker:
        raw = root / "experiments/cloud_runs/neutral-motor-smoke"
        raw.mkdir(exist_ok=False)
        try:
            with (raw / "smoke.log").open("w", encoding="utf-8") as stream:
                subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.neutral_smoke", "--worker",
                                "--out", str(output)], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
        except subprocess.SubprocessError as error:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "abort.json", {"all_pass": False, "error_type": type(error).__name__, "cap_s": 600})
            raise
        print("Smoke complete; see public report", flush=True)
        return
    implementation = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    start = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        report = run(root)
    output.mkdir(parents=True)
    rows = report.pop("rows")
    manifest = report.pop("manifest")
    report.update(steps=len(rows), terminations=sum(r["terminated"] for r in rows),
                  zero_reward_fraction=sum(r["reward"] == 0 for r in rows) / len(rows),
                  wall_seconds=round(time.monotonic() - start, 2), learned_readiness=False)
    write_json(output / "trials.json", rows)
    write_json(output / "summary.json", report)
    write_json(output / "reference_manifest.json", manifest)
    write_json(output / "protocol.json", {"preregistration_commit": "f92e19e", "implementation_commit": implementation,
                                          "seed": 7, "steps": 80, "wall_cap_s": 600, "cloud_work": False,
                                          "optimisation": False, "policy_observation_size": 101})


if __name__ == "__main__":
    main()
