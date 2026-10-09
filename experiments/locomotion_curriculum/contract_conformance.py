"""Fixed NumPy/JAX observation-clock conformance; no dynamics or learning."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from nerva.motor_contract import NeutralPhaseClock, planar_tracking, training_motor_tick


def stream():
    rng = np.random.default_rng(123)
    commands = np.zeros((1000, 7), dtype=np.float32)
    commands[:, :2] = rng.uniform(-.15, .15, (1000, 2))
    commands[:, 2] = rng.uniform(-.60, .60, 1000)
    commands[::10] = 0
    fixed = np.array([[0, 0, 0, 0, 0, 0, 0], [.005, 0, .02, 0, 0, 0, 0],
                      [.006, 0, 0, 0, 0, 0, 0], [0, 0, .025, 0, 0, 0, 0], [.15, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, .4, 0], [0, 0, .6, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    return np.concatenate([commands, np.repeat(fixed, 10, axis=0)])


def evaluate(commands, period):
    import jax
    import jax.numpy as jp
    clock = NeutralPhaseClock(period)
    runtime = [clock.tick(command) for command in commands]
    initial = {"motor_phase_index": jp.int32(0), "motor_previous_rest": jp.bool_(True)}

    def tick(carry, command):
        output = training_motor_tick(carry, command, period)
        state = {key: output[key] for key in initial}
        return state, output

    _, training = jax.jit(lambda c: jax.lax.scan(tick, initial, c))(jp.array(commands))
    arrays = {name: np.asarray(value) for name, value in training.items()}
    indices = np.array([r["index"] for r in runtime])
    rest = np.array([r["rest"] for r in runtime])
    phases = np.array([r["phase"] for r in runtime])
    canonical = np.array([r["command"] for r in runtime])
    phase_error = float(np.max(np.abs(phases - arrays["imitation_phase"])))
    command_error = float(np.max(np.abs(canonical - arrays["command"])))
    previous_rest = np.r_[True, rest[:-1]]
    criteria = {"index_parity": bool(np.array_equal(indices, arrays["motor_phase_index"])),
                "mode_parity": bool(np.array_equal(rest, arrays["motor_previous_rest"])),
                "phase_parity": phase_error <= 1e-6, "command_parity": command_error <= 1e-7,
                "unit_phase": bool(np.max(np.abs(np.linalg.norm(phases, axis=1) - 1)) <= 1e-6
                                   and np.max(np.abs(np.linalg.norm(arrays["imitation_phase"], axis=1) - 1)) <= 1e-6),
                "rest_index_zero": bool(np.all(indices[rest] == 0)),
                "movement_onset_zero": bool(np.all(indices[previous_rest & ~rest] == 0)),
                "index_range": bool(np.all((indices >= 0) & (indices < period))),
                "finite": all(np.isfinite(value).all() for value in arrays.values())}
    return {"period_steps": period, "ticks": len(commands), "criteria": criteria,
            "all_pass": all(criteria.values()), "max_phase_error": phase_error,
            "max_command_error": command_error, "rest_ticks": int(rest.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_contract"))
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("never overwrite conformance results")
    if not args.worker:
        try:
            subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.contract_conformance",
                            "--worker", "--out", str(args.out)], check=True, timeout=120)
        except subprocess.TimeoutExpired:
            args.out.mkdir(parents=True, exist_ok=True)
            write_json(args.out / "timeout.json", {"all_pass": False, "reason": "120-second wall cap"})
            raise
        return
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit code before evaluating")
    start = time.monotonic()
    commands = stream()
    rows = [evaluate(commands, period) for period in (27, 20)]
    x = float(planar_tracking(np.zeros(7), [.05, 0, 0]))
    y = float(planar_tracking(np.zeros(7), [0, .05, 0]))
    symmetric = abs(x - y) <= 1e-6 and x < 1 and y < 1
    report = {"all_pass": all(r["all_pass"] for r in rows) and symmetric, "periods": rows,
              "symmetric_tracking": {"pass": symmetric, "vx_reward": x, "vy_reward": y, "rest_reward": 1.},
              "wall_seconds": round(time.monotonic() - start, 2), "motor_readiness": "not evaluated"}
    args.out.mkdir(parents=True)
    write_json(args.out / "report.json", report)
    write_json(args.out / "protocol.json", {"preregistration_commit": "009bb5c",
               "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "stream_sha256": hashlib.sha256(commands.tobytes()).hexdigest(), "seed": 123,
               "wall_cap_s": 120, "cloud_work": False, "dynamics": False, "training": False})
    print(report, flush=True)


if __name__ == "__main__":
    main()
