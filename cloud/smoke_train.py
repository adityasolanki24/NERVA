"""Run a short upstream PPO pass without changing the scientific baseline.

The upstream runner hard-codes 15 evaluations. That is appropriate for a full
training run, but makes pipeline checks and hardware comparisons needlessly
expensive. This wrapper changes only ``num_evals`` and ``num_timesteps`` and
records the effective (batch-rounded) throughput from checkpoint names.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path


SMOKE_TIMESTEPS = 200_000
SMOKE_EVALS = 2  # initial and final checkpoint/export
_CHECKPOINT_RE = re.compile(r"^(\d{4}_\d{2}_\d{2}_\d{6})_(\d+)$")


def checkpoint_measurement(output_dir: Path) -> dict[str, object]:
    """Measure effective steps/s between first and last checkpoint."""
    checkpoints: list[tuple[datetime, int, str]] = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            if not path.is_dir():
                continue
            match = _CHECKPOINT_RE.fullmatch(path.name)
            if match:
                checkpoints.append((
                    datetime.strptime(match.group(1), "%Y_%m_%d_%H%M%S"),
                    int(match.group(2)),
                    path.name,
                ))
    checkpoints.sort(key=lambda item: (item[1], item[0]))
    result: dict[str, object] = {
        "checkpoints": [item[2] for item in checkpoints],
        "effective_timesteps": checkpoints[-1][1] if checkpoints else None,
        "training_interval_seconds": None,
        "training_steps_per_second": None,
    }
    if len(checkpoints) >= 2:
        seconds = (checkpoints[-1][0] - checkpoints[0][0]).total_seconds()
        steps = checkpoints[-1][1] - checkpoints[0][1]
        result["training_interval_seconds"] = seconds
        result["training_steps_per_second"] = steps / seconds if seconds > 0 else None
    return result


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, i


def _proto_fields(buf: bytes):
    """Minimal protobuf wire-format reader: yields (field_number, value) pairs."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        elif wire == 2:
            length, i = _varint(buf, i)
            value, i = buf[i:i + length], i + length
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield number, value


def training_timing_from_events(output_dir: Path) -> list[tuple[int, float]]:
    """(env step, cumulative Brax training/walltime) at each evaluation after the first.

    Read from the TensorBoard events file that upstream's runner writes. Brax's
    training/walltime counts only training epochs (no evaluation or export), but
    its first chunk includes compiling the training step.
    """
    import struct

    points: list[tuple[int, float]] = []
    for path in sorted(output_dir.glob("events.out.tfevents*")) if output_dir.exists() else []:
        data, i = path.read_bytes(), 0
        while i + 12 <= len(data):
            length = struct.unpack("<Q", data[i:i + 8])[0]
            record, i = data[i + 12:i + 12 + length], i + 12 + length + 4
            step, walltime = 0, None
            for number, value in _proto_fields(record):
                if number == 2:
                    step = value
                elif number == 5:
                    for n2, v2 in _proto_fields(value):
                        if n2 != 1:
                            continue
                        tag = simple = None
                        for n3, v3 in _proto_fields(v2):
                            if n3 == 1:
                                tag = v3.decode("utf-8", "replace")
                            elif n3 == 2:
                                simple = struct.unpack("<f", v3)[0]
                        if tag == "training/walltime":
                            walltime = float(simple)
            if walltime is not None:
                points.append((int(step), walltime))
    return sorted(points)


def steady_state_throughput(points: list[tuple[int, float]]) -> dict[str, object]:
    """Steps/s between the last two training chunks: excludes compilation, eval and export."""
    result: dict[str, object] = {"training_walltime_points": points,
                                 "steady_training_steps_per_second": None}
    if len(points) >= 2:
        (s0, w0), (s1, w1) = points[-2], points[-1]
        if w1 > w0:
            result["steady_training_steps_per_second"] = (s1 - s0) / (w1 - w0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=SMOKE_TIMESTEPS)
    parser.add_argument("--evals", type=int, default=SMOKE_EVALS)
    parser.add_argument("--task", default="flat_terrain")
    parser.add_argument("--output-dir", default="/work/out/checkpoints")
    parser.add_argument("--summary-json")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.evals < 2:
        parser.error("--timesteps must be positive and --evals must be at least 2")

    # Kept inside main so checkpoint parsing can be unit-tested without the
    # heavyweight upstream/JAX environment installed.
    from argparse import Namespace
    from mujoco_playground.config import locomotion_params
    from playground.open_duck_mini_v2.runner import OpenDuckMiniV2Runner

    original_config = locomotion_params.brax_ppo_config

    def short_config(env_name: str):
        config = original_config(env_name)
        config.num_evals = args.evals
        return config

    locomotion_params.brax_ppo_config = short_config
    runner_args = Namespace(
        output_dir=args.output_dir,
        num_timesteps=args.timesteps,
        env="joystick",
        task=args.task,
        restore_checkpoint_path=None,
    )
    started = time.time()
    try:
        OpenDuckMiniV2Runner(runner_args).train()
    finally:
        locomotion_params.brax_ppo_config = original_config

    summary = {
        "requested_timesteps": args.timesteps,
        "num_evals": args.evals,
        "task": args.task,
        "wall_seconds": time.time() - started,
        **checkpoint_measurement(Path(args.output_dir)),
        **steady_state_throughput(training_timing_from_events(Path(args.output_dir))),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print("NERVA_SHORT_RUN_SUMMARY=" + rendered, flush=True)
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
