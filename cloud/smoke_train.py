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
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print("NERVA_SHORT_RUN_SUMMARY=" + rendered, flush=True)
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
