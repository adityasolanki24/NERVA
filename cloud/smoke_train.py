"""Run a minimal upstream PPO pass without changing the scientific baseline.

The upstream runner hard-codes 15 evaluations.  That is appropriate for a full
training run, but each evaluation also exports ONNX and makes a short pipeline
smoke test unnecessarily long.  This wrapper changes only ``num_evals`` and
``num_timesteps`` for the disposable cloud smoke job.
"""

from argparse import Namespace

from mujoco_playground.config import locomotion_params
from playground.open_duck_mini_v2.runner import OpenDuckMiniV2Runner


SMOKE_TIMESTEPS = 200_000
SMOKE_EVALS = 2  # initial and final checkpoint/export


def main() -> None:
    original_config = locomotion_params.brax_ppo_config

    def smoke_config(env_name: str):
        config = original_config(env_name)
        config.num_evals = SMOKE_EVALS
        return config

    locomotion_params.brax_ppo_config = smoke_config
    args = Namespace(
        output_dir="/work/out/checkpoints",
        num_timesteps=SMOKE_TIMESTEPS,
        env="joystick",
        task="flat_terrain",
        restore_checkpoint_path=None,
    )
    OpenDuckMiniV2Runner(args).train()


if __name__ == "__main__":
    main()
