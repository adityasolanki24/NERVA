"""Train a style-conditioned policy (S1) or its neutral control (B1). Run from Open_Duck_Playground/.

Mirrors upstream playground/common/runner.py (same Brax PPO config
"BerkeleyHumanoidJoystickFlatTerrain", same domain randomisation, same checkpoint + ONNX
export at every eval) but builds nerva.training.style_joystick.StyleJoystick.

  python -m nerva.training.train_style --refs DIR --output_dir OUT [--task flat_terrain_backlash]
      DIR contains styles.json: [{"style": [e1, e2, e3], "pickle": "file.pkl"}, ...]
  python -m nerva.training.train_style --neutral ...   one neutral style = upstream reference (B1)
  --smoke  tiny PPO settings, no ONNX export (CPU check that the env trains end to end)
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from playground.common import randomize

from nerva.training.style_joystick import StyledReference, StyleJoystick

UPSTREAM_PKL = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SMOKE_OVERRIDES = dict(num_envs=4, batch_size=4, num_minibatches=1, unroll_length=5,
                       num_updates_per_batch=1, num_evals=2, num_timesteps=200, episode_length=50)


def build_reference(args) -> StyledReference:
    if args.neutral:
        return StyledReference([UPSTREAM_PKL], [[0.0, 0.0, 0.0]])
    spec = json.loads((Path(args.refs) / "styles.json").read_text())
    return StyledReference([str(Path(args.refs) / s["pickle"]) for s in spec], [s["style"] for s in spec])


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--refs")
    g.add_argument("--neutral", action="store_true")
    p.add_argument("--task", default="flat_terrain")
    p.add_argument("--num_timesteps", type=int, default=150_000_000)
    p.add_argument("--output_dir", default="checkpoints_style")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--backward_fraction", type=float, default=0.0,
                   help="S4: probability of replacing the forward command with a backward one")
    p.add_argument("--apply_head_commands", action="store_true",
                   help="S3: add head commands to head motor targets, as the hardware runtime does")
    p.add_argument("--feet_height_scale", type=float, default=0.0,
                   help="S2 per-style feet-height cost weight (negative); 0 = off, as S1/B1")
    args = p.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = build_reference(args)
    env, eval_env = (StyleJoystick(ref, task=args.task, feet_height_scale=args.feet_height_scale,
                                   apply_head_commands=args.apply_head_commands,
                                   backward_fraction=args.backward_fraction) for _ in range(2))
    obs_size = int(env.observation_size["state"][0])
    print(f"styles={ref.n_styles} style_dim={ref.style_dim} obs={obs_size} periods={ref.periods} "
          f"feet_height_scale={args.feet_height_scale} apply_head_commands={args.apply_head_commands} "
          f"backward_fraction={args.backward_fraction}")

    params = dict(locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain"))  # as upstream
    params["num_timesteps"] = args.num_timesteps
    if args.smoke:
        params.update(SMOKE_OVERRIDES)
    net = params.pop("network_factory", None)
    network_factory = (functools.partial(ppo_networks.make_ppo_networks, **net) if net
                       else ppo_networks.make_ppo_networks)
    print("PPO params:", params)

    def progress(num_steps, metrics):
        print(f"STEP {num_steps} reward {metrics.get('eval/episode_reward')} "
              f"std {metrics.get('eval/episode_reward_std')}", flush=True)

    def save(current_step, make_policy, policy_params):
        if args.smoke:
            return
        from flax.training import orbax_utils
        from orbax import checkpoint as ocp
        from playground.common.export_onnx import export_onnx  # imports tensorflow

        stamp = f"{datetime.now():%Y_%m_%d_%H%M%S}_{current_step}"
        ocp.PyTreeCheckpointer().save(str(out / stamp), policy_params, force=True,
                                      save_args=orbax_utils.save_args_from_target(policy_params))
        export_onnx(policy_params, env.action_size, locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"), obs_size, output_path=str(out / f"{stamp}.onnx"))

    t0 = time.time()
    ppo.train(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training,
              network_factory=network_factory, randomization_fn=randomize.domain_randomize,
              progress_fn=progress, policy_params_fn=save, **params)
    print(f"done in {time.time() - t0:.0f} s on {jax.devices()}")


if __name__ == "__main__":
    main()
