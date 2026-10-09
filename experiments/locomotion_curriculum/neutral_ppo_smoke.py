"""One capped CPU PPO/parameter warm-start check, never a locomotion efficacy run."""
import argparse
import functools
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from nerva.sim.open_duck import OPEN_DUCK_ROOT
from nerva.training.parameter_checkpoint import (
    checkpoint_hashes, count_value, leaf_comparison, require_contract, save_parameters, tree_finite, write_contract,
)

PPO_CONFIG = dict(num_timesteps=16, num_envs=2, batch_size=2, unroll_length=4, num_minibatches=1,
                  num_updates_per_batch=1, num_evals=1, num_eval_envs=2, episode_length=8, action_repeat=1,
                  run_evals=False, seed=7, normalize_observations=True, learning_rate=.0001, entropy_cost=.0001,
                  discounting=.9, reward_scaling=1., clipping_epsilon=.3, gae_lambda=.95,
                  vf_loss_coefficient=.5, normalize_advantage=True, restore_value_fn=True)
NETWORK_CONFIG = dict(policy_hidden_layer_sizes=(32, 32), value_hidden_layer_sizes=(32, 32),
                      policy_obs_key="state", value_obs_key="privileged_state")


def probe_observations(normalizer):
    return [{key: np.full_like(value, constant) for key, value in normalizer.mean.items()}
            for constant in (0., .1, -.1)] + [
        {key: np.linspace(-.2, .2, value.size).reshape(value.shape).astype(value.dtype)
         for key, value in normalizer.mean.items()}, normalizer.mean]


def action_parity(make_policy, params, loaded_policy):
    import jax
    maximum = 0.
    finite = True
    for deterministic in (True, False):
        before = jax.jit(make_policy(params, deterministic=deterministic))
        after = jax.jit(loaded_policy(deterministic))
        for observation in probe_observations(params[0]):
            key = jax.random.PRNGKey(19)
            x, _ = before(observation, key)
            y, _ = after(observation, key)
            x, y = np.asarray(x), np.asarray(y)
            finite &= bool(np.isfinite(x).all() and np.isfinite(y).all() and np.abs(x).max() <= 1
                           and np.abs(y).max() <= 1)
            maximum = max(maximum, float(np.max(np.abs(x - y))))
    return {"probes": 10, "finite_bounded": finite, "max_action_error": maximum,
            "all_pass": finite and maximum <= 1e-6}


def run(root, raw, output, *, normalization_eps=0., preregistration_commit="4243988", stage_observer=None,
        normalize_observations=True, stage_validator=None, initial_validator=None):
    import jax
    from brax.training import checkpoint as common_checkpoint
    from brax.training.agents.ppo import checkpoint, networks, train
    from nerva import motor_contract
    from nerva.training import neutral_joystick, neutral_reference, neutral_wrapper, parameter_checkpoint
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("CPU smoke requires CPU devices only")
    sources = {module.__name__: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
               for module in (train, checkpoint, common_checkpoint, networks, motor_contract,
                              neutral_joystick, neutral_reference, neutral_wrapper, parameter_checkpoint)}
    records, manifest = neutral_reference.verified_references(root)
    options = dict(PPO_CONFIG)
    options["normalize_observations"] = normalize_observations
    if normalization_eps:
        options["normalize_observations_std_eps"] = normalization_eps
    contract = {"contract": neutral_reference.CONTRACT, "observation_size": {"state": [101], "privileged_state": [212]},
                "action_size": 14, "network": NETWORK_CONFIG, "normalize_observations": normalize_observations,
                "preprocessing": "running_statistics" if normalize_observations else "identity_statistics_unused",
                "normalization_mode": "welford", "normalization_variance_eps": normalization_eps,
                "references": [row["reference_sha256"] for row in manifest["references"]],
                "source_hashes": sources, "restoration": "parameters_only_optimizer_rng_and_counters_restart"}
    protocol = {"preregistration_commit": preregistration_commit,
                "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "config": options, "network": NETWORK_CONFIG, "source_hashes": sources,
                "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {key: importlib.metadata.version(key) for key in
                             ("brax", "jax", "mujoco", "numpy", "flax", "orbax-checkpoint")},
                "python_version": sys.version.split()[0],
                "checkpoint_storage": "plain_state_dict_host_numpy_default_none_initializers_omitted",
                "wall_cap_s": 900, "max_optimization_transitions": 32, "cloud_work": False,
                "normalization_mode": "welford", "normalize_observations_std_eps": normalization_eps,
                "preprocessing": "running_statistics" if normalize_observations else "identity_statistics_unused",
                "num_resets_per_eval": 0, "use_pmap_on_reset": True, "max_grad_norm": None,
                "learning_rate_schedule": None, "domain_randomization": False}
    write_json(output / "protocol.json", protocol)
    write_json(output / "reference_manifest.json", manifest)
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    reference = neutral_reference.NeutralReference(records)
    env = neutral_joystick.NeutralJoystick(reference)
    network = functools.partial(networks.make_ppo_networks, **NETWORK_CONFIG)
    stages = []
    total_start = time.monotonic()
    fresh_params = fresh_path = None
    for name in ("fresh", "warm"):
        initial, steps = [], []
        stage_start = time.monotonic()

        def callback(step, make_policy, params, *, stage_name=name, stage_steps=steps,
                     stage_initial=initial, expected=fresh_params):
            del make_policy
            if not tree_finite(params):
                raise ValueError("nonfinite PPO parameters")
            stage_steps.append(int(step))
            if step == 0:
                stage_initial.append(params)
                if stage_name == "warm" and not leaf_comparison(expected, params)["equal"]:
                    raise ValueError("warm initialization differs from checkpoint")
                if initial_validator is not None:
                    initial_validator(stage_name, params)
            print(stage_name, "callback", int(step), flush=True)
            write_json(output / "progress.json", {"stage": stage_name, "callback_steps": stage_steps,
                                                 "wall_seconds": round(time.monotonic() - total_start, 2)})

        def progress(step, metrics, stage_name=name):
            if not tree_finite(metrics):
                raise ValueError("nonfinite PPO metrics")
            print(stage_name, "metrics at", int(step), flush=True)

        stage_options = dict(options)
        if name == "warm":
            require_contract(fresh_path, contract)
            stage_options["restore_checkpoint_path"] = str(fresh_path)
        make_policy, params, metrics = train.train(
            environment=env, network_factory=network, wrap_env_fn=neutral_wrapper.wrap_neutral_for_training,
            policy_params_fn=callback, progress_fn=progress, **stage_options)
        if not initial:
            raise ValueError("missing initial PPO callback")
        policy_change = leaf_comparison(initial[0][1], params[1])
        value_change = leaf_comparison(initial[0][2], params[2])
        count = count_value(params[0].count)
        criteria = {"steps": steps == [0, 16], "finite": bool(tree_finite(params) and tree_finite(metrics)),
                    "policy_updates": policy_change["max_error"] is not None and policy_change["max_error"] > 1e-12,
                    "value_updates": value_change["max_error"] is not None and value_change["max_error"] > 1e-12,
                    "normalizer_updates": count > count_value(initial[0][0].count)}
        if not all(criteria.values()):
            write_json(output / "stage_failure.json", {"stage": name, "criteria": criteria,
                                                      "callback_steps": steps, "normalizer_count": count})
            raise ValueError("PPO stage criteria failed")
        location = raw / "checkpoints" / name
        save_parameters(location, 16, params, checkpoint.network_config(
            observation_size={"state": (101,), "privileged_state": (212,)}, action_size=14,
            normalize_observations=normalize_observations, network_factory=network))
        path = location / "000000000016"
        write_contract(path, contract)
        require_contract(path, contract)
        restored = checkpoint.load(path)
        equality = leaf_comparison(params, restored)
        parity = action_parity(make_policy, params,
                               lambda deterministic, location=path: checkpoint.load_policy(location, deterministic=deterministic))
        criteria.update(roundtrip=equality["equal"] and equality["finite"], action_parity=parity["all_pass"])
        row = {"stage": name, "criteria": criteria, "all_pass": all(criteria.values()), "callback_steps": steps,
               "normalizer_count": count, "policy_max_update": policy_change["max_error"],
               "value_max_update": value_change["max_error"], "roundtrip": equality, "action_parity": parity,
               "metrics": {key: float(value) for key, value in metrics.items()},
               "checkpoint_hashes": checkpoint_hashes(path), "wall_seconds": round(time.monotonic() - stage_start, 2)}
        if stage_observer is not None:
            row["normalization_diagnostics"] = stage_observer(name, params)
        stages.append(row)
        write_json(output / "stages.json", stages)
        if not row["all_pass"]:
            raise ValueError("checkpoint criteria failed")
        if stage_validator is not None:
            stage_validator(row)
        if name == "fresh":
            fresh_params, fresh_path = params, path
        print(name, "pass", row["all_pass"], flush=True)
    report = {"all_pass": len(stages) == 2 and all(stage["all_pass"] for stage in stages),
              "optimization_transitions": 32, "wall_seconds": round(time.monotonic() - total_start, 2),
              "warm_initialization_equal": True, "optimizer_rng_and_counters_resumed": False,
              "learned_motor_readiness": False, "cloud_work": False}
    write_json(output / "summary.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_neutral_ppo"))
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/cloud_runs/neutral-ppo-smoke"))
    args = parser.parse_args()
    root, output, raw = Path.cwd(), args.out.resolve(), args.raw_dir.resolve()
    if not args.worker:
        if output.exists() or raw.exists():
            raise FileExistsError("never overwrite prior PPO artifacts")
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise RuntimeError("commit implementation before evaluating")
        output.mkdir(parents=True)
        raw.mkdir(parents=True)
        environment = {**os.environ, "JAX_PLATFORMS": "cpu"}
        start = time.monotonic()
        try:
            with (raw / "smoke.log").open("w", encoding="utf-8") as stream:
                subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.neutral_ppo_smoke", "--worker",
                                "--out", str(output), "--raw-dir", str(raw)], stdout=stream,
                               stderr=subprocess.STDOUT, check=True, env=environment, timeout=900)
        except subprocess.SubprocessError as error:
            write_json(output / "abort.json", {"all_pass": False, "error_type": type(error).__name__,
                                               "wall_seconds": round(time.monotonic() - start, 2), "wall_cap_s": 900})
            raise
        write_json(output / "parent_completion.json", {"wall_seconds": round(time.monotonic() - start, 2),
                                                       "wall_cap_s": 900, "worker_returncode": 0})
        print("PPO/restore smoke finished; inspect public results", flush=True)
        return
    try:
        run(root, raw, output)
    except Exception as error:
        write_json(output / "worker_abort.json", {"all_pass": False, "error_type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
