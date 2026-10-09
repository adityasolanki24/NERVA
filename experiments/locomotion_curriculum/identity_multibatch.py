"""Capped warm-parameter on-policy batches with deployment checks after every SGD."""
import argparse
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from experiments.locomotion_curriculum.identity_preprocessing import preprocessing_invariance
from experiments.locomotion_curriculum.neutral_ppo_smoke import NETWORK_CONFIG, PPO_CONFIG, action_parity
from experiments.locomotion_curriculum.normalization_timing import (
    archive, first_batch_keys, gaussian_kl, gaussian_log_prob, make_collector, make_replay,
)
from experiments.locomotion_curriculum.normalization_timing_offline import admit_inputs
from nerva.training.parameter_checkpoint import (
    checkpoint_hashes, count_value, leaf_comparison, require_contract, save_parameters, tree_finite, write_contract,
)

PREREGISTRATION = "8c9c16b"


def update_screen(pre, post, control_errors):
    return {"control_logits": control_errors["logits"] <= 1e-5,
            "control_log_prob": control_errors["log_prob"] <= 1e-4,
            "control_kl": pre["kl_mean"] <= .001,
            "deployed_kl": post["kl_mean"] <= 1, "deployed_value_loss": post["v_loss"] <= 100}


def run(root, raw, output):
    import jax
    import optax
    from brax.training import types
    from brax.training.agents.ppo import checkpoint, losses, networks, train
    from nerva.sim.open_duck import OPEN_DUCK_ROOT
    from nerva.training.neutral_reference import CONTRACT, NeutralReference, verified_references
    from nerva.training.neutral_joystick import NeutralJoystick
    from nerva.training.neutral_wrapper import wrap_neutral_for_training

    _, _, _, _, admission = admit_inputs(root)
    prior = root / "experiments/locomotion_curriculum/results_identity_preprocessing"
    protocol = json.loads((prior / "protocol.json").read_text(encoding="utf-8"))
    stage = json.loads((prior / "stages.json").read_text(encoding="utf-8"))[-1]
    records, manifest = verified_references(root)
    expected_config = {**PPO_CONFIG, "normalize_observations": False, "normalize_observations_std_eps": .0001}
    if (protocol["config"] != expected_config
            or any(admission["source_hashes"][name] != digest for name, digest in protocol["source_hashes"].items())
            or manifest != json.loads((prior / "reference_manifest.json").read_text(encoding="utf-8"))
            or not json.loads((prior / "comparison.json").read_text(encoding="utf-8"))["all_pass"]):
        raise ValueError("identity source/config/reference/screen admission failed")
    location = root / "experiments/cloud_runs/identity-preprocessing/checkpoints/warm/000000000016"
    expected_contract = {"contract": CONTRACT, "observation_size": {"state": [101], "privileged_state": [212]},
                         "action_size": 14, "network": NETWORK_CONFIG, "normalize_observations": False,
                         "preprocessing": "identity_statistics_unused", "normalization_mode": "welford",
                         "normalization_variance_eps": .0001,
                         "references": [row["reference_sha256"] for row in manifest["references"]],
                         "source_hashes": protocol["source_hashes"],
                         "restoration": "parameters_only_optimizer_rng_and_counters_restart"}
    require_contract(location, expected_contract)
    if checkpoint_hashes(location) != stage["checkpoint_hashes"]:
        raise ValueError("identity checkpoint hash mismatch")
    params = checkpoint.load(location)
    if not tree_finite(params) or count_value(params[0].count) != 32:
        raise ValueError("initial checkpoint finite/count failure")
    factory = functools.partial(networks.make_ppo_networks, **NETWORK_CONFIG)
    shape = {"state": (101,), "privileged_state": (212,)}
    net = factory(shape, 14, preprocess_observations_fn=types.identity_observation_preprocessor)
    make_policy = networks.make_inference_fn(net)
    replay = make_replay(net)
    loss_options = {name: PPO_CONFIG[name] for name in
                    ("entropy_cost", "discounting", "reward_scaling", "gae_lambda", "clipping_epsilon", "normalize_advantage")}
    loss_fn = functools.partial(losses.compute_ppo_loss, ppo_network=net,
                                vf_coefficient=PPO_CONFIG["vf_loss_coefficient"], **loss_options)
    optimizer = optax.adam(.0001)

    @jax.jit
    def optimize(weights, normalizer, data, key, state):
        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(weights, normalizer, data, key)
        updates, state = optimizer.update(gradients, state)
        return optax.apply_updates(weights, updates), state, gradients, metrics

    weights = losses.PPONetworkParams(policy=params[1], value=params[2])
    optimizer_state = optimizer.init(weights)
    normalizer = params[0]
    keys = first_batch_keys(19)
    write_json(output / "protocol.json", {"preregistration_commit": PREREGISTRATION,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(inspect.getfile(make_collector)).read_bytes()).hexdigest(),
        "admission": admission, "input_checkpoint_hashes": checkpoint_hashes(location),
        "seed": 19, "num_envs": 2, "unroll_length": 4, "episode_length": 32,
        "max_new_transitions": 128, "max_optimizer_updates": 16, "normalizer_initial_count_unused": 32,
        "network": NETWORK_CONFIG, "ppo_loss_options": loss_options, "learning_rate": .0001,
        "preprocessing": "identity_statistics_unused", "warm_start": "parameters_only_fresh_adam_rng_environment",
        "wall_cap_s": 900, "cloud_work": False})
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    env = wrap_neutral_for_training(NeutralJoystick(NeutralReference(records)), episode_length=32, action_repeat=1)
    state = train._strip_weak_type(jax.pmap(env.reset, axis_name=train._PMAP_AXIS_NAME)(keys["reset"]))
    epoch_key = keys["epoch"]
    collect = make_collector(env, make_policy, train._PMAP_AXIS_NAME, return_state=True, extra_fields=("command",))
    rows = []
    for index in range(16):
        epoch = epoch_key[0]
        key_sgd, _, _ = jax.random.split(epoch, 3)
        _, permutation_key, gradient_key = jax.random.split(key_sgd, 3)
        _, loss_key = jax.random.split(gradient_key)
        data, next_normalizer, state, epoch_key = collect(jax.device_put_replicated(
            (normalizer, weights.policy, weights.value), jax.local_devices()), state, epoch_key)
        data, next_normalizer = jax.tree.map(lambda x: x[0], (data, next_normalizer))
        if (not tree_finite((data, next_normalizer)) or data.discount.shape != (2, 4)
                or count_value(next_normalizer.count) != 32 + 8 * (index + 1)):
            raise ValueError("collection finite/shape/count failure")
        batch = jax.tree.map(lambda x, key=permutation_key: jax.random.permutation(key, x), data)
        captured = archive(raw / f"batch_{index:02d}.npz", batch)
        pre = replay(normalizer, weights, batch, loss_key)
        behavior_logits = np.swapaxes(np.asarray(batch.extras["policy_extras"]["distribution_params"]), 0, 1)
        behavior_lp = np.swapaxes(np.asarray(batch.extras["policy_extras"]["log_prob"]), 0, 1)
        errors = {"logits": float(np.max(np.abs(np.asarray(pre["logits"]) - behavior_logits))),
                  "log_prob": float(np.max(np.abs(np.asarray(pre["log_prob"]) - behavior_lp)))}
        pre_metrics = {key: float(x) for key, x in pre["metrics"].items()}
        controls = update_screen(pre_metrics, {"kl_mean": 0., "v_loss": 0.}, errors)
        if not tree_finite(pre) or not all(controls.values()):
            write_json(output / "control_failure.json", {"batch": index, "errors": errors, "metrics": pre_metrics})
            raise ValueError("on-policy pre-update control failed")
        after, optimizer_state, gradients, _ = optimize(weights, normalizer, batch, loss_key, optimizer_state)
        if not tree_finite((after, optimizer_state, gradients)):
            raise ValueError("nonfinite SGD")
        changes = [leaf_comparison(a, b) for a, b in ((weights.policy, after.policy), (weights.value, after.value))]
        if not all(x["max_error"] is not None and x["max_error"] > 1e-12 for x in changes):
            raise ValueError("both networks must update")
        post = replay(next_normalizer, after, batch, loss_key)
        if not tree_finite(post):
            raise ValueError("nonfinite deployed replay")
        post_metrics = {key: float(x) for key, x in post["metrics"].items()}
        behavior = net.parametric_action_distribution.create_dist(behavior_logits)
        manual = gaussian_kl(behavior.loc, behavior.scale, post["loc"], post["scale"], 1e-5)
        manual_lp = gaussian_log_prob(post["loc"], post["scale"],
                                     np.swapaxes(np.asarray(batch.extras["policy_extras"]["raw_action"]), 0, 1))
        criteria = update_screen(pre_metrics, post_metrics, errors)
        criteria.update({"manual_kl": bool(np.isclose(np.mean(manual), post_metrics["kl_mean"], atol=1e-5, rtol=1e-5)),
                         "manual_log_prob": bool(np.allclose(manual_lp, np.asarray(post["log_prob"]), atol=1e-4, rtol=1e-5))})
        weights, normalizer = after, next_normalizer
        row = {"batch": index, "new_transitions": 8 * (index + 1), "optimizer_updates": index + 1,
               "criteria": criteria, "all_pass": all(criteria.values()), "pre": pre_metrics, "post": post_metrics,
               "control_errors": errors, "gradient_norm": float(optax.global_norm(gradients)),
               "normalizer_count_unused": count_value(normalizer.count), "reward_mean": float(np.mean(batch.reward)),
               "terminations": int(np.sum(1 - np.asarray(batch.discount))),
               "truncations": int(np.sum(np.asarray(batch.extras["state_extras"]["truncation"]))),
               "commands": np.asarray(batch.extras["state_extras"]["command"])[:, :, :3].tolist(),
               "raw_observation_max_abs": {key: float(np.max(np.abs(value))) for key, value in batch.observation.items()},
               "data_artifact": captured, "parameter_adam_artifact": archive(
                   raw / f"parameters_{index:02d}.npz", ((normalizer, weights.policy, weights.value), optimizer_state)),
               "deployed_artifact": archive(raw / f"deployed_{index:02d}.npz", post)}
        rows.append(row)
        write_json(output / "batches.json", rows)
        print("batch", index + 1, "post KL", post_metrics["kl_mean"], "pass", row["all_pass"], flush=True)
        if not row["all_pass"]:
            break
    completed = 8 * len(rows)
    deployment = (normalizer, weights.policy, weights.value)
    save_parameters(raw / "checkpoint", completed, deployment, checkpoint.network_config(shape, 14, False, factory))
    saved = raw / "checkpoint" / f"{completed:012d}"
    write_contract(saved, {**expected_contract, "purpose": "bounded_identity_multibatch"})
    require_contract(saved, {**expected_contract, "purpose": "bounded_identity_multibatch"})
    equality = leaf_comparison(deployment, checkpoint.load(saved))
    parity = action_parity(make_policy, deployment,
                           lambda deterministic: checkpoint.load_policy(saved, deterministic=deterministic))
    invariant = preprocessing_invariance(net, deployment)
    checkpoint_pass = equality["equal"] and equality["finite"] and parity["all_pass"] and invariant["all_pass"]
    write_json(output / "summary.json", {"all_pass": len(rows) == 16 and all(x["all_pass"] for x in rows) and checkpoint_pass,
        "completed_batches": len(rows), "new_environment_transitions": completed, "optimizer_updates": len(rows),
        "stopped_on_batch_screen": len(rows) < 16 or not rows[-1]["all_pass"],
        "checkpoint_pass": checkpoint_pass, "roundtrip": equality, "action_parity": parity, "invariance": invariant,
        "checkpoint_hashes": checkpoint_hashes(saved), "learned_motor_readiness": False,
        "optimizer_rng_environment_resumed": False, "cloud_work": False})
    if not checkpoint_pass:
        raise ValueError("checkpoint or preprocessing integrity failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.cwd()
    raw = root / "experiments/cloud_runs/identity-multibatch"
    output = root / "experiments/locomotion_curriculum/results_identity_multibatch"
    if args.worker:
        try:
            run(root, raw, output)
        except Exception as error:
            write_json(output / "worker_abort.json", {"error_type": type(error).__name__, "all_pass": False})
            raise
        return
    if raw.exists() or output.exists():
        raise FileExistsError("never overwrite stability artifacts")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before collection")
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        with (raw / "stability.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.identity_multibatch", "--worker"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
    except subprocess.SubprocessError as error:
        write_json(output / "abort.json", {"all_pass": False, "error_type": type(error).__name__,
                                          "wall_seconds": round(time.monotonic() - start, 2), "wall_cap_s": 900})
        raise
    write_json(output / "parent_completion.json", {"wall_seconds": round(time.monotonic() - start, 2),
                                                   "wall_cap_s": 900, "worker_returncode": 0})
    print("Bounded stability check finished; inspect summary.json", flush=True)


if __name__ == "__main__":
    main()
