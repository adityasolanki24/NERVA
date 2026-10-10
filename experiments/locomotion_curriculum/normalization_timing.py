"""One frozen eight-transition batch, two normalization replays, zero SGD updates."""
import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from experiments.locomotion_curriculum.neutral_ppo_smoke import NETWORK_CONFIG, PPO_CONFIG
from nerva.training.parameter_checkpoint import (
    checkpoint_hashes, count_value, leaf_comparison, require_contract, save_parameters, tree_finite, write_contract,
)
from nerva.training.motor_artifacts import archive, fingerprint, gaussian_kl, tree_arrays  # noqa: F401,E402

PREREGISTRATION = "0901b11"
VARIANCE_EPS = .0001


def first_batch_keys(seed=7):
    """Pinned Brax seed/process/epoch schedule; no sampling or environment execution."""
    import jax
    global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))
    local_key = jax.random.fold_in(local_key, 0)
    local_key, key_env, _ = jax.random.split(local_key, 3)
    epoch_key, _ = jax.random.split(local_key)
    epoch_key = jax.random.split(epoch_key, 1)[0]
    key_sgd, _, _ = jax.random.split(epoch_key, 3)
    _, key_perm, key_grad = jax.random.split(key_sgd, 3)
    _, key_loss = jax.random.split(key_grad)
    key_policy, key_value = jax.random.split(global_key)
    return {"reset": jax.random.split(key_env, 2).reshape(1, 2, 2), "epoch": epoch_key.reshape(1, 2),
            "permutation": key_perm, "loss": key_loss, "policy_init": key_policy, "value_init": key_value}


def make_collector(env, make_policy, axis_name, *, return_state=False, extra_fields=()):
    import jax
    import jax.numpy as jp
    from brax.training import acting
    from brax.training.acme import running_statistics

    def capture(params, state, epoch_key):
        _, key_generate_unroll, next_epoch_key = jax.random.split(epoch_key, 3)
        policy = make_policy(params)

        def unroll(carry, unused):
            del unused
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env, current_state, policy, current_key, 4,
                extra_fields=("truncation", "episode_metrics", "episode_done", *extra_fields))
            return (next_state, next_key), data

        (next_state, _), data = jax.lax.scan(unroll, (state, key_generate_unroll), (), length=1)
        data = jax.tree.map(lambda x: jp.swapaxes(x, 1, 2), data)
        data = jax.tree.map(lambda x: jp.reshape(x, (-1,) + x.shape[2:]), data)
        updated = running_statistics.update(params[0], data.observation, pmap_axis_name=axis_name)
        return (data, updated, next_state, next_epoch_key) if return_state else (data, updated)

    return jax.pmap(capture, axis_name=axis_name)


def decide(integrity, old_kl, updated_kl):
    finite = bool(np.isfinite(old_kl) and np.isfinite(updated_kl))
    valid = all(integrity.values()) and finite and old_kl <= .001
    effect = finite and updated_kl > 1
    return {"integrity_pass": valid, "criteria": integrity, "old_statistics_kl": old_kl,
            "updated_statistics_kl": updated_kl, "timing_effect_exceeds_screen": effect,
            "hypothesis_supported": valid and effect, "environment_transitions": 8, "optimizer_updates": 0,
            "general_learning_stability_established": False, "learned_motor_readiness": False, "cloud_work": False}


def run(root, raw, output):
    import jax
    from brax.training import acting, distribution
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import losses, networks, train
    from nerva.sim.open_duck import OPEN_DUCK_ROOT
    from nerva.training.neutral_joystick import NeutralJoystick
    from nerva.training.neutral_reference import NeutralReference, verified_references
    from nerva.training.neutral_wrapper import wrap_neutral_for_training
    if (jax.process_count() != 1 or jax.local_device_count() != 1
            or jax.devices()[0].platform != "cpu"):
        raise RuntimeError("replay requires one CPU process/device")
    prior = root / "experiments/locomotion_curriculum/results_normalization_control"
    names = ("protocol.json", "comparison.json", "reference_manifest.json", "control_admission.json")
    prior_inputs = {name: json.loads((prior / name).read_text(encoding="utf-8")) for name in names}
    protocol = prior_inputs["protocol.json"]
    sources = {name: hashlib.sha256(Path(inspect.getfile(importlib.import_module(name))).read_bytes()).hexdigest()
               for name in protocol["source_hashes"]}
    versions = {name: importlib.metadata.version(name) for name in protocol["versions"]}
    records, manifest = verified_references(root)
    operator_hash = hashlib.sha256(Path(inspect.getfile(running_statistics)).read_bytes()).hexdigest()
    if (sources != protocol["source_hashes"] or versions != protocol["versions"]
            or protocol["python_version"] != sys.version.split()[0]
            or protocol["config"] != {**PPO_CONFIG, "normalize_observations_std_eps": VARIANCE_EPS}
            or protocol["network"] != json.loads(json.dumps(NETWORK_CONFIG))
            or manifest != prior_inputs["reference_manifest.json"]
            or operator_hash != prior_inputs["control_admission.json"]["normalization_source_sha256"]):
        raise ValueError("prior protocol/source/reference/operator mismatch")
    write_json(output / "protocol.json", {
        "preregistration_commit": PREREGISTRATION,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {**sources, **{module.__name__: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
                                       for module in (acting, distribution, losses, running_statistics)}},
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prior_report_hashes": {name: hashlib.sha256((prior / name).read_bytes()).hexdigest() for name in names},
        "versions": versions, "seed": 7, "network": NETWORK_CONFIG, "variance_eps": VARIANCE_EPS,
        "num_envs": 2, "unroll_length": 4, "episode_length": 8, "max_environment_transitions": 8,
        "optimizer_updates": 0, "initializer_requested_timesteps": 0, "wall_cap_s": 600, "cloud_work": False})
    write_json(output / "reference_manifest.json", manifest)
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    underlying = NeutralJoystick(NeutralReference(records))
    factory = functools.partial(networks.make_ppo_networks, **NETWORK_CONFIG)
    options = {**PPO_CONFIG, "num_timesteps": 0, "normalize_observations_std_eps": VARIANCE_EPS}
    make_policy, params, _ = train.train(environment=underlying, network_factory=factory,
                                        wrap_env_fn=wrap_neutral_for_training, **options)
    params = train._strip_weak_type(params)
    if count_value(params[0].count) != 0 or not tree_finite(params):
        raise ValueError("zero-step initializer integrity failure")
    shape = {"state": (101,), "privileged_state": (212,)}
    net = factory(shape, 14, preprocess_observations_fn=running_statistics.normalize)
    keys = first_batch_keys()
    independently_initialized = (net.policy_network.init(keys["policy_init"]), net.value_network.init(keys["value_init"]))
    if not leaf_comparison(params[1:], independently_initialized)["equal"]:
        raise ValueError("seed schedule does not reproduce official initial weights")
    weight_hash = fingerprint(params[1:])
    write_json(output / "initializer.json", {"normalizer_count": 0, "weights_sha256": weight_hash,
        "initial_parameter_artifact": archive(raw / "initial_params.npz", params),
        "official_initialization_matches_seed_schedule": True, "optimizer_updates": 0})
    env = wrap_neutral_for_training(underlying, episode_length=8, action_repeat=1)
    state = jax.pmap(env.reset, axis_name=train._PMAP_AXIS_NAME)(keys["reset"])
    state = train._strip_weak_type(state)
    print("initializer verified; collecting eight transitions", flush=True)
    data, updated = make_collector(env, make_policy, train._PMAP_AXIS_NAME)(
        jax.device_put_replicated(params, jax.local_devices()), state, keys["epoch"])
    data, updated = jax.tree.map(lambda x: x[0], (data, updated))
    if data.discount.shape != (2, 4) or data.action.shape != (2, 4, 14) or not tree_finite((data, updated)):
        raise ValueError("capture integrity failure")
    batch_artifact = archive(raw / "batch.npz", data)
    batch = jax.tree.map(lambda x: jax.random.permutation(keys["permutation"], x), data)
    batch_hash = fingerprint(batch)
    write_json(output / "capture.json", {"shape": [2, 4], "transitions": 8, "batch_artifact": batch_artifact,
        "minibatch_tree_sha256": batch_hash, "initial_weights_sha256": weight_hash,
        "key_ledger_sha256": fingerprint(keys), "normalization_count": [0, count_value(updated.count)],
        "terminations": int(np.sum(1 - np.asarray(data.discount))),
        "truncations": int(np.sum(np.asarray(data.extras["state_extras"]["truncation"])))})
    print("capture saved; replaying old and updated statistics", flush=True)

    evaluate_cases(params, updated, batch, keys, net, shape, factory, raw, output)


def make_replay(net):
    """Keep the complete replay dataset and weights as dynamic compiled inputs."""
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import losses

    loss_options = {name: PPO_CONFIG[name] for name in
                    ("entropy_cost", "discounting", "reward_scaling", "gae_lambda", "clipping_epsilon", "normalize_advantage")}
    loss_fn = functools.partial(losses.compute_ppo_loss, ppo_network=net,
                                vf_coefficient=PPO_CONFIG["vf_loss_coefficient"], **loss_options)

    @jax.jit
    def replay(normalizer, weights, batch, loss_key):
        obs = jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), batch.observation)
        logits = net.policy_network.apply(normalizer, weights.policy, obs)
        dist = net.parametric_action_distribution.create_dist(logits)
        log_prob = net.parametric_action_distribution.log_prob(
            logits, jp.swapaxes(batch.extras["policy_extras"]["raw_action"], 0, 1))
        value = net.value_network.apply(normalizer, weights.value, obs)
        _, metrics = loss_fn(weights, normalizer, batch, loss_key)
        return {"metrics": metrics, "logits": logits, "loc": dist.loc, "scale": dist.scale,
                "log_prob": log_prob, "value": value, "normalized": running_statistics.normalize(obs, normalizer)}

    return replay


def gaussian_log_prob(loc, scale, raw_action):
    """Independent float64 tanh-Gaussian density evaluated on raw actions."""
    loc, scale, raw_action = [np.asarray(x, dtype=np.float64) for x in (loc, scale, raw_action)]
    if np.any(scale <= 0):
        raise ValueError("Gaussian scales must be positive")
    gaussian = -.5 * ((raw_action - loc) / scale)**2 - np.log(scale) - .5 * np.log(2 * np.pi)
    jacobian = 2 * (np.log(2) - raw_action - np.logaddexp(0, -2 * raw_action))
    return np.sum(gaussian - jacobian, axis=-1)


def evaluate_cases(params, updated, batch, keys, net, shape, factory, raw, output):
    from brax.training.agents.ppo import checkpoint, losses
    weight_hash, batch_hash = fingerprint(params[1:]), fingerprint(batch)
    fixed_weights = losses.PPONetworkParams(policy=params[1], value=params[2])
    replay = make_replay(net)
    stored_logits = np.swapaxes(np.asarray(batch.extras["policy_extras"]["distribution_params"]), 0, 1)
    behavior_dist = net.parametric_action_distribution.create_dist(stored_logits)
    stored_lp = np.swapaxes(np.asarray(batch.extras["policy_extras"]["log_prob"]), 0, 1)
    results, rows, roundtrips = [], [], []
    for name, normalizer, count in (("old", params[0], 0), ("updated", updated, 8)):
        result = replay(normalizer, fixed_weights, batch, keys["loss"])
        if not tree_finite(result):
            raise ValueError("nonfinite replay")
        artifact = archive(raw / f"{name}_replay.npz", result)
        conventional = gaussian_kl(behavior_dist.loc, behavior_dist.scale, result["loc"], result["scale"])
        stabilized = gaussian_kl(behavior_dist.loc, behavior_dist.scale, result["loc"], result["scale"], 1e-5)
        manual_lp = gaussian_log_prob(result["loc"], result["scale"],
                                      np.swapaxes(batch.extras["policy_extras"]["raw_action"], 0, 1))
        lp_match = bool(np.allclose(manual_lp, result["log_prob"], atol=1e-4, rtol=1e-5))
        metrics = {key: float(value) for key, value in result["metrics"].items()}
        manual_match = bool(np.isclose(np.mean(stabilized), metrics["kl_mean"], atol=1e-5, rtol=1e-5))
        save_parameters(raw / name, count, (normalizer, *params[1:]), checkpoint.network_config(shape, 14, True, factory))
        location = raw / name / f"{count:012d}"
        contract = {"purpose": "fixed_weight_normalization_replay", "normalizer_count": count,
                    "variance_eps": VARIANCE_EPS, "network": NETWORK_CONFIG, "weights_sha256": weight_hash}
        write_contract(location, contract)
        require_contract(location, contract)
        equality = leaf_comparison((normalizer, *params[1:]), checkpoint.load(location))
        roundtrips.append(equality["equal"] and equality["finite"])
        row = {"case": name, "metrics": metrics, "conventional_kl_mean": float(np.mean(conventional)),
               "conventional_kl_max": float(np.max(conventional)), "manual_stabilized_kl_mean": float(np.mean(stabilized)),
               "manual_kl_agreement": manual_match, "manual_log_prob_agreement": lp_match,
               "manual_log_prob_max_error": float(np.max(np.abs(manual_lp - result["log_prob"]))),
               "behavior_logits_max_error": float(np.max(np.abs(result["logits"] - stored_logits))),
               "behavior_log_prob_max_error": float(np.max(np.abs(result["log_prob"] - stored_lp))),
               "value_max_abs": float(np.max(np.abs(result["value"]))),
               "loc_max_abs": float(np.max(np.abs(result["loc"]))),
               "scale_min": float(np.min(result["scale"])), "scale_max": float(np.max(result["scale"])),
               "normalized_max_abs": {key: float(np.max(np.abs(value))) for key, value in result["normalized"].items()},
               "roundtrip": equality, "checkpoint_hashes": checkpoint_hashes(location), "replay_artifact": artifact}
        rows.append(row)
        results.append(result)
        write_json(output / "cases.json", rows)
        case_valid = (manual_match and lp_match and equality["equal"] and equality["finite"]
                      and fingerprint(params[1:]) == weight_hash and fingerprint(batch) == batch_hash)
        if name == "old":
            case_valid &= (row["behavior_logits_max_error"] <= 1e-5
                           and row["behavior_log_prob_max_error"] <= 1e-4
                           and abs(row["conventional_kl_mean"]) <= 1e-6 and metrics["kl_mean"] <= .001)
        if not case_valid:
            write_json(output / "integrity_failure.json", {"case": name, "integrity_pass": False})
            raise ValueError("case integrity criteria failed")
    integrity = {"finite": tree_finite((results, params, updated, batch)),
                 "statistics_count": count_value(updated.count) == 8,
                 "fixed_weights": fingerprint(params[1:]) == weight_hash,
                 "fixed_batch": fingerprint(batch) == batch_hash,
                 "checkpoint_roundtrips": all(roundtrips), "manual_kl_agreement": all(r["manual_kl_agreement"] for r in rows),
                 "manual_log_prob_agreement": all(r["manual_log_prob_agreement"] for r in rows),
                 "old_logits": rows[0]["behavior_logits_max_error"] <= 1e-5,
                 "old_log_prob": rows[0]["behavior_log_prob_max_error"] <= 1e-4,
                 "old_conventional_self_kl": abs(rows[0]["conventional_kl_mean"]) <= 1e-6}
    summary = decide(integrity, rows[0]["metrics"]["kl_mean"], rows[1]["metrics"]["kl_mean"])
    summary["deterministic_action_max_shift"] = float(np.max(np.abs(np.tanh(results[1]["loc"]) - np.tanh(results[0]["loc"]))))
    summary["value_max_shift"] = float(np.max(np.abs(results[1]["value"] - results[0]["value"])))
    summary["log_prob_max_shift"] = float(np.max(np.abs(results[1]["log_prob"] - results[0]["log_prob"])))
    write_json(output / "normalizer_shift.json", {
        key: {"mean_delta": np.asarray(updated.mean[key] - params[0].mean[key]).tolist(),
              "std_delta": np.asarray(updated.std[key] - params[0].std[key]).tolist()}
        for key in updated.mean})
    write_json(output / "summary.json", summary)
    if not summary["integrity_pass"]:
        raise ValueError("replay integrity criteria failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.cwd()
    raw = root / "experiments/cloud_runs/normalization-timing"
    output = root / "experiments/locomotion_curriculum/results_normalization_timing"
    if args.worker:
        try:
            run(root, raw, output)
        except Exception as error:
            write_json(output / "worker_abort.json", {"error_type": type(error).__name__, "integrity_pass": False})
            raise
        return
    if raw.exists() or output.exists():
        raise FileExistsError("never overwrite normalization-timing artifacts")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before capture")
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        with (raw / "replay.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.normalization_timing", "--worker"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
    except subprocess.SubprocessError as error:
        write_json(output / "abort.json", {"integrity_pass": False, "error_type": type(error).__name__,
                                          "wall_seconds": round(time.monotonic() - start, 2), "wall_cap_s": 600})
        raise
    write_json(output / "parent_completion.json", {"wall_seconds": round(time.monotonic() - start, 2),
                                                   "wall_cap_s": 600, "worker_returncode": 0})
    print("Fixed-weight replay finished; inspect diagnostic summary", flush=True)


if __name__ == "__main__":
    main()
