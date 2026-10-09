"""Two offline Adam steps; compare optimization and deployment preprocessing."""
import argparse
import functools
import hashlib
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
from experiments.locomotion_curriculum.neutral_ppo_smoke import (
    NETWORK_CONFIG, PPO_CONFIG, action_parity, probe_observations,
)
from experiments.locomotion_curriculum.normalization_timing import (
    archive, fingerprint, gaussian_kl, gaussian_log_prob, make_replay, tree_arrays,
)
from experiments.locomotion_curriculum.normalization_timing_offline import admit_inputs
from nerva.training.parameter_checkpoint import (
    checkpoint_hashes, leaf_comparison, require_contract, save_parameters, tree_finite, write_contract,
)

PREREGISTRATION = "e241208"


def rebase_mlp(weights, old_mean, old_std, new_mean, new_std):
    """Experimental inference rebase of an unclipped first affine layer, not Adam."""
    import jax.numpy as jp
    from flax import core
    frozen = isinstance(weights, core.FrozenDict)
    result = core.unfreeze(weights)
    layer = result["params"]["hidden_0"]
    kernel, bias = layer["kernel"], layer["bias"]
    arrays = tuple(np.asarray(x) for x in (old_mean, old_std, new_mean, new_std))
    if (set(layer) != {"kernel", "bias"} or len(kernel.shape) != 2
            or any(x.shape != (kernel.shape[0],) for x in arrays) or bias.shape != (kernel.shape[1],)
            or not all(np.isfinite(x).all() for x in arrays)
            or np.any(arrays[1] <= 0) or np.any(arrays[3] <= 0)):
        raise ValueError("unsupported affine normalizer/layer schema")
    dtype = kernel.dtype
    m, s, new_m, new_s = (jp.asarray(x, dtype=dtype) for x in arrays)
    layer["kernel"] = (new_s / s)[:, None] * kernel
    layer["bias"] = bias + ((new_m - m) / s) @ kernel
    return core.freeze(result) if frozen else result


def decide(integrity, fixed_metrics, deferred_metrics, rebase_checks, rebased_metrics):
    fixed = all(integrity.values()) and fixed_metrics["kl_mean"] <= 1 and fixed_metrics["v_loss"] <= 100
    return {"integrity_pass": all(integrity.values()), "integrity": integrity,
            "fixed_preprocessing_screen_pass": fixed,
            "deferred_deployment_drift_supported": fixed and deferred_metrics["kl_mean"] > 1,
            "rebase_support": fixed and all(rebase_checks.values())
            and rebased_metrics["kl_mean"] <= 1 and rebased_metrics["v_loss"] <= 100,
            "rebase_checks": rebase_checks, "optimizer_updates": 2,
            "new_environment_transitions": 0, "original_environment_transitions": 8,
            "learned_motor_readiness": False, "optimizer_rebase_equivalence": False, "cloud_work": False}


def run(root, raw, output):
    import jax
    import optax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, losses, networks

    params, updated, batch, keys, admission = admit_inputs(root)
    fixed_initial = losses.PPONetworkParams(policy=params[1], value=params[2])
    initial_hash, batch_hash = fingerprint(params[1:]), fingerprint(batch)
    shape = {"state": (101,), "privileged_state": (212,)}
    factory = functools.partial(networks.make_ppo_networks, **NETWORK_CONFIG)
    net = factory(shape, 14, preprocess_observations_fn=running_statistics.normalize)
    replay = make_replay(net)
    write_json(output / "protocol.json", {
        "preregistration_commit": PREREGISTRATION,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **admission,
        "optimizer": {"name": "adam", "learning_rate": .0001, "version": importlib.metadata.version("optax"),
                      "source_sha256": hashlib.sha256(Path(inspect.getfile(optax.adam)).read_bytes()).hexdigest(),
                      "state": "independent_fresh_per_gradient_arm", "updates": 2},
        "ppo_config": PPO_CONFIG, "wall_cap_s": 180, "cloud_work": False})
    old_control = replay(params[0], fixed_initial, batch, keys["loss"])
    new_control = replay(updated, fixed_initial, batch, keys["loss"])
    prior = root / "experiments/locomotion_curriculum/results_normalization_timing_offline_schema/cases.json"
    prior_cases = json.loads(prior.read_text(encoding="utf-8"))
    control_match = all(np.isclose(float(control["metrics"][key]), value, atol=1e-4, rtol=1e-5)
                        for control, row in zip((old_control, new_control), prior_cases)
                        for key, value in row["metrics"].items())
    if not control_match or not tree_finite((old_control, new_control)):
        raise ValueError("frozen controls disagree with preserved replay")
    print("inputs and controls admitted; two independent offline Adam steps", flush=True)
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

    optimized = {}
    gradient_rows = []
    for name, normalizer in (("native", updated), ("fixed", params[0])):
        weights, state, gradients, metrics = optimize(
            fixed_initial, normalizer, batch, keys["loss"], optimizer.init(fixed_initial))
        if not tree_finite((weights, state, gradients, metrics)):
            raise ValueError("nonfinite optimization")
        changes = [leaf_comparison(a, b) for a, b in
                   ((fixed_initial.policy, weights.policy), (fixed_initial.value, weights.value))]
        if not all(row["max_error"] is not None and row["max_error"] > 1e-12 for row in changes):
            raise ValueError("both networks must update")
        optimized[name] = weights
        gradient_rows.append({"arm": name, "pre_update_metrics": {key: float(x) for key, x in metrics.items()},
                              "gradient_global_norm": float(optax.global_norm(gradients)),
                              "policy_max_update": changes[0]["max_error"], "value_max_update": changes[1]["max_error"],
                              "artifact": archive(raw / f"{name}_optimization.npz", (weights, state, gradients))})
        write_json(output / "gradients.json", gradient_rows)
    rebased = losses.PPONetworkParams(**{
        name: rebase_mlp(getattr(optimized["fixed"], name), params[0].mean[key], params[0].std[key],
                         updated.mean[key], updated.std[key])
        for name, key in (("policy", "state"), ("value", "privileged_state"))})
    rebase_checks = {"other_leaves_literal": True, "dtypes_shapes": True, "independent_affine_formula": True}
    for name, key in (("policy", "state"), ("value", "privileged_state")):
        before, _ = tree_arrays(getattr(optimized["fixed"], name))
        after, _ = tree_arrays(getattr(rebased, name))
        for leaf in before:
            rebase_checks["dtypes_shapes"] &= before[leaf].dtype == after[leaf].dtype and before[leaf].shape == after[leaf].shape
            if "['hidden_0']" not in leaf:
                rebase_checks["other_leaves_literal"] &= before[leaf].tobytes() == after[leaf].tobytes()
        w = before["['params']['hidden_0']['kernel']"].astype(np.float64)
        b = before["['params']['hidden_0']['bias']"].astype(np.float64)
        m, s, new_m, new_s = [np.asarray(x, np.float64) for x in
                             (params[0].mean[key], params[0].std[key], updated.mean[key], updated.std[key])]
        for leaf, expected in (("kernel", (new_s / s)[:, None] * w), ("bias", b + ((new_m - m) / s) @ w)):
            rebase_checks["independent_affine_formula"] &= bool(np.allclose(
                after[f"['params']['hidden_0']['{leaf}']"], expected, atol=1e-5, rtol=1e-5))

    @jax.jit
    def probe(normalizer, weights, obs, key):
        logits = net.policy_network.apply(normalizer, weights.policy, obs)
        dist = net.parametric_action_distribution.create_dist(logits)
        value = net.value_network.apply(normalizer, weights.value, obs)
        return {"logits": logits, "loc": dist.loc, "scale": dist.scale, "value": value,
                "deterministic": net.parametric_action_distribution.mode(logits),
                "sampled": net.parametric_action_distribution.sample(logits, key)}

    drift = []
    for obs in [batch.observation, batch.next_observation, *probe_observations(params[0])]:
        a = probe(params[0], optimized["fixed"], obs, jax.random.PRNGKey(19))
        b = probe(updated, rebased, obs, jax.random.PRNGKey(19))
        if not tree_finite((a, b)):
            raise ValueError("nonfinite rebase probe")
        pair_kl = gaussian_kl(a["loc"], a["scale"], b["loc"], b["scale"])
        drift.append({**{key: float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))))
                         for key in ("logits", "value", "deterministic", "sampled")},
                      "analytic_kl_mean": float(np.mean(pair_kl))})
    rebase_checks.update({"logits": all(x["logits"] <= 1e-4 for x in drift),
                          "value": all(x["value"] <= 1e-4 for x in drift),
                          "deterministic": all(x["deterministic"] <= 1e-5 for x in drift),
                          "sampled": all(x["sampled"] <= 1e-5 for x in drift),
                          "analytic_kl": all(abs(x["analytic_kl_mean"]) <= 1e-6 for x in drift)})
    write_json(output / "rebase_drift.json", drift)
    stored_logits = np.swapaxes(np.asarray(batch.extras["policy_extras"]["distribution_params"]), 0, 1)
    behavior = net.parametric_action_distribution.create_dist(stored_logits)
    raw_action = np.swapaxes(np.asarray(batch.extras["policy_extras"]["raw_action"]), 0, 1)
    rows = []
    make_policy = networks.make_inference_fn(net)
    all_case_checks = []
    for name, normalizer, weights in (("native", updated, optimized["native"]),
                                      ("fixed", params[0], optimized["fixed"]),
                                      ("deferred", updated, optimized["fixed"]), ("rebased", updated, rebased)):
        result = replay(normalizer, weights, batch, keys["loss"])
        if not tree_finite(result):
            raise ValueError("nonfinite deployment replay")
        metrics = {key: float(value) for key, value in result["metrics"].items()}
        stabilized = gaussian_kl(behavior.loc, behavior.scale, result["loc"], result["scale"], 1e-5)
        manual_lp = gaussian_log_prob(result["loc"], result["scale"], raw_action)
        checks = {"manual_kl": bool(np.isclose(np.mean(stabilized), metrics["kl_mean"], atol=1e-5, rtol=1e-5)),
                  "manual_log_prob": bool(np.allclose(manual_lp, np.asarray(result["log_prob"]), atol=1e-4, rtol=1e-5))}
        deployment = (normalizer, weights.policy, weights.value)
        save_parameters(raw / name, 1, deployment, checkpoint.network_config(shape, 14, True, factory))
        location = raw / name / "000000000001"
        contract = {"purpose": "offline_normalization_schedule", "arm": name, "network": NETWORK_CONFIG,
                    "optimizer_resume": False, "initial_weights_sha256": initial_hash, "batch_sha256": batch_hash}
        write_contract(location, contract)
        require_contract(location, contract)
        equality = leaf_comparison(deployment, checkpoint.load(location))
        parity = action_parity(make_policy, deployment, lambda deterministic, path=location:
                              checkpoint.load_policy(path, deterministic=deterministic))
        checks.update({"roundtrip": equality["equal"] and equality["finite"], "action_restore": parity["all_pass"]})
        all_case_checks.append(all(checks.values()))
        rows.append({"arm": name, "metrics": metrics, "checks": checks, "roundtrip": equality, "action_parity": parity,
                     "manual_kl_mean": float(np.mean(stabilized)),
                     "manual_log_prob_max_error": float(np.max(np.abs(manual_lp - np.asarray(result["log_prob"])))),
                     "checkpoint_hashes": checkpoint_hashes(location),
                     "artifact": archive(raw / f"{name}_replay.npz", result)})
        write_json(output / "cases.json", rows)
        if not all(checks.values()):
            raise ValueError("deployment integrity failed")
    integrity = {"input_admission": True, "control_match": control_match,
                 "fixed_initial_weights": fingerprint(params[1:]) == initial_hash,
                 "fixed_batch": fingerprint(batch) == batch_hash, "finite": tree_finite((optimized, rebased, updated)),
                 "case_checks": all(all_case_checks)}
    summary = decide(integrity, rows[1]["metrics"], rows[2]["metrics"], rebase_checks, rows[3]["metrics"])
    write_json(output / "summary.json", summary)
    if not summary["integrity_pass"]:
        raise ValueError("comparison integrity failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.cwd()
    raw = root / "experiments/cloud_runs/normalization-schedule"
    output = root / "experiments/locomotion_curriculum/results_normalization_schedule"
    if args.worker:
        try:
            run(root, raw, output)
        except Exception as error:
            write_json(output / "worker_abort.json", {"error_type": type(error).__name__, "integrity_pass": False})
            raise
        return
    if raw.exists() or output.exists():
        raise FileExistsError("never overwrite comparison artifacts")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before comparison")
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        with (raw / "comparison.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.normalization_schedule", "--worker"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
    except subprocess.SubprocessError as error:
        write_json(output / "abort.json", {"integrity_pass": False, "error_type": type(error).__name__,
                                          "wall_seconds": round(time.monotonic() - start, 2), "wall_cap_s": 180})
        raise
    write_json(output / "parent_completion.json", {"wall_seconds": round(time.monotonic() - start, 2),
                                                   "wall_cap_s": 180, "worker_returncode": 0})
    print("Offline schedule comparison finished", flush=True)


if __name__ == "__main__":
    main()
