"""One opt-in raw-identity PPO/warm-start screen, with deployed-policy diagnostics."""
import argparse
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from experiments.locomotion_curriculum import neutral_ppo_smoke
from experiments.locomotion_curriculum.normalization_timing import (
    archive, fingerprint, gaussian_kl, gaussian_log_prob, make_replay,
)
from experiments.locomotion_curriculum.normalization_timing_offline import admit_inputs
from nerva.training.parameter_checkpoint import checkpoint_hashes, count_value, tree_finite

PREREGISTRATION = "e627af3"


def preprocessing_invariance(net, params):
    import jax
    altered = params[0].replace(mean=jax.tree.map(lambda x: x + 10, params[0].mean),
                                std=jax.tree.map(lambda x: x * .01, params[0].std))

    @jax.jit
    def outputs(normalizer, policy, value, observation):
        return net.policy_network.apply(normalizer, policy, observation), net.value_network.apply(normalizer, value, observation)

    maximum = 0.
    finite = True
    for obs in neutral_ppo_smoke.probe_observations(params[0]):
        a = outputs(params[0], params[1], params[2], obs)
        b = outputs(altered, params[1], params[2], obs)
        finite &= tree_finite((a, b))
        maximum = max(maximum, *(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(a, b)))
    return {"probes": 5, "finite": finite, "max_logits_or_value_error": maximum,
            "all_pass": finite and maximum <= 1e-6}


def screen(stage):
    metrics, replay = stage["metrics"], stage["normalization_diagnostics"]
    return {"plumbing": stage["all_pass"], "training_kl": metrics["training/kl_mean"] <= 1,
            "training_value_loss": metrics["training/v_loss"] <= 100,
            "deployed_kl": replay["metrics"]["kl_mean"] <= 1,
            "deployed_value_loss": replay["metrics"]["v_loss"] <= 100,
            "density_integrity": replay["manual_kl_agreement"] and replay["manual_log_prob_agreement"],
            "preprocessing_invariance": replay["preprocessing_invariance"]["all_pass"]}


def run(root, raw, output):
    from brax.training import types
    from brax.training.agents.ppo import losses, networks
    initial, _, batch, keys, admission = admit_inputs(root)
    control_root = root / "experiments/locomotion_curriculum/results_normalization_control"
    control = json.loads((control_root / "protocol.json").read_text(encoding="utf-8"))
    comparator = json.loads((control_root / "stages.json").read_text(encoding="utf-8"))
    expected = {**neutral_ppo_smoke.PPO_CONFIG, "normalize_observations_std_eps": .0001}
    current_sources = {name: hashlib.sha256(Path(inspect.getfile(importlib.import_module(name))).read_bytes()).hexdigest()
                       for name in control["source_hashes"]}
    if control["config"] != expected or current_sources != control["source_hashes"]:
        raise ValueError("frozen control config/core source mismatch")
    for row in comparator:
        location = root / "experiments/cloud_runs/normalization-control/checkpoints" / row["stage"] / "000000000016"
        if checkpoint_hashes(location) != row["checkpoint_hashes"]:
            raise ValueError("frozen comparator checkpoint hash mismatch")
    write_json(output / "admission.json", {**admission, "control_source_match": True,
        "control_report_hashes": {name: hashlib.sha256((control_root / name).read_bytes()).hexdigest()
                                  for name in ("protocol.json", "stages.json", "comparison.json")},
        "declared_config_difference": {"normalize_observations": [True, False]},
        "initial_weights_sha256": fingerprint(initial[1:])})
    net = functools.partial(networks.make_ppo_networks, **neutral_ppo_smoke.NETWORK_CONFIG)(
        {"state": (101,), "privileged_state": (212,)}, 14,
        preprocess_observations_fn=types.identity_observation_preprocessor)
    replay = make_replay(net)
    stored = net.parametric_action_distribution.create_dist(
        np.swapaxes(np.asarray(batch.extras["policy_extras"]["distribution_params"]), 0, 1))
    raw_actions = np.swapaxes(np.asarray(batch.extras["policy_extras"]["raw_action"]), 0, 1)
    diagnostics, decisions = [], []

    def validate_initial(name, params):
        if name == "fresh":
            matches = fingerprint(params[1:]) == fingerprint(initial[1:])
            write_json(output / "initial_admission.json", {"fresh_weights_match_retained_behavior": matches})
            if not matches:
                raise ValueError("fresh initial weights differ from retained behavior")

    def observe(name, params):
        result = replay(params[0], losses.PPONetworkParams(policy=params[1], value=params[2]), batch, keys["loss"])
        invariant = preprocessing_invariance(net, params)
        if not tree_finite((params, result)) or not invariant["finite"]:
            raise ValueError("nonfinite identity diagnostic")
        manual_kl = gaussian_kl(stored.loc, stored.scale, result["loc"], result["scale"], 1e-5)
        lp = gaussian_log_prob(result["loc"], result["scale"], raw_actions)
        metrics = {key: float(x) for key, x in result["metrics"].items()}
        row = {"stage": name, "normalizer_count_unused": count_value(params[0].count), "metrics": metrics,
               "manual_stabilized_kl_mean": float(np.mean(manual_kl)),
               "manual_kl_agreement": bool(np.isclose(np.mean(manual_kl), metrics["kl_mean"], atol=1e-5, rtol=1e-5)),
               "manual_log_prob_agreement": bool(np.allclose(lp, np.asarray(result["log_prob"]), atol=1e-4, rtol=1e-5)),
               "manual_log_prob_max_error": float(np.max(np.abs(lp - np.asarray(result["log_prob"])))),
               "preprocessing_invariance": invariant, "retained_batch_sha256": fingerprint(batch),
               "replay_artifact": archive(raw / f"{name}_deployed_replay.npz", result)}
        diagnostics.append(row)
        write_json(output / "deployment.json", diagnostics)
        return row

    def validate(stage):
        criteria = screen(stage)
        decisions.append({"stage": stage["stage"], "criteria": criteria, "all_pass": all(criteria.values()),
                          "training_kl": stage["metrics"]["training/kl_mean"],
                          "deployed_kl": stage["normalization_diagnostics"]["metrics"]["kl_mean"]})
        write_json(output / "comparison.json", {"stages": decisions,
            "all_pass": len(decisions) == 2 and all(x["all_pass"] for x in decisions),
            "new_environment_transitions": sum(16 for _ in decisions), "optimizer_updates": 2 * len(decisions),
            "frozen_control_metrics": {row["stage"]: row["metrics"] for row in comparator},
            "general_learning_stability": False, "learned_motor_readiness": False, "cloud_work": False})
        if not all(criteria.values()):
            raise ValueError("identity numerical or deployment screen failed")

    neutral_ppo_smoke.run(root, raw, output, normalization_eps=.0001, normalize_observations=False,
                         preregistration_commit=PREREGISTRATION, stage_observer=observe, stage_validator=validate,
                         initial_validator=validate_initial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.cwd()
    raw = root / "experiments/cloud_runs/identity-preprocessing"
    output = root / "experiments/locomotion_curriculum/results_identity_preprocessing"
    if args.worker:
        try:
            run(root, raw, output)
        except Exception as error:
            write_json(output / "worker_abort.json", {"error_type": type(error).__name__, "all_pass": False})
            raise
        return
    if raw.exists() or output.exists():
        raise FileExistsError("never overwrite identity experiment artifacts")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before training")
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        with (raw / "identity.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.identity_preprocessing", "--worker"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
    except subprocess.SubprocessError as error:
        write_json(output / "abort.json", {"all_pass": False, "error_type": type(error).__name__,
                                          "wall_seconds": round(time.monotonic() - start, 2), "wall_cap_s": 900})
        raise
    write_json(output / "parent_completion.json", {"wall_seconds": round(time.monotonic() - start, 2),
                                                   "wall_cap_s": 900, "worker_returncode": 0})
    print("Identity preprocessing screen finished; inspect comparison.json", flush=True)


if __name__ == "__main__":
    main()
