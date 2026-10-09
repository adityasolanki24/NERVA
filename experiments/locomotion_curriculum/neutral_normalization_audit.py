"""Capped forensic audit of saved smoke parameters, with no new optimization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from nerva.training.parameter_checkpoint import (
    checkpoint_hashes, count_value, leaf_comparison, require_contract, save_parameters, tree_finite,
)


def std_statistics(std):
    values = np.asarray(std)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("normalizer standard deviations must be finite and positive")
    indices = np.flatnonzero(values <= 1.0001e-6).tolist()
    return {"slots": int(values.size), "floor_count": len(indices), "floor_indices": indices,
            "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max()),
            "max_inverse_std": float(np.max(1 / values))}


def run(root, raw, output):
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks
    from experiments.locomotion_curriculum.neutral_ppo_smoke import probe_observations
    from nerva.training.neutral_reference import verified_references

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("audit is CPU only")
    reports = root / "experiments/locomotion_curriculum/results_neutral_ppo"
    original = root / "experiments/cloud_runs/neutral-ppo-smoke/checkpoints"
    stages = json.loads((reports / "stages.json").read_text(encoding="utf-8"))
    protocol = json.loads((reports / "protocol.json").read_text(encoding="utf-8"))
    _, manifest = verified_references(root)
    results = []
    for stage in stages:
        name = stage["stage"]
        if name not in ("fresh", "warm"):
            raise ValueError("unknown checkpoint stage")
        path = original / name / "000000000016"
        if checkpoint_hashes(path) != stage["checkpoint_hashes"]:
            raise ValueError("input checkpoint hash mismatch")
        contract = json.loads((original / "fresh/000000000016/nerva_contract.json").read_text(encoding="utf-8"))
        require_contract(path, contract)
        if (contract["references"] != [row["reference_sha256"] for row in manifest["references"]]
                or contract["source_hashes"] != protocol["source_hashes"]
                or contract["observation_size"] != {"state": [101], "privileged_state": [212]}
                or not contract["normalize_observations"]):
            raise ValueError("input reference/normalization contract mismatch")
        params = checkpoint.load(path)
        if not tree_finite(params):
            raise ValueError("nonfinite checkpoint")
        config = checkpoint.load_config(path)
        if (dict(config.network_factory_kwargs)["policy_obs_key"] != "state"
                or dict(config.network_factory_kwargs)["value_obs_key"] != "privileged_state"
                or config.observation_size.to_dict() != contract["observation_size"]):
            raise ValueError("input network config mismatch")
        net = networks.make_ppo_networks(config.observation_size.to_dict(), config.action_size,
                                        preprocess_observations_fn=running_statistics.normalize,
                                        **dict(config.network_factory_kwargs))
        value_fn = jax.jit(net.value_network.apply)
        policy = jax.jit(checkpoint.load_policy(path, deterministic=True))
        normalizer = params[0]
        stats = {key: std_statistics(std) for key, std in normalizer.std.items()}
        probes = []
        for index, observation in enumerate(probe_observations(normalizer)):
            normalized = running_statistics.normalize(observation, normalizer)
            manual = {key: (np.asarray(x) - np.asarray(normalizer.mean[key])) / np.asarray(normalizer.std[key])
                      for key, x in observation.items()}
            agreement = all(np.allclose(manual[key], normalized[key], atol=1e-6, rtol=1e-6) for key in manual)
            action, _ = policy(observation, jax.random.PRNGKey(19))
            value = value_fn(normalizer, params[2], observation)
            if not agreement or not tree_finite((normalized, action, value)):
                raise ValueError("normalization/probe integrity failure")
            maxima = {key: float(np.max(np.abs(x))) for key, x in normalized.items()}
            if index == 4 and any(maxima.values()):
                raise ValueError("mean probe does not normalize to zero")
            probes.append({"index": index, "manual_agreement": agreement, "normalized_max_abs": maxima,
                           "action_max_abs": float(np.max(np.abs(action))), "value": float(value)})
        save_parameters(raw / name, 16, params, config)
        restored = checkpoint.load(raw / name / "000000000016")
        equality = leaf_comparison(params, restored)
        if not equality["equal"] or not equality["finite"]:
            raise ValueError("literal-byte roundtrip failure")
        results.append({"stage": name, "normalizer_count": count_value(normalizer.count),
                        "std": stats, "probes": probes, "literal_byte_roundtrip": equality,
                        "input_hashes": stage["checkpoint_hashes"], "all_integrity_pass": True})
        write_json(output / "stages.json", results)
    write_json(output / "summary.json", {"preregistration_commit": "7a3b7c5",
               "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "all_integrity_pass": len(results) == 2, "new_optimization_transitions": 0,
               "rollouts": 0, "cloud_work": False, "causal_loss_diagnosis": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.cwd()
    raw = root / "experiments/cloud_runs/neutral-normalization-audit"
    output = root / "experiments/locomotion_curriculum/results_neutral_normalization"
    if args.worker:
        run(root, raw, output)
        return
    if raw.exists() or output.exists():
        raise FileExistsError("never overwrite existing audit")
    # The prior experiment's small results may be untracked; implementation must be committed.
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], text=True).strip():
        raise RuntimeError("commit implementation before audit")
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        with (raw / "audit.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.neutral_normalization_audit", "--worker"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
    except subprocess.SubprocessError as error:
        write_json(output / "abort.json", {"all_integrity_pass": False, "error_type": type(error).__name__,
                                          "wall_seconds": round(time.monotonic() - start, 2), "wall_cap_s": 60})
        raise
    write_json(output / "parent_completion.json", {"wall_seconds": round(time.monotonic() - start, 2), "wall_cap_s": 60})


if __name__ == "__main__":
    main()
