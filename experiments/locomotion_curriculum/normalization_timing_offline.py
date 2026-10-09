"""Hash-admitted offline retry: no environment initialization, stepping or SGD."""
import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from experiments.locomotion_curriculum.neutral_ppo_smoke import NETWORK_CONFIG
from experiments.locomotion_curriculum.normalization_timing import (
    evaluate_cases, fingerprint, first_batch_keys, tree_arrays,
)
from nerva.training.parameter_checkpoint import checkpoint_hashes, count_value, tree_finite

PREREGISTRATION = "75cf8ed"


def load_batch(path):
    """Read the generated simple dictionary schema; never permit object pickles."""
    import jax.numpy as jp
    from brax.training.types import Transition
    state = {}
    with np.load(path, allow_pickle=False) as saved:
        for name in saved.files:
            parts = re.findall(r"\['([a-z_]+(?:/[a-z_]+)*)'\]", name)
            if not parts or "".join(f"['{part}']" for part in parts) != name:
                raise ValueError("unsupported archive key")
            cursor = state
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = jp.asarray(saved[name])
    return Transition(**state)


def require_archive(path, metadata, tree=None):
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["file_sha256"]:
        raise ValueError("archive file hash mismatch")
    if tree is not None:
        arrays, _ = tree_arrays(tree)
        if fingerprint(tree) != metadata["tree_sha256"]:
            raise ValueError("archive tree hash mismatch")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError("archive schema mismatch")
            for name, array in arrays.items():
                other = saved[name]
                if (array.dtype != other.dtype or array.shape != other.shape
                        or array.tobytes() != other.tobytes()):
                    raise ValueError("archive literal bytes mismatch")


def run(root, raw, output):
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks, train
    from nerva.training.neutral_reference import verified_references

    if (jax.process_count() != 1 or jax.local_device_count() != 1
            or jax.devices()[0].platform != "cpu"):
        raise RuntimeError("offline replay requires one CPU process/device")
    prior = root / "experiments/locomotion_curriculum/results_normalization_timing"
    names = ("protocol.json", "initializer.json", "capture.json", "cases.json", "reference_manifest.json")
    reports = {name: json.loads((prior / name).read_text(encoding="utf-8")) for name in names}
    protocol, initializer, capture = (reports[name] for name in names[:3])
    sources = {name: hashlib.sha256(Path(inspect.getfile(importlib.import_module(name))).read_bytes()).hexdigest()
               for name in protocol["source_hashes"]}
    versions = {name: importlib.metadata.version(name) for name in protocol["versions"]}
    _, manifest = verified_references(root)
    original_protocol = json.loads((root / "experiments/locomotion_curriculum/results_normalization_control/protocol.json")
                                   .read_text(encoding="utf-8"))
    if (sources != protocol["source_hashes"] or versions != protocol["versions"]
            or sys.version.split()[0] != original_protocol["python_version"]
            or manifest != reports["reference_manifest.json"]
            or protocol["network"] != json.loads(json.dumps(NETWORK_CONFIG))
            or protocol["variance_eps"] != .0001 or protocol["seed"] != 7):
        raise ValueError("source/version/reference/config mismatch")
    inputs = root / "experiments/cloud_runs/normalization-timing"
    location = inputs / "old/000000000000"
    if checkpoint_hashes(location) != reports["cases.json"][0]["checkpoint_hashes"]:
        raise ValueError("checkpoint hash mismatch")
    params = checkpoint.load(location)
    require_archive(inputs / "initial_params.npz", initializer["initial_parameter_artifact"], params)
    batch = load_batch(inputs / "batch.npz")
    require_archive(inputs / "batch.npz", capture["batch_artifact"], batch)
    if (batch.discount.shape != (2, 4) or batch.action.shape != (2, 4, 14)
            or count_value(params[0].count) != 0 or not tree_finite((params, batch))
            or fingerprint(params[1:]) != initializer["weights_sha256"]):
        raise ValueError("input shape/finite/count/weight mismatch")
    keys = first_batch_keys()
    if fingerprint(keys) != capture["key_ledger_sha256"]:
        raise ValueError("key ledger mismatch")
    update = jax.pmap(lambda state, obs: running_statistics.update(
        state, obs, pmap_axis_name=train._PMAP_AXIS_NAME), axis_name=train._PMAP_AXIS_NAME)
    updated = update(jax.device_put_replicated(params[0], jax.local_devices()),
                     jax.device_put_replicated(batch.observation, jax.local_devices()))
    updated = jax.tree.map(lambda x: x[0], updated)
    batch = jax.tree.map(lambda x: jax.random.permutation(keys["permutation"], x), batch)
    if count_value(updated.count) != 8 or fingerprint(batch) != capture["minibatch_tree_sha256"]:
        raise ValueError("updated count/minibatch mismatch")
    write_json(output / "protocol.json", {
        "preregistration_commit": PREREGISTRATION,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_runner_source_sha256": hashlib.sha256(Path(inspect.getfile(evaluate_cases)).read_bytes()).hexdigest(),
        "prior_report_hashes": {name: hashlib.sha256((prior / name).read_bytes()).hexdigest() for name in names},
        "source_hashes": sources, "versions": versions, "network": NETWORK_CONFIG,
        "variance_eps": .0001, "seed": 7, "normalization_count": [0, 8],
        "original_environment_transitions": 8, "new_environment_transitions": 0, "optimizer_updates": 0,
        "wall_cap_s": 180, "cloud_work": False, "input_admission_pass": True,
        "weights_sha256": fingerprint(params[1:]), "batch_sha256": fingerprint(batch),
        "checkpoint_hashes": checkpoint_hashes(location)})
    print("retained inputs admitted; replaying old then updated statistics", flush=True)
    shape = {"state": (101,), "privileged_state": (212,)}
    factory = functools.partial(networks.make_ppo_networks, **NETWORK_CONFIG)
    net = factory(shape, 14, preprocess_observations_fn=running_statistics.normalize)
    evaluate_cases(params, updated, batch, keys, net, shape, factory, raw, output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary.update({"original_environment_transitions": 8, "new_environment_transitions": 0,
                    "retry_of_integrity_invalid_capture": True})
    write_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.cwd()
    raw = root / "experiments/cloud_runs/normalization-timing-offline-schema"
    output = root / "experiments/locomotion_curriculum/results_normalization_timing_offline_schema"
    if args.worker:
        try:
            run(root, raw, output)
        except Exception as error:
            write_json(output / "worker_abort.json", {"error_type": type(error).__name__, "integrity_pass": False})
            raise
        return
    if raw.exists() or output.exists():
        raise FileExistsError("never overwrite offline replay artifacts")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before replay")
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        with (raw / "replay.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.normalization_timing_offline", "--worker"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
    except subprocess.SubprocessError as error:
        write_json(output / "abort.json", {"integrity_pass": False, "error_type": type(error).__name__,
                                          "wall_seconds": round(time.monotonic() - start, 2), "wall_cap_s": 180})
        raise
    write_json(output / "parent_completion.json", {"wall_seconds": round(time.monotonic() - start, 2),
                                                   "wall_cap_s": 180, "worker_returncode": 0})
    print("Offline timing replay finished; inspect diagnostic summary", flush=True)


if __name__ == "__main__":
    main()
