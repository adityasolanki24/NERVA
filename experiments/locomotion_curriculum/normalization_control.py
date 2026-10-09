"""One preregistered variance-epsilon intervention against a frozen strict control."""
import argparse
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
from experiments.locomotion_curriculum.neutral_ppo_smoke import NETWORK_CONFIG, PPO_CONFIG, probe_observations
from nerva.training.parameter_checkpoint import checkpoint_hashes, require_contract, tree_finite

VARIANCE_EPS = .0001
PREREGISTRATION = "1ee56c5"


def validate_baseline(protocol, stages, manifest, sources, versions, current_manifest):
    expected_network = json.loads(json.dumps(NETWORK_CONFIG))
    if (protocol["config"] != PPO_CONFIG or protocol["network"] != expected_network
            or protocol["normalization_mode"] != "welford" or protocol["normalize_observations_std_eps"] != 0
            or protocol["source_hashes"] != sources or protocol["versions"] != versions
            or protocol["python_version"] != sys.version.split()[0] or manifest != current_manifest):
        raise ValueError("frozen control protocol/source/reference mismatch")
    if ([stage["stage"] for stage in stages] != ["fresh", "warm"]
            or [stage["normalizer_count"] for stage in stages] != [16, 32]
            or any(not stage["all_pass"] or not all(stage["criteria"].values())
                   or stage["callback_steps"] != [0, 16] for stage in stages)):
        raise ValueError("frozen control is not a successful strict pair")


def diagnostics(params):
    from brax.training.acme import running_statistics
    from experiments.locomotion_curriculum.neutral_normalization_audit import std_statistics
    normalizer = params[0]
    stats = {key: std_statistics(std) for key, std in normalizer.std.items()}
    probes = []
    for index, observation in enumerate(probe_observations(normalizer)):
        normalized = running_statistics.normalize(observation, normalizer)
        agreement = all(np.allclose((np.asarray(x) - np.asarray(normalizer.mean[key])) /
                                   np.asarray(normalizer.std[key]), normalized[key], atol=1e-6, rtol=1e-6)
                        for key, x in observation.items())
        if not agreement or not tree_finite(normalized):
            raise ValueError("normalization diagnostic integrity failure")
        maxima = {key: float(np.max(np.abs(value))) for key, value in normalized.items()}
        if index == 4 and any(maxima.values()):
            raise ValueError("mean probe does not normalize to zero")
        probes.append({"index": index, "normalized_max_abs": maxima, "manual_agreement": agreement})
    return {"std": stats, "probes": probes, "variance_eps": float(np.asarray(normalizer.std_eps)),
            "peak_normalized": max(value for probe in probes for value in probe["normalized_max_abs"].values()),
            "integrity_pass": True}


def assess(intervention, baseline, baseline_diagnostics):
    if ([row["stage"] for row in intervention] != ["fresh", "warm"]
            or [row["stage"] for row in baseline] != ["fresh", "warm"]):
        raise ValueError("comparison requires complete ordered stage pairs")
    rows = []
    for index, (actual, control, control_diag) in enumerate(zip(intervention, baseline, baseline_diagnostics)):
        diag = actual["normalization_diagnostics"]
        value_loss, kl = (actual["metrics"][key] for key in ("training/v_loss", "training/kl_mean"))
        criteria = {
            "plumbing": actual["all_pass"] and all(actual["criteria"].values())
                        and actual["normalizer_count"] == (16 if index == 0 else 32)
                        and actual["callback_steps"] == [0, 16],
            "diagnostic_integrity": diag["integrity_pass"] and abs(diag["variance_eps"] - VARIANCE_EPS) <= 1e-10,
            "std_floor": all(stat["min"] >= .01 * (1 - 1e-6) for stat in diag["std"].values()),
            "inverse_std_bound": all(stat["max_inverse_std"] <= 100.0001 for stat in diag["std"].values()),
            "probe_magnitude_bound": diag["peak_normalized"] <= 100,
            "probe_reduction": diag["peak_normalized"] <= control_diag["peak_normalized"] / 100,
            "value_loss_screen": value_loss <= 100,
            "kl_screen": kl <= 1,
        }
        if index == 0:
            criteria["fresh_value_loss_reduction"] = value_loss <= .01 * control["metrics"]["training/v_loss"]
            criteria["fresh_kl_reduction"] = kl <= .01 * control["metrics"]["training/kl_mean"]
        rows.append({"stage": actual["stage"], "criteria": criteria, "all_pass": all(criteria.values()),
                     "value_loss": value_loss, "kl": kl, "peak_normalized": diag["peak_normalized"],
                     "control_peak_normalized": control_diag["peak_normalized"],
                     "control_value_loss": control["metrics"]["training/v_loss"],
                     "control_kl": control["metrics"]["training/kl_mean"]})
    return {"all_required_pass": all(row["all_pass"] for row in rows), "stages": rows,
            "new_optimization_transitions": 32, "frozen_control": True, "learned_motor_readiness": False,
            "general_learning_stability_established": False, "cloud_work": False}


def run(root, raw, output):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint
    from experiments.locomotion_curriculum.neutral_ppo_smoke import run as run_smoke
    from nerva.training.neutral_reference import verified_references
    control = root / "experiments/locomotion_curriculum/results_neutral_ppo_bytes"
    paths = {name: control / name for name in ("protocol.json", "stages.json", "reference_manifest.json")}
    inputs = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    protocol, baseline = inputs["protocol.json"], inputs["stages.json"]
    sources = {name: hashlib.sha256(Path(inspect.getfile(importlib.import_module(name))).read_bytes()).hexdigest()
               for name in protocol["source_hashes"]}
    versions = {name: importlib.metadata.version(name) for name in protocol["versions"]}
    _, manifest = verified_references(root)
    validate_baseline(protocol, baseline, inputs["reference_manifest.json"], sources, versions, manifest)
    baseline_diagnostics = []
    first_contract = None
    for stage in baseline:
        path = root / "experiments/cloud_runs/neutral-ppo-byte-smoke/checkpoints" / stage["stage"] / "000000000016"
        if checkpoint_hashes(path) != stage["checkpoint_hashes"]:
            raise ValueError("frozen checkpoint hash mismatch")
        contract = json.loads((path / "nerva_contract.json").read_text(encoding="utf-8"))
        if first_contract is None:
            first_contract = contract
        require_contract(path, first_contract)
        if (contract["source_hashes"] != sources or contract["network"] != protocol["network"]
                or contract["references"] != [row["reference_sha256"] for row in manifest["references"]]):
            raise ValueError("frozen checkpoint contract mismatch")
        params = checkpoint.load(path)
        if not tree_finite(params) or float(np.asarray(params[0].std_eps)) != 0:
            raise ValueError("frozen checkpoint normalization mismatch")
        baseline_diagnostics.append(diagnostics(params))
    write_json(output / "control_admission.json", {
        "all_pass": True, "control_implementation_commit": protocol["implementation_commit"],
        "input_report_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "normalization_diagnostics": baseline_diagnostics,
        "preregistration_commit": PREREGISTRATION,
        "normalization_source_sha256": hashlib.sha256(Path(inspect.getfile(running_statistics)).read_bytes()).hexdigest(),
        "control_runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    try:
        run_smoke(root, raw, output, normalization_eps=VARIANCE_EPS, preregistration_commit=PREREGISTRATION,
                  stage_observer=lambda name, params: diagnostics(params))
    finally:
        os.chdir(root)
    intervention = json.loads((output / "stages.json").read_text(encoding="utf-8"))
    write_json(output / "comparison.json", assess(intervention, baseline, baseline_diagnostics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.cwd()
    raw = root / "experiments/cloud_runs/normalization-control"
    output = root / "experiments/locomotion_curriculum/results_normalization_control"
    if args.worker:
        try:
            run(root, raw, output)
        except Exception as error:
            write_json(output / "worker_abort.json", {"error_type": type(error).__name__, "all_required_pass": False})
            raise
        return
    if raw.exists() or output.exists():
        raise FileExistsError("never overwrite normalization-control artifacts")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before evaluation")
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        with (raw / "control.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.normalization_control", "--worker"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
    except subprocess.SubprocessError as error:
        write_json(output / "abort.json", {"all_required_pass": False, "error_type": type(error).__name__,
                                          "wall_seconds": round(time.monotonic() - start, 2), "wall_cap_s": 900})
        raise
    write_json(output / "parent_completion.json", {"wall_seconds": round(time.monotonic() - start, 2),
                                                   "wall_cap_s": 900, "worker_returncode": 0})
    print("Normalization control complete; comparison.json contains the prospective decision", flush=True)


if __name__ == "__main__":
    main()
