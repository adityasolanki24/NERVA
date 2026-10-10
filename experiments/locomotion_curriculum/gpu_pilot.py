"""Capped GPU neutral continuation; protocol frozen in docs/gpu_neutral_pilot.md before implementation.

Cloud:  bash cloud/jobs/neutral_gpu_pilot.sh   (runs `--out /work/out` on the VM)
Smoke:  python -m experiments.locomotion_curriculum.gpu_pilot --smoke   (CPU, tiny, local; not a result)

Starts from the local pilot's final accepted checkpoint (hash-verified), keeps B2's cropped observation
statistics frozen, trains with minibatch PPO epochs on balanced persistent-command environments
(environment i always runs COMMANDS[i mod 7]), and stops at the declared deadline, step ceiling or any
integrity stop. Saves parameter checkpoints plus Adam/key/iteration snapshots: a parameter + optimizer
restart, not a physical-environment resume.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from nerva.training.b2_warm_start import NETWORK, SHAPE, balanced_environment, export_policy
from nerva.training.motor_artifacts import archive, fingerprint, gaussian_kl, write_json
from nerva.training.neutral_reference import COMMANDS
from nerva.training.parameter_checkpoint import checkpoint_hashes, leaf_comparison, save_parameters, tree_finite

# Named experiments. Each fixes its preregistration, starting checkpoint (hash-verified against the report that
# produced it), the source of the frozen statistics hash, and the reward variant. Existing entries never change.
EXPERIMENTS = {
    "gpu_neutral_pilot": {  # docs/gpu_neutral_pilot.md
        "preregistration": "a3ba5ca",
        "start": "experiments/cloud_runs/neutral-learning-pilot-corrected/checkpoints/000000028672",
        "start_report": "experiments/locomotion_curriculum/results_neutral_learning_corrected/training_summary.json",
        "gait_averaged_tracking": False},
    "gait_averaged_tracking": {  # docs/gait_averaged_tracking_pilot.md
        "preregistration": "a143ae4",
        "start": "experiments/cloud_runs/neutral_gpu_pilot-20261010-152812/checkpoints/000060318720",
        "start_report": "experiments/locomotion_curriculum/results_gpu_pilot/training_summary.json",
        "gait_averaged_tracking": True},
    "base_origin_velocity": {  # docs/base_origin_velocity_pilot.md
        "preregistration": "69a6f0f",
        "start": "experiments/cloud_runs/gait_averaged_tracking-20261010-170130/checkpoints/000061931520",
        "start_report": "experiments/locomotion_curriculum/results_gait_averaged_tracking/training_summary.json",
        "gait_averaged_tracking": True, "base_origin_velocity": True},
    "turn_translation": {  # docs/turn_translation_pilot.md
        "preregistration": "PENDING",
        "start": "experiments/cloud_runs/base_origin_velocity-20261010-201858/checkpoints/000060480000",
        "start_report": "experiments/locomotion_curriculum/results_base_origin_velocity/training_summary.json",
        "gait_averaged_tracking": True, "base_origin_velocity": True, "turn_translation": True},
}
STATISTICS_SOURCE = "experiments/locomotion_curriculum/results_neutral_learning_corrected/protocol.json"
LOSS = {"entropy_cost": .005, "discounting": .97, "reward_scaling": 1., "gae_lambda": .95,
        "clipping_epsilon": .2, "normalize_advantage": True, "vf_coefficient": .5}
FULL = {"replicas": 1152, "episode_length": 1000, "unroll_length": 20, "num_minibatches": 32, "epochs": 4,
        "learning_rate": 1e-4, "seed": 41, "checkpoint_every": 20, "deadline_uptime_s": 2220,
        "max_steps": 80_000_000, "kl_limit": .1, "logit_tolerance": 1e-3, "log_prob_tolerance": 1e-2,
        "timed_iterations": (2, 3, 4), "max_iterations": None, "wall_s": None}
SMOKE = {**FULL, "replicas": 1, "episode_length": 40, "unroll_length": 4, "num_minibatches": 1, "epochs": 1,
         "checkpoint_every": 1, "deadline_uptime_s": None, "max_iterations": 2, "wall_s": 900,
         "timed_iterations": (2,), "logit_tolerance": 1e-4, "log_prob_tolerance": 1e-3}


def uptime_s() -> float | None:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def verify_start(root: Path, experiment: dict) -> dict:
    """Every file of the starting checkpoint must match the hashes recorded by the run that produced it."""
    summary = json.loads((root / experiment["start_report"]).read_text(encoding="utf-8"))
    actual = checkpoint_hashes(root / experiment["start"])
    if actual != summary["checkpoint_hashes"]:
        raise ValueError("starting checkpoint hash mismatch")
    statistics = json.loads((root / STATISTICS_SOURCE).read_text(encoding="utf-8"))["statistics_sha256"]
    return {"checkpoint": experiment["start"], "files": len(actual), "statistics_sha256": statistics}


def per_command(values, replicas):
    """[N, ...] with environment i on COMMANDS[i mod 7] → per-command means (canonical order)."""
    values = np.asarray(values, dtype=np.float64).reshape(replicas, 7, -1)
    return values.mean(axis=(0, 2)).tolist()


def run(root: Path, raw: Path, cfg: dict, name: str = "gpu_neutral_pilot") -> None:
    import jax
    import jax.numpy as jp
    import optax
    from brax.training import acting
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, losses, networks
    from nerva.sim.open_duck import OPEN_DUCK_ROOT
    from nerva.training.neutral_reference import NeutralReference, verified_references

    start = time.monotonic()

    def remaining() -> float:
        if cfg["deadline_uptime_s"] is not None and uptime_s() is not None:
            return cfg["deadline_uptime_s"] - uptime_s()
        return cfg["wall_s"] - (time.monotonic() - start)

    report = raw / "report"
    report.mkdir(parents=True, exist_ok=True)
    devices = [str(d) for d in jax.devices()]
    records, manifest = verified_references(root)
    experiment = EXPERIMENTS[name]
    admitted = verify_start(root, experiment)
    params = checkpoint.load(root / experiment["start"])
    normalizer = params[0]
    frozen = fingerprint(normalizer)
    if frozen != admitted["statistics_sha256"]:
        raise ValueError("starting normalizer differs from the pilot's frozen statistics")
    factory = functools.partial(networks.make_ppo_networks, **NETWORK)
    net = factory(SHAPE, 14, preprocess_observations_fn=running_statistics.normalize)
    make_policy = networks.make_inference_fn(net)
    weights = losses.PPONetworkParams(policy=params[1], value=params[2])
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(cfg["learning_rate"]))
    opt_state = optimizer.init(weights)
    loss_fn = functools.partial(losses.compute_ppo_loss, ppo_network=net, **LOSS)
    config = checkpoint.network_config(SHAPE, 14, True, factory)
    replicas, unroll = cfg["replicas"], cfg["unroll_length"]
    n = 7 * replicas
    per_iteration = n * unroll
    if n % cfg["num_minibatches"]:
        raise ValueError("environments must divide into minibatches")
    expected = np.tile(np.asarray(COMMANDS, dtype=np.float32), (replicas, 1))
    write_json(report / "protocol.json", {
        "experiment": name, "preregistration_commit": experiment["preregistration"],
        "gait_averaged_tracking": experiment["gait_averaged_tracking"],
        "base_origin_velocity": experiment.get("base_origin_velocity", False),
        "turn_translation": experiment.get("turn_translation", False), "config": {k: v for k, v in cfg.items()},
        "loss": LOSS, "network": NETWORK, "devices": devices, "start": admitted,
        "environments": n, "transitions_per_iteration": per_iteration, "reference_manifest": manifest,
        "restoration": "parameters_only; fresh Adam/RNG/environment; snapshots allow parameter+optimizer "
                       "restart, not physical-environment resume",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "code_commit": os.environ.get("NERVA_CODE_SHA", "")})

    def save(step, current):
        folder = raw / "checkpoints" / f"{step:012d}"
        if not folder.exists():
            save_parameters(raw / "checkpoints", step, (normalizer, current.policy, current.value), config)
        if fingerprint(normalizer) != frozen:
            raise ValueError("frozen statistics changed")
        return folder

    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    env = balanced_environment(NeutralReference(records), episode_length=cfg["episode_length"],
                               replicas=replicas, persistent_command=True,
                               gait_averaged_tracking=experiment["gait_averaged_tracking"],
                               base_origin_velocity=experiment.get("base_origin_velocity", False),
                               turn_translation=experiment.get("turn_translation", False))
    key, reset_key = jax.random.split(jax.random.PRNGKey(cfg["seed"]))
    state = jax.jit(env.reset)(jax.random.split(reset_key, n))

    @jax.jit
    def collect(current, physical, rng):
        rng, unroll_key = jax.random.split(rng)
        policy = make_policy((normalizer, current.policy, current.value))
        physical, data = acting.generate_unroll(env, physical, policy, unroll_key, unroll,
                                                extra_fields=("truncation", "episode_done", "command"))
        return physical, jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), data), rng  # [N, T]

    @jax.jit
    def replay(current, data):
        logits = net.policy_network.apply(normalizer, current.policy, data.observation)
        dist = net.parametric_action_distribution.create_dist(logits)
        log_prob = net.parametric_action_distribution.log_prob(logits, data.extras["policy_extras"]["raw_action"])
        return logits, log_prob, dist.loc, dist.scale

    @jax.jit
    def sgd(current, optimizer_state, data, rng):
        def epoch(carry, epoch_key):
            w, o = carry
            permute_key, loss_key = jax.random.split(epoch_key)
            order = jax.random.permutation(permute_key, n)
            batches = jax.tree.map(lambda x: x[order].reshape((cfg["num_minibatches"], -1) + x.shape[1:]), data)

            def minibatch(carry2, item):
                w2, o2 = carry2
                batch, k = item
                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(w2, normalizer, batch, k)
                updates, o2 = optimizer.update(grads, o2, w2)
                return (optax.apply_updates(w2, updates), o2), (metrics, optax.global_norm(grads))

            keys = jax.random.split(loss_key, cfg["num_minibatches"])
            return jax.lax.scan(minibatch, (w, o), (batches, keys))

        (w, o), (metrics, norms) = jax.lax.scan(epoch, (current, optimizer_state),
                                                 jax.random.split(rng, cfg["epochs"]))
        return w, o, jax.tree.map(jp.mean, metrics), jp.max(norms)

    rows, accepted, ceiling, stop = [], 0, None, "deadline"
    timings, throughput = {}, None
    while True:
        if remaining() <= 0:
            stop = "deadline"
            break
        if cfg["max_iterations"] is not None and accepted >= cfg["max_iterations"]:
            stop = "iteration_cap"
            break
        if ceiling is not None and accepted >= ceiling:
            stop = "step_ceiling"
            break
        t0 = time.monotonic()
        state, data, key = collect(weights, state, key)
        jax.block_until_ready(data.reward)
        commands = np.asarray(data.extras["state_extras"]["command"])[..., :3]
        coverage = bool(np.allclose(commands, expected[:, None, :], atol=1e-8, rtol=0))
        if not tree_finite(data):
            stop = "nonfinite_data"
            break
        before = replay(weights, data)
        behavior = data.extras["policy_extras"]
        errors = {"logits": float(np.max(np.abs(np.asarray(before[0]) - np.asarray(behavior["distribution_params"])))),
                  "log_prob": float(np.max(np.abs(np.asarray(before[1]) - np.asarray(behavior["log_prob"]))))}
        if (not coverage or not tree_finite(before) or errors["logits"] > cfg["logit_tolerance"]
                or errors["log_prob"] > cfg["log_prob_tolerance"]):
            write_json(report / "integrity_stop.json", {"iteration": accepted + 1, "coverage": coverage, "errors": errors})
            stop = "collection_integrity"
            break
        key, sgd_key = jax.random.split(key)
        after, next_opt, metrics, grad_norm = sgd(weights, opt_state, data, sgd_key)
        if not tree_finite((after, next_opt, metrics)):
            archive(raw / "nonfinite_update.npz", (after, next_opt))
            stop = "nonfinite_update"
            break
        post = replay(after, data)
        if not tree_finite(post):
            stop = "nonfinite_policy"
            break
        kl = float(np.mean(gaussian_kl(before[2], before[3], post[2], post[3])))
        discount = np.asarray(data.discount)
        truncation = np.asarray(data.extras["state_extras"]["truncation"])
        seconds = time.monotonic() - t0
        row = {"iteration": accepted + 1, "transitions": (accepted + 1) * per_iteration, "post_update_kl": kl,
               "accepted": kl <= cfg["kl_limit"], "control_errors": errors,
               "reward_by_command": per_command(data.reward, replicas),
               "true_terminations_by_command": per_command((discount == 0) & (truncation == 0), replicas),
               "timeouts_by_command": per_command(truncation != 0, replicas),
               "max_gradient_norm": float(grad_norm),
               "metrics": {name: float(value) for name, value in metrics.items()},
               "iteration_seconds": round(seconds, 3), "remaining_s": round(remaining(), 1),
               "uptime_s": uptime_s(), "wall_s": round(time.monotonic() - start, 1)}
        rows.append(row)
        write_json(report / "training.json", rows)
        if kl > cfg["kl_limit"]:
            archive(raw / "rejected_update.npz", (after, next_opt))
            stop = "deployed_kl"
            break
        weights, opt_state = after, next_opt
        accepted += 1
        if accepted in cfg["timed_iterations"]:
            timings[accepted] = seconds
        if throughput is None and accepted == max(cfg["timed_iterations"]):
            throughput = per_iteration * len(timings) / sum(timings.values())
            ceiling = min(cfg["max_steps"] // per_iteration,
                          accepted + int(max(0., remaining()) * throughput // per_iteration))
            write_json(report / "step_ceiling.json", {"timed_iterations": timings,
                       "steps_per_second": throughput, "ceiling_iterations": ceiling,
                       "ceiling_transitions": ceiling * per_iteration, "remaining_s_at_decision": remaining()})
        if accepted % cfg["checkpoint_every"] == 0:
            save(accepted * per_iteration, weights)
            archive(raw / "checkpoints" / f"snapshot_{accepted:05d}.npz", (weights, opt_state, key, np.int64(accepted)))
        print(f"iteration {accepted} transitions {accepted * per_iteration} KL {kl:.5f} "
              f"reward {np.mean(row['reward_by_command']):.4f} {seconds:.2f}s remaining {remaining():.0f}s", flush=True)
    final = save(accepted * per_iteration, weights)
    restored = checkpoint.load(final)
    equality = leaf_comparison((normalizer, weights.policy, weights.value), restored)
    if not equality["equal"] or not equality["finite"]:
        raise ValueError("final checkpoint roundtrip failed")
    export_policy(raw / "candidate.onnx", restored)
    write_json(report / "training_summary.json", {
        "accepted_iterations": accepted, "accepted_transitions": accepted * per_iteration,
        "stop_reason": stop, "steps_per_second_measured": throughput, "step_ceiling_iterations": ceiling,
        "wall_seconds": round(time.monotonic() - start, 1), "uptime_at_exit_s": uptime_s(),
        "frozen_statistics_unchanged": fingerprint(normalizer) == frozen, "roundtrip": equality,
        "final_checkpoint": final.name, "checkpoint_hashes": checkpoint_hashes(final),
        "candidate_onnx_sha256": hashlib.sha256((raw / "candidate.onnx").read_bytes()).hexdigest(),
        "devices": devices, "motor_readiness": "pending_local_paired_evaluation"})
    print("training finished", accepted * per_iteration, stop, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, help="output directory (cloud: /work/out)")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU check; not a result")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="gpu_neutral_pilot")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.smoke:
        raw = args.out or root / f"experiments/cloud_runs/gpu-pilot-smoke-{time.strftime('%Y%m%d-%H%M%S')}"
        cfg = SMOKE
    else:
        if args.out is None:
            raise SystemExit("--out is required for the GPU pilot")
        # Cloud runs execute a committed `git archive` (NERVA_CODE_SHA); locally, require a clean tree.
        if not os.environ.get("NERVA_CODE_SHA") and subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
            raise SystemExit("commit before training")
        raw, cfg = args.out, FULL
    if (raw / "report" / "training_summary.json").exists():
        raise FileExistsError("never overwrite pilot artifacts")
    if not args.smoke and EXPERIMENTS[args.experiment]["preregistration"] == "PENDING":
        raise SystemExit("experiment is not preregistered yet")
    raw = raw.resolve()  # the trainer later changes into the upstream checkout; never write there
    raw.mkdir(parents=True, exist_ok=True)
    run(root, raw, cfg, args.experiment)


if __name__ == "__main__":
    main()
