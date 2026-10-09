"""Bounded local B2 warm refinement; docs/neutral_learning_pilot.md is frozen first."""
import argparse
import functools
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.locomotion_curriculum.gate import write_json
from experiments.locomotion_curriculum.learning_support import (
    B2_SHA256, NETWORK, SHAPE, balanced_environment, conversion_parity, converted_b2, export_policy,
)
from experiments.locomotion_curriculum.normalization_timing import archive, fingerprint, gaussian_kl
from nerva.training.neutral_reference import COMMANDS
from nerva.training.parameter_checkpoint import checkpoint_hashes, leaf_comparison, save_parameters, tree_finite

PREREGISTRATION = "8fd4fd3"
LOSS = {"entropy_cost": .005, "discounting": .97, "reward_scaling": 1., "gae_lambda": .95,
        "clipping_epsilon": .2, "normalize_advantage": True, "vf_coefficient": .5}


def collect_balanced_batch(env, make_policy, params, physical, rng):
    import jax
    import jax.numpy as jp
    from brax.training import acting
    rng, unroll_key = jax.random.split(rng)
    next_state, data = acting.generate_unroll(env, physical, make_policy(params), unroll_key, 32,
        extra_fields=("truncation", "episode_done", "command"))
    return next_state, jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), data), rng


def run(root, raw, output):
    import jax
    import jax.numpy as jp
    import optax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, losses, networks
    from nerva.sim.open_duck import OPEN_DUCK_ROOT
    from nerva.training.neutral_reference import NeutralReference, verified_references

    start = time.monotonic()
    records, manifest = verified_references(root)
    factory = functools.partial(networks.make_ppo_networks, **NETWORK)
    net = factory(SHAPE, 14, preprocess_observations_fn=running_statistics.normalize)
    make_policy = networks.make_inference_fn(net)
    params = converted_b2(root)
    if not tree_finite(params):
        raise ValueError("nonfinite source checkpoint")
    parity = conversion_parity(root, make_policy, params)
    write_json(output / "conversion.json", parity)
    if not parity["all_pass"]:
        raise ValueError("B2 conversion action parity failed")
    normalizer = params[0]
    frozen_hash = fingerprint(normalizer)
    weights = losses.PPONetworkParams(policy=params[1], value=params[2])
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(1e-5))
    opt_state = optimizer.init(weights)
    loss_fn = functools.partial(losses.compute_ppo_loss, ppo_network=net, **LOSS)
    config = checkpoint.network_config(SHAPE, 14, True, factory)

    def save(step, current):
        folder = raw / "checkpoints" / f"{step:012d}"
        if not folder.exists():
            save_parameters(raw / "checkpoints", step, (normalizer, current.policy, current.value), config)
        return folder

    save(0, weights)
    export_policy(raw / "initial.onnx", params)
    write_json(output / "protocol.json", {"preregistration_commit": PREREGISTRATION,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "b2_policy_sha256": B2_SHA256, "network": NETWORK, "loss": LOSS, "seed": 27,
        "num_envs": 7, "unroll_length": 32, "episode_length": 256, "max_transitions": 28672,
        "max_optimizer_updates": 128, "learning_rate": 1e-5, "grad_clip": 1.,
        "preprocessing": "frozen_cropped_B2_statistics", "statistics_sha256": frozen_hash,
        "warm_start": "parameters_only_fresh_adam_rng_environment", "reference_manifest": manifest,
        "wall_cap_s": 1200, "cloud_work": False})
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    env = balanced_environment(NeutralReference(records))
    key, reset_key = jax.random.split(jax.random.PRNGKey(27))
    state = jax.jit(env.reset)(jax.random.split(reset_key, 7))

    @jax.jit
    def collect(current, physical, rng):
        return collect_balanced_batch(env, make_policy, (normalizer, current.policy, current.value), physical, rng)

    @jax.jit
    def replay(current, data):
        obs = jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), data.observation)
        logits = net.policy_network.apply(normalizer, current.policy, obs)
        dist = net.parametric_action_distribution.create_dist(logits)
        log_prob = net.parametric_action_distribution.log_prob(
            logits, jp.swapaxes(data.extras["policy_extras"]["raw_action"], 0, 1))
        return logits, log_prob, dist.loc, dist.scale

    @jax.jit
    def optimize(current, data, rng, optimizer_state):
        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(current, normalizer, data, rng)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, current)
        return optax.apply_updates(current, updates), optimizer_state, gradients, metrics

    rows, accepted = [], 0
    stop = "transition_cap"
    for index in range(128):
        if time.monotonic() - start >= 1100:
            stop = "wall_cap"
            break
        batch_start = time.monotonic()
        state, data, key = collect(weights, state, key)
        if not tree_finite(data):
            stop = "nonfinite_data"
            break
        commands = np.asarray(data.extras["state_extras"]["command"])[..., :3]
        coverage = np.allclose(commands, np.asarray(COMMANDS)[:, None, :], atol=1e-8, rtol=0)
        before = replay(weights, data)
        behavior = data.extras["policy_extras"]
        errors = {"logits": float(np.max(np.abs(before[0] - np.swapaxes(behavior["distribution_params"], 0, 1)))),
                  "log_prob": float(np.max(np.abs(before[1] - np.swapaxes(behavior["log_prob"], 0, 1))))}
        if not coverage or not tree_finite(before) or errors["logits"] > 1e-4 or errors["log_prob"] > 1e-3:
            write_json(output / "integrity_stop.json", {"batch": index, "coverage": bool(coverage), "errors": errors})
            stop = "collection_integrity"
            break
        key, loss_key = jax.random.split(key)
        after, next_opt, gradients, metrics = optimize(weights, data, loss_key, opt_state)
        if not tree_finite((after, next_opt, gradients, metrics)):
            archive(raw / "nonfinite_update.npz", (after, next_opt, gradients, metrics))
            stop = "nonfinite_update"
            break
        post = replay(after, data)
        if not tree_finite(post):
            archive(raw / "nonfinite_policy.npz", post)
            stop = "nonfinite_policy"
            break
        kl = float(np.mean(gaussian_kl(before[2], before[3], post[2], post[3])))
        discount = np.asarray(data.discount)
        truncation = np.asarray(data.extras["state_extras"]["truncation"])
        row = {"batch": index + 1, "collected_transitions": (index + 1) * 224,
            "post_update_kl": kl, "accepted": kl <= .05, "control_errors": errors,
            "reward_by_command": np.mean(np.asarray(data.reward), axis=1).tolist(),
            "true_terminations_by_command": np.sum((discount == 0) & (truncation == 0), axis=1).tolist(),
            "timeouts_by_command": np.sum(truncation != 0, axis=1).tolist(),
            "gradient_norm": float(optax.global_norm(gradients)),
            "metrics": {name: float(value) for name, value in metrics.items()},
            "batch_wall_seconds": round(time.monotonic() - batch_start, 3),
            "wall_seconds": round(time.monotonic() - start, 3)}
        rows.append(row)
        write_json(output / "training.json", rows)
        if kl > .05:
            archive(raw / "rejected_update.npz", (after, next_opt, data))
            stop = "deployed_kl"
            break
        weights, opt_state = after, next_opt
        accepted += 1
        if accepted % 16 == 0:
            save(accepted * 224, weights)
            archive(raw / f"training_state_{accepted:03d}.npz", (weights, opt_state, data, key))
        if index % 8 == 0 or accepted == 128:
            print(f"batch {index + 1}/128 steps {(index + 1) * 224} KL {kl:.6f} "
                  f"reward {np.mean(data.reward):.4f} elapsed {row['wall_seconds']:.1f}s", flush=True)
    if fingerprint(normalizer) != frozen_hash:
        raise ValueError("frozen statistics changed")
    final = save(accepted * 224, weights)
    restored = checkpoint.load(final)
    equality = leaf_comparison((normalizer, weights.policy, weights.value), restored)
    if not equality["equal"] or not equality["finite"]:
        raise ValueError("final checkpoint roundtrip failed")
    export_policy(raw / "candidate.onnx", restored)
    write_json(output / "training_summary.json", {"accepted_updates": accepted,
        "accepted_transitions": accepted * 224, "collected_transitions": len(rows) * 224,
        "stop_reason": stop, "wall_seconds": round(time.monotonic() - start, 2),
        "frozen_statistics_unchanged": True, "roundtrip": equality,
        "checkpoint_hashes": checkpoint_hashes(final), "cloud_work": False,
        "motor_readiness": "pending_paired_evaluation"})
    print("Training finished", accepted * 224, stop, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retry", action="store_true", help="retain the pre-collection interface abort separately")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    suffix = "-retry" if args.retry else ""
    raw = root / f"experiments/cloud_runs/neutral-learning-pilot{suffix}"
    output = root / ("experiments/locomotion_curriculum/results_neutral_learning" + suffix.replace("-", "_"))
    if args.worker:
        run(root, raw, output)
        return
    if raw.exists() or output.exists():
        raise FileExistsError("never overwrite pilot artifacts")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before training")
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    with (raw / "training.log").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "experiments.locomotion_curriculum.neutral_learning", "--worker",
                        *(["--retry"] if args.retry else [])],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1200,
                       env={**os.environ, "JAX_PLATFORMS": "cpu"})
    print("Training worker complete; run paired_motor evaluation", flush=True)


if __name__ == "__main__":
    main()
