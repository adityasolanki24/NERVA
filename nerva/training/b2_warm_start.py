"""B2 warm start: parameter conversion, frozen preprocessing, balanced resets and ONNX export.

Maintained home (moved from experiments/locomotion_curriculum/learning_support.py, which re-exports it).
"""
import copy
from pathlib import Path

import numpy as np

from nerva.training.neutral_reference import COMMANDS

B2_NAME = "2026_09_30_083416_300482560"
B2_SHA256 = "6ebfe384f4bcb8f06657ba597825744a01a69146ab07583b5f993f15b7f2c9a1"
NETWORK = {"policy_hidden_layer_sizes": (512, 256, 128), "value_hidden_layer_sizes": (512, 256, 128),
           "policy_obs_key": "state", "value_obs_key": "privileged_state"}
SHAPE = {"state": (101,), "privileged_state": (212,)}


def b2_location(root):
    return Path(root) / "experiments/cloud_runs/b2_neutral-20260930-163257/checkpoints" / B2_NAME


def remove_zero_style(weights, mean, std):
    """Fold normalized constant-zero style into the bias without changing other inputs."""
    result = copy.deepcopy(weights)
    first = result["params"]["hidden_0"]
    kernel = np.asarray(first["kernel"])
    mean, std = np.asarray(mean), np.asarray(std)
    if kernel.shape[0] != len(mean) or mean.shape != std.shape or np.any(std <= 0):
        raise ValueError("incompatible first layer or statistics")
    first["bias"] = np.asarray(first["bias"]) + (-mean[-3:] / std[-3:]) @ kernel[-3:]
    first["kernel"] = kernel[:-3].copy()
    return result


def converted_b2(root):
    import hashlib
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from flax import serialization
    from orbax import checkpoint as ocp

    location = b2_location(root)
    if hashlib.sha256(location.with_suffix(".onnx").read_bytes()).hexdigest() != B2_SHA256:
        raise ValueError("historical B2 policy hash mismatch")
    reader = ocp.PyTreeCheckpointer()
    metadata = reader.metadata(str(location)).item_metadata.tree
    # Explicit host restore does not try to reuse the cloud checkpoint's CUDA sharding.
    original = reader.restore(str(location), restore_args=jax.tree.map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata))
    normalizer = copy.deepcopy(original[0])
    for name in ("mean", "std", "summed_variance"):
        normalizer[name] = {key: value[:-3] for key, value in normalizer[name].items()}
    template = running_statistics.init_state({key: jp.zeros(shape) for key, shape in SHAPE.items()})
    normalizer = serialization.from_state_dict(template, normalizer)
    policy = remove_zero_style(original[1], original[0]["mean"]["state"], original[0]["std"]["state"])
    value = remove_zero_style(original[2], original[0]["mean"]["privileged_state"],
                              original[0]["std"]["privileged_state"])
    return normalizer, policy, value


def conversion_parity(root, make_policy, params):
    import jax
    import jax.numpy as jp
    import onnxruntime as ort
    session = ort.InferenceSession(str(b2_location(root).with_suffix(".onnx")),
                                   providers=["CPUExecutionProvider"])
    mean, std = np.asarray(params[0].mean["state"]), np.asarray(params[0].std["state"])
    # Includes real-scale states near the retained distribution, not huge raw inputs.
    probes = mean + np.random.default_rng(27).normal(size=(100, 101)).astype(np.float32) * std
    action_fn = jax.jit(make_policy(params, deterministic=True))
    errors = []
    for obs in probes:
        old = session.run(None, {session.get_inputs()[0].name: np.r_[obs, np.zeros(3)].astype(np.float32)[None]})[0][0]
        new = np.asarray(action_fn({"state": jp.asarray(obs), "privileged_state": jp.zeros(212)},
                                   jax.random.PRNGKey(0))[0])
        errors.append(float(np.max(np.abs(old - new))))
    return {"probes": 100, "max_action_error": max(errors), "all_pass": max(errors) <= 1e-5}


def balanced_environment(reference, episode_length=256, replicas=1, persistent_command=False,
                         gait_averaged_tracking=False):
    """Environment i always uses COMMANDS[i mod 7] (replicas environments per command). Defaults are the
    local pilot's (7 environments, upstream resampling boundary beyond its 256-step episodes);
    persistent_command blocks upstream's step-500 resampling for longer episodes."""
    import jax
    import jax.numpy as jp
    from brax.envs.wrappers import training
    from nerva.training.neutral_joystick import (
        GaitAveragedTrackingNeutralJoystick, NeutralJoystick, PersistentNeutralJoystick,
    )
    from nerva.training.neutral_wrapper import NeutralAutoResetWrapper

    class BalancedVmap(training.VmapWrapper):
        def reset(self, rng):
            def reset_one(key, command):
                state = self.env.reset(key)
                info = state.info
                info["command"] = jp.concatenate([command, jp.zeros(4)])
                self.env._initialize_reference(info)
                info["imitation_i"] = jp.int32(0)
                info["current_reference_motion"] = self.env.SREF.get_reference_motion(*command, 0)
                # The superclass observation is discarded; retain its actual initial contacts.
                contact = state.obs["state"][-4:-2].astype(bool)
                obs = self.env._get_obs(state.data, info, contact)
                return state.replace(obs=obs)
            return jax.vmap(reset_one)(rng, jp.tile(jp.asarray(COMMANDS), (replicas, 1)))

    if gait_averaged_tracking and not persistent_command:
        raise ValueError("gait-averaged tracking is defined for the persistent-command environment")
    base = (GaitAveragedTrackingNeutralJoystick if gait_averaged_tracking
            else PersistentNeutralJoystick if persistent_command else NeutralJoystick)
    env = BalancedVmap(base(reference, task="flat_terrain_backlash"))
    return NeutralAutoResetWrapper(training.EpisodeWrapper(env, episode_length, action_repeat=1))


def export_policy(path, params):
    """Deterministic frozen-statistics SiLU policy, without TensorFlow or upstream edits."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    path = Path(path)
    if path.exists():
        raise FileExistsError("never overwrite an exported policy")
    nodes, constants = [], []

    def constant(name, value):
        constants.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    constant("mean", np.asarray(params[0].mean["state"], dtype=np.float32))
    constant("std", np.asarray(params[0].std["state"], dtype=np.float32))
    nodes.extend([helper.make_node("Sub", ["obs", "mean"], ["centered"]),
                  helper.make_node("Div", ["centered", "std"], ["normalized"])])
    previous = "normalized"
    for i in range(4):
        layer = params[1]["params"][f"hidden_{i}"]
        weight = constant(f"w{i}", np.asarray(layer["kernel"], dtype=np.float32))
        bias = constant(f"b{i}", np.asarray(layer["bias"], dtype=np.float32))
        nodes.extend([helper.make_node("MatMul", [previous, weight], [f"mm{i}"]),
                      helper.make_node("Add", [f"mm{i}", bias], [f"z{i}"])])
        previous = f"z{i}"
        if i < 3:
            nodes.extend([helper.make_node("Sigmoid", [previous], [f"sig{i}"]),
                          helper.make_node("Mul", [previous, f"sig{i}"], [f"silu{i}"])])
            previous = f"silu{i}"
    for name, values in (("starts", [0]), ("ends", [14]), ("axes", [1])):
        constant(name, np.array(values, dtype=np.int64))
    nodes.extend([helper.make_node("Slice", [previous, "starts", "ends", "axes"], ["loc"]),
                  helper.make_node("Tanh", ["loc"], ["continuous_actions"])])
    graph = helper.make_graph(nodes, "neutral_frozen_b2", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 101])],
                             [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, 14])], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)
