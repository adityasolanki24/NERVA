"""Auditable parameter warm-start checks; never claims full optimizer continuation."""
import hashlib
import json
from pathlib import Path

import numpy as np


def leaf_comparison(expected, actual):
    import jax
    from flax import serialization
    # State dictionaries preserve named fields while normalizing tuple/list serialization.
    expected, actual = serialization.to_state_dict(expected), serialization.to_state_dict(actual)
    left, right = jax.tree.leaves(expected), jax.tree.leaves(actual)
    if jax.tree.structure(expected) != jax.tree.structure(actual):
        return {"equal": False, "finite": False, "max_error": None, "leaves": len(left)}
    max_error, equal, finite = 0., True, True
    for x, y in zip(left, right):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype:
            return {"equal": False, "finite": False, "max_error": None, "leaves": len(left)}
        finite &= bool(np.isfinite(x).all() and np.isfinite(y).all())
        equal &= x.tobytes(order="C") == y.tobytes(order="C")
        if x.size:
            max_error = max(max_error, float(np.max(np.abs(x.astype(float) - y.astype(float)))))
    return {"equal": equal, "finite": finite, "max_error": max_error, "leaves": len(left)}


def tree_finite(tree):
    import jax
    return all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(tree))


def count_value(count):
    if hasattr(count, "hi") and hasattr(count, "lo"):
        return (int(np.asarray(count.hi)) << 32) + int(np.asarray(count.lo))
    return float(np.asarray(count))


def write_contract(directory, contract):
    path = Path(directory) / "nerva_contract.json"
    if path.exists():
        raise FileExistsError("never overwrite checkpoint contract")
    path.write_text(json.dumps(contract, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def require_contract(directory, expected):
    path = Path(directory) / "nerva_contract.json"
    actual = json.loads(path.read_text(encoding="utf-8"))
    # JSON roundtrip normalizes tuples to lists without weakening field comparisons.
    if actual != json.loads(json.dumps(expected, allow_nan=False)):
        raise ValueError("checkpoint contract mismatch")


def checkpoint_hashes(directory):
    root = Path(directory)
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def save_parameters(directory, step, params, config):
    """Save identical CPU host arrays through Brax, avoiding incompatible JAX sharding telemetry."""
    import jax
    from brax.training.agents.ppo import checkpoint
    from flax import serialization
    if not tree_finite(params):
        raise ValueError("cannot checkpoint nonfinite parameters")
    # UInt64 reconstructs JAX arrays in its constructor, so flatten custom state first.
    host = tuple(jax.tree.map(np.asarray, serialization.to_state_dict(part)) for part in params)
    if not leaf_comparison(params, host)["equal"]:
        raise ValueError("host parameter conversion changed values")
    # Brax's loader treats explicit None initializers as registry names. Omit only
    # these default-valued entries; reconstructed networks retain the same defaults.
    config = type(config)(config.to_dict())
    kwargs = config["network_factory_kwargs"]
    for key in tuple(kwargs):
        if key.endswith("_kernel_init_fn") and kwargs[key] is None:
            del kwargs[key]
    checkpoint.save(directory, step, host, config)
