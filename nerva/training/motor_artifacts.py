"""Artifact helpers shared by the neutral motor workflow (moved unchanged from completed runners, 2026-10-10).

JSON reports (finite only), named array archives that never overwrite, tree fingerprints, and an
independent float64 Gaussian KL. Pure NumPy, with jax/flax imported lazily for tree flattening.
"""
import hashlib
import json

import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def tree_arrays(tree):
    import jax
    from flax import serialization
    state = serialization.to_state_dict(tree)
    leaves, structure = jax.tree_util.tree_flatten_with_path(state)
    return {jax.tree_util.keystr(path) or "root": np.asarray(value) for path, value in leaves}, str(structure)


def fingerprint(tree):
    arrays, structure = tree_arrays(tree)
    digest = hashlib.sha256(structure.encode())
    for name, array in arrays.items():
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def archive(path, tree):
    arrays, _ = tree_arrays(tree)
    if path.exists():
        raise FileExistsError("never overwrite a captured artifact")
    np.savez_compressed(path, **arrays)
    return {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tree_sha256": fingerprint(tree),
            "bytes": path.stat().st_size, "leaves": len(arrays)}


def gaussian_kl(old_loc, old_scale, new_loc, new_scale, log_epsilon=0.):
    """Independent float64 KL(old || new), optionally matching Brax's log stabilizer."""
    old_loc, old_scale, new_loc, new_scale = [np.asarray(x, dtype=np.float64)
                                            for x in (old_loc, old_scale, new_loc, new_scale)]
    if np.any(old_scale <= 0) or np.any(new_scale <= 0):
        raise ValueError("Gaussian scales must be positive")
    return np.sum(np.log(new_scale / old_scale + log_epsilon)
                  + (old_scale**2 + (old_loc - new_loc)**2) / (2 * new_scale**2) - .5, axis=-1)
