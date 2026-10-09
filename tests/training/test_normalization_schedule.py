import numpy as np
import pytest

from experiments.locomotion_curriculum.normalization_schedule import decide, rebase_mlp


def test_affine_rebase_preserves_preactivation_across_scales_and_keeps_other_layers_literal():
    pytest.importorskip("jax")
    from flax import core
    rng = np.random.default_rng(13)
    w, b = rng.normal(size=(4, 3)).astype(np.float32), rng.normal(size=3).astype(np.float32)
    untouched = np.array([-0., 1.], np.float32)
    original = core.freeze({"params": {"hidden_0": {"kernel": w, "bias": b}, "later": {"bias": untouched}}})
    m, s = np.array([1, -2, 3, .5], np.float32), np.array([.5, 2, .01, 10], np.float32)
    new_m, new_s = m + .1, s * np.array([2, .1, 10, .5], np.float32)
    rebased = rebase_mlp(original, m, s, new_m, new_s)
    assert isinstance(rebased, core.FrozenDict)
    raw = rng.normal(size=(2, 4, 4)).astype(np.float32)
    expected = ((raw - m) / s) @ w + b
    layer = rebased["params"]["hidden_0"]
    actual = ((raw - new_m) / new_s) @ np.asarray(layer["kernel"]) + np.asarray(layer["bias"])
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)
    assert np.asarray(rebased["params"]["later"]["bias"]).tobytes() == untouched.tobytes()
    assert original["params"]["hidden_0"]["kernel"].tobytes() == w.tobytes()


@pytest.mark.parametrize("std", ([0., 1.], [-1., 1.], [float("nan"), 1.]))
def test_rebase_rejects_invalid_normalizer_scale(std):
    pytest.importorskip("jax")
    w = {"params": {"hidden_0": {"kernel": np.ones((2, 3), np.float32), "bias": np.ones(3, np.float32)}}}
    with pytest.raises(ValueError, match="unsupported"):
        rebase_mlp(w, [0., 0.], std, [1., 1.], [1., 1.])


def test_schedule_decisions_do_not_confuse_training_kl_with_deployment_or_rebase_support():
    integrity, checks = {"fixed_batch": True}, {"value": True}
    fixed, deferred = {"kl_mean": .02, "v_loss": .1}, {"kl_mean": 10., "v_loss": 20.}
    result = decide(integrity, fixed, deferred, checks, fixed)
    assert result["fixed_preprocessing_screen_pass"] and result["deferred_deployment_drift_supported"]
    assert result["rebase_support"] and not result["optimizer_rebase_equivalence"]
    assert not decide({"fixed_batch": False}, fixed, deferred, checks, fixed)["fixed_preprocessing_screen_pass"]
    assert not decide(integrity, {**fixed, "kl_mean": 1.01}, deferred, checks, fixed)["rebase_support"]
    assert not decide(integrity, fixed, deferred, {"value": False}, fixed)["rebase_support"]
    assert not decide(integrity, fixed, deferred, checks, {**fixed, "v_loss": 101.})["rebase_support"]
