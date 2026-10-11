"""E1′ conditioning path (docs/expressive_posture_e1prime.md): sampling, observation, targets, widening, export,
native inference."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
START = ROOT / "experiments/cloud_runs/turn_translation-20261011-000711/checkpoints/000059189760"
STYLED = ROOT / "experiments/cloud_runs/e1-styled-references_p1.0_h0.5"
jax = pytest.importorskip("jax")


def _samples(scale, n=20000):
    import jax.numpy as jp

    from nerva.training.style_curriculum import sample_style
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    return np.asarray(jax.vmap(sample_style)(keys, jp.full(n, scale)))


def test_sampling_domain_anchor_curriculum_and_held_out():
    from nerva.training.style_curriculum import curriculum_scale
    e = _samples(1.)
    anchor = np.all(e == 0, axis=1)
    assert abs(anchor.mean() - .2) < .01
    rest = e[~anchor]
    assert rest[:, 0].min() < -.99 and rest[:, 0].max() > .99 and rest[:, 1].min() < .01 and rest[:, 1].max() > .99
    assert not np.any((rest[:, 0] > .5) & (rest[:, 1] > .5))
    small = _samples(.25)
    assert np.abs(small[:, 0]).max() <= .25 and small[:, 1].max() <= .25 and small[:, 1].min() >= 0
    assert curriculum_scale(0.) == .25 and curriculum_scale(.25) == .625 and curriculum_scale(.5) == 1.
    assert curriculum_scale(.9) == 1.


@pytest.mark.skipif(not START.exists(), reason="needs the validated neutral checkpoint")
def test_widened_start_reproduces_the_neutral_policy_for_every_style():
    import functools

    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks

    from nerva.training.b2_warm_start import NETWORK, SHAPE, add_style_inputs
    from nerva.training.motor_artifacts import fingerprint
    params = checkpoint.load(START)
    before = fingerprint(params[0])
    wide = add_style_inputs(params, 2)
    assert fingerprint(params[0]) == before  # the original statistics object is not modified
    for key in ("state", "privileged_state"):
        assert np.array_equal(np.asarray(wide[0].mean[key][:-2]), np.asarray(params[0].mean[key]))
        assert np.array_equal(np.asarray(wide[0].std[key][-2:]), [1., 1.])
        assert np.array_equal(np.asarray(wide[0].mean[key][-2:]), [0., 0.])
    make = functools.partial(networks.make_ppo_networks, **NETWORK)
    n0 = make(SHAPE, 14, preprocess_observations_fn=running_statistics.normalize)
    n1 = make({k: (v[0] + 2,) for k, v in SHAPE.items()}, 14, preprocess_observations_fn=running_statistics.normalize)
    rng = np.random.default_rng(1)
    obs = {k: jp.asarray(rng.normal(size=(4,) + v), jp.float32) for k, v in SHAPE.items()}
    for e in ((0., 0.), (1., 1.), (-1., .3)):
        styled = {k: jp.concatenate([v, jp.tile(jp.asarray(e, jp.float32), (4, 1))], 1) for k, v in obs.items()}
        assert np.array_equal(np.asarray(n0.policy_network.apply(params[0], params[1], obs)),
                              np.asarray(n1.policy_network.apply(wide[0], wide[1], styled)))
        assert np.array_equal(np.asarray(n0.value_network.apply(params[0], params[2], obs)),
                              np.asarray(n1.value_network.apply(wide[0], wide[2], styled)))


@pytest.mark.skipif(not START.exists(), reason="needs the validated neutral checkpoint")
def test_styled_export_takes_103_inputs_and_matches_the_neutral_export(tmp_path):
    import onnxruntime as ort
    from brax.training.agents.ppo import checkpoint

    from nerva.training.b2_warm_start import add_style_inputs, export_policy
    params = checkpoint.load(START)
    export_policy(tmp_path / "wide.onnx", add_style_inputs(params, 2))
    export_policy(tmp_path / "base.onnx", params)
    a = ort.InferenceSession(str(tmp_path / "wide.onnx"), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(tmp_path / "base.onnx"), providers=["CPUExecutionProvider"])
    assert a.get_inputs()[0].shape == [1, 103]
    obs = np.random.default_rng(2).normal(size=(1, 101)).astype(np.float32)
    for e in ((0., 0.), (.75, .75)):
        styled = np.concatenate([obs, np.asarray([e], np.float32)], 1)
        assert np.allclose(a.run(None, {"obs": styled})[0], b.run(None, {"obs": obs})[0], atol=1e-6)


@pytest.mark.slow
@pytest.mark.skipif(not STYLED.exists(), reason="needs the local styled references")
def test_environment_style_observation_targets_and_autoreset_resampling():
    import os

    import jax.numpy as jp

    from nerva.sim.open_duck import OPEN_DUCK_ROOT
    from nerva.training.b2_warm_start import balanced_environment
    from nerva.training.style_curriculum import set_scale
    from nerva.training.styled_reference import StyledNeutralReference, styled_grid
    reference = StyledNeutralReference(styled_grid(ROOT))
    cwd = os.getcwd()
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    try:
        env = balanced_environment(reference, episode_length=3, replicas=4, persistent_command=True,
                                   gait_averaged_tracking=True, base_origin_velocity=True, turn_translation=True,
                                   style_conditioning=True)
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 28))
        assert state.obs["state"].shape == (28, 103) and state.obs["privileged_state"].shape == (28, 214)
        assert np.array_equal(np.asarray(state.obs["state"][:, -2:]), np.asarray(state.info["e1_style"]))
        assert np.abs(np.asarray(state.info["e1_style"])).max() <= .25  # initial curriculum scale
        state = set_scale(state, 1.)
        step = jax.jit(env.step)
        first = np.asarray(state.info["e1_style"])
        rng_before = np.asarray(state.info["style_rng"])
        for i in range(3):
            state = step(state, jp.zeros((28, 14)))
            if i < 2:
                assert np.array_equal(np.asarray(state.info["e1_style"]), first)  # held within the episode
                expected = jax.vmap(lambda c, k, e: reference.get_styled_motion(c[0], c[1], c[2], k, e))(
                    state.info["command"], state.info["imitation_i"], state.info["e1_style"])
                assert np.allclose(np.asarray(state.info["current_reference_motion"]), np.asarray(expected),
                                   atol=1e-6)
        assert np.asarray(state.done).astype(bool).all()  # episode_length 3
        after = np.asarray(state.info["e1_style"])
        assert not np.array_equal(after, first)  # resampled at autoreset
        assert np.array_equal(np.asarray(state.obs["state"][:, -2:]), after)
        assert not np.array_equal(np.asarray(state.info["style_rng"]), rng_before)  # not restored
        assert np.allclose(np.asarray(state.info["style_scale"]), 1.)  # not restored
        assert np.all(np.isfinite(np.asarray(state.reward)))
    finally:
        os.chdir(cwd)


@pytest.mark.skipif(not START.exists(), reason="needs a policy file")
def test_native_styled_contract_appends_style_and_validates_domain(tmp_path):
    import hashlib

    from brax.training.agents.ppo import checkpoint

    from nerva.analysis.motor_eval import deployment_metadata
    from nerva.sim.open_duck import OpenDuckSim, SCENE_BACKLASH
    from nerva.training.b2_warm_start import add_style_inputs, export_policy
    path = tmp_path / "styled.onnx"
    export_policy(path, add_style_inputs(checkpoint.load(START), 2))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sim = OpenDuckSim(policy_path=path, scene=SCENE_BACKLASH)
    with pytest.raises(ValueError):
        sim.set_neutral_style(0., 0.)  # no contract yet
    sim.set_neutral_motor_contract(deployment_metadata(digest, styled=True))
    seen = []
    infer = sim.inf.policy.infer
    sim.inf.policy.infer = lambda obs: (seen.append(np.asarray(obs)), infer(obs))[1]
    sim.set_neutral_style(.75, .75)  # the held-out point is representable
    sim.step_physics(sim.inf.decimation)
    assert seen[-1].shape == (103,) and np.array_equal(seen[-1][-2:], [.75, .75])
    for bad in ((1.2, 0.), (0., -.1), (0., 1.1)):
        with pytest.raises(ValueError):
            sim.set_neutral_style(*bad)
    plain = OpenDuckSim(policy_path=path, scene=SCENE_BACKLASH)
    with pytest.raises(ValueError):
        plain.set_neutral_motor_contract({**deployment_metadata(digest, styled=True), "style": "other"})
