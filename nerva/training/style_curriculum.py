"""E1′ style sampling with a neutral-anchored curriculum (docs/expressive_posture_e1prime.md §4).

At every episode start (initial reset and autoreset) each environment draws e and holds it for the episode:
with probability 0.2 exactly e = (0, 0); otherwise e = (s·U(−1, 1), s·U(0, 1)), rejecting the held-out region
e_pitch > 0.5 and e_crouch > 0.5. The scale s is per-environment state set by the trainer every iteration
(s = min(1, 0.25 + 1.5·u), u = completed fraction of the step-ceiling iterations).
"""
from __future__ import annotations

import jax
import jax.numpy as jp

NEUTRAL_ANCHOR = .2
HELD_OUT = (.5, .5)  # e_pitch > 0.5 and e_crouch > 0.5 is never sampled
CANDIDATES = 16
INITIAL_SCALE = .25


def curriculum_scale(u):
    """s(u) for u = completed fraction of the step-ceiling iterations."""
    return min(1., INITIAL_SCALE + 1.5 * max(0., u))


def held_out(e):
    return jp.logical_and(e[..., 0] > HELD_OUT[0], e[..., 1] > HELD_OUT[1])


def sample_style(key, scale):
    """One e for a given scale: neutral anchor with p = 0.2, else rejection-sampled outside the held-out region."""
    anchor_key, candidate_key = jax.random.split(key)
    u = jax.random.uniform(candidate_key, (CANDIDATES, 2))
    candidates = jp.stack([scale * (2 * u[:, 0] - 1), scale * u[:, 1]], axis=1)
    ok = ~held_out(candidates)
    first = jp.argmax(ok)
    chosen = jp.where(ok[first], candidates[first], candidates[first] * jp.array([-1., 1.]))  # reflection: p ≈ 1e-15
    return jp.where(jax.random.uniform(anchor_key) < NEUTRAL_ANCHOR, jp.zeros(2), chosen)


class StyleCurriculumWrapper:
    """Outermost wrapper over the batched (autoresetting) environment."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _assign(self, state, mask):
        keys = jax.vmap(jax.random.split)(state.info["style_rng"])
        new = jax.vmap(sample_style)(keys[:, 1], state.info["style_scale"])
        style = jp.where(mask[:, None], new, state.info["e1_style"])
        obs = {k: v.at[:, -2:].set(jp.where(mask[:, None], new, v[:, -2:])) for k, v in state.obs.items()}
        info = {**state.info, "e1_style": style, "style_rng": keys[:, 0]}
        return state.replace(obs=obs, info=info)

    def reset(self, rng):
        state = self.env.reset(rng)
        n = state.obs["state"].shape[0]
        info = {**state.info, "style_rng": jax.vmap(lambda k: jax.random.fold_in(k, 7))(rng),
                "style_scale": jp.full(n, INITIAL_SCALE)}
        return self._assign(state.replace(info=info), jp.ones(n, bool))

    def step(self, state, action):
        state = self.env.step(state, action)
        return self._assign(state, state.done.astype(bool))


def set_scale(state, scale):
    return state.replace(info={**state.info, "style_scale": jp.full_like(state.info["style_scale"], scale)})
