"""E1′ styled reference sampler: exact neutral anchor, exact grid nodes, bilinear in between."""
from pathlib import Path

import numpy as np
import pytest

from nerva.training.styled_reference import style_weights

ROOT = Path(__file__).resolve().parents[2]
HAVE_ARTIFACTS = (ROOT / "experiments/cloud_runs/e1-styled-references_p1.0_h0.5").exists()


def test_weights_are_exact_at_nodes_and_sum_to_one():
    for p, i in ((-1, 0), (0, 1), (1, 2)):
        for c, j in ((0, 0), (1, 1)):
            w = style_weights(np.array([p, c], dtype=float))
            assert w[i, j] == 1 and w.sum() == 1
    w = style_weights(np.array([.75, .75]))
    assert w.sum() == pytest.approx(1) and w[2, 1] == pytest.approx(.5625) and w[0].sum() == 0


@pytest.mark.skipif(not HAVE_ARTIFACTS, reason="needs the local E1 Phase A artifacts")
def test_neutral_anchor_and_nodes_are_exact():
    import jax.numpy as jp

    from nerva.training.neutral_reference import NeutralReference, verified_references
    from nerva.training.styled_reference import StyledNeutralReference, styled_grid
    grid = styled_grid(ROOT)
    sampler = StyledNeutralReference(grid)
    neutral = NeutralReference(verified_references(ROOT)[0])
    for index in (0, 5, 13):
        for command in ((.074, 0., 0.), (0., 0., .6), (0., 0., 0.)):
            a = sampler.get_styled_motion(*command, index, jp.zeros(2))
            assert np.array_equal(np.asarray(a), np.asarray(neutral.get_reference_motion(*command, index)))
    node = NeutralReference(grid[(1, 1)])
    assert np.allclose(np.asarray(sampler.get_styled_motion(.074, 0., 0., 7, jp.array([1., 1.]))),
                       np.asarray(node.get_reference_motion(.074, 0., 0., 7)), atol=1e-6)
    assert np.all(np.isnan(np.asarray(sampler.get_styled_motion(.05, 0., 0., 0, jp.zeros(2)))))
