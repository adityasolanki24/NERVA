"""Stage G: Model B satisfies the same contract as Model A, stays bounded and decays."""

import numpy as np
import pytest

from nerva.affect.emotions import CategoricalAffectModel
from nerva.affect.model_b import DimensionalAffectModel
from nerva.interfaces import ActionTendencyState, AffectSystem, AppraisalFrame, AppraisalState, OutcomeHypothesis

HARM = AppraisalState(relevance=1.0, desirability=-1.0, likelihood=0.99, expectedness=0.0, controllability=0.0)


@pytest.mark.parametrize("model_cls", [CategoricalAffectModel, DimensionalAffectModel])
def test_both_models_satisfy_the_affect_contract(model_cls):
    m = model_cls()
    assert isinstance(m, AffectSystem)
    m.add(AppraisalFrame(OutcomeHypothesis("near_collision", probability=0.7), relevance=0.9, desirability=-0.7,
                         expectedness=0.2, controllability=0.3))
    m.add(HARM)
    m.step(0.1)
    assert isinstance(m.tendencies, ActionTendencyState) and m.tendencies.avoid > 0


def test_model_b_is_bounded_under_extreme_repeated_input():
    m = DimensionalAffectModel()
    for _ in range(1000):
        m.add(HARM)
        m.step(0.1)
        assert np.all(np.abs(m.x) <= 1.0)
        assert m.tendencies.avoid <= 10.0  # bounded by the contract


def test_model_b_returns_to_baseline():
    m = DimensionalAffectModel()
    m.add(HARM)
    for _ in range(600):  # 60 s
        m.step(0.1)
    assert np.all(np.abs(m.x) < 0.01) and max(m.z) < 1e-3


def test_model_b_uses_no_emotion_labels():
    import inspect

    import nerva.affect.model_b as mb

    src = inspect.getsource(mb)
    for label in ("\"fear\"", "\"joy\"", "\"hope\"", "\"interest\"", "\"surprise\"", "\"distress\""):
        assert label not in src


def test_model_b_signs_follow_desirability_and_control():
    good, bad = DimensionalAffectModel(), DimensionalAffectModel()
    good.add(AppraisalState(relevance=0.8, desirability=0.8, likelihood=1.0, expectedness=0.8, controllability=0.9))
    bad.add(AppraisalState(relevance=0.8, desirability=-0.8, likelihood=0.8, expectedness=0.8, controllability=0.1))
    for _ in range(30):
        good.step(0.1)
        bad.step(0.1)
    assert good.pad.valence > 0 > bad.pad.valence and good.pad.dominance > 0 > bad.pad.dominance
    assert bad.tendencies.withdraw > 0 and good.tendencies.withdraw == 0
