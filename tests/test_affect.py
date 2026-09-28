import math

import numpy as np
import pytest

from nerva.affect import EMOTION_PAD, AffectConfig, CategoricalAffectModel, categorise, emotion_pad_point
from nerva.appraisal import EVENT_APPRAISALS, appraise
from nerva.interfaces import AffectSystem, AppraisalState, Event

# ── categorisation (simplified EMA-inspired; relevance scaling is a NERVA extension) ──


@pytest.mark.parametrize("d,l,label", [
    (0.5, 0.6, "hope"), (0.5, 1.0, "joy"), (-0.5, 0.6, "fear"), (-0.5, 1.0, "distress")])
def test_ema_labels_and_base_intensity_at_full_relevance(d, l, label):
    out = categorise(AppraisalState(relevance=1.0, desirability=d, likelihood=l, expectedness=1.0))
    assert out == [(label, pytest.approx(abs(d * l)))]


def test_relevance_scales_intensity():
    a = AppraisalState(relevance=0.25, desirability=-0.8, likelihood=0.5, expectedness=0.1)
    assert dict(categorise(a)) == {"fear": pytest.approx(0.25 * 0.4), "surprise": pytest.approx(0.25 * 0.9)}
    unscaled = dict(categorise(a, AffectConfig(scale_by_relevance=False)))
    assert unscaled == {"fear": pytest.approx(0.4), "surprise": pytest.approx(0.9)}


def test_irrelevant_or_neutral_appraisal_gives_no_emotion():
    assert categorise(AppraisalState(relevance=0.0, desirability=-1.0, likelihood=1.0, expectedness=0.0)) == []
    assert categorise(AppraisalState(relevance=1.0, desirability=0.0, expectedness=1.0)) == []


# ── anchors ──


def test_pad_anchors_match_alma_and_surprise_is_arousal_only():
    assert EMOTION_PAD["fear"] == (-0.64, 0.60, -0.43)
    assert EMOTION_PAD["joy"] == (0.40, 0.20, 0.10)
    v, a, d = EMOTION_PAD["surprise"]
    assert math.isnan(v) and math.isnan(d) and a == 0.80


def test_controllability_shifts_dominance_only():
    lo, hi = emotion_pad_point("fear", 0.0), emotion_pad_point("fear", 1.0)
    np.testing.assert_allclose(lo[:2], hi[:2])
    assert lo[2] == pytest.approx(0.5 * -0.43 - 0.5) and hi[2] == pytest.approx(0.5 * -0.43 + 0.5)
    assert math.isnan(emotion_pad_point("surprise", 0.0)[2])  # surprise has no dominance anchor


def test_surprise_does_not_dilute_fear_valence():
    m = CategoricalAffectModel()
    m.add(AppraisalState(relevance=1.0, desirability=-0.8, likelihood=0.5, expectedness=0.1))  # fear + surprise
    centre, strength = m.emotion_centre()
    assert centre[0] == pytest.approx(-0.64)  # valence centre = fear alone
    assert centre[1] > 0.6  # arousal centre includes surprise


# ── appraisal table ──


def test_every_synthetic_event_has_an_appraisal_and_magnitude_scales_desirability():
    assert set(EVENT_APPRAISALS) == {"successful_walking", "near_fall", "person_approaching_slowly",
                                     "person_approaching_rapidly", "obstacle_blocking_goal"}
    assert appraise(Event("near_fall", 0.5)).desirability == pytest.approx(-0.4)
    with pytest.raises(ValueError):
        appraise(Event("unknown_event"))


# ── dynamics ──


def test_model_satisfies_the_affect_system_protocol():
    assert isinstance(CategoricalAffectModel(), AffectSystem)


def test_state_stays_at_baseline_without_events():
    m = CategoricalAffectModel()
    for _ in range(100):
        m.step(0.1)
    np.testing.assert_array_equal(m.x, 0.0)


def test_event_moves_state_toward_emotion_then_returns_to_baseline():
    m = CategoricalAffectModel()
    m.add(appraise(Event("person_approaching_rapidly")))
    peak = max((m.step(0.05) for _ in range(40)), key=lambda s: abs(s.arousal))  # 2 s
    assert peak.valence < -0.05 and peak.arousal > 0.05 and peak.dominance < -0.05
    while m.emotions:
        m.step(0.05)
    # With no active emotions each dimension relaxes as exp(-t / tau_return[dim]).
    before = m.x.copy()
    m.step(10.0)
    np.testing.assert_allclose(m.x, before * np.exp(-10.0 / np.array(m.cfg.tau_return_s)), rtol=1e-12)


def test_arousal_recovers_faster_than_valence_and_dominance():
    m = CategoricalAffectModel()
    m.x = np.array([-0.5, 0.5, -0.5])
    m.step(6.0)
    assert abs(m.x[1]) / 0.5 < abs(m.x[0]) / 0.5
    assert abs(m.x[1]) / 0.5 < abs(m.x[2]) / 0.5


def test_state_is_continuous_not_instant():
    m = CategoricalAffectModel()
    m.add(appraise(Event("near_fall")))
    first = m.step(0.02)
    assert abs(first.arousal) < 0.05


def test_update_rate_does_not_change_the_trajectory_much():
    def after_3s(dt):
        m = CategoricalAffectModel()
        m.add(appraise(Event("near_fall")))
        for _ in range(round(3.0 / dt)):
            m.step(dt)
        return m.x

    np.testing.assert_allclose(after_3s(0.1), after_3s(0.01), atol=0.02)


def test_state_is_bounded_under_repeated_extreme_events():
    m = CategoricalAffectModel(AffectConfig(k_pull_per_s=50.0))
    for _ in range(200):
        m.add(AppraisalState(relevance=1.0, desirability=-1.0, likelihood=0.99, expectedness=0.0,
                             controllability=0.0))
        s = m.step(0.1)
        assert all(-1.0 <= v <= 1.0 for v in (s.valence, s.arousal, s.dominance))


def test_repeated_routine_success_stays_modest():
    """Design target (NERVA v0.1): a routine success every 3 s for 60 s must not push valence
    past 0.2. Measured value at the time of writing: +0.16 (see docs/affect_model.md)."""
    m = CategoricalAffectModel()
    a = appraise(Event("successful_walking"))
    for k in range(int(60 / 0.05)):
        if k % int(3 / 0.05) == 0:
            m.add(a)
        s = m.step(0.05)
    assert 0.0 < s.valence < 0.2
