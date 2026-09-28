import numpy as np
import pytest

from nerva.affect import EMOTION_PAD, AffectConfig, AffectModel, categorise, emotion_pad_point
from nerva.appraisal import EVENT_APPRAISALS, appraise
from nerva.interfaces import AppraisalState, Event

# ── EMA categorisation (Marsella & Gratch 2009 Table 2; Gratch & Marsella 2004 Table 3) ──


@pytest.mark.parametrize("d,l,label", [
    (0.5, 0.6, "hope"), (0.5, 1.0, "joy"), (-0.5, 0.6, "fear"), (-0.5, 1.0, "distress")])
def test_ema_labels_and_intensity(d, l, label):
    out = categorise(AppraisalState(relevance=1.0, desirability=d, likelihood=l, expectedness=1.0))
    assert out == [(label, pytest.approx(abs(d * l)))]


def test_surprise_when_expectedness_low_with_nerva_intensity_rule():
    out = dict(categorise(AppraisalState(relevance=1.0, desirability=0.0, expectedness=0.1)))
    assert out == {"surprise": pytest.approx(0.9)}


def test_irrelevant_or_neutral_appraisal_gives_no_emotion():
    assert categorise(AppraisalState(relevance=0.0, desirability=-1.0, likelihood=1.0, expectedness=0.0)) == []
    assert categorise(AppraisalState(relevance=1.0, desirability=0.0, expectedness=1.0)) == []


def test_pad_table_matches_alma_values():
    assert EMOTION_PAD["fear"] == (-0.64, 0.60, -0.43)
    assert EMOTION_PAD["joy"] == (0.40, 0.20, 0.10)


def test_controllability_shifts_dominance_only():
    lo, hi = emotion_pad_point("fear", 0.0), emotion_pad_point("fear", 1.0)
    np.testing.assert_allclose(lo[:2], hi[:2])
    assert lo[2] == pytest.approx(0.5 * -0.43 - 0.5) and hi[2] == pytest.approx(0.5 * -0.43 + 0.5)


# ── appraisal table ──


def test_every_synthetic_event_has_an_appraisal_and_magnitude_scales_desirability():
    assert set(EVENT_APPRAISALS) == {"successful_walking", "near_fall", "person_approaching_slowly",
                                     "person_approaching_rapidly", "obstacle_blocking_goal"}
    assert appraise(Event("near_fall", 0.5)).desirability == pytest.approx(-0.4)
    with pytest.raises(ValueError):
        appraise(Event("unknown_event"))


# ── dynamics ──


def test_state_stays_at_baseline_without_events():
    m = AffectModel()
    for _ in range(100):
        m.step(0.1)
    np.testing.assert_array_equal(m.x, 0.0)


def test_event_moves_state_toward_emotion_then_returns_to_baseline():
    m = AffectModel()
    m.add(appraise(Event("person_approaching_rapidly")))
    peak = max((m.step(0.05) for _ in range(40)), key=lambda s: abs(s.arousal))  # 2 s
    assert peak.valence < -0.05 and peak.arousal > 0.05 and peak.dominance < -0.05
    while m.emotions:  # emotions fade (tau_emotion = 4 s)
        m.step(0.05)
    # With no active emotions the state relaxes to baseline exactly as exp(-t / tau_return).
    before = m.x.copy()
    m.step(10.0)
    np.testing.assert_allclose(m.x, before * np.exp(-10.0 / m.cfg.tau_return_s), rtol=1e-12)
    for _ in range(20):
        m.step(10.0)  # 200 s more
    np.testing.assert_allclose(m.x, 0.0, atol=1e-4)


def test_state_is_continuous_not_instant():
    m = AffectModel()
    m.add(appraise(Event("near_fall")))
    first = m.step(0.02)
    assert abs(first.arousal) < 0.05  # an event nudges the state; it does not jump to the emotion


def test_update_rate_does_not_change_the_trajectory_much():
    def after_3s(dt):
        m = AffectModel()
        m.add(appraise(Event("near_fall")))
        for _ in range(round(3.0 / dt)):
            m.step(dt)
        return m.x

    np.testing.assert_allclose(after_3s(0.1), after_3s(0.01), atol=0.02)


def test_state_is_bounded_under_repeated_extreme_events():
    m = AffectModel(AffectConfig(k_pull_per_s=50.0))
    for _ in range(200):
        m.add(AppraisalState(relevance=1.0, desirability=-1.0, likelihood=0.99, expectedness=0.0,
                             controllability=0.0))
        s = m.step(0.1)
        assert all(-1.0 <= v <= 1.0 for v in (s.valence, s.arousal, s.dominance))
