"""Model B v2: bounded PAD target; persistent re-appraisal converges instead of accumulating."""

import numpy as np

from nerva.affect.model_b import AttractorAffectModel, DimensionalAffectModel
from nerva.interfaces import AffectSystem, AppraisalFrame, AppraisalState, OutcomeHypothesis

IN_VIEW = AppraisalFrame(OutcomeHypothesis("novel_stimulus", subject="ball-0", probability=0.5), relevance=0.6,
                         desirability=0.0, expectedness=0.3, controllability=0.8, persistent=True)
LUNGE = AppraisalState(relevance=0.9, desirability=-0.8, likelihood=0.7, expectedness=0.15, controllability=0.25)


def run(model, period_s, seconds=120.0, dt=0.1):
    xs = []
    for k in range(int(seconds / dt)):
        if k % int(round(period_s / dt)) == 0:
            model.add(IN_VIEW)
        xs.append(model.step(dt))
    return np.array([(p.valence, p.arousal, p.dominance) for p in xs])


def test_contract():
    assert isinstance(AttractorAffectModel(), AffectSystem)


def test_repeated_persistent_appraisal_converges_at_any_rate():
    slow, fast = run(AttractorAffectModel(), 2.0), run(AttractorAffectModel(), 1.0)
    assert np.all(np.abs(slow[-100:].mean(0) - slow[500:600].mean(0)) < 0.02)
    assert np.all(np.abs(fast[-100:].mean(0) - slow[-100:].mean(0)) < 0.05)
    assert np.abs(slow).max() < 0.9


def test_old_model_b_saturates_under_repeated_persistent_appraisal():  # the diagnosed problem, as a marker
    assert run(DimensionalAffectModel(), 2.0)[-100:, 2].mean() > 0.99


def test_events_are_phasic_and_pad_returns_to_baseline():
    m = AttractorAffectModel()
    m.add(LUNGE)
    peak = min(m.step(0.1).valence for _ in range(60))
    for _ in range(600):
        m.step(0.1)
    assert peak < -0.1 and np.all(np.abs(m.x) < 0.01)


def test_context_fades_when_the_situation_is_no_longer_reappraised():
    m = AttractorAffectModel()
    for k in range(100):
        if k % 20 == 0:
            m.add(IN_VIEW)
        m.step(0.1)
    held = m.pad.dominance
    for _ in range(400):  # nothing in view for 40 s
        m.step(0.1)
    assert held > 0.05 and abs(m.pad.dominance) < 0.01 and not m.context


def test_tendencies_are_those_of_model_b():
    a, b = DimensionalAffectModel(), AttractorAffectModel()
    for m in (a, b):
        m.add(LUNGE, source="person-0")
        m.add(IN_VIEW)
        m.step(0.1)
    assert a.tendencies == b.tendencies and a.tendencies_for("person-0") == b.tendencies_for("person-0")


def test_bv3_builds_fast_and_returns_slowly():
    from nerva.affect.model_b import OnsetAttractorAffectModel

    slow, fast = AttractorAffectModel(), OnsetAttractorAffectModel()
    for m in (slow, fast):
        m.add(LUNGE)
    v_slow = [slow.step(0.1).dominance for _ in range(20)]
    v_fast = [fast.step(0.1).dominance for _ in range(20)]
    assert min(v_fast) < min(v_slow) - 0.1  # responds within 2 s
    for _ in range(600):
        fast.step(0.1)
    assert np.all(np.abs(fast.x) < 0.01)  # still returns to baseline
    assert isinstance(fast, AffectSystem)
