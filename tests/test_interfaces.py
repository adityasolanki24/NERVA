import dataclasses

import pytest

from nerva.interfaces import (
    AppraisalState,
    BehaviourCommand,
    Event,
    ExpressiveStyle,
    PADState,
    PerceptionState,
)


def test_defaults_are_neutral():
    assert PADState() == PADState(0.0, 0.0, 0.0)
    assert ExpressiveStyle().label == "Neutral"
    cmd = BehaviourCommand()
    assert (cmd.vx, cmd.vy, cmd.yaw_rate, cmd.skill) == (0.0, 0.0, 0.0, "walk")
    assert cmd.style == ExpressiveStyle(0.0)


@pytest.mark.parametrize(
    "make",
    [
        lambda: PADState(valence=1.5),
        lambda: PADState(arousal=-1.01),
        lambda: AppraisalState(relevance=-0.1),
        lambda: AppraisalState(desirability=2.0),
        lambda: AppraisalState(controllability=1.1),
        lambda: ExpressiveStyle(style=1.2),
        lambda: Event(kind="near_fall", magnitude=1.5),
        lambda: Event(kind=""),
        lambda: BehaviourCommand(skill="backflip"),
    ],
)
def test_out_of_range_values_are_rejected(make):
    with pytest.raises(ValueError):
        make()


def test_boundaries_are_accepted():
    PADState(-1.0, 1.0, -1.0)
    AppraisalState(relevance=1.0, desirability=-1.0, likelihood=0.0,
                   expectedness=0.0, controllability=0.0)
    ExpressiveStyle(-1.0)
    ExpressiveStyle(1.0)


def test_states_are_immutable():
    pad = PADState()
    with pytest.raises(dataclasses.FrozenInstanceError):
        pad.valence = 0.5


def test_style_labels_do_not_claim_emotions():
    assert ExpressiveStyle(-1.0).label == "Style -1"
    assert ExpressiveStyle(1.0).label == "Style +1"
    assert ExpressiveStyle(0.5).label == "Style +0.5"


def test_perception_carries_events_without_interpretation():
    p = PerceptionState(time_s=1.0, events=(Event("person_approaching", 0.8),))
    assert p.events[0].kind == "person_approaching"


# ── stage B contracts ────────────────────────────────────────────────────────

from nerva.interfaces import (  # noqa: E402
    ActionTendencyState,
    AffectSystem,
    AppraisalFrame,
    Goal,
    GoalState,
    OutcomeHypothesis,
    OutcomeSignal,
    RiskEstimate,
    SelfState,
)


@pytest.mark.parametrize(
    "make",
    [
        lambda: SelfState(stability_risk=1.5),
        lambda: SelfState(tilt_deg=-1.0),
        lambda: SelfState(speed=-0.1),
        lambda: SelfState(escape_room=2.0),
        lambda: Goal("fly"),
        lambda: Goal("explore", priority=1.2),
        lambda: OutcomeSignal("hugged"),
        lambda: OutcomeSignal("benign_contact", magnitude=1.5),
        lambda: OutcomeHypothesis(""),
        lambda: OutcomeHypothesis("collision", probability=1.2),
        lambda: OutcomeHypothesis("collision", predicts=("hugged",)),
        lambda: OutcomeSignal("near_collision"),  # a near miss is a RiskEstimate, not an outcome
        lambda: RiskEstimate("contact_impact"),
        lambda: AppraisalFrame(OutcomeHypothesis("x"), relevance=1.2),
        lambda: ActionTendencyState(avoid=-0.1),
        lambda: ActionTendencyState(avoid=float("nan")),
        lambda: ActionTendencyState(approach=1e6),
    ],
)
def test_new_contracts_reject_invalid_values(make):
    with pytest.raises(ValueError):
        make()


def test_frame_likelihood_is_the_hypothesis_probability():
    hyp = OutcomeHypothesis("collision", subject="person-3", probability=0.72, predicts=("contact_impact",))
    frame = AppraisalFrame(hyp, relevance=0.9, desirability=-0.8, expectedness=0.2, controllability=0.35)
    assert frame.likelihood == 0.72
    legacy = frame.as_appraisal_state()
    assert (legacy.relevance, legacy.desirability, legacy.likelihood) == (0.9, -0.8, 0.72)


def test_outcome_kinds_split_into_adverse_and_benign():
    assert OutcomeSignal("contact_impact").adverse and OutcomeSignal("stability_loss").adverse
    assert not OutcomeSignal("benign_contact").adverse


def test_goal_priority_lookup():
    goals = GoalState((Goal("remain_upright", priority=1.0), Goal("inspect", "ball-1", 0.6)))
    assert goals.priority(("inspect", "approach")) == 0.6
    assert goals.priority(("retreat",)) == 0.0


def test_model_a_satisfies_the_affect_contract():
    from nerva.affect.emotions import CategoricalAffectModel

    model = CategoricalAffectModel()
    assert isinstance(model, AffectSystem)
    model.add(AppraisalFrame(OutcomeHypothesis("collision", probability=0.7), relevance=0.9,
                             desirability=-0.7, expectedness=0.2, controllability=0.3))
    model.step(0.1)
    assert model.tendencies.avoid > 0 and model.tendencies.orient > 0
