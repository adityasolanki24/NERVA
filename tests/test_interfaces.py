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
