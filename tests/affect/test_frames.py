"""Stage E: appraisal frames over explicit hypotheses, relative to goals and self state."""

import dataclasses

import pytest

from nerva.affect.appraisal import ContextualAppraiser
from nerva.affect.frames import HYPOTHESES, HYPOTHESIS_KINDS, FrameAppraiser, frame_from
from nerva.behaviour.goals import active_goals
from nerva.interfaces import AppraisalState, Event, Goal, GoalState, SelfState, Track

LUNGE = Event("person_approaching_rapidly", 0.9, source="person-0")
CLOSE_PERSON = (Track("person", 0.0, 0.6, approach_speed=1.5, tid="person-0"),)


def test_every_likelihood_has_a_referent():
    frame = FrameAppraiser(ContextualAppraiser()).appraise(LUNGE, 10.0, CLOSE_PERSON)
    assert frame.hypothesis.kind == "collision" and frame.hypothesis.subject == "person-0"
    assert frame.likelihood == frame.hypothesis.probability
    assert frame.hypothesis.predicts == ("contact_impact",)


def test_every_mapped_event_resolves_to_a_known_hypothesis_and_unmapped_events_fail():
    for kind in HYPOTHESES:
        for d in (-0.5, 0.5):
            f = frame_from(kind, AppraisalState(relevance=0.5, desirability=d, likelihood=0.5), "", None, None)
            assert f.hypothesis.kind in HYPOTHESIS_KINDS
    with pytest.raises(ValueError):
        frame_from("dragon_appeared", AppraisalState(), "", None, None)


def test_nominal_context_reproduces_the_wrapped_appraiser_exactly():
    plain = ContextualAppraiser().appraise(LUNGE, 10.0, CLOSE_PERSON)
    frame = FrameAppraiser(ContextualAppraiser()).appraise(LUNGE, 10.0, CLOSE_PERSON, SelfState(), active_goals())
    assert frame.as_appraisal_state() == plain


def test_the_same_observation_is_appraised_differently_when_the_robot_is_unsteady():
    steady = FrameAppraiser(ContextualAppraiser()).appraise(LUNGE, 10.0, CLOSE_PERSON, SelfState(), active_goals())
    shaky = FrameAppraiser(ContextualAppraiser()).appraise(
        LUNGE, 10.0, CLOSE_PERSON, SelfState(tilt_deg=25.0, stability_risk=0.75), active_goals())
    assert shaky.controllability < steady.controllability and shaky.relevance > steady.relevance
    assert shaky.desirability == steady.desirability and shaky.likelihood == steady.likelihood


def test_the_same_observation_matters_less_when_no_goal_it_bears_on_is_active():
    ball = Event("ball_appeared", source="ball-0")
    tracks = (Track("ball", 0.0, 1.2, tid="ball-0"),)
    exploring = FrameAppraiser(ContextualAppraiser()).appraise(ball, 5.0, tracks, None, active_goals())
    busy = FrameAppraiser(ContextualAppraiser()).appraise(
        ball, 5.0, tracks, None, GoalState((Goal("remain_upright"), Goal("retreat", "person-0", 0.9))))
    assert busy.relevance < exploring.relevance and busy.goals == ()


def test_context_selects_the_social_hypothesis():
    benign = frame_from("person_appeared", AppraisalState(relevance=0.6, desirability=0.1, likelihood=0.6), "p", None, None)
    adverse = dataclasses.replace(benign.as_appraisal_state(), desirability=-0.3)
    assert benign.hypothesis.kind == "benign_interaction"
    assert frame_from("person_appeared", adverse, "p", None, None).hypothesis.kind == "adverse_interaction"


def test_agent_controllability_is_the_reaction_margin():
    from nerva.affect.frames import C_MAX, C_MIN, agent_controllability

    assert agent_controllability(Track("person", 0.0, 1.0, approach_speed=0.0), None) == C_MAX  # petting
    assert agent_controllability(Track("person", 0.0, 1.0, approach_speed=0.25), None) == C_MAX  # TTC 4 s
    lunge = agent_controllability(Track("person", 0.0, 1.0, approach_speed=1.8), None)
    assert C_MIN <= lunge < 0.25
    shaky = agent_controllability(Track("person", 0.0, 1.0, approach_speed=1.8), SelfState(stability_risk=1.0))
    assert abs(shaky - 0.5 * lunge) < 1e-12


def test_in_view_frames_of_people_use_the_margin_and_objects_do_not():
    fa = FrameAppraiser(ContextualAppraiser(), margin_controllability=True)
    tracks = (Track("person", 0.0, 1.0, approach_speed=1.8, tid="person-0"), Track("ball", 0.5, 1.0, tid="ball-0"))
    assert {f.controllability for _, f in FrameAppraiser(ContextualAppraiser()).observe(10.0, tracks, 0.1)} == {0.8}
    frames = {f.hypothesis.subject: f for _, f in fa.observe(10.0, tracks, 0.1)}
    assert frames["person-0"].controllability < 0.3 and frames["ball-0"].controllability == 0.8
    assert all(f.persistent for f in frames.values())


def test_continued_touch_is_an_ongoing_contact_situation():
    fa = FrameAppraiser(ContextualAppraiser(), touch_context=True)
    touch = Event("touch_gentle", source="person-1")
    first = fa.appraise(touch, 10.0, ())
    second = fa.appraise(touch, 11.0, ())
    later = fa.appraise(touch, 20.0, ())
    assert not first.persistent and first.hypothesis.kind == "benign_interaction"
    assert second.persistent and second.hypothesis.kind == "ongoing_contact" and second.hypothesis.subject == "person-1"
    assert not later.persistent  # a new contact after a pause starts with an event again
    assert not FrameAppraiser(ContextualAppraiser()).appraise(touch, 11.0, ()).persistent  # off by default
