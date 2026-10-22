from nerva.behaviour.attention import AttentionManager
from nerva.interfaces import Event, Goal, GoalState, Track


def test_attention_prefers_a_novel_object_over_a_familiar_person():
    manager = AttentionManager()
    tracks = (Track("person", 0.1, 2.0, tid="person-0"), Track("ball", -0.2, 1.6, tid="ball-0"))
    d = manager.update(0.0, tracks, [Event("ball_appeared", source="ball-0")],
                       {"person-0": 0.05, "ball-0": 1.0})
    assert d.target == "ball-0"
    assert "novel" in d.reason and "unexpected" in d.reason


def test_attention_holds_its_target_until_the_dwell_time_expires():
    manager = AttentionManager()
    a, b = Track("ball", 0.0, 1.5, tid="ball-0"), Track("ball", 0.2, 1.5, tid="ball-1")
    assert manager.update(0.0, (a, b), novelty={"ball-0": 0.9, "ball-1": 0.1}).target == "ball-0"
    # B becomes interesting, but the current fixation is still protected.
    assert manager.update(0.5, (a, b), novelty={"ball-0": 0.1, "ball-1": 1.0}).target == "ball-0"
    assert manager.update(1.3, (a, b), novelty={"ball-0": 0.1, "ball-1": 1.0}).target == "ball-1"


def test_a_sudden_event_can_interrupt_a_familiar_fixation():
    manager = AttentionManager()
    a, b = Track("ball", 0.0, 1.5, tid="ball-0"), Track("person", 0.2, 1.5, tid="person-0")
    manager.update(0.0, (a, b), novelty={"ball-0": 1.0, "person-0": 0.05})
    d = manager.update(0.2, (a, b), [Event("person_approaching_rapidly", source="person-0")],
                       {"ball-0": 0.2, "person-0": 0.05})
    assert d.target == "person-0" and d.reason == "unexpected"


def test_active_goal_increases_attention_without_overriding_a_safety_event():
    manager = AttentionManager()
    a, b = Track("ball", 0.0, 1.5, tid="ball-0"), Track("person", 0.2, 1.5, tid="person-0")
    goals = GoalState((Goal("inspect", "ball-0", 1.0),))
    d = manager.update(0.0, (a, b), novelty={"ball-0": 0.2, "person-0": 0.2}, goals=goals)
    assert d.target == "ball-0"
