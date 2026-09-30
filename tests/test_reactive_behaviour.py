from nerva.interfaces import PADState, Track
from nerva.reactive_behaviour import FREEZE_S, MIN_DWELL_S, ReactiveBehaviour

DT = 0.1


def run(beh, seconds, emotions, tracks, t0=0.0, pad=PADState()):
    d = None
    for i in range(int(seconds / DT)):
        d = beh.step(t0 + i * DT, DT, pad, emotions, tracks)
    return d


def test_explores_when_nothing_is_there():
    d = run(ReactiveBehaviour(), 2.0, {}, ())
    assert d.mode == "explore" and d.command.vx > 0


def test_orients_to_something_new_then_approaches_when_interested():
    beh = ReactiveBehaviour()
    ball = (Track("ball", bearing=0.8, distance=1.5),)
    beh.notice(0.0, "ball_appeared")
    d = run(beh, 0.5, {}, ball)
    assert d.mode == "orient" and d.command.yaw_rate > 0 and d.command.vx == 0
    d = run(beh, 1.0, {"interest": 0.4}, ball, t0=MIN_DWELL_S + 0.5)
    assert d.mode == "approach" and d.command.vx > 0 and d.target == "ball"
    d = run(beh, 1.5, {"interest": 0.4}, (Track("ball", 0.0, 0.3),), t0=3.0)
    assert d.mode == "inspect" and d.command.vx == 0


def test_startle_freezes_then_retreats_from_a_close_person():
    beh = ReactiveBehaviour()
    person = (Track("person", bearing=0.1, distance=0.6, approach_speed=1.2),)
    d = run(beh, 0.3, {"fear": 0.4, "surprise": 0.8}, person)
    assert d.mode == "freeze" and d.command.vx == 0
    d = run(beh, 1.0, {"fear": 0.4, "surprise": 0.3}, person, t0=FREEZE_S + 0.1)
    assert d.mode == "retreat" and d.command.vx < 0  # steps back first
    assert abs(d.head[2] - 0.1) < 0.05  # keeps looking at the threat
    d = run(beh, 3.0, {"fear": 0.3}, person, t0=3.0)
    assert d.mode == "retreat" and d.command.yaw_rate != 0  # turns away


def test_retreat_ends_once_safe_and_calm():
    beh = ReactiveBehaviour(mode="retreat")
    d = run(beh, 2.0, {"fear": 0.03}, (Track("person", 3.0, 3.0),), t0=10.0)
    assert d.mode != "retreat"


def test_low_dominance_keeps_more_distance():
    near = ReactiveBehaviour._stop_distance("person", PADState(dominance=0.2))
    far = ReactiveBehaviour._stop_distance("person", PADState(dominance=-0.6))
    assert far > near
