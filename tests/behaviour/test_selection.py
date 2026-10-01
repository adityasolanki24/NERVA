from nerva.affect.tendencies import tendencies_from_emotions
from nerva.behaviour.selection import UtilityBehaviour
from nerva.interfaces import PADState, Track

DT = 0.1


def run(beh, seconds, emotions, tracks, t0=0.0, salience=None):
    """`emotions`: Model A intensities by label (converted with the Model A adapter), or tendencies."""
    if isinstance(emotions, dict):
        emotions = tendencies_from_emotions(emotions)
    d = None
    for i in range(int(seconds / DT)):
        d = beh.step(t0 + i * DT, DT, PADState(), emotions, tracks, salience)
    return d


def test_novel_ball_is_approached_but_a_habituated_one_is_not():
    ball = (Track("ball", 0.0, 1.5),)
    assert run(UtilityBehaviour(), 1.0, {"interest": 0.4}, ball, t0=5.0, salience={"ball": 1.0}).mode == "approach"
    assert run(UtilityBehaviour(), 1.0, {"interest": 0.4}, ball, t0=5.0, salience={"ball": 0.1}).mode == "explore"


def test_fear_beats_curiosity_up_close_but_only_watches_from_afar():
    emo = {"interest": 0.5, "fear": 0.3}
    near = run(UtilityBehaviour(), 2.0, emo, (Track("person", 0.0, 0.6),), t0=5.0, salience={"person": 1.0})
    far = run(UtilityBehaviour(), 2.0, emo, (Track("person", 0.0, 2.5),), t0=5.0, salience={"person": 1.0})
    assert near.mode == "retreat" and far.mode == "watch"


def test_startle_close_by_freezes_and_is_committed():
    beh = UtilityBehaviour()
    d = run(beh, 0.2, {"fear": 0.6, "surprise": 0.9}, (Track("person", 0.0, 0.5),), t0=5.0)
    assert d.mode == "freeze"
    d = run(beh, 0.3, {}, (), t0=5.2)  # emotions vanish, but a freeze lasts its minimum duration
    assert d.mode == "freeze"


def test_utilities_are_exposed_for_logging():
    beh = UtilityBehaviour()
    run(beh, 0.5, {"interest": 0.3}, (Track("ball", 0.0, 1.0),), t0=5.0)
    assert "approach:ball" in beh.utilities and "explore" in beh.utilities


def test_behaviour_runs_on_tendencies_without_any_emotion_labels():
    """Stage C: any affect model that produces ActionTendencyState can drive behaviour."""
    from nerva.interfaces import ActionTendencyState

    person = (Track("person", 0.0, 0.6, tid="person-0"),)
    d = run(UtilityBehaviour(), 2.0, ActionTendencyState(avoid=0.3), person, t0=5.0)
    assert d.mode == "retreat"
    d = run(UtilityBehaviour(), 1.0, ActionTendencyState(explore=0.4), (Track("ball", 0.0, 1.5),), t0=5.0,
            salience={"ball": 1.0})
    assert d.mode == "approach"


def test_selection_code_reads_no_emotion_labels():
    import inspect

    import nerva.behaviour.modes as modes
    import nerva.behaviour.selection as selection

    for mod in (modes, selection):
        src = inspect.getsource(mod)
        for label in ("\"fear\"", "\"interest\"", "\"hope\"", "\"joy\"", "\"surprise\"", "\"distress\""):
            assert label not in src, f"{mod.__name__} still reads {label}"
