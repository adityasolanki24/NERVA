from nerva.affect.emotions import categorise
from nerva.affect.appraisal import ContextualAppraiser
from nerva.interfaces import Event, Track


def labels(a):
    return {lbl for lbl, _ in categorise(a)}


def test_novel_stimulus_elicits_interest_and_habituates():
    ap = ContextualAppraiser()
    first = ap.appraise(Event("ball_appeared"), 0.0, ())
    assert "interest" in labels(first) and "surprise" in labels(first)
    track = (Track("ball", 0.0, 1.0),)
    early = ap.observe(0.1, track, 0.1)
    assert early and "interest" in labels(early[0][1])
    for i in range(400):  # 40 s of looking at it
        ap.observe(0.2 + 0.1 * i, track, 0.1)
    late = ap.observe(100.0, track, 0.1)
    assert late and "interest" not in labels(late[0][1])


def test_fast_approach_is_worse_up_close_and_habituates():
    far = ContextualAppraiser().appraise(Event("person_approaching_rapidly", 0.8), 0.0, (Track("person", 0, 2.5),))
    ap = ContextualAppraiser()
    near = ap.appraise(Event("person_approaching_rapidly", 0.8), 0.0, (Track("person", 0, 0.5),))
    assert near.desirability < far.desirability and near.controllability < far.controllability
    assert "fear" in labels(near) and "surprise" in labels(near)
    for k in range(4):
        again = ap.appraise(Event("person_approaching_rapidly", 0.8), 100.0 * (k + 1), (Track("person", 0, 0.5),))
    assert again.expectedness > near.expectedness and "surprise" not in labels(again)


def test_same_event_means_different_things_after_a_threat():
    ap = ContextualAppraiser()
    friendly = ap.appraise(Event("person_approaching_slowly"), 0.0, ())
    ap.appraise(Event("person_approaching_rapidly", 1.0), 10.0, (Track("person", 0, 0.6),))
    wary = ap.appraise(Event("person_approaching_slowly"), 20.0, ())
    assert friendly.desirability > 0 > wary.desirability
    assert "joy" in labels(ap.appraise(Event("person_lost"), 25.0, ()))  # relief
    assert ap.appraise(Event("person_lost"), 100.0, ()) is None  # long after: irrelevant


def test_memory_appraiser_is_person_specific():
    import numpy as np

    from nerva.affect.appraisal import MemoryAppraiser
    from nerva.memory.entity import EntityMemory

    mem = EntityMemory()
    ap = MemoryAppraiser(mem)
    a = mem.resolve("person", np.array([1.0, 0, 0]), 0.0)
    b = mem.resolve("person", np.array([0, 1.0, 0]), 0.0)
    ap.identity = {"person-0": a}
    lunge = ap.appraise(Event("person_approaching_rapidly", 1.0, source="person-0"), 1.0,
                        (Track("person", 0, 0.6, tid="person-0"),))
    mem.learn(a, 1.0, "person_approaching_rapidly", categorise(lunge), arousal=0.4, surprise_negative=True)
    ap.identity = {"person-3": a}  # A seen again later, on a new track
    assert ap.appraise(Event("person_approaching_slowly", source="person-3"), 500.0, ()).desirability < 0
    ap.identity = {"person-4": b}
    assert ap.appraise(Event("person_approaching_slowly", source="person-4"), 500.0, ()).desirability > 0
    ap.identity = {"person-3": a, "person-4": b}  # both in view: each event is judged by its own person
    assert ap.appraise(Event("person_approaching_slowly", source="person-4"), 501.0, ()).desirability > 0
    assert ap.appraise(Event("person_approaching_slowly", source="person-3"), 501.0, ()).desirability < 0
    seen = ap.observe(502.0, (Track("person", 0, 2.0, tid="person-3"),), 0.1)
    assert seen and "fear" in labels(seen[0][1])  # seeing A brings the fear back
