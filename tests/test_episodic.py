import numpy as np

from nerva.episodic import CAPACITY, EpisodicMemory
from nerva.memory import EntityMemory


def _lunge(m, t, entity="person#0"):
    return m.encode(t, "person_approaching_rapidly", entity, (0, 0), [("fear", 0.6), ("surprise", 0.8)],
                    relevance=0.9, arousal=0.4, prediction_error=0.7, novelty=0.2)


def test_only_salient_events_are_stored():
    m = EpisodicMemory()
    assert m.encode(0, "ball_in_view", "ball#0", (0, 0), [], relevance=0.2, arousal=0.0,
                    prediction_error=0.0, novelty=0.1) is None
    assert _lunge(m, 1) is not None and m.routine_count == 1 and len(m.episodes) == 1


def test_power_law_forgetting_and_retrieval_strengthening():
    m = EpisodicMemory()
    ep = _lunge(m, 0.0)
    a1, a100, a1000 = ep.activation(1), ep.activation(100), ep.activation(1000)
    assert a1 > a100 > a1000
    assert np.isclose(a100 - a1000, 0.5 * np.log(10), atol=1e-6)  # slope -d on log-log: power law
    m.retrieve(100.0, entity="person#0")
    assert ep.activation(1000) > a1000  # a retrieved memory stays more available


def test_retrieval_by_entity_cue():
    m = EpisodicMemory()
    _lunge(m, 0, "person#0")
    m.encode(1, "touch_gentle", "person#1", (1, 1), [("joy", 0.4)], relevance=0.6, arousal=0.3,
             prediction_error=0.2, novelty=0.5)
    assert m.retrieve(10, entity="person#1", k=1)[0].entity == "person#1"


def test_consolidation_merges_repeats_replays_and_prunes():
    ents = EntityMemory()
    a = ents.resolve("person", np.array([1.0, 0, 0]), 0.0)
    m = EpisodicMemory()
    for k in range(5):
        m.encode(k, "touch_gentle", a.eid, (0, 0), [("joy", 0.4)], relevance=0.6, arousal=0.3,
                 prediction_error=0.2, novelty=0.5)
    report = m.consolidate(10.0, ents)
    assert report["merged"] == 4 and m.episodes[0].count == 5 and a.warmth > 0
    assert len(a.episodes) == 0  # replay changes the association, not the event history


def test_replay_keeps_an_important_lesson_from_drifting_away():
    def run(replay):
        ents = EntityMemory()
        a = ents.resolve("person", np.array([1.0, 0, 0]), 0.0)
        ents.learn(a, 1.0, "person_approaching_rapidly", [("fear", 0.6)], 0.4, True)
        ents.observe(a, 1.0, 0.1)
        m = EpisodicMemory()
        _lunge(m, 1.0, a.eid)
        for hour in range(1, 6):
            ents.resolve("person", np.array([1.0, 0, 0]), hour * 3600.0)  # absence drift at each encounter
            ents.observe(a, hour * 3600.0, 0.1)
            if replay:
                m.consolidate(hour * 3600.0 + 1, ents)
        return a.threat

    assert run(True) > 2 * run(False)


def test_capacity_is_bounded_and_persistence_roundtrips(tmp_path):
    m = EpisodicMemory(capacity=50)
    for k in range(200):
        _lunge(m, float(k))
    assert len(m.episodes) == 50 and CAPACITY >= 50
    path = str(tmp_path / "memory.sqlite")
    m.save(path)
    back = EpisodicMemory.load(path, capacity=50)
    assert len(back.episodes) == 50 and back.episodes[0].event == m.episodes[0].event
