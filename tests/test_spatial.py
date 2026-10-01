import time

from nerva.episodic import EpisodicMemory
from nerva.spatial import PlaceMemory


def test_familiarity_and_forgetting():
    m = PlaceMemory()
    k = m.key((0.1, 0.1))
    assert m.novelty(k) == 1.0
    for i in range(200):
        m.observe((0.1, 0.1), i * 0.1, 0.1)
    assert m.novelty(k) < 0.4
    m.observe((0.1, 0.1), 20.0 + 3600.0, 0.0)  # an hour away
    assert m.novelty(k) > 0.9


def test_place_where_it_was_frightened_is_marked_and_avoided():
    m = PlaceMemory()
    m.learn((2.0, 0.1), 0.0, [("fear", 0.7)], arousal=0.5)
    assert m.threat(m.key((2.0, 0.1))) > 0.3
    bearing, _ = m.explore_heading((0.8, 0.1), 0.0)  # the frightening cell is straight ahead
    assert abs(bearing) > 0.2


def test_exploration_prefers_unvisited_places():
    m = PlaceMemory()
    for i in range(6000):  # everything ahead (+x) is familiar (40 s per cell)
        m.observe((1.0 + (i % 3) * 0.75, (i // 3 % 5 - 2) * 0.75), i * 0.1, 0.1)
    bearing, _ = m.explore_heading((0.3, 0.1), 0.0)
    assert abs(bearing) > 1.0  # it turns away from the familiar area toward unexplored cells


def test_episodic_budget_long_run():
    m = EpisodicMemory(capacity=500)
    start = time.perf_counter()
    for k in range(20000):  # a long stream of mixed events
        salient = k % 7 == 0
        m.encode(float(k), "person_approaching_rapidly" if salient else "ball_in_view", "person#0", (0, 0),
                 [("fear", 0.5)] if salient else [], relevance=0.9 if salient else 0.2,
                 arousal=0.4 if salient else 0.0, prediction_error=0.6 if salient else 0.0, novelty=0.1)
        if k % 1000 == 999:
            m.consolidate(float(k))
    assert len(m.episodes) <= 500 and m.routine_count > 15000
    assert (time.perf_counter() - start) / 20000 < 0.005  # well under the 100 ms perception period
