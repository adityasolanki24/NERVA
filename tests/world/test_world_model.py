from nerva.interfaces import Track
from nerva.world.model import TTL_S, WorldModel

A = Track("person", 0.0, 0.8, approach_speed=0.3, tid="person-3")
B = Track("person", 0.5, 2.0, tid="person-4")


def test_track_and_entity_identity_are_distinct_and_renamed_when_known():
    wm = WorldModel()
    s = wm.update(0.0, (0, 0), 0.0, (A,))
    node = next(n for n in s.entities if n.track_id == "person-3")
    assert node.node_id == "track:person-3" and node.entity_id == ""
    assert s.related("near", "self")[0].subject == "track:person-3"
    s = wm.update(0.1, (0, 0), 0.0, (A,), identities={"person-3": "person#0"})
    ids = {n.node_id for n in s.entities}
    assert "person#0" in ids and "track:person-3" not in ids
    assert {r.subject for r in s.related("near", "self")} == {"person#0"}


def test_relations_carry_provenance_and_decay_then_expire():
    wm = WorldModel()
    wm.update(0.0, (0, 0), 0.0, (A, B), touching="person-3")
    s = wm.state(0.0)
    touch = s.related("touching", "self")[0]
    assert touch.confidence == 1.0 and touch.source == "touch sensor"
    assert s.related("seen_with") and s.related("approaching", "self")
    half = wm.state(TTL_S["touching"] / 2).related("touching", "self")[0]
    assert abs(half.confidence - 0.5) < 1e-9
    assert wm.state(TTL_S["touching"] + 0.01).related("touching", "self") == ()


def test_people_who_leave_fade_from_the_current_situation():
    wm = WorldModel()
    wm.update(0.0, (0, 0), 0.0, (A,))
    s = wm.update(10.0, (0, 0), 0.0, ())
    assert all(n.kind in ("self", "place") for n in s.entities)


def test_self_and_others_are_placed_on_the_place_grid():
    wm = WorldModel()
    s = wm.update(0.0, (0.1, 0.1), 0.0, (Track("ball", 0.0, 1.0, tid="ball-0"),))
    assert any(r.subject == "self" and r.obj == "place:0,0" for r in s.related("at_place"))
    assert any(r.subject == "track:ball-0" and r.obj == "place:1,0" for r in s.related("at_place"))


def test_queries_used_by_appraisal_and_behaviour():
    wm = WorldModel()
    near_a = Track("person", 0.0, 0.4, tid="person-3")
    s = wm.update(0.0, (0, 0), 0.0, (near_a, B), identities={"person-3": "person#0"})
    assert s.identity_map() == {"person-3": "person#0"}
    assert {tr.tid for tr in s.tracks()} == {"person-3", "person-4"}
    assert s.nearest("person", "near").track_id == "person-3"
    s = wm.update(0.1, (0, 0), 0.0, (B,))  # person#0 no longer measured: node kept, measurement dropped
    assert s.node("person#0").track is None and {tr.tid for tr in s.tracks()} == {"person-4"}
    s = wm.assert_touch(0.2, "person-4")
    assert s.related("touching", "self")[0].subject == "track:person-4"
