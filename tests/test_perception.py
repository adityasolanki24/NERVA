import numpy as np

from nerva.perception import SimulatedPerception

CAM = np.array([0.0, 0.0, 0.3])
# camera looking along world +x, +y up = world +z  (columns: cam x, y, z)
XMAT = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)
DT = 0.1


def _run(perception, path, t0=0.0, kind="person"):
    out = []
    for i, pos in enumerate(path):
        out.append(perception.detect(t0 + i * DT, CAM, XMAT, 0.0, {kind: np.array(pos)}, DT))
    return out


def kinds(states):
    return [e.kind for s in states for e in s.events]


def test_only_things_in_view_are_detected():
    p = SimulatedPerception(seed=1)
    behind = _run(p, [(-2.0, 0.0, 1.5)] * 20)
    assert not any(s.tracks for s in behind)
    front = _run(p, [(2.0, 0.5, 1.5)] * 20, t0=2.0)
    assert "person_appeared" in kinds(front)
    track = front[-1].tracks[0]
    assert abs(track.distance - np.hypot(2.0, 0.5)) < 0.15 and track.bearing > 0  # left of the robot
    assert track.elevation > 0  # the face is above the camera


def test_fast_and_slow_approach_events():
    fast = kinds(_run(SimulatedPerception(seed=2), [(max(0.3, 2.5 - 1.5 * DT * i), 0, 1.5) for i in range(25)]))
    assert "person_approaching_rapidly" in fast and "person_close" in fast
    slow = kinds(_run(SimulatedPerception(seed=3), [(2.8 - 0.2 * DT * i, 0, 1.5) for i in range(60)]))
    assert "person_approaching_slowly" in slow and "person_approaching_rapidly" not in slow


def test_standing_person_does_not_trigger_approach_events():
    still = kinds(_run(SimulatedPerception(seed=4), [(1.5, 0.2, 1.5)] * 100))
    assert still == ["person_appeared"]


def test_track_is_forgotten_then_reported_lost():
    p = SimulatedPerception(seed=5)
    seen = _run(p, [(1.5, 0, 1.5)] * 10)
    gone = _run(p, [(-3.0, 0, 1.5)] * 50, t0=1.0)
    assert seen[-1].tracks and not gone[-1].tracks
    assert kinds(gone) == ["person_lost"]
