import numpy as np

from nerva.events.prototypes import MAX_PROTOTYPES, PrototypeMemory
from nerva.events.segmentation import BoundaryDetector, OnlinePredictor


def test_predictor_learns_a_linear_dynamic():
    rng = np.random.default_rng(0)
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    p = OnlinePredictor(2)
    x = rng.normal(size=2)
    for _ in range(300):
        nxt = A @ x + 0.01 * rng.normal(size=2)
        p.update(x, nxt)
        x = nxt if np.linalg.norm(nxt) > 0.1 else rng.normal(size=2)
    assert np.allclose(p.A[:, :2], A, atol=0.05)


def test_a_sudden_change_is_a_boundary_and_steady_drift_is_not():
    rng = np.random.default_rng(1)
    det = BoundaryDetector(1)
    flags = []
    for k in range(300):
        x = np.array([0.01 * k + 0.01 * rng.normal() + (3.0 if k >= 200 else 0.0)])
        flags.append(det.step(x))
    assert flags[200] and sum(flags[40:199]) == 0


def test_prototypes_are_bounded_and_track_outcome_rates():
    mem = PrototypeMemory(new_distance=0.5, max_prototypes=3)
    for k in range(10):
        mem.assign(np.array([float(k)]))
    assert len(mem.prototypes) == 3 <= MAX_PROTOTYPES
    p = mem.assign(np.array([100.0]))
    before = p.p("near_collision")
    for _ in range(5):
        mem.resolve(p, ["near_collision"])
    assert p.p("near_collision") > before and p.p("benign_contact") < before
