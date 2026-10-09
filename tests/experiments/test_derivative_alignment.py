import numpy as np
import pytest

from experiments.locomotion_curriculum.derivative_alignment import compare, known_checks
from nerva.training.reference_kinematics import fit_reference


def test_backward_difference_matches_interval_average_not_endpoint_derivative():
    rows = known_checks()
    assert len(rows) == 4 and all(r["pass"] for r in rows)
    assert all(r["interval_average"] < 1e-8 for r in rows)


def test_empty_heldout_or_bad_timestamps_are_rejected():
    t = np.arange(50) * .02
    ref = fit_reference(t, np.zeros((50, 16)), np.ones((50, 2)), np.zeros((50, 3)), np.zeros((50, 3)), .54)
    with pytest.raises(ValueError):
        compare(ref, t, np.zeros((50, 16)))
    with pytest.raises(ValueError):
        compare(ref, np.zeros(50), np.zeros((50, 16)))
