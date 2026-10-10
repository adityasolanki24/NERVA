"""E1 Phase A helpers: style mapping, coefficient interpolation, gate bookkeeping."""
import numpy as np
import pytest

from experiments.locomotion_curriculum.e1_references import CHECKPOINTS, GRID, interpolate, style_preset


def test_style_mapping_matches_the_preregistration():
    base = {"walk_trunk_pitch": -4, "walk_com_height": .215}
    assert style_preset(base, (1, -1), 1., 1.) == {"walk_trunk_pitch": 2, "walk_com_height": .203}
    assert style_preset(base, (-1, 1), 1., .5)["walk_com_height"] == pytest.approx(.221)
    assert style_preset(base, (0, 0), 1., 1.) == base
    assert len(GRID) == 8 and (0, 0) not in GRID and len(CHECKPOINTS) == 4


def test_interpolation_is_linear_in_coefficients():
    refs = [{"version": 2, "coefficients": {"joint_position": np.full((25, 16), float(i)).tolist()}} for i in range(4)]
    mix = interpolate(refs, [.25] * 4)
    assert np.allclose(mix["coefficients"]["joint_position"], 1.5)
    assert mix["version"] == 2
