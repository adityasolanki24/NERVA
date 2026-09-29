import pytest

from cloud.generate_styled_references import PILOT_STYLES, apply_style


BASE = {
    "single_support_duration": 0.18,
    "walk_foot_height": 0.04,
    "walk_trunk_pitch": -4,
    "untouched": 123,
}


def test_pilot_contains_neutral_and_axis_endpoints():
    assert [item[0] for item in PILOT_STYLES] == [
        "neutral", "e1_neg", "e1_pos", "e2_neg", "e2_pos", "e3_neg", "e3_pos"
    ]


@pytest.mark.parametrize(
    ("style", "duration", "height", "pitch"),
    [
        ((0, 0, 0), 0.18, 0.04, -4),
        ((-1, 0, 0), 0.225, 0.04, -4),
        ((1, 0, 0), 0.135, 0.04, -4),
        ((0, -1, 0), 0.18, 0.02, -4),
        ((0, 1, 0), 0.18, 0.06, -4),
        ((0, 0, -1), 0.18, 0.04, -10),
        ((0, 0, 1), 0.18, 0.04, 2),
    ],
)
def test_apply_style_matches_preregistered_mapping(style, duration, height, pitch):
    result = apply_style(BASE, style)
    assert result["single_support_duration"] == duration
    assert result["walk_foot_height"] == height
    assert result["walk_trunk_pitch"] == pitch
    assert result["untouched"] == 123
    assert BASE["single_support_duration"] == 0.18


def test_apply_style_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        apply_style(BASE, (0, 0, 1.1))
