import pytest

from nerva.interfaces import ExpressiveStyle
from nerva.style import style_to_phase_factor


@pytest.mark.parametrize("style,factor", [(-1.0, 0.7), (0.0, 1.0), (1.0, 1.3), (0.5, 1.15)])
def test_style_maps_linearly_to_phase_factor(style, factor):
    assert style_to_phase_factor(ExpressiveStyle(style)) == pytest.approx(factor)
