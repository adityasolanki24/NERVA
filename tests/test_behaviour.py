import pytest

from nerva.behaviour import WALK_VX, pad_to_style_vector, s1_behaviour
from nerva.interfaces import BehaviourCommand, ExpressiveStyle, PADState, StyleVector


def test_neutral_pad_gives_neutral_style():
    assert pad_to_style_vector(PADState()).as_tuple() == (0.0, 0.0, 0.0)


def test_directions_and_clipping():
    aroused = pad_to_style_vector(PADState(arousal=0.2))
    assert aroused.tempo == pytest.approx(0.6) and aroused.torso_pitch == 0.0
    sad = pad_to_style_vector(PADState(valence=-0.4, arousal=-0.1, dominance=-0.3))
    assert sad.tempo < 0 and sad.torso_pitch > 0  # slower, more forward lean
    assert pad_to_style_vector(PADState(arousal=1.0)).tempo == 1.0
    assert pad_to_style_vector(PADState(valence=1.0, dominance=1.0)).torso_pitch == -1.0


def test_step_height_is_never_used():
    assert pad_to_style_vector(PADState(0.9, 0.9, -0.9)).step_height == 0.0


def test_what_is_separate_from_how():
    pad = PADState(valence=-0.3, arousal=0.3)
    walk, stop = s1_behaviour(pad, False), s1_behaviour(pad, True)
    assert walk.command.vx == WALK_VX and stop.command.vx == 0.0
    assert walk.command.style_vector == stop.command.style_vector


def test_style_vector_and_phase_clock_style_are_exclusive():
    with pytest.raises(ValueError):
        BehaviourCommand(style=ExpressiveStyle(0.5), style_vector=StyleVector())
    with pytest.raises(ValueError):
        StyleVector(tempo=1.5)
