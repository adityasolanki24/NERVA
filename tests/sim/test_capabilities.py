import pytest

from nerva.behaviour.selection import UtilityBehaviour
from nerva.interfaces import ActionTendencyState, PADState, PolicyCapabilities, Track
from nerva.sim.capabilities import CAPABILITIES, LEGACY, capabilities_for


def test_policies_are_identified_by_their_run_folder():
    assert capabilities_for("experiments/cloud_runs/b2_neutral-20260930-163257/checkpoints/x.onnx").name == "B2"
    assert capabilities_for("experiments/cloud_runs/s1_pilot-20260929-231102/checkpoints/x.onnx").name == "S1"
    assert capabilities_for("somewhere/unknown.onnx") is LEGACY and not LEGACY.tested


def test_b2_envelope_is_the_measured_one():
    b2 = CAPABILITIES["B2"]
    assert b2.tested and b2.style_input == "neutral_only" and b2.training_scene == "backlash"
    assert b2.head_yaw_range == (-0.4, 0.4) and b2.head_pitch_range[0] == -0.2 and "2026-10-02" in b2.evidence


def test_invalid_capabilities_are_rejected():
    with pytest.raises(ValueError):
        PolicyCapabilities("x", head_yaw_range=(0.1, 0.4))
    with pytest.raises(ValueError):
        PolicyCapabilities("x", style_input="telepathy")


def test_behaviour_never_commands_head_offsets_outside_the_policy_envelope():
    beh = UtilityBehaviour()
    beh.apply_capabilities(CAPABILITIES["B2"])
    far_left_low = (Track("ball", bearing=1.2, distance=0.4, elevation=-1.0, tid="ball-0"),)
    d = None
    for i in range(40):
        d = beh.step(5.0 + 0.1 * i, 0.1, PADState(), ActionTendencyState(explore=0.5), far_left_low, {"ball-0": 1.0})
    assert -0.4 <= d.head[2] <= 0.4 and d.head[1] >= -0.2 - 1e-9
    beh.mode, beh.threat_bearing = "withdraw", 1.0  # withdraw looks away to the right: must stay inside too
    assert beh._control(10.0, PADState(), {})[2][2] >= -0.4
