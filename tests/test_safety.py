import inspect

from nerva.behaviour.selection import UtilityBehaviour
from nerva.interfaces import ActionTendencyState, BehaviourCommand, PADState, SelfState, Track
from nerva.safety import RESUME_HOLD_S, STOP_TILT_DEG, SafetySupervisor


def test_safety_reads_no_affect_memory_or_appraisal():
    params = set(inspect.signature(SafetySupervisor.filter).parameters)
    assert params == {"self", "command", "head", "self_state"}
    import nerva.safety as mod

    code = inspect.getsource(mod).split('"""', 2)[2]  # everything after the module docstring
    for word in ("tendenc", "PAD", "emotion", "memory", "apprais"):
        assert word not in code


def test_strong_approach_tendency_cannot_suppress_a_safety_stop():
    beh = UtilityBehaviour()
    ball = (Track("ball", 0.0, 2.0, tid="ball-0"),)
    d = None
    for i in range(10):
        d = beh.step(5.0 + 0.1 * i, 0.1, PADState(), ActionTendencyState(explore=5.0, approach=5.0), ball, {"ball-0": 1.0})
    assert d.command.vx > 0  # behaviour wants to go
    sup = SafetySupervisor()
    cmd, head, reason = sup.filter(d.command, d.head, SelfState(time_s=6.0, tilt_deg=STOP_TILT_DEG + 2))
    assert (cmd.vx, cmd.yaw_rate) == (0.0, 0.0) and head == (0.0, 0.0, 0.0, 0.0) and reason


def test_stop_is_held_until_upright_long_enough():
    sup = SafetySupervisor()
    go = BehaviourCommand(vx=0.15)
    sup.filter(go, (0, 0, 0, 0), SelfState(time_s=0.0, tilt_deg=30.0))
    assert sup.filter(go, (0, 0, 0, 0), SelfState(time_s=0.1, tilt_deg=5.0))[0].vx == 0.0
    assert sup.filter(go, (0, 0, 0, 0), SelfState(time_s=0.1 + RESUME_HOLD_S, tilt_deg=5.0))[0].vx == 0.15
    assert sup.interventions == 1


def test_normal_walking_passes_unchanged():
    sup = SafetySupervisor()
    go = BehaviourCommand(vx=0.15, yaw_rate=0.3)
    assert sup.filter(go, (0, 0.1, 0.2, 0), SelfState(time_s=1.0, tilt_deg=15.0)) == (go, (0, 0.1, 0.2, 0), "")
