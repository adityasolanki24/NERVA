import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("playground.open_duck_mini_v2.joystick")
jax = pytest.importorskip("jax")
import jax.numpy as jp  # noqa: E402

from nerva.interfaces import BehaviourCommand, ExpressiveStyle  # noqa: E402
from nerva.sim.open_duck import OpenDuckSim  # noqa: E402
from nerva.training.neutral_joystick import body_imitation  # noqa: E402
from nerva.training.neutral_reference import COMMANDS, CONTRACT, REQUIRED_CRITERIA, NeutralReference, verified_references  # noqa: E402
from nerva.training.reference_kinematics import fit_reference, sample_reference  # noqa: E402
from nerva.training.reference_validation import REFERENCE_JOINTS  # noqa: E402


def synthetic_records():
    t = np.arange(100) * .02
    rows = []
    for command in COMMANDS:
        positions = .1 * np.sin(2 * np.pi * t[:, None] / .54) * np.ones((1, 16))
        previous = .1 * np.sin(2 * np.pi * (t[:, None] - .02) / .54) * np.ones((1, 16))
        ref = fit_reference(t, positions, np.ones((100, 2)), np.zeros((100, 3)), np.zeros((100, 3)),
                            .54, static=command == (0., 0., 0.), interval_start=t - .02,
                            joint_velocity=(positions - previous) / .02)
        rows.append({"command": list(command), "reference": ref})
    return rows


def test_exact_jitted_lookup_and_position_derived_velocity_parity():
    rows = synthetic_records()
    sampler = NeutralReference(rows)
    get = jax.jit(sampler.get_reference_motion)
    for row in rows:
        expected = sample_reference(row["reference"], np.arange(27) * .02)
        for index in range(27):
            value = np.asarray(get(*row["command"], index))
            np.testing.assert_allclose(value[:16], expected["joint_position"][index], atol=1e-6)
            np.testing.assert_allclose(value[16:32], expected["joint_velocity"][index], atol=1e-5)
    assert np.isnan(np.asarray(get(.01, 0., 0., 0))).all()


def test_body_imitation_uses_declared_frames_contacts_and_rest_gate():
    target = jp.zeros(40).at[32:34].set(1).at[34:37].set(jp.array([.1, .2, .3]))
    target = target.at[37:40].set(jp.array([.2, .3, .4]))
    q = jp.zeros(14)
    args = (q, q, target[34:37], target[37:40], jp.ones(2), target)
    assert float(body_imitation(*args, jp.array([.074, 0, 0, 0, 0, 0, 0]))) == pytest.approx(5.)
    assert float(body_imitation(*args, jp.zeros(7))) == 0.
    mismatch = body_imitation(q, q, jp.zeros(3), target[37:40], jp.ones(2), target,
                              jp.array([.074, 0, 0, 0, 0, 0, 0]))
    assert float(mismatch) < 5.


def test_deployment_requires_matching_metadata_and_rejects_out_of_scope_commands(tmp_path):
    sim = OpenDuckSim.__new__(OpenDuckSim)
    sim.policy_path = tmp_path / "candidate.onnx"
    sim.policy_path.write_bytes(b"test artifact, not a trained policy")
    sim.style_vector, sim.head_offset = None, np.zeros(4)
    sim._neutral_motor_clock = None
    sim.inf = SimpleNamespace(commands=[0.] * 7)
    metadata = {"contract": CONTRACT, "observation_size": 101, "period_steps": 27,
                "head_commands_zero": True, "commands": COMMANDS,
                "policy_sha256": hashlib.sha256(sim.policy_path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError):
        sim.set_neutral_motor_contract({**metadata, "policy_sha256": "wrong"})
    sim.set_neutral_motor_contract(metadata)
    sim.set_behaviour(BehaviourCommand(vx=.074))
    assert sim.inf.commands == [.074, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        sim.set_behaviour(BehaviourCommand(vx=.15))
    with pytest.raises(ValueError):
        sim.set_behaviour(BehaviourCommand(style=ExpressiveStyle(.5)))
    with pytest.raises(ValueError):
        sim.set_head_offset(head_yaw=.1)
    with pytest.raises(ValueError):
        sim.set_style_vector([0, 0, 0])


def test_admission_checks_every_criterion_and_both_artifact_hashes(tmp_path):
    records = synthetic_records()
    names = ("stand", "forward", "backward", "left", "right", "turn_left", "turn_right")
    reports = {"repair": [], "pivot_retry": []}
    for name, record in zip(names, records):
        suffix = "pivot_retry" if name.startswith("turn") else "repair"
        raw = tmp_path / f"experiments/cloud_runs/neutral-reference-{suffix.replace('_', '-')}" / name
        raw.mkdir(parents=True)
        ref = record["reference"]
        ref.update(joint_names=list(REFERENCE_JOINTS), joint_velocity_semantics="analytic_instantaneous",
                   contact_semantics="static_geometric_support" if ref["static"] else "planned_support")
        reference = raw / "reference.json"
        reference.write_text(json.dumps(ref), encoding="utf-8")
        recording = raw / "motion.json"
        recording.write_text("{}", encoding="utf-8")
        reports[suffix].append({"condition": name, "command": record["command"], "all_pass": True,
                                "criteria": dict.fromkeys(REQUIRED_CRITERIA, True),
                                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                                "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest()})
    for suffix, rows in reports.items():
        report = tmp_path / f"experiments/locomotion_curriculum/results_reference_{suffix}/trials.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps(rows), encoding="utf-8")
    admitted, manifest = verified_references(tmp_path)
    assert len(admitted) == 7 and manifest["mixed_provenance"]
    raw = tmp_path / "experiments/cloud_runs/neutral-reference-repair/stand/motion.json"
    raw.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_references(tmp_path)
    raw.write_text("{}", encoding="utf-8")
    reports["repair"][0]["criteria"] = {}
    report = tmp_path / "experiments/locomotion_curriculum/results_reference_repair/trials.json"
    report.write_text(json.dumps(reports["repair"]), encoding="utf-8")
    with pytest.raises(ValueError, match="admission failed"):
        verified_references(tmp_path)
