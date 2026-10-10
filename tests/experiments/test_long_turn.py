import numpy as np
import pytest

from experiments.locomotion_curriculum.archive.long_turn import KinematicRecorder, analyse, summarise


def trace(rate=.6, drift=.0):
    t = np.arange(1, 3251) * .02
    yaw = rate * t
    position = np.column_stack([.06 * np.cos(yaw) + drift * t, .06 * np.sin(yaw), .15 + 0 * t])
    quat = np.column_stack([np.cos(yaw / 2), 0 * t, 0 * t, np.sin(yaw / 2)])
    velocity = np.column_stack([-.06 * rate * np.sin(yaw) + drift, .06 * rate * np.cos(yaw), 0 * t])
    feet = np.stack([position + [0, .03, -.15], position + [0, -.03, -.15]], axis=1)
    return {"t": t, "base_pos": position, "base_quat": quat, "base_linvel": velocity,
            "com": position.copy(), "foot_xyz": feet, "measured_contacts": np.ones((len(t), 2), dtype=bool),
            "contact_centroid": feet.mean(axis=1), "foot_z": feet[:, :, 2]}


@pytest.mark.parametrize("rate,command", [(.6, "turn_left"), (-.6, "turn_right"), (0., "stop")])
def test_known_local_region_and_migration(rate, command):
    local = analyse(trace(rate), command, True, 0)
    assert local["classification"] == "local_region"
    assert local["blocks"] >= 4
    migrant = analyse(trace(rate, .004), command, True, 0)
    assert migrant["classification"] == "coherent_migration"


def test_airborne_incomplete_or_safety_cannot_support_description():
    a = trace()
    assert analyse(a, "turn_left", False, 0)["classification"] == "inconclusive"
    assert analyse(a, "turn_left", True, 1)["classification"] == "inconclusive"
    a["measured_contacts"][:] = False
    a["contact_centroid"][:] = np.nan
    assert analyse(a, "turn_left", True, 0)["classification"] == "inconclusive"


def test_incomplete_aggregate_cannot_pass():
    rows = [{"command": c, "seed": s, "completed": True, "classification": "local_region"}
            for c in ("turn_left", "turn_right", "stop") for s in range(5)]
    assert summarise(rows)["aggregate"] == "local_region"
    assert summarise(rows[:-1])["aggregate"] == "inconclusive"
    assert summarise(rows + rows[:1])["aggregate"] == "inconclusive"


def test_isolated_kinematics_matches_subtree_com_without_changing_live_state():
    pytest.importorskip("mujoco")
    pytest.importorskip("playground.open_duck_mini_v2.mujoco_infer")
    from nerva.sim.open_duck import OpenDuckSim
    sim = OpenDuckSim()
    sim.run(.1)
    live = {name: getattr(sim.data, name).copy() for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "site_xpos")}
    recorder = KinematicRecorder(sim)
    com, feet, contacts, centroid = recorder.sample(sim)
    np.testing.assert_allclose(com, recorder.data.subtree_com[recorder.root], atol=1e-12)
    assert feet.shape == (2, 3) and contacts.shape == (2,)
    if contacts.any():
        np.testing.assert_allclose(centroid, feet[contacts].mean(axis=0))
    for name, before in live.items():
        np.testing.assert_array_equal(getattr(sim.data, name), before)
    assert recorder.model.nbody > len(recorder.bodies)  # world is excluded
