from experiments.v2_autonomy.scenarios import curiosity_repeat_scenario, social_history_scenario


def test_curiosity_scene_has_a_separate_repeat_encounter():
    ball = curiosity_repeat_scenario()["ball"]
    assert [step[0] for step in ball.steps if step[1] == "appear"] == [4.0, 42.0]


def test_social_histories_share_the_same_later_meeting():
    later = []
    for history in ("positive", "cautious"):
        steps = social_history_scenario(history)["person"].steps
        later.append([step for step in steps if step[0] >= 38.0])
    assert later[0] == later[1]


def test_unknown_social_history_is_rejected():
    try:
        social_history_scenario("labelled-friendly")
    except ValueError:
        pass
    else:
        raise AssertionError("unrecognised history must not create a scenario")
