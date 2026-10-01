from nerva.affect.emotions import CategoricalAffectModel
from nerva.affect.tendencies import tendencies_from_emotions
from nerva.interfaces import AppraisalState


def test_mapping_reproduces_the_former_utility_drives():
    t = tendencies_from_emotions({"interest": 0.4, "hope": 0.1, "joy": 0.2, "fear": 0.3, "surprise": 0.5,
                                  "distress": 0.05})
    assert t.explore == 0.4 and abs(t.approach - 0.3) < 1e-12 and t.avoid == 0.3
    assert t.orient == 0.5 and t.freeze == 0.5 * 0.3 and t.withdraw == 0.05


def test_missing_labels_mean_zero_and_unknown_labels_are_ignored():
    assert tendencies_from_emotions({}) == tendencies_from_emotions({"pride": 3.0})


def test_model_a_tendencies_follow_its_active_emotions():
    model = CategoricalAffectModel()
    model.add(AppraisalState(relevance=0.9, desirability=-0.6, likelihood=0.6, expectedness=0.2, controllability=0.3))
    inten = model.intensities()
    assert model.tendencies == tendencies_from_emotions(inten)
    assert inten["fear"] > 0 and model.tendencies.avoid == inten["fear"]
