import numpy as np

from nerva.reference_validation import KNEES, evaluate_gait, substitute_invalid, validate_reference


def _gait(knee_left, knee_right, n_signals=40):
    """Fitted-gait dict (lowest-order-first coefficients, as upstream fit_poly stores them)."""
    coeffs = {f"dim_{d}": [0.1, 0.0] for d in range(n_signals)}
    coeffs[f"dim_{KNEES['left_knee']}"] = knee_left
    coeffs[f"dim_{KNEES['right_knee']}"] = knee_right
    return {"coefficients": coeffs, "period": 0.54}


def test_evaluate_gait_uses_lowest_order_first_coefficients():
    joints = evaluate_gait(_gait([1.0, 0.5], [1.2, 0.0]), samples=3)
    np.testing.assert_allclose(joints[:, 3], [1.0, 1.25, 1.5])  # 1 + 0.5 t


def test_normal_gait_is_valid_even_beyond_the_knee_limit():
    report = validate_reference({"ok": _gait([1.4, 0.4], [1.3, 0.5])})  # reaches 1.8 > π/2
    assert report["valid"] and report["backward_knee_gaits"] == []
    assert report["knee_over_limit_gaits"] == 1


def test_mirrored_knee_is_rejected():
    report = validate_reference({"ok": _gait([1.4, 0.0], [1.3, 0.0]),
                                 "mirrored": _gait([1.4, 0.0], [-1.3, -0.2])})
    assert not report["valid"]
    assert report["backward_knee_gaits"] == ["mirrored"]
    assert report["backward_knee"][0]["joint"] == "right_knee"


def test_mid_cycle_branch_switch_is_rejected():
    report = validate_reference({"switch": _gait([-1.4, 3.0], [1.3, 0.0])})  # -1.4 → +1.6
    assert report["backward_knee_gaits"] == ["switch"]


def test_nonfinite_gait_is_rejected():
    report = validate_reference({"bad": _gait([float("nan"), 0.0], [1.3, 0.0])})
    assert report["nonfinite_gaits"] == ["bad"] and not report["valid"]


def test_invalid_gait_is_replaced_by_nearest_valid_neighbour():
    good, bad = _gait([1.4, 0.0], [1.3, 0.0]), _gait([1.4, 0.0], [-1.3, 0.0])
    reference = {"0.222_0.111_0.963": bad, "0.222_0.111_0.704": dict(good, tag="near"),
                 "-0.148_-0.111_-1.111": dict(good, tag="far")}
    fixed, mapping = substitute_invalid(reference, ["0.222_0.111_0.963"])
    assert mapping == {"0.222_0.111_0.963": "0.222_0.111_0.704"}
    assert fixed["0.222_0.111_0.963"]["tag"] == "near"
    assert validate_reference(fixed)["valid"]
    assert reference["0.222_0.111_0.963"] is bad  # input not mutated
