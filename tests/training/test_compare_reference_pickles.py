from scripts.compare_reference_pickles import compare_references


def _entry(period, values):
    return {
        "period": period,
        "coefficients": {"dim_0": values[:2], "dim_1": values[2:]},
    }


def test_compare_references_reports_grid_and_numerical_differences():
    shipped = {
        "0_0_0": _entry(0.5, [1, 2, 3, 4]),
        "1_0_0": _entry(0.6, [5, 6, 7, 8]),
    }
    generated = {
        "0_0_0": _entry(0.5, [1, 2.5, 3, 4]),
        "2_0_0": _entry(0.7, [9, 10, 11, 12]),
    }

    result = compare_references(shipped, generated)

    assert result["shipped_keys"] == 2
    assert result["generated_keys"] == 2
    assert result["common_keys"] == 1
    assert result["only_shipped_keys"] == 1
    assert result["only_generated_keys"] == 1
    assert result["max_abs_coefficient_delta_on_common_keys"] == 0.5
    assert result["shipped_period_range"] == [0.5, 0.6]
    assert result["generated_period_range"] == [0.5, 0.7]
