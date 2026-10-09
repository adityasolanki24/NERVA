"""Preregistered same-source facet invariants; no scenario-outcome tuning."""

from dataclasses import replace

import numpy as np

from nerva.affect.model_b import FacetAttractorAffectModel, OnsetAttractorAffectModel
from nerva.interfaces import AffectSystem
from tests.affect.test_model_bv2 import IN_VIEW, LUNGE


def facet(source, kind="ongoing_contact", **kwargs):
    return replace(IN_VIEW, hypothesis=replace(IN_VIEW.hypothesis, subject=source, kind=kind), **kwargs)


def test_single_facet_and_phasic_equivalence():
    old, new = OnsetAttractorAffectModel(), FacetAttractorAffectModel()
    assert isinstance(new, AffectSystem)
    for model in (old, new):
        model.add(IN_VIEW)
        model.add(LUNGE, source="person-0")
    for _ in range(600):
        old.step(0.1)
        new.step(0.1)
        np.testing.assert_allclose(old.x, new.x, atol=1e-15, rtol=0)
        assert old.tendencies == new.tendencies
        assert old.tendencies_for("person-0") == new.tendencies_for("person-0")
    assert not new.context


def test_duplicate_facets_are_idempotent_but_distinct_sources_add():
    model = FacetAttractorAffectModel()
    u = model.add(IN_VIEW)
    model.add(facet("ball-0"))
    np.testing.assert_array_equal(model.context_input(), u)
    model.add(facet("other"))
    np.testing.assert_array_equal(model.context_input(), 2 * u)


def test_mixed_facets_average_and_refresh_replaces():
    model = FacetAttractorAffectModel()
    a = model.add(IN_VIEW)
    b = model.add(facet("ball-0", desirability=-0.8, controllability=0.1))
    np.testing.assert_allclose(model.context_input(), (a + b) / 2)
    c = model.add(facet("ball-0", desirability=0.7))
    assert len(model.context) == 2
    np.testing.assert_allclose(model.context_input(), (a + c) / 2)


def test_stale_facet_loses_influence_while_single_source_still_fades():
    model = FacetAttractorAffectModel()
    a = model.add(IN_VIEW)
    model.t = 5.5  # old facet weight exp(-1)
    b = model.add(facet("ball-0", desirability=0.7))
    w = np.exp(-1)
    np.testing.assert_allclose(model.context_input(), (w * a + b) / (w + 1))
    model.t += 5.5
    old_weight = np.exp(-8.5 / 3)
    np.testing.assert_allclose(model.context_input(), w * (old_weight * a + w * b) / (old_weight + w))


def test_source_override_and_separator_are_unambiguous():
    model = FacetAttractorAffectModel()
    a = model.add(facet("ignored", kind="novel_stimulus"), source="a|b")
    b = model.add(facet("a", kind="ongoing_contact", desirability=-0.7))
    np.testing.assert_allclose(model.context_input(), a + b)
    assert ("a|b", "novel_stimulus") in model.context
    assert ("a", "ongoing_contact") in model.context


def test_tendencies_and_phasic_input_match_bv3_with_multiple_facets():
    old, new = OnsetAttractorAffectModel(), FacetAttractorAffectModel()
    for model in (old, new):
        model.add(IN_VIEW)
        model.add(facet("ball-0", desirability=0.7))
        model.add(LUNGE)
        model.step(0.1)
    np.testing.assert_array_equal(old.z_phasic, new.z_phasic)
    assert old.tendencies == new.tendencies
    assert old.tendencies_for("ball-0") == new.tendencies_for("ball-0")
