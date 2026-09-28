"""Appraisal layer: what does an event mean for the robot?

EMA-inspired (Marsella & Gratch 2009): events are not mapped to emotions
directly. Each event is first appraised along EMA's appraisal variables; the
affect layer derives emotions from the appraisal.

The appraisal VALUES below are NERVA design choices for synthetic events, not
measurements or psychology. Reasoning per event: docs/affect_model.md §1.
Replace this table (or `appraise`) to change how events are interpreted.
"""

from __future__ import annotations

import dataclasses

from nerva.interfaces import AppraisalState, Event

EVENT_APPRAISALS: dict[str, AppraisalState] = {
    "successful_walking": AppraisalState(
        relevance=0.5, desirability=0.4, likelihood=1.0, expectedness=0.9, controllability=0.9),
    "near_fall": AppraisalState(
        relevance=1.0, desirability=-0.8, likelihood=0.5, expectedness=0.1, controllability=0.3),
    "person_approaching_slowly": AppraisalState(
        relevance=0.4, desirability=0.2, likelihood=0.6, expectedness=0.7, controllability=0.7),
    "person_approaching_rapidly": AppraisalState(
        relevance=0.9, desirability=-0.6, likelihood=0.6, expectedness=0.2, controllability=0.3),
    "obstacle_blocking_goal": AppraisalState(
        relevance=0.8, desirability=-0.5, likelihood=1.0, expectedness=0.5, controllability=0.6),
}


def appraise(event: Event, table: dict[str, AppraisalState] = EVENT_APPRAISALS) -> AppraisalState:
    """Look up the event's appraisal; `event.magnitude` scales its desirability."""
    try:
        base = table[event.kind]
    except KeyError:
        raise ValueError(f"no appraisal defined for event {event.kind!r}; known: {sorted(table)}") from None
    return dataclasses.replace(base, desirability=base.desirability * event.magnitude)
