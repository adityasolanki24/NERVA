"""Model A adapter: discrete emotion intensities → model-independent action tendencies.

Behaviour consumes `ActionTendencyState` (nerva/interfaces.py), never emotion labels, so that an affect
model without labels (Model B) can drive it. This module is the only place where Model A's labels are
translated; it belongs to Model A.

Mapping [NERVA design, chosen to reproduce the drives the utility selector used before the refactor, so
behaviour is unchanged with Model A]:

  approach  = hope + joy          (affiliative: liked or promising)
  explore   = interest            (novelty-seeking; behaviour scales it by each target's novelty)
  avoid     = fear
  orient    = surprise
  freeze    = surprise · fear     (startle under threat)
  withdraw  = distress

The pairing follows the spirit of Frijda's action tendencies (fear ↔ avoidance, interest ↔ exploration,
surprise ↔ interruption/attending) but the formulas are ours and not validated.
"""

from __future__ import annotations

from collections.abc import Mapping

from nerva.interfaces import MAX_TENDENCY, ActionTendencyState

LABELS = ("joy", "hope", "fear", "distress", "surprise", "interest")


def _bounded(x: float) -> float:
    return min(max(float(x), 0.0), MAX_TENDENCY)


def tendencies_from_emotions(intensity: Mapping[str, float]) -> ActionTendencyState:
    """Summed intensity per emotion label (missing labels = 0) → tendencies."""
    g = {lbl: float(intensity.get(lbl, 0.0)) for lbl in LABELS}
    return ActionTendencyState(
        approach=_bounded(g["hope"] + g["joy"]),
        explore=_bounded(g["interest"]),
        avoid=_bounded(g["fear"]),
        orient=_bounded(g["surprise"]),
        freeze=_bounded(g["surprise"] * g["fear"]),
        withdraw=_bounded(g["distress"]),
    )
