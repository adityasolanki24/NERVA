"""Data passed between NERVA's layers.

    PERCEPTION ──PerceptionState──▶ APPRAISAL ──AppraisalState──▶ AFFECT
    AFFECT ──PADState──▶ BEHAVIOUR ──BehaviourCommand (incl. ExpressiveStyle)──▶ LOCOMOTION

Each layer talks to the next *only* through these types. That keeps the layers
replaceable: a rule-based appraisal and a learned one must both produce an
AppraisalState, and the locomotion controller never sees raw events or PAD.

This module holds only containers with documented meanings and range checks,
plus the AffectSystem protocol. Implementations live in other modules.

Status of the ideas behind each type (see docs/architecture.md):
  - Appraisal variables: taken from appraisal theory, specifically the EMA model
    (Marsella & Gratch). Which subset we use, and their numeric ranges, is our choice.
  - PAD: an established dimensional description of affect (Mehrabian & Russell).
    Using PAD as EMA's output is a NERVA design hypothesis; EMA does not use PAD.
  - ExpressiveStyle: a single engineering control variable for the first
    locomotion experiment. It carries no emotional meaning until evidence
    (measurements, then human evaluation) supports one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not lo <= value <= hi:
        raise ValueError(f"{name}={value} outside [{lo}, {hi}]")


# ── Perception: "what happened?" ─────────────────────────────────────────────


@dataclass(frozen=True)
class Event:
    """One thing perception detected, e.g. kind="person_approaching".

    `kind` is a free string for now. A fixed vocabulary will be defined when
    the first synthetic-event prototype exists, not before.
    """

    kind: str
    magnitude: float = 1.0  # [0, 1], how strongly/clearly it is present
    source: str = ""  # ID of the track it is about (e.g. "person-1"), if any

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("Event.kind must be non-empty")
        _check_range("magnitude", self.magnitude, 0.0, 1.0)


@dataclass(frozen=True)
class Track:
    """One tracked thing in the robot's surroundings (simulated detector + filter).

      kind            "person" | "ball" | ...
      bearing         rad, horizontal direction relative to the robot's BODY heading, + = left
      distance        m, horizontal distance from the robot's head
      elevation       rad, vertical angle from the head camera's axis to the target, + = up
      approach_speed  m/s, rate at which the distance shrinks (+ = coming closer), filtered
      visible         detected in the most recent frame
      seen_for_s      time since the track was created
      unseen_for_s    time since the last detection (0 while visible)
    """

    kind: str
    bearing: float
    distance: float
    elevation: float = 0.0
    approach_speed: float = 0.0
    visible: bool = True
    seen_for_s: float = 0.0
    unseen_for_s: float = 0.0
    tid: str = ""  # track ID; several objects of one kind can be tracked at once


@dataclass(frozen=True)
class PerceptionState:
    """Everything perception reports at one instant. Carries no interpretation."""

    time_s: float
    events: tuple[Event, ...] = field(default_factory=tuple)
    tracks: tuple[Track, ...] = field(default_factory=tuple)


# ── Appraisal: "what does this mean for the robot?" ──────────────────────────


@dataclass(frozen=True)
class AppraisalState:
    """EMA-inspired appraisal of the current situation.

    Variable names follow EMA. EMA defines more (e.g. causal attribution,
    changeability); we start with five. Ranges are a NERVA convention:
      relevance       [0, 1]   does this matter to the robot's goals at all
      desirability    [-1, 1]  harmful (-1) ... beneficial (+1)
      likelihood      [0, 1]   how probable the appraised outcome is
      expectedness    [0, 1]   0 = surprising, 1 = fully expected
      controllability [0, 1]   how much the robot can influence the outcome
    """

    relevance: float = 0.0
    desirability: float = 0.0
    likelihood: float = 0.0
    expectedness: float = 1.0
    controllability: float = 1.0

    def __post_init__(self) -> None:
        _check_range("relevance", self.relevance, 0.0, 1.0)
        _check_range("desirability", self.desirability, -1.0, 1.0)
        _check_range("likelihood", self.likelihood, 0.0, 1.0)
        _check_range("expectedness", self.expectedness, 0.0, 1.0)
        _check_range("controllability", self.controllability, 0.0, 1.0)


# ── Affect: persistent internal state ────────────────────────────────────────


@dataclass(frozen=True)
class PADState:
    """Pleasure/valence, arousal, dominance, each normalised to [-1, 1].

    (0, 0, 0) is neutral. The values describe the robot's internal state only;
    they are not claims about what a human would feel or perceive.
    """

    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def __post_init__(self) -> None:
        _check_range("valence", self.valence, -1.0, 1.0)
        _check_range("arousal", self.arousal, -1.0, 1.0)
        _check_range("dominance", self.dominance, -1.0, 1.0)


@runtime_checkable
class AffectSystem(Protocol):
    """Contract for any affect model: appraisals in, a persistent PADState out.

    How appraisal becomes PAD is deliberately NOT fixed here. Affect model v0
    (`nerva.affect.CategoricalAffectModel`) goes through discrete emotion labels
    ("Model A"); a future model may map appraisal to PAD directly ("Model B").
    Both must satisfy this protocol so the layers around them do not change.
    """

    def add(self, appraisal: AppraisalState) -> object: ...

    def step(self, dt: float) -> PADState: ...

    @property
    def pad(self) -> PADState: ...


# ── Behaviour: "what should I do, and how?" ──────────────────────────────────


@dataclass(frozen=True)
class ExpressiveStyle:
    """How to move, separate from what to do.

    One scalar for the first experiment. Internal labels only:
      -1 = "Style -1", 0 = "Neutral", +1 = "Style +1".
    Words such as hesitant/confident are hypotheses to be tested, not names.
    """

    style: float = 0.0

    def __post_init__(self) -> None:
        _check_range("style", self.style, -1.0, 1.0)

    @property
    def label(self) -> str:
        if self.style == 0.0:
            return "Neutral"
        return f"Style {self.style:+g}"


@dataclass(frozen=True)
class StyleVector:
    """Input e of the style-conditioned policy S1 (docs/style_policy_design.md).

      tempo        e1: gait period 0.54·(1 − 0.25·e1) s
      step_height  e2: reference foot height 40·(1 + 0.5·e2) mm (NOT expressed by S1, 2026-09-30)
      torso_pitch  e3: reference trunk pitch −4 + 6·e3 deg (+ = more forward lean in MuJoCo)

    Each in [-1, 1]. S1 was trained only at 0 and ±1 on one axis at a time; other
    values interpolate or combine axes and are outside its training distribution.
    """

    tempo: float = 0.0
    step_height: float = 0.0
    torso_pitch: float = 0.0

    def __post_init__(self) -> None:
        for name in ("tempo", "step_height", "torso_pitch"):
            _check_range(name, getattr(self, name), -1.0, 1.0)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.tempo, self.step_height, self.torso_pitch)


SKILLS = ("walk",)  # only what the baseline can actually do; extend when a policy exists


@dataclass(frozen=True)
class BehaviourCommand:
    """What the behaviour layer asks the locomotion layer to do.

    Velocities are in the robot's heading frame, SI units. They are requests:
    the baseline policy does not track them exactly (see
    docs/open_duck_baseline.md §11), so experiments must log measured velocity.
    Clipping to the policy's trained command range is the locomotion adapter's
    job, not this layer's.
    """

    vx: float = 0.0  # m/s, forward +
    vy: float = 0.0  # m/s, left +
    yaw_rate: float = 0.0  # rad/s, counter-clockwise +
    skill: str = "walk"
    style: ExpressiveStyle = field(default_factory=ExpressiveStyle)
    style_vector: StyleVector | None = None  # only for S1-type policies

    def __post_init__(self) -> None:
        if self.skill not in SKILLS:
            raise ValueError(f"unknown skill {self.skill!r}; available: {SKILLS}")
        if self.style_vector is not None and self.style.style != 0.0:
            raise ValueError("use either the phase-clock style (method A) or a StyleVector (S1), not both")
