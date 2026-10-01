"""Typed contracts between NERVA's modules.

The cognitive side is a stateful graph, not a chain (docs/architecture.md):

    PERCEPTION --PerceptionState (tracks, events)--> APPRAISAL <-- SelfState, GoalState, memory
    APPRAISAL --AppraisalFrame (or the legacy AppraisalState)--> AFFECT
    AFFECT --PADState + ActionTendencyState--> BEHAVIOUR --BehaviourCommand--> LOCOMOTION
    ROBOT/WORLD --OutcomeSignal--> MEMORY (grounded learning)

This module holds only containers with documented meanings and range checks, plus the AffectSystem
protocol. Implementations live in other modules. Which of these contracts the running loop already uses
is tracked in docs/architecture.md section 2.3 (migration stages).

Status of the ideas behind each type:
  - Appraisal variables: from appraisal theory, specifically EMA (Marsella & Gratch). The subset and
    ranges are our choice. AppraisalFrame ties likelihood to an explicit OutcomeHypothesis.
  - PAD: an established dimensional description of affect (Mehrabian & Russell). Using PAD as the output
    of EMA-inspired appraisal is a NERVA design hypothesis; EMA does not use PAD.
  - Action tendencies: the idea that appraised situations come with readiness for approach, avoidance,
    attending or interruption is from appraisal theory (Frijda). The six fields and every mapping to
    them are NERVA design choices, not validated psychology.
  - OutcomeSignal: measurable consequences for the robot; the kinds are NERVA definitions.
  - ExpressiveStyle / StyleVector: engineering control variables; no emotional meaning without evidence.
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


# ── Self, goals, outcomes: what appraisal is relative to ─────────────────────


@dataclass(frozen=True)
class SelfState:
    """The robot's own condition, from quantities that are measured or estimated (not perfect sim state
    unless a real sensor would provide the same quantity).

      tilt_deg          torso tilt from vertical [IMU orientation estimate]
      angular_speed     |base angular velocity| rad/s [gyro]
      speed             horizontal body speed m/s [estimate: odometry / state estimator; sim: base velocity]
      feet_contact      (left, right) foot contact, or None if not measured [foot switches on hardware]
      stability_risk    [0, 1] derived estimate of fall risk (NERVA heuristic, not a sensor)
      escape_room       [0, 1] derived estimate of how freely it can move away, or None if not computed
      locomotion_mode   the behaviour mode currently being executed ("explore", "retreat", ...)
    """

    time_s: float = 0.0
    tilt_deg: float = 0.0
    angular_speed: float = 0.0
    speed: float = 0.0
    feet_contact: tuple[bool, bool] | None = None
    stability_risk: float = 0.0
    escape_room: float | None = None
    locomotion_mode: str = ""

    def __post_init__(self) -> None:
        _check_range("tilt_deg", self.tilt_deg, 0.0, 180.0)
        _check_range("stability_risk", self.stability_risk, 0.0, 1.0)
        if self.escape_room is not None:
            _check_range("escape_room", self.escape_room, 0.0, 1.0)
        if not (self.angular_speed >= 0 and self.speed >= 0):
            raise ValueError("angular_speed and speed are magnitudes (>= 0)")


GOAL_KINDS = ("remain_upright", "follow_command", "explore", "inspect", "approach", "keep_distance", "retreat")


@dataclass(frozen=True)
class Goal:
    """One active goal. `target` is a track/entity ID when the goal is about something specific."""

    kind: str
    target: str = ""
    priority: float = 1.0  # [0, 1]

    def __post_init__(self) -> None:
        if self.kind not in GOAL_KINDS:
            raise ValueError(f"unknown goal {self.kind!r}; known: {GOAL_KINDS}")
        _check_range("priority", self.priority, 0.0, 1.0)


@dataclass(frozen=True)
class GoalState:
    """The robot's currently active goals (small and explicit; not a planner)."""

    goals: tuple[Goal, ...] = ()

    def priority(self, kinds) -> float:
        """Highest priority among active goals of these kinds (0 if none is active)."""
        return max((g.priority for g in self.goals if g.kind in kinds), default=0.0)


ADVERSE_OUTCOMES = ("stability_loss", "near_collision")
BENIGN_OUTCOMES = ("benign_contact",)
OUTCOME_KINDS = ADVERSE_OUTCOMES + BENIGN_OUTCOMES


@dataclass(frozen=True)
class OutcomeSignal:
    """A measurable consequence for the robot: the ground truth for memory learning.

      stability_loss  tilt beyond the near-fall threshold [IMU]
      near_collision  something came within reach while closing fast [depth/tracking estimate]
      benign_contact  gentle, slow touch [touch sensor; simulated in MuJoCo]
    Only kinds the system can measure or estimate are defined; add more when a sensor exists.
    `source` is the track the outcome is attributed to ("" = none / the robot itself).
    """

    kind: str
    magnitude: float = 1.0  # [0, 1]
    time_s: float = 0.0
    source: str = ""
    provenance: str = ""  # which measurement produced it, for logs

    def __post_init__(self) -> None:
        if self.kind not in OUTCOME_KINDS:
            raise ValueError(f"unknown outcome {self.kind!r}; known: {OUTCOME_KINDS}")
        _check_range("magnitude", self.magnitude, 0.0, 1.0)

    @property
    def adverse(self) -> bool:
        return self.kind in ADVERSE_OUTCOMES


@dataclass(frozen=True)
class OutcomeHypothesis:
    """An explicit proposition about what may happen, e.g. "person-3 may collide with the robot".

      kind         proposition ID ("near_collision", "benign_interaction", "novel_stimulus", ...)
      subject      track/entity the proposition is about ("" = none)
      target       who is affected ("self" by default)
      probability  [0, 1] estimated probability that the proposition is or becomes true
      predicts     OutcomeSignal kinds that would confirm it (may be empty for non-physical outcomes)
    """

    kind: str
    subject: str = ""
    target: str = "self"
    probability: float = 0.5
    predicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("OutcomeHypothesis.kind must be non-empty")
        _check_range("probability", self.probability, 0.0, 1.0)
        for k in self.predicts:
            if k not in OUTCOME_KINDS:
                raise ValueError(f"unknown predicted outcome {k!r}")


@dataclass(frozen=True)
class AppraisalFrame:
    """Appraisal of ONE explicit hypothesis. Likelihood is the hypothesis's probability, so it always has
    a referent. Same five dimensions and ranges as AppraisalState.

      goals   goal kinds this hypothesis bears on (empty = none active)
    """

    hypothesis: OutcomeHypothesis
    relevance: float = 0.0
    desirability: float = 0.0
    expectedness: float = 1.0
    controllability: float = 1.0
    goals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_range("relevance", self.relevance, 0.0, 1.0)
        _check_range("desirability", self.desirability, -1.0, 1.0)
        _check_range("expectedness", self.expectedness, 0.0, 1.0)
        _check_range("controllability", self.controllability, 0.0, 1.0)

    @property
    def likelihood(self) -> float:
        return self.hypothesis.probability

    def as_appraisal_state(self) -> AppraisalState:
        """Compatibility view for consumers of the legacy AppraisalState."""
        return AppraisalState(relevance=self.relevance, desirability=self.desirability, likelihood=self.likelihood,
                              expectedness=self.expectedness, controllability=self.controllability)


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


TENDENCIES = ("approach", "explore", "avoid", "orient", "freeze", "withdraw")
MAX_TENDENCY = 10.0  # sanity bound: a tendency can sum several elicitations, but never explode


@dataclass(frozen=True)
class ActionTendencyState:
    """Model-independent readiness for kinds of action (>= 0; 0 = none). Behaviour consumes these, never
    emotion labels.

      approach  move toward an agent/thing because it is liked or promising (affiliative)
      explore   seek out and examine what is novel (behaviour scales it by each target's novelty)
      avoid     keep away from / move away from a threat
      orient    stop and attend to something sudden or unexpected
      freeze    stop all movement (startle under threat)
      withdraw  disengage: stop, lower the head, look away
    Which situations produce which tendency is defined by each affect model; the names and the mapping
    are NERVA design choices (Frijda-inspired), not validated psychology.
    """

    approach: float = 0.0
    explore: float = 0.0
    avoid: float = 0.0
    orient: float = 0.0
    freeze: float = 0.0
    withdraw: float = 0.0

    def __post_init__(self) -> None:
        for name in TENDENCIES:
            v = getattr(self, name)
            if not (0.0 <= v <= MAX_TENDENCY):  # also rejects NaN
                raise ValueError(f"tendency {name}={v} outside [0, {MAX_TENDENCY}]")


@runtime_checkable
class AffectSystem(Protocol):
    """Contract for any affect model: appraisals in; a persistent PADState and ActionTendencyState out.

    How appraisal becomes PAD and tendencies is deliberately NOT fixed here. Model A
    (`nerva.affect.emotions.CategoricalAffectModel`) goes through discrete emotion labels; Model B maps
    appraisal to PAD and tendencies directly. Behaviour and memory may depend only on this protocol.
    `add` accepts an AppraisalFrame or a legacy AppraisalState.
    """

    def add(self, appraisal: AppraisalState | AppraisalFrame) -> object: ...

    def step(self, dt: float) -> PADState: ...

    @property
    def pad(self) -> PADState: ...

    @property
    def tendencies(self) -> ActionTendencyState: ...


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
