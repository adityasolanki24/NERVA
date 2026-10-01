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


# ── World model: entities, relations, sensor evidence ────────────────────────


@dataclass(frozen=True)
class SensorEvidence:
    """One piece of evidence about an entity from one modality.

      modality    e.g. "vision.appearance" (clothing-colour histogram today); later "vision.face",
                  "audio.voice", "touch.contact" ... Only modalities that exist produce evidence.
      feature     embedding / estimate vector, or None for evidence without a vector (e.g. a contact)
      confidence  [0, 1]
    """

    modality: str
    feature: tuple[float, ...] | None = None
    confidence: float = 1.0
    time_s: float = 0.0
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.modality:
            raise ValueError("SensorEvidence.modality must be non-empty")
        _check_range("confidence", self.confidence, 0.0, 1.0)


NODE_KINDS = ("self", "person", "object", "place")
RELATIONS = ("near", "visible_from", "at_place", "approaching", "touching", "interacting_with", "seen_with")


@dataclass(frozen=True)
class WorldEntity:
    """A node of the world model.

      node_id    "self", "place:i,j", the persistent entity ID ("person#0") once identity is known, or
                 "track:<track ID>" while it is not (track ID = temporary perceptual continuity;
                 entity ID = persistent remembered identity; the two are kept distinct)
      xy         estimated world position (m), or None
      confidence [0, 1] that the node is currently present where estimated
    """

    node_id: str
    kind: str
    track_id: str = ""
    entity_id: str = ""
    xy: tuple[float, float] | None = None
    confidence: float = 1.0
    time_s: float = 0.0
    track: Track | None = None  # latest measurement (bearing/distance relative to the robot), if perceived

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind {self.kind!r}; known: {NODE_KINDS}")
        _check_range("confidence", self.confidence, 0.0, 1.0)


@dataclass(frozen=True)
class WorldRelation:
    """A typed, time-stamped relation between two nodes, with confidence and provenance."""

    subject: str
    relation: str
    obj: str
    confidence: float = 1.0
    time_s: float = 0.0
    source: str = ""  # which measurement asserted it

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"unknown relation {self.relation!r}; known: {RELATIONS}")
        _check_range("confidence", self.confidence, 0.0, 1.0)


@dataclass(frozen=True)
class WorldModelState:
    """Snapshot of the world model at one instant."""

    time_s: float
    entities: tuple[WorldEntity, ...] = ()
    relations: tuple[WorldRelation, ...] = ()

    def related(self, relation: str, obj: str | None = None) -> tuple[WorldRelation, ...]:
        return tuple(r for r in self.relations if r.relation == relation and (obj is None or r.obj == obj))

    def node(self, node_id: str) -> WorldEntity | None:
        return next((n for n in self.entities if n.node_id == node_id), None)

    def perceived(self) -> tuple[WorldEntity, ...]:
        """Nodes with a current measurement (tracked people and objects)."""
        return tuple(n for n in self.entities if n.track is not None)

    def tracks(self) -> tuple[Track, ...]:
        return tuple(n.track for n in self.perceived())

    def identity_map(self) -> dict[str, str]:
        """Track ID → persistent entity ID, for perceived nodes whose identity is known."""
        return {n.track_id: n.entity_id for n in self.perceived() if n.entity_id}

    def nearest(self, kind: str, relation: str | None = None) -> WorldEntity | None:
        """Nearest perceived node of `kind`, optionally only among nodes with `relation` to self."""
        subjects = None if relation is None else {r.subject for r in self.related(relation, "self")}
        cands = [n for n in self.perceived() if n.kind == kind and (subjects is None or n.node_id in subjects)]
        return min(cands, key=lambda n: n.track.distance, default=None)


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


ADVERSE_OUTCOMES = ("stability_loss", "contact_impact")
BENIGN_OUTCOMES = ("benign_contact",)
OUTCOME_KINDS = ADVERSE_OUTCOMES + BENIGN_OUTCOMES
RISK_KINDS = ("collision_risk",)


@dataclass(frozen=True)
class OutcomeSignal:
    """Something that ACTUALLY happened to the robot, measured: the primary ground truth for memory.

      stability_loss  tilt beyond the near-fall threshold [IMU]
      contact_impact  a hard contact / collision [force or touch sensor; none occurs in the current MuJoCo
                      scenes, whose people have no collision geometry]
      benign_contact  gentle, slow touch [touch sensor; simulated in MuJoCo]
    Predictions of what might happen are NOT outcomes: they are RiskEstimates.
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
class RiskEstimate:
    """An ESTIMATE that something adverse was about to happen, derived from measurements, not an outcome.

      collision_risk  a tracked agent came within reach while closing fast enough that, without stopping,
                      it would have made contact (time to contact from depth/tracking)
    Memory may learn from risk estimates (a near miss is informative), but keeps them separate from actual
    outcomes so the two can be weighted and audited differently.
    """

    kind: str
    magnitude: float = 1.0  # [0, 1]
    time_s: float = 0.0
    source: str = ""
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.kind not in RISK_KINDS:
            raise ValueError(f"unknown risk {self.kind!r}; known: {RISK_KINDS}")
        _check_range("magnitude", self.magnitude, 0.0, 1.0)


@dataclass(frozen=True)
class OutcomeHypothesis:
    """An explicit proposition about what may happen, e.g. "person-3 may collide with the robot".

      kind         proposition ID ("collision", "benign_interaction", "novel_stimulus", ...)
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

      goals       goal kinds this hypothesis bears on (empty = none active)
      persistent  True = re-appraisal of an ongoing, unchanged situation (e.g. "<kind>_in_view" every 2 s);
                  False = a discrete event. Affect models may treat the two differently (Model B v2 does).
    """

    hypothesis: OutcomeHypothesis
    relevance: float = 0.0
    desirability: float = 0.0
    expectedness: float = 1.0
    controllability: float = 1.0
    goals: tuple[str, ...] = ()
    persistent: bool = False

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
    `add` accepts an AppraisalFrame or a legacy AppraisalState; `source` is the track the appraisal is about
    (for a frame, its hypothesis subject is used when `source` is empty). `tendencies` is the total;
    `tendencies_for(source)` only what is directed at that source ("" = undirected, e.g. a near-fall).
    """

    def add(self, appraisal: AppraisalState | AppraisalFrame, source: str = "") -> object: ...

    def step(self, dt: float) -> PADState: ...

    @property
    def pad(self) -> PADState: ...

    @property
    def tendencies(self) -> ActionTendencyState: ...

    def tendencies_for(self, source: str) -> ActionTendencyState: ...


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
class PolicyCapabilities:
    """What a locomotion policy can safely be asked to do, as measured (registry: nerva/sim/capabilities.py).

      style_input        "phase_clock" (upstream: gait-clock style), "vector" (S-policies: StyleVector e),
                         "neutral_only" (B1/B2: trained on e = 0; any other e is out of distribution)
      training_scene     "backlash" or "plain": evaluate in the scene the policy was trained in
      head_pitch_range   (down, up) head_pitch offsets [rad] (positive = face up) tested without falls
      head_yaw_range     (right, left) head_yaw offsets [rad] tested without falls
      walking_head_limit (pitch, yaw, roll) bound while walking, or None
      tested             False = no envelope measured; the values are the legacy defaults
      evidence           where the numbers come from (development log entries)
    """

    name: str
    style_input: str = "phase_clock"
    training_scene: str = "backlash"
    head_pitch_range: tuple[float, float] = (-0.35, 0.6)
    head_yaw_range: tuple[float, float] = (-1.3, 1.3)
    walking_head_limit: tuple[float, float, float] | None = (0.15, 0.08, 0.15)
    tested: bool = False
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.style_input not in ("phase_clock", "vector", "neutral_only"):
            raise ValueError(f"unknown style_input {self.style_input!r}")
        if self.training_scene not in ("backlash", "plain"):
            raise ValueError(f"unknown training_scene {self.training_scene!r}")
        for lo, hi in (self.head_pitch_range, self.head_yaw_range):
            if not lo <= 0.0 <= hi:
                raise ValueError("head ranges must contain 0 (lo <= 0 <= hi)")


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
