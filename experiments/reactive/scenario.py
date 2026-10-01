"""Reactive-behaviour scenario: scripted agents in the scene, the robot's full loop reacting to them.

Loop per 20 ms control step (docs/reactive_behaviour_design.md §2):
  agents move (their scripts may refer to where the robot is NOW, so the scene stays in view)
  every 0.1 s: head camera → SimulatedPerception → events + tracks
               → ContextualAppraiser → CategoricalAffectModel (v0.2, with interest) → PAD + action tendencies
               → UtilityBehaviour (reads tendencies, not emotion labels) → velocity command + style + head gaze
  OpenDuckSim with the S1 policy (unchanged), 500 Hz physics.
Nothing about the robot's reactions is scripted: only the person and the ball follow scripts.

With use_memory=True (docs/memory_design.md, M1) appraisal reads an entity memory: identities come
from the vision's appearance feature, stay bound to a track while appearance is uninformative (up
close), and events with an unknown identity are held back and learned once the identity is resolved.

memory_learning (refactor stage D): "legacy" (default, the recorded baseline) learns entity/place
associations from the elicited emotions; "grounded" learns them only from measured outcomes
(nerva/world/outcomes.py: near_collision, stability_loss, benign_contact), attributed through the same
identity binding. Outcomes are detected in both modes and returned on `sim.outcomes` for analysis.

Refactor stages E-G (defaults reproduce the recorded baselines):
  appraisal_mode  "legacy" (AppraisalState from the v1/v2 appraiser) or "frames" (AppraisalFrame over an
                  explicit outcome hypothesis, relative to the explicit SelfState and GoalState)
  affect_model    "A" (CategoricalAffectModel) or "B" (DimensionalAffectModel, no emotion labels; needs
                  memory_learning="grounded" when memory is used, since legacy memory learns from labels)
Always on: the world model (nerva/world/model.py, returned as sim.world) and the deterministic safety
supervisor (nerva/safety.py) between behaviour and the policy.

profile="v2" (consolidation, 2026-10-02) bundles the new paths:
  - the policy's measured capabilities (nerva/sim/capabilities.py) set the head envelope, walking head
    limit, training scene and whether the style input must stay neutral (no flags needed)
  - grounded memory, appraisal frames
  - target-conditioned arbitration (tendencies split by the track they are about)
  - the world model is updated right after identity binding and is the source that touch attribution,
    appraisal identity and the behaviour's inputs (tracks, remembered threat per entity) query
  - near misses are RiskEstimates, kept apart from actual OutcomeSignals (also in legacy runs' logs)
profile="v2" with affect Model B is the DEFAULT since 2026-10-02 (preregistered consolidation suite,
development log). profile="legacy" (affect A by default) reproduces the pre-refactor behaviour byte-for-byte;
pass it explicitly to reproduce any result recorded before that date.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from nerva.analysis import gait_metrics as gm
from nerva.affect.emotions import CategoricalAffectModel
from nerva.affect.appraisal import ContextualAppraiser, MemoryAppraiser
from nerva.memory.episodic import EpisodicMemory
from nerva.memory.entity import EntityMemory
from nerva.memory.spatial import PlaceMemory
from nerva.world.outcomes import OutcomeMonitor
from nerva.sim.capabilities import capabilities_for
from nerva.affect.frames import FrameAppraiser
from nerva.affect.model_b import DimensionalAffectModel
from nerva.behaviour.goals import active_goals
from nerva.safety import SafetySupervisor
from nerva.world.model import WorldModel
from nerva.world.self_state import estimate_self_state
from nerva.interfaces import ActionTendencyState, Event, RiskEstimate, StyleVector, TENDENCIES
from nerva.sim.open_duck import SCENE, SCENE_BACKLASH, OpenDuckSim
from nerva.perception.tracker import FRAME_HZ, SimulatedPerception
from nerva.perception.vision import VisionPerception
from nerva.behaviour.selection import UtilityBehaviour
from nerva.behaviour.modes import ReactiveBehaviour
from nerva.sim.world import ENTITY_KINDS, World, extend_scene

CTRL_DT = 0.02
PERCEIVE_EVERY = int(round(1 / (FRAME_HZ * CTRL_DT)))  # control steps per perception frame
HIDDEN = (30.0, 30.0)
NEAR_FALL_TILT_DEG = 20.0
EYE_W, EYE_H = 160, 120  # vision resolution
EMOTIONS = ("joy", "hope", "fear", "distress", "surprise", "interest")


@dataclass
class Agent:
    """A scripted mover. Steps: (start_t, action, params).

    Actions:
      hide                          out of the world
      appear   dist, bearing        appear at this distance/bearing relative to the robot's current pose
      approach speed, stop_dist     walk toward the robot's current position, stop at stop_dist
      wait                          stand still, facing the robot
      leave    speed                walk directly away from the robot
      pet      stop_dist            walk up to the robot and stay there, touching it (touch events)
    """

    steps: list[tuple[float, str, tuple]]
    xy: np.ndarray = field(default_factory=lambda: np.array(HIDDEN))
    yaw: float = 0.0
    _i: int = -1

    def update(self, t: float, dt: float, robot_xy: np.ndarray, robot_yaw: float) -> None:
        while self._i + 1 < len(self.steps) and self.steps[self._i + 1][0] <= t:
            self._i += 1
            _, action, p = self.steps[self._i]
            if action == "hide":
                self.xy = np.array(HIDDEN)
            elif action == "appear":
                dist, bearing = p
                a = robot_yaw + bearing
                self.xy = robot_xy + dist * np.array([np.cos(a), np.sin(a)])
        if self._i < 0:
            return
        _, action, p = self.steps[self._i]
        to_robot = robot_xy - self.xy
        d = float(np.linalg.norm(to_robot))
        if action in ("approach", "wait", "appear") and d > 1e-6:
            self.yaw = float(np.arctan2(to_robot[1], to_robot[0]))
        if action == "pet":
            step = min(0.25 * dt, max(0.0, d - p[0]))
            self.xy = self.xy + step * to_robot / max(d, 1e-6)
        if action == "approach":
            speed, stop = p
            step = min(speed * dt, max(0.0, d - stop))
            self.xy = self.xy + step * to_robot / max(d, 1e-6)
        elif action == "leave":
            (speed,) = p
            self.yaw = float(np.arctan2(-to_robot[1], -to_robot[0]))
            self.xy = self.xy - speed * dt * to_robot / max(d, 1e-6)

    @property
    def action(self) -> str | None:
        return self.steps[self._i][1] if self._i >= 0 else None


def default_scenario() -> dict[str, Agent]:
    """~100 s: a ball to be curious about, a friendly person, a lunge, the person leaving and returning."""
    ball = Agent([(0.0, "hide", ()), (4.0, "appear", (1.3, 0.5)), (4.0, "wait", ())])
    person = Agent([
        (0.0, "hide", ()),
        (30.0, "appear", (3.5, 0.2)), (30.0, "approach", (0.25, 1.4)),  # slow, friendly approach
        (43.0, "leave", (0.8,)), (44.8, "wait", ()),  # steps back ...
        (46.5, "approach", (1.8, 0.45)),  # ... then lunges at the robot
        (48.5, "wait", ()),
        (56.0, "leave", (0.5,)),
        (68.0, "hide", ()),
        (72.0, "appear", (3.5, -0.2)), (72.0, "approach", (0.25, 1.2)),  # returns while the lunge is recent
        (90.0, "wait", ()),
    ])
    return {"ball": ball, "person": person}


def memory_scenario() -> dict[str, Agent]:
    """~135 s, two people never in view together: A lunges early; B (green) is new, then pets the robot;
    A returns 70 s after the lunge (beyond v1's 45 s threat timer); then B returns."""
    a = Agent([(0.0, "hide", ()), (5.0, "appear", (3.5, 0.1)), (5.0, "approach", (0.25, 1.4)),
               (18.0, "leave", (0.8,)), (19.8, "wait", ()), (21.0, "approach", (1.8, 0.45)), (23.0, "wait", ()),
               (28.0, "leave", (0.5,)), (40.0, "hide", ()),
               (92.0, "appear", (3.5, -0.1)), (92.0, "approach", (0.25, 1.2)), (106.0, "leave", (0.5,)),
               (116.0, "hide", ())])
    b = Agent([(0.0, "hide", ()), (46.0, "appear", (3.5, 0.15)), (46.0, "approach", (0.25, 0.9)),
               (58.0, "pet", (0.3,)), (68.0, "leave", (0.5,)), (80.0, "hide", ()),
               (118.0, "appear", (3.5, 0.0)), (118.0, "approach", (0.25, 1.0))])
    return {"person": a, "person_b": b}


IDLE_BEFORE_SLEEP_S = 5.0  # no one in view for this long → consolidate ("sleep")
SLEEP_EVERY_S = 20.0
def together_scenario() -> dict[str, Agent]:
    """memory_scenario's history (A lunges, B is new and pets the robot), then at 95 s A and B come back
    TOGETHER from the left and the right and approach slowly: multi-person tracking + person-specific memory."""
    agents = memory_scenario()
    a, b = agents["person"], agents["person_b"]
    a.steps = [st for st in a.steps if st[0] < 92.0] + [(95.0, "appear", (3.0, 0.6)), (95.0, "approach", (0.25, 1.3))]
    b.steps = [st for st in b.steps if st[0] < 118.0] + [(95.0, "appear", (3.0, -0.6)), (95.0, "approach", (0.25, 0.9))]
    return agents


DEFERRED_CONFIDENCE = 0.8  # learning applied later, once an unknown identity is resolved
TOUCH_REACH = 0.5  # m
TOUCH_PERIOD_S = 1.0


class IdentityBinder:
    """Which remembered entity each perceived track is (docs/memory_design.md section 3.7)."""

    def __init__(self, memory: EntityMemory):
        self.memory = memory
        self.current: dict = {}
        self.pending: dict[str, list] = {}
        self.pending_outcomes: dict[str, list] = {}

    def update(self, t: float, tracks, perception) -> dict:
        """Track ID -> entity record for every track whose identity is known."""
        present = {tr.tid for tr in tracks}
        for tid in list(self.current):
            if tid not in present:  # track lost: continuity broken, identity must be re-established
                self.current.pop(tid)
        for pend in (self.pending, self.pending_outcomes):
            for tid in list(pend):
                if tid not in present:
                    pend.pop(tid)  # never resolved: dropped, no phantom certainty about who it was
        for tr in tracks:
            st = perception.tracks.get(tr.tid)
            appearance = getattr(st, "appearance", None) if tr.visible else None
            if appearance is None:
                continue  # up close / occluded: keep the track's identity (continuity)
            rec = self.memory.resolve(tr.kind, appearance, t)
            if rec is None:
                continue  # ambiguous appearance: keep whatever identity the track already has
            if self.current.get(tr.tid) is not rec:
                self.current[tr.tid] = rec
                for args in self.pending.pop(tr.tid, []):  # "oh, it was you"
                    self.memory.learn(rec, *args, confidence=DEFERRED_CONFIDENCE)
                for item in self.pending_outcomes.pop(tr.tid, []):
                    learn = self.memory.learn_risk if isinstance(item, RiskEstimate) else self.memory.learn_outcome
                    learn(rec, item, confidence=DEFERRED_CONFIDENCE)
        return self.current

    def learn(self, tid: str, t: float, event: str, emotions, arousal: float, surprise_negative: bool) -> None:
        rec = self.current.get(tid)
        if rec is None:
            self.pending.setdefault(tid, []).append((t, event, emotions, arousal, surprise_negative))
        else:
            self.memory.learn(rec, t, event, emotions, arousal, surprise_negative)

    def learn_outcome(self, tid: str, outcome) -> None:
        """Credit a measured outcome (or a RiskEstimate) to the entity of track `tid`, or hold it until the
        identity is known."""
        rec = self.current.get(tid)
        if rec is None:
            self.pending_outcomes.setdefault(tid, []).append(outcome)
        elif isinstance(outcome, RiskEstimate):
            self.memory.learn_risk(rec, outcome)
        else:
            self.memory.learn_outcome(rec, outcome)


def touch_source_from_world(state) -> str:
    """Who is touching the robot, asked of the world model: the nearest person `near` the robot, else the
    nearest perceived person (v2; same rule as touch_source, but from the world model)."""
    node = state.nearest("person", "near") or state.nearest("person")
    return node.track_id if node is not None else ""


def touch_source(tracks) -> str:
    """Who is touching the robot: the nearest person track within reach, else the nearest person seen."""
    people = [tr for tr in tracks if tr.kind == "person"]
    if not people:
        return ""
    return min(people, key=lambda tr: tr.distance).tid


def neutralised(command):
    """For policies trained on the neutral style only (B1/B2): keep WHAT, drop the style input."""
    import dataclasses

    return dataclasses.replace(command, style_vector=StyleVector())


def run(policy: str, seed: int = 0, duration: float = 100.0, agents: dict[str, Agent] | None = None,
        record_every: int | None = None, head_moves_while_walking: bool = False, selector: str = "utility",
        walking_head_limit=None, perception_mode: str = "simulated", use_memory: bool = False,
        use_spatial: bool | None = None, backlash_scene: bool = False, neutral_style: bool = False,
        memory_learning: str = "legacy", appraisal_mode: str = "legacy", affect_model: str | None = None,
        safety_supervisor: bool = True, head_pitch_down: float | None = None, head_yaw_max: float | None = None,
        profile: str = "v2"):
    """selector: "utility" (behaviour v2, emotion-modulated action selection) or "rules" (v1).
    perception_mode: "simulated" (ground-truth positions + noise) or "vision" (colour + depth images
    from the robot's head camera, nerva.perception.vision)."""
    """Simulate the closed loop. Returns (sim, rows, frames, fired); frames = qpos + mocap snapshots."""
    if profile not in ("legacy", "v2"):
        raise ValueError("profile must be 'legacy' or 'v2'")
    v2 = profile == "v2"
    affect_model = affect_model or ("B" if v2 else "A")  # defaults decided 2026-10-02 (development log)
    capabilities = capabilities_for(policy) if v2 else None
    if v2:
        backlash_scene = capabilities.training_scene == "backlash"
        neutral_style = capabilities.style_input == "neutral_only"
        appraisal_mode, memory_learning = "frames", "grounded"
    if affect_model == "B" and use_memory and memory_learning != "grounded":
        raise ValueError("Model B has no emotion labels; legacy memory learns from labels: use memory_learning='grounded'")
    agents = agents or default_scenario()
    sim = OpenDuckSim(raw_accel=True, obs_noise=True, init_joint_noise=0.02, seed=seed, policy_path=policy,
                      scene_extender=extend_scene, scene=SCENE_BACKLASH if backlash_scene else SCENE)
    world = World(sim.model)
    for name in ENTITY_KINDS:  # entities this scenario does not use are removed from the scene
        if name not in agents:
            world.place(sim.data, name, HIDDEN, 0.0)
    vision = perception_mode == "vision"
    perception = VisionPerception(seed=seed) if vision else SimulatedPerception(seed=seed)
    eye = mujoco.Renderer(sim.model, EYE_H, EYE_W) if vision else None
    grounded = memory_learning == "grounded"
    memory = EntityMemory(learning=memory_learning) if use_memory else None
    binder = IdentityBinder(memory) if use_memory else None
    appraiser = MemoryAppraiser(memory) if use_memory else ContextualAppraiser()
    frames_mode = appraisal_mode == "frames"
    if frames_mode:
        appraiser = FrameAppraiser(appraiser)
    world_model, safety = WorldModel(), SafetySupervisor()
    self_state, goals = None, None
    episodic = EpisodicMemory() if use_memory else None
    places = PlaceMemory(learning=memory_learning) if (use_memory if use_spatial is None else use_spatial) else None
    monitor, outcomes_log, risks_log, memory_trace = OutcomeMonitor(), [], [], []
    world_state = None
    last_seen_anything, last_sleep, sleep_log = 0.0, -1e9, []
    last_touch = -1e9
    affect = {"A": CategoricalAffectModel, "B": DimensionalAffectModel}[affect_model]()
    behaviour_cls = {"utility": UtilityBehaviour, "rules": ReactiveBehaviour}[selector]
    if head_moves_while_walking:
        behaviour = behaviour_cls(walking_head_limit=None)
    elif walking_head_limit is not None:
        behaviour = behaviour_cls(walking_head_limit=walking_head_limit)
    else:
        behaviour = behaviour_cls()
    if capabilities is not None:
        behaviour.apply_capabilities(capabilities)
    if head_pitch_down is not None:  # diagnostic overrides; normal runs use the capabilities
        behaviour.head_pitch_down = head_pitch_down
    if head_yaw_max is not None:
        behaviour.head_yaw_max, behaviour.head_yaw_min = head_yaw_max, -head_yaw_max
    cam = sim.model.camera("robot_eye").id
    decision = behaviour.step(0.0, 0.1, affect.pad, ActionTendencyState(), ())
    sim.set_behaviour(decision.command)
    rows, frames, fired = [], [], []
    tracks = ()
    near_fall_armed = True
    d = sim.data
    for k in range(int(round(duration / CTRL_DT))):
        t = k * CTRL_DT
        w, x, y, z = d.qpos[3:7]
        robot_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        robot_xy = d.qpos[0:2].copy()
        for name, agent in agents.items():
            agent.update(t, CTRL_DT, robot_xy, robot_yaw)
            world.place(d, name, agent.xy, agent.yaw)
        if k % PERCEIVE_EVERY == 0:
            mujoco.mj_kinematics(sim.model, d)
            entities = {n: world.entity_position(d, n) for n in agents}
            if vision:
                mujoco.mj_forward(sim.model, d)
                eye.update_scene(d, camera=cam)
                rgb = eye.render()
                eye.enable_depth_rendering()
                eye.update_scene(d, camera=cam)
                depth = eye.render()
                eye.disable_depth_rendering()
                pstate = perception.detect_frame(t, rgb, depth, d.cam_xpos[cam].copy(),
                                                 d.cam_xmat[cam].reshape(3, 3).copy(), float(sim.model.cam_fovy[cam]),
                                                 robot_yaw, 1.0 / FRAME_HZ, ego_velocity=d.qvel[0:2].copy())
            else:
                pstate = perception.detect(t, d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy(),
                                           robot_yaw, entities, 1.0 / FRAME_HZ, ego_velocity=d.qvel[0:2].copy())
            tracks = pstate.tracks
            events = list(pstate.events)
            if binder is not None:
                appraiser.identity = binder.update(t, tracks, perception)
            if v2:  # the world model holds the current situation; the rest of the frame queries it
                world_state = world_model.update(t, robot_xy, robot_yaw, tracks,
                                                 {tid: rec.eid for tid, rec in appraiser.identity.items()}
                                                 if use_memory else None)
                if use_memory:
                    appraiser.identity = {tid: memory.records[eid] for tid, eid in world_state.identity_map().items()}
                tracks = world_state.tracks()
            risks = monitor.proximity(t, tracks)
            outcomes = []
            self_state = estimate_self_state(t, d.qpos, d.qvel, behaviour.mode)
            goals = active_goals(behaviour.mode, behaviour.target)
            for name, agent in agents.items():  # simulated touch sensing: gentle contact while petted
                if (agent.action == "pet" and np.linalg.norm(agent.xy - robot_xy) < TOUCH_REACH
                        and t - last_touch >= TOUCH_PERIOD_S):
                    who_touches = touch_source_from_world(world_state) if v2 else touch_source(tracks)
                    events.append(Event("touch_gentle", source=who_touches))
                    outcomes.append(monitor.contact(t, who_touches))
                    if v2 and who_touches:
                        world_state = world_model.assert_touch(t, who_touches)
                    last_touch = t
            tilt = float(gm.tilt_deg(d.qpos[3:7][None])[0])
            if near_fall_armed and tilt > NEAR_FALL_TILT_DEG:
                events.append(Event("near_fall"))
                near_fall_armed = False
            elif tilt < NEAR_FALL_TILT_DEG / 2:
                near_fall_armed = True
            outcomes += monitor.stability(t, tilt)
            for oc in outcomes + risks:  # grounded learning: what happened (outcomes) and what nearly did (risks)
                is_risk = isinstance(oc, RiskEstimate)
                (risks_log if is_risk else outcomes_log).append(oc)
                if binder is not None and oc.source:
                    binder.learn_outcome(oc.source, oc)
                if places is not None:
                    (places.learn_risk if is_risk else places.learn_outcome)(robot_xy, t, oc)
                if episodic is not None and grounded:
                    who_oc = appraiser.identity.get(oc.source) if oc.source else None
                    encode = episodic.encode_risk if is_risk else episodic.encode_outcome
                    encode(t, oc, who_oc.eid if who_oc else None, robot_xy, affect.pad.arousal)
            for ev in events:
                behaviour.notice(t, ev.kind, ev.source)
                a = (appraiser.appraise(ev, t, tracks, self_state, goals) if frames_mode
                     else appraiser.appraise(ev, t, tracks))
                if a is not None:
                    new = affect.add(a, source=ev.source)
                    elicited = [(e.label, e.intensity) for e in new] if affect_model == "A" else []
                    fired.append((t, ev.kind, a, elicited))
                    about = ev.source or None  # the track this event is about
                    if binder is not None and about and not ev.kind.endswith("_lost"):
                        binder.learn(about, t, ev.kind, elicited, affect.pad.arousal,
                                     a.desirability < 0 and any(lbl == "surprise" for lbl, _ in elicited))
                    if places is not None:
                        places.learn(robot_xy, t, elicited, affect.pad.arousal)
                    if episodic is not None:
                        who_ev = appraiser.identity.get(about) if (about and use_memory) else None
                        episodic.encode(t, ev.kind, who_ev.eid if who_ev else None, robot_xy, elicited,
                                        a.relevance, affect.pad.arousal, 1.0 - a.expectedness,
                                        appraiser.novelty(about) if about else 0.0)
            observed = (appraiser.observe(t, tracks, 1.0 / FRAME_HZ, self_state, goals) if frames_mode
                        else appraiser.observe(t, tracks, 1.0 / FRAME_HZ))
            for kind, a in observed:
                affect.add(a)
            if episodic is not None:  # consolidation ("sleep") when nothing has been in view for a while
                if any(tr.visible for tr in tracks):
                    last_seen_anything = t
                elif t - last_seen_anything > IDLE_BEFORE_SLEEP_S and t - last_sleep > SLEEP_EVERY_S:
                    sleep_log.append((round(t, 1), episodic.consolidate(t, memory)))
                    last_sleep = t
            if places is not None:
                places.observe(robot_xy, t, 1.0 / FRAME_HZ)
                behaviour.explore_bearing = places.explore_heading(robot_xy, robot_yaw)[0]
            pad = affect.step(1.0 / FRAME_HZ)
            if memory is not None and k % (PERCEIVE_EVERY * 10) == 0:  # once per second, for analysis
                memory_trace.append((round(t, 1), {r.eid: (r.threat, r.warmth, r.adverse, r.benign, r.risk)
                                                   for r in memory.records.values() if r.kind == "person"}))
            salience = {tr.tid: appraiser.novelty(tr.tid if use_memory else tr.kind) for tr in tracks}
            threats = ({tid: rec.threat for tid, rec in appraiser.identity.items() if rec.kind == "person"}
                       if use_memory else None)
            directed = None
            if v2:  # tendencies split by the track they are about ("" = undirected)
                about = {""} | {tr.tid for tr in tracks} | ({behaviour.target} if behaviour.target else set())
                directed = {src: affect.tendencies_for(src) for src in about}
                if use_memory:  # remembered threat per perceived entity, asked of the world model
                    threats = {n.track_id: memory.records[n.entity_id].threat for n in world_state.perceived()
                               if n.kind == "person" and n.entity_id in memory.records}
            decision = behaviour.step(t, 1.0 / FRAME_HZ, pad, affect.tendencies, tracks, salience, threats, directed)
            command, head, safety_reason = (safety.filter(decision.command, decision.head, self_state)
                                            if safety_supervisor else (decision.command, decision.head, ""))
            sim.set_behaviour(neutralised(command) if neutral_style else command)
            sim.set_head_offset(*head)
            if v2:
                if decision.mode in ("approach", "inspect") and decision.target:
                    world_model.assert_interacting(t, decision.target)
            else:
                touching = next((ev.source for ev in events if ev.kind == "touch_gentle"), "")
                world_model.update(t, robot_xy, robot_yaw, tracks,
                                   {tid: rec.eid for tid, rec in appraiser.identity.items()} if use_memory else None,
                                   touching=touching,
                                   interacting=(decision.target or "") if decision.mode in ("approach", "inspect") else "")
        sim.step_physics(10)
        pad = affect.pad
        tilt = float(gm.tilt_deg(d.qpos[3:7][None])[0])
        person_xy = agents["person"].xy if "person" in agents else np.array(HIDDEN)
        person_b_xy = agents["person_b"].xy if "person_b" in agents else np.array(HIDDEN)
        people = [tr for tr in tracks if tr.kind == "person"]
        nearest = min(people, key=lambda tr: tr.distance).tid if people else ""
        who = appraiser.identity.get(nearest) if use_memory else None
        rows.append({
            "t": round(t + CTRL_DT, 3), "valence": pad.valence, "arousal": pad.arousal, "dominance": pad.dominance,
            **{lbl: sum(e.intensity for e in getattr(affect, "emotions", []) if e.label == lbl)
               for lbl in EMOTIONS},  # logging only (Model A); zero for Model B
            **{f"tend_{k}": getattr(affect.tendencies, k) for k in TENDENCIES},
            "mode": decision.mode, "reason": safety_reason or decision.reason, "target": decision.target or "",
            "cmd_vx": decision.command.vx, "cmd_yaw": decision.command.yaw_rate,
            "e_tempo": decision.command.style_vector.tempo, "e_torso_pitch": decision.command.style_vector.torso_pitch,
            "head_pitch": decision.head[1], "head_yaw": decision.head[2], "head_roll": decision.head[3],
            "robot_x": float(d.qpos[0]), "robot_y": float(d.qpos[1]),
            "person_x": float(person_xy[0]), "person_y": float(person_xy[1]),
            "dist_person": float(np.linalg.norm(person_xy - d.qpos[0:2])),
            "dist_person_b": float(np.linalg.norm(person_b_xy - d.qpos[0:2])),
            "identity": who.eid if who is not None else "",
            "mem_threat": who.threat if who is not None else float("nan"),
            "mem_warmth": who.warmth if who is not None else float("nan"),
            "dist_ball": float(np.linalg.norm(agents["ball"].xy - d.qpos[0:2])) if "ball" in agents else np.nan,
            "sees_person": any(tr.kind == "person" and tr.visible for tr in tracks),
            "sees_ball": any(tr.kind == "ball" and tr.visible for tr in tracks),
            "tilt_deg": tilt,
        })
        if record_every and k % record_every == 0:
            boxes = None
            if vision:  # what the vision actually found: (kind, row0, col0, row1, col1) in EYE_H x EYE_W pixels
                boxes = [(bl.kind, *bl.pixels.min(0), *bl.pixels.max(0)) for bl in perception.last_blobs]
            frames.append((d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(), tracks, boxes))
    if use_memory:
        sim.memory, sim.episodic, sim.sleep_log = memory, episodic, sleep_log  # for inspection by callers
        sim.memory_trace = memory_trace
    sim.outcomes, sim.risks, sim.capabilities = outcomes_log, risks_log, capabilities
    sim.world, sim.safety_interventions = world_model, safety.interventions
    sim.places = places
    return sim, rows, frames, fired
