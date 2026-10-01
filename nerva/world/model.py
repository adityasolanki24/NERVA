"""Compact world model / scene graph (refactor stage F).

A small dynamic graph of what the robot currently believes is around it, sized for embedded hardware.
It does not replace memory: persistent identity and associations stay in `nerva.memory`; this holds the
*current* situation, with confidence, time and provenance on every relation.

Nodes (`WorldEntity`): "self"; one node per tracked person/object, named by its persistent entity ID once
identity is known ("person#0") and "track:<tid>" until then (track ID ≠ entity ID); places "place:i,j" on
the same grid as place memory.

Relations (`WorldRelation`), only those the current experiments need:
  visible_from(x, self)       x detected in the current camera frame               [vision]
  near(x, self)               distance < NEAR_M                                     [vision depth]
  approaching(x, self)        ego-motion-corrected closing speed > APPROACH_MS      [tracker]
  at_place(x, place)          from the robot pose and x's bearing/distance          [pose + vision]
  touching(x, self)           a touch attributed to x                               [touch sensor]
  interacting_with(self, x)   the robot is approaching/inspecting x                 [behaviour]
  seen_with(x, y)             two people visible at the same time                   [vision]
A relation's confidence decays linearly to 0 over TTL_S[relation] after it was last asserted, then it is
dropped. When a track's identity becomes known, its node and relations are renamed to the entity ID.
Track nodes no longer perceived fade over NODE_TTL_S. All constants are NERVA design choices.
"""

from __future__ import annotations

import math

from nerva.interfaces import Track, WorldEntity, WorldModelState, WorldRelation

NEAR_M = 1.0
APPROACH_MS = 0.1
CELL_M = 0.75
NODE_TTL_S = 3.0
TTL_S = {"visible_from": 0.5, "near": 1.0, "approaching": 0.5, "at_place": 2.0, "touching": 1.5,
         "interacting_with": 1.0, "seen_with": 10.0}
KIND = {"person": "person"}  # every other tracked kind is an "object"


class WorldModel:
    def __init__(self, cell: float = CELL_M):
        self.cell = cell
        self.nodes: dict[str, WorldEntity] = {}
        self._rel: dict[tuple[str, str, str], WorldRelation] = {}
        self._track_node: dict[str, str] = {}

    def place_id(self, xy) -> str:
        return f"place:{math.floor(xy[0] / self.cell)},{math.floor(xy[1] / self.cell)}"

    def _assert(self, t: float, subj: str, rel: str, obj: str, source: str, confidence: float = 1.0) -> None:
        self._rel[(subj, rel, obj)] = WorldRelation(subj, rel, obj, confidence, t, source)

    def _rename(self, old: str, new: str) -> None:
        """A track's identity became known: its node and relations now use the entity ID."""
        if old in self.nodes:
            self.nodes.pop(old)
        for key in [k for k in self._rel if old in (k[0], k[2])]:
            r = self._rel.pop(key)
            s, o = (new if r.subject == old else r.subject), (new if r.obj == old else r.obj)
            self._rel[(s, r.relation, o)] = WorldRelation(s, r.relation, o, r.confidence, r.time_s, r.source)

    def update(self, t: float, robot_xy, robot_yaw: float, tracks: tuple[Track, ...],
               identities: dict[str, str] | None = None, touching: str = "", interacting: str = "") -> WorldModelState:
        """identities: track ID → persistent entity ID (only for tracks whose identity is known).
        touching / interacting: track IDs ("" = none)."""
        identities = identities or {}
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        self.nodes["self"] = WorldEntity("self", "self", xy=(rx, ry), time_s=t)
        here = self.place_id((rx, ry))
        self.nodes[here] = WorldEntity(here, "place", xy=((math.floor(rx / self.cell) + 0.5) * self.cell,
                                                          (math.floor(ry / self.cell) + 0.5) * self.cell), time_s=t)
        self._assert(t, "self", "at_place", here, "pose")
        visible_people = []
        for tr in tracks:
            tid = tr.tid or tr.kind
            eid = identities.get(tid, "")
            node = eid or f"track:{tid}"
            old = self._track_node.get(tid)
            if old is not None and old != node:
                self._rename(old, node)
            self._track_node[tid] = node
            a = robot_yaw + tr.bearing
            xy = (rx + tr.distance * math.cos(a), ry + tr.distance * math.sin(a))
            conf = 1.0 if tr.visible else max(0.0, 1.0 - tr.unseen_for_s / NODE_TTL_S)
            self.nodes[node] = WorldEntity(node, KIND.get(tr.kind, "object"), track_id=tid, entity_id=eid, xy=xy,
                                           confidence=conf, time_s=t)
            if tr.visible:
                self._assert(t, node, "visible_from", "self", "vision")
                if tr.kind == "person":
                    visible_people.append(node)
            if tr.distance < NEAR_M:
                self._assert(t, node, "near", "self", "vision depth")
            if tr.approach_speed > APPROACH_MS:
                self._assert(t, node, "approaching", "self", "tracker")
            place = self.place_id(xy)
            self.nodes.setdefault(place, WorldEntity(place, "place", time_s=t))
            self._assert(t, node, "at_place", place, "pose + vision", conf)
        for i, a_node in enumerate(visible_people):
            for b_node in visible_people[i + 1:]:
                self._assert(t, a_node, "seen_with", b_node, "vision")
        if touching:
            self._assert(t, self._track_node.get(touching, f"track:{touching}"), "touching", "self", "touch sensor")
        if interacting:
            self._assert(t, "self", "interacting_with", self._track_node.get(interacting, f"track:{interacting}"),
                         "behaviour")
        present = {self._track_node[tr.tid or tr.kind] for tr in tracks}
        for nid, n in list(self.nodes.items()):
            if n.kind in ("person", "object") and nid not in present and t - n.time_s > NODE_TTL_S:
                self.nodes.pop(nid)
        return self.state(t)

    def state(self, t: float) -> WorldModelState:
        rels = []
        for key, r in list(self._rel.items()):
            conf = r.confidence * max(0.0, 1.0 - (t - r.time_s) / TTL_S[r.relation])
            if conf <= 0.0:
                self._rel.pop(key)
                continue
            rels.append(WorldRelation(r.subject, r.relation, r.obj, conf, r.time_s, r.source))
        return WorldModelState(t, tuple(self.nodes.values()), tuple(rels))
