"""Image-based perception: find the person and the ball in the robot's own camera images.

Replaces the ground-truth detector of nerva.perception.SimulatedPerception with real (if simple)
vision on the rendered head-camera frames (docs/reactive_behaviour_design.md; stage "real vision"):

  RGB frame  → colour segmentation (HSV)  → connected components → largest blob per class
  depth frame → median depth over the blob → 3-D point in the camera frame → world → bearing/distance
  appearance → normalised hue histogram of the blob's strongly coloured pixels (clothing), a stand-in
               for a face/body re-ID embedding; None when too little of it is visible (e.g. up close
               only the trousers, which everyone shares), so identity must come from track continuity

The detector knows only colours, not which geometry is which: a person is "a blob of blue-ish
clothing/dark-blue trousers", the ball "an orange blob". That is honest colour-based vision on simple
shapes; realistic humans and learned detectors are the Isaac Sim stage. Thresholds are NERVA
engineering choices tuned on the MuJoCo renders of nerva.world.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from nerva.interfaces import PerceptionState
from nerva.perception import RANGE_M, Observation, SimulatedPerception, _wrap

MIN_BLOB_PX = 12  # at 160x120: rejects grid lines and noise
HUE_BINS = 8
APPEARANCE_MIN_SAT = 0.35  # clothing is strongly coloured; trousers, skin tones and shadows are not
APPEARANCE_MIN_PX = 10
MIN_DEPTH_M = {"ball": 0.2}  # closer than this is the robot's own body (camera is 0.09 m ahead of the head)
MERGE_DEPTH_M = 0.4  # components closer than this in depth may be one person
MERGE_GAP_PX = 12.0  # at 1 m; scaled by 1/depth


@dataclass(frozen=True)
class Blob:
    kind: str
    pixels: np.ndarray  # (N, 2) rows, cols
    depth_m: float  # median camera-axis depth
    centroid: tuple[float, float]  # (row, col)
    top_row: int
    appearance: np.ndarray | None  # (HUE_BINS,) normalised hue histogram, or None if not informative


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """uint8 or float RGB (H, W, 3) → HSV with hue in degrees [0, 360), s and v in [0, 1]."""
    x = rgb.astype(np.float32) / (255.0 if rgb.dtype == np.uint8 else 1.0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(-1)
    c = v - x.min(-1)
    s = np.where(v > 0, c / np.maximum(v, 1e-6), 0.0)
    h = np.zeros_like(v)
    nz = c > 1e-6
    rm, gm = nz & (v == r), nz & (v == g) & (v != r)
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / c[rm]) % 6
    h[gm] = (b - r)[gm] / c[gm] + 2
    h[bm] = (r - g)[bm] / c[bm] + 4
    return np.stack([h * 60.0, s, v], -1)


def masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Per-class pixel masks from colour alone."""
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ball = (h > 10) & (h < 40) & (s > 0.6) & (v > 0.4)
    # clothing (blue or green) and dark-blue trousers; grey floor/sky (s≈0), skin and the orange ball excluded
    person = (h > 80) & (h < 260) & (s > 0.12) & (v > 0.08)
    return {"ball": ball, "person": person}


def _merge_person_components(labels: np.ndarray, ids: list[int], depth: np.ndarray) -> list[np.ndarray]:
    """Group person-coloured components that belong to one person (e.g. two legs seen up close):
    similar median depth and overlapping or nearby column ranges."""
    comps = []
    for i in ids:
        rows, cols = np.nonzero(labels == i)
        d = depth[rows, cols]
        d = d[np.isfinite(d) & (d > 0.02)]
        if d.size:
            comps.append([rows, cols, float(np.median(d)), cols.min(), cols.max()])
    groups: list[list] = []
    for c in sorted(comps, key=lambda c: c[3]):
        for g in groups:
            gap_px = max(0, c[3] - g[4])
            if abs(c[2] - g[2]) < MERGE_DEPTH_M and gap_px < MERGE_GAP_PX * (1.0 / max(g[2], 0.3)):
                g[0] = np.concatenate([g[0], c[0]])
                g[1] = np.concatenate([g[1], c[1]])
                g[4] = max(g[4], c[4])
                break
        else:
            groups.append(list(c))
    return [np.stack([g[0], g[1]], 1) for g in groups]


def find_blobs(rgb: np.ndarray, depth: np.ndarray) -> list[Blob]:
    """All person blobs (merged per person) and the largest ball blob."""
    blobs = []
    hsv = rgb_to_hsv(rgb)
    for kind, mask in masks(rgb).items():
        labels, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        big = [i + 1 for i in range(n) if sizes[i] >= MIN_BLOB_PX]
        if not big:
            continue
        if kind == "person":
            pixel_sets = _merge_person_components(labels, big, depth)
        else:
            best = max(big, key=lambda i: sizes[i - 1])
            pixel_sets = [np.stack(np.nonzero(labels == best), 1)]
        for px in pixel_sets:
            rows, cols = px[:, 0], px[:, 1]
            d = depth[rows, cols]
            d = d[np.isfinite(d) & (d > 0.02)]
            if d.size == 0 or (kind in MIN_DEPTH_M and np.median(d) < MIN_DEPTH_M[kind]):
                continue  # e.g. the robot's own orange feet seen when looking down are not a ball
            strong = hsv[rows, cols, 1] > APPEARANCE_MIN_SAT
            appearance = None
            if strong.sum() >= APPEARANCE_MIN_PX:
                hist, _ = np.histogram(hsv[rows, cols, 0][strong], bins=HUE_BINS, range=(0, 360))
                appearance = hist / hist.sum()
            blobs.append(Blob(kind, px, float(np.median(d)), (float(rows.mean()), float(cols.mean())),
                              int(rows.min()), appearance))
    return blobs


def pixel_ray(row: float, col: float, height: int, width: int, fovy_deg: float) -> np.ndarray:
    """Unit-depth ray in MuJoCo camera coordinates (x right, y up, looking along -z)."""
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
    return np.array([(col + 0.5 - width / 2) / f, (height / 2 - row - 0.5) / f, -1.0])


class VisionPerception(SimulatedPerception):
    """Same tracker, events and memory as SimulatedPerception; detections come from the images."""

    def detect_frame(self, t: float, rgb: np.ndarray, depth: np.ndarray, cam_pos: np.ndarray,
                     cam_xmat: np.ndarray, fovy_deg: float, body_yaw: float, dt: float,
                     ego_velocity: np.ndarray | None = None) -> PerceptionState:
        ego = np.zeros(2) if ego_velocity is None else np.asarray(ego_velocity, dtype=float)[:2]
        height, width = depth.shape
        self.last_blobs = find_blobs(rgb, depth)
        observations = []
        for blob in self.last_blobs:
            ray = pixel_ray(*blob.centroid, height, width, fovy_deg)
            point = cam_pos + cam_xmat @ (ray * blob.depth_m)  # depth is along the optical axis
            rel = point - cam_pos
            h_dist = float(np.hypot(rel[0], rel[1]))
            if h_dist > RANGE_M[blob.kind]:
                continue
            bearing = _wrap(np.arctan2(rel[1], rel[0]) - body_yaw)
            top = pixel_ray(blob.top_row, blob.centroid[1], height, width, fovy_deg)  # person: head end
            aim = top if blob.kind == "person" else ray
            elevation = float(np.arctan2(aim[1], np.hypot(aim[0], 1.0)))
            toward = rel[:2] / max(h_dist, 1e-6)
            observations.append(Observation(blob.kind, bearing, h_dist, elevation, (float(point[0]), float(point[1])),
                                            float(ego @ toward), blob.appearance))
        return self.ingest(t, observations, dt)
