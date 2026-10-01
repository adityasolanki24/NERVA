"""Policy capability registry: what each locomotion policy can safely be asked to do, as measured.

Behaviour and the simulation read a policy's `PolicyCapabilities` (nerva/interfaces.py) instead of
global constants or command-line flags: head-offset envelope, walking head limit, whether it takes a style
vector, and the scene it was trained in. Every number below cites its evidence in
docs/development_log.md. A policy without a measured envelope gets LEGACY (the constants the behaviour
used before 2026-10-02, `tested=False`).
"""

from __future__ import annotations

from pathlib import Path

from nerva.interfaces import PolicyCapabilities

S1_WALK = (0.15, 0.08, 0.15)  # S1 stops walking with larger head offsets (2026-09-30)
S3_WALK = (0.2, 0.3, 0.3)  # S3 trained with head commands applied (2026-10-01)

LEGACY = PolicyCapabilities("legacy", style_input="vector", training_scene="backlash",
                            head_pitch_range=(-0.35, 0.6), head_yaw_range=(-1.3, 1.3), walking_head_limit=S1_WALK,
                            tested=False, evidence="pre-2026-10-02 behaviour constants; no envelope measured")

CAPABILITIES: dict[str, PolicyCapabilities] = {
    "upstream": PolicyCapabilities(
        "upstream", style_input="phase_clock", training_scene="backlash", walking_head_limit=None, tested=False,
        evidence="Open Duck BEST_WALK_ONNX_2; head offsets slow walking (RQ1c, 2026-09-28); no standing envelope"),
    "S1": PolicyCapabilities(
        "S1", style_input="vector", training_scene="backlash", walking_head_limit=S1_WALK, tested=False,
        evidence="S1 walking head limit 2026-09-30; standing envelope not measured"),
    "S3": PolicyCapabilities(
        "S3", style_input="vector", training_scene="backlash", walking_head_limit=S3_WALK, tested=False,
        evidence="S3 walking head tolerance 2026-10-01; standing envelope not measured"),
    "B2": PolicyCapabilities(
        "B2", style_input="neutral_only", training_scene="backlash", head_pitch_range=(-0.2, 0.6),
        head_yaw_range=(-0.4, 0.4), walking_head_limit=S1_WALK, tested=True,
        evidence="2026-10-02: standing 15 s, 4 seeds: yaw -0.8 fell 4/4, +-0.4 fell 0/4 at pitch 0 and -0.2; "
                 "pitch -0.35 with yaw 0.4 fell (8 s test); up to +0.6 used without falls in F2 (21 runs). "
                 "+0.8 yaw also held standing but is not used (asymmetric, unexplained)"),
}

RUN_PREFIXES = {"b2_neutral": "B2", "s1_pilot": "S1", "s3_pilot": "S3", "BEST_WALK": "upstream"}


def capabilities_for(policy_path: str) -> PolicyCapabilities:
    """Capabilities of a policy file, identified by its cloud-run folder or file name; LEGACY if unknown."""
    parts = Path(policy_path).parts
    for part in reversed(parts):
        for prefix, name in RUN_PREFIXES.items():
            if part.startswith(prefix):
                return CAPABILITIES[name]
    return LEGACY
