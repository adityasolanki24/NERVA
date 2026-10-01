# Experiments

Each folder is one study: a runnable script, its README, and (small) committed results. Large outputs
(videos, cloud runs) are not committed.

| Folder | What it studies | Status |
|---|---|---|
| `reactive/` | The full loop in a scene with people and a ball: perception → appraisal → emotions → memory → behaviour; evaluations with and without memory | **current** |
| `style_policy/` | Does the style-conditioned policy express each style? (S1–S3 evaluations) | done |
| `expressive_locomotion/` | RQ1/RQ1b/RQ1c: style on the pretrained policy (gait clock, speed matching, head posture) | done |
| `affect_prototype/` | Affect model on a scripted event timeline (PAD plot) | done |
| `demo_video/` | First closed-loop demos (gait clock; S1 style vector); superseded by `reactive/` | legacy |

Heavy runs (training, rendering) go to the cloud: `cloud/jobs/`.
