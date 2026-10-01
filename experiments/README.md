# Experiments

Each folder is one study: a runnable script, its README, and (small) committed results. Large outputs
(videos, cloud runs) are not committed.

| Folder | What it studies | Status |
|---|---|---|
| `reactive/` | The full loop in a scene with people and a ball: perception → appraisal → emotions → memory → behaviour; evaluations with and without memory | **current** |
| `affect_models/` | RQ8: affect Model A vs Model B on a fixed appraisal trace (scenario comparisons use `reactive/`) | current |
| `event_learning/` | RQ9: learned event prototypes vs hand-coded events, predicting measured outcomes | current |
| `isaac/` | Isaac Sim kinematic replay of MuJoCo runs (rendering only) | current |
| `style_policy/` | Does the style-conditioned policy express each style? (S1–S3 evaluations) | done |
| `expressive_locomotion/` | RQ1/RQ1b/RQ1c: style on the pretrained policy (gait clock, speed matching, head posture) | done |
| `affect_prototype/` | Affect model on a scripted event timeline (PAD plot) | done |
| `demo_video/` | First closed-loop demos (gait clock; S1 style vector); superseded by `reactive/` | legacy |

Heavy runs (training, rendering) go to the cloud: `cloud/jobs/`.
