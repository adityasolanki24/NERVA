# Design: reactive, affect-driven behaviour in a simulated scene (stage 1)

Status: design, 2026-09-30. User decision: stay in MuJoCo for now, make the robot genuinely react to its
environment through the emotion model, and show complex behaviours (curiosity, fear with stepping back);
Isaac Sim is a later stage.

## 1. Why [measured]

The S1 demo varied only *style* (tempo, torso pitch). Its differences are small (about 3° of pitch) and hard
to see. Emotion is far more readable in *what* the robot does relative to things around it: orienting,
approaching, keeping distance, freezing and fleeing. That needs a world with things in it, perception of
them, and appraisal that depends on context.

## 2. Loop (canonical architecture, unchanged)

```
MuJoCo scene (robot + person + object, scripted movers)
  → PERCEPTION   simulated head-camera detection: field of view, range, noise, dropouts → tracks + events
  → APPRAISAL    context-aware: distance, approach speed, novelty/habituation, escape room → AppraisalState
  → AFFECT       affect model v0.1 + "interest" (new) → PAD, active emotions
  → BEHAVIOUR    WHAT: behaviour mode (explore / orient / approach / freeze / retreat / withdraw)
                 HOW:  style e for S1 (tempo, lean) + head gaze/posture
  → S1 policy (unchanged) → deterministic sim
```

Rules kept: no PAD → joints; no LLM in the loop; the learned policy only receives commands, style and head
offsets it already takes.

## 3. Components [design]

- **Scene** (`nerva/sim/world.py`):
  - Upstream scene plus visual-only mocap bodies: a "person" (body + head) and a "ball". These have no
    collision with the robot, so the physics of the robot is unchanged. A test checks this.
  - A camera on the robot's head, for a "robot's-eye view" inset.
  - Movers follow scripted waypoints; the robot's reactions are not scripted.
- **Perception** (`nerva/perception/tracker.py`):
  - Stands in for a face/object detector, which we can't run meaningfully on MuJoCo renders. It uses
    ground-truth entity positions, but detects only inside the head camera's field of view and range.
  - It adds bearing/range noise and distance-dependent miss probability, then tracks bearing, distance and
    approach speed (filtered).
  - Events come from tracks: appeared, approaching (slow/fast), very close, lost.
- **Contextual appraisal** (`nerva/affect/appraisal.py`, v1):
  - The same event kind is appraised differently by context.
  - Faster and closer approaches are less desirable and less controllable.
  - Repeated harmless exposure raises expectedness (habituation).
  - A novel, non-threatening stimulus is relevant and unexpected, which elicits interest.
- **Interest** (`nerva/affect/emotions.py`): a new emotion, "interest" (curiosity).
  - Elicited when a stimulus is novel (expectedness < 0.5) and not harmful (desirability ≥ 0).
  - EMA and ALMA don't define it, so its rule and PAD anchor are NERVA choices. The anchor is positive
    valence, raised arousal and slight dominance, the "interested/alert" region of the affect circumplex
    (qualitative placement; the numbers are ours).
- **Behaviour v1** (`nerva/behaviour/pad_style.py`): mode selection with hysteresis and minimum durations, from the
  dominant emotion, PAD and the tracks. Continuous controllers per mode:
  - explore: walk and gently turn, head scanning
  - orient: stop, turn head then body toward the stimulus
  - approach: walk toward it, stop at a social distance that grows as dominance falls; gaze at face, head tilt
  - freeze: stop briefly, head lowered, on a startle close by
  - retreat: step back, then turn away and walk off until at a safe distance; glance back; fast tempo
  - withdraw: stop, look away / head down, slow tempo (distress)

## 4. Known limits [measured / fact]

- Backward walking is weak: 21% of the command (Phase 2 baseline). The references do move backward
  (−0.10 to −0.12 m/s for straight gaits, R1 logs), so the cause is in training. Retreat therefore uses a
  short backward step, then turn-away + forward walking. Fixing backward walking is a separate training task.
- Head offsets slow walking (RQ1c).
- S1 never saw combined or intermediate styles.

## 5. Evaluation (stated before running)

Per scenario, 5 seeds (the perception-noise seed and the robot's initial-state noise):

| Reaction | Measure |
|---|---|
| Curiosity | the robot's distance to the ball/person decreases, and it stops within 0.3–1.0 m |
| Fear | after a fast approach, the distance to the person **increases** within 3 s of onset |
| Habituation | a second identical slow approach elicits lower peak interest/fear than the first |
| Safety | no falls |

Nothing here is a claim about human perception of emotion (roadmap stage 15).
