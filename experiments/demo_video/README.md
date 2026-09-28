# Demo video: events → appraisal → PAD → behaviour → walking robot

`results/nerva_affect_demo.mp4`: 66 s, 1280×650, 25 fps, about 10 MB.

**This is a demonstration of the closed loop running, not an experiment and not a result.** The mapping from PAD to behaviour is hand-designed for the video and has not been validated. It is *not* the answer to how affect should drive movement: that is still open (RQ1c; `docs/research_questions.md`). Nothing in the video shows that the robot "feels" anything, or that its movement *looks* like any emotion.

## What the video shows

- **Left:** the Open Duck Mini v2 in MuJoCo with its unmodified walking policy, followed by a camera.
- **Right:** three panels revealed up to the current time (red cursor):
  - emotion intensities, from affect model v0.1
  - the PAD state
  - behaviour: commanded vs measured speed, tempo style, head posture
- **Bottom:** a caption for 4 s after each event (the event, its appraisal values, the emotions elicited), and a live status line.

## Loop

| rate | step |
|---|---|
| scripted times | synthetic events → `appraise()` (appraisal v0 table) → `CategoricalAffectModel.add()` |
| 10 Hz | affect `step()` → `demo_behaviour()` → `BehaviourCommand` + head posture |
| 50 Hz / 500 Hz | `OpenDuckSim`: policy / physics |

The `near_fall` event at 27 s coincides with a **real sideways push of 0.6 m/s**, so the stumble is physical: tilt peaks at 13.4°, and the robot recovers. In a check with the same seed, 0.75 m/s made the robot fall, so the margin is narrow.

**Demo mapping (hand-designed).** WHAT (functional) and HOW (expressive) are kept separate, and PAD never drives joints directly:
- **WHAT:** walk forward at 0.13 m/s. Stop for 6 s when the goal is blocked; pause while arousal > 0.2.
- **HOW:**
  - tempo style = clip(4·arousal, −1, 1): the gait clock (RQ1)
  - head posture = clip(1.25·(valence + dominance), −0.5, 0.5), where + is head up
  - Head posture is limited to ±0.5 because RQ1c showed that head offsets slow the walk (by 15–25% at ±0.5).

## What happens in this run [measured, `results/timeline.csv`]

| t (s) | event | emotions | visible behaviour |
|---|---|---|---|
| 2, 8 | successful_walking | joy 0.08 | walking; PAD drifts slightly positive; head rises a little |
| 15 | person_approaching_slowly | hope 0.05 | almost no change (weak emotion, as designed) |
| 25 | person_approaching_rapidly | fear 0.32, surprise 0.72 | arousal rises above 0.2 → robot pauses; tempo goes to +1 (steps quicker in place); head lowers as V + D falls |
| 27 | near_fall + push | fear 0.40, surprise 0.90 | real stumble (13° tilt), recovery; V −0.39, D −0.25 |
| about 36 | none | none | arousal decays below 0.2 → walking resumes |
| 40 | obstacle_blocking_goal | distress 0.40 | stops for 6 s |
| 52 | successful_walking | joy 0.08 | walking; PAD slowly recovers toward baseline |

## Reproduce

```bash
<venv>/Scripts/python -m pip install -e "<NERVA>[experiments]"
<venv>/Scripts/python experiments/demo_video/run.py
```

This takes about 75 s. The run is seeded (seed 0), so the simulation is deterministic. The MP4 encoding comes from the ffmpeg bundled with `imageio-ffmpeg`. The offscreen framebuffer of the loaded model is raised to 640×540 at runtime, as upstream `base.py` does. No upstream file is changed.
