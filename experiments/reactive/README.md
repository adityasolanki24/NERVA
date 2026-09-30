# Reactive behaviour (stage 1)

A scene with a person and a ball; the robot perceives them through a simulated head-camera detector,
appraises what happens in context, updates its emotions and PAD, and chooses what to do by
emotion-modulated action selection. Only the person and the ball follow scripts. Design:
`docs/reactive_behaviour_design.md`.

- `scenario.py`: the closed loop and the default ~100 s scenario (ball, friendly approach, lunge, return)
- `evaluate.py`: 5-seed evaluation against criteria stated in advance (curiosity, fear, habituation, safety)
- `render.py`: video with scene view, robot's-eye detections and live affect charts (render in the cloud:
  `cloud/jobs/reactive_demo.sh`)

Latest results (S1 policy, utility selector): curiosity 5/5, habituation 5/5, no falls 5/5, fear 0/5
(the robot freezes and retreats but gains distance too slowly in the first 3 s). See the development log.
Perception is simulated from ground-truth positions; no image-based detection yet.
