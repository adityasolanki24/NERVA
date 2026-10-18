# NERVA roadmap

Only the milestones that remain. The previous long-term roadmap is in
[archive/roadmap_2026-10.md](archive/roadmap_2026-10.md).

| # | milestone | done when | status |
|---|---|---|---|
| M1 | **Continuous, robust locomotion** | a policy accepting continuous (vx, vy, ω) commands passes the neutral long motor gate *and* a new repeated-push gate (e.g. 5 pushes of 0.1 m/s within 10 s, 0 falls) | open. Blocks richer expression in APM. A playground-trained humanoid controller with random pushes in training is evidence this is achievable on GPU in about 1 h |
| M2 | Expressive posture that actually expresses | torso pitch / posture at ≥ 50% of the reference with the neutral gate retained (E1′ reached 9–30%) | open |
| M3 | Exteroceptive sensing in the loop | a camera-based observation of nearby entities' motion, used by the robot's evidence model | open (APM K models it as an extension) |
| M4 | Human evaluation | a preregistered perception study after an ethics determination | blocked on ethics |
| M5 | Hardware / sim-to-real | the controller runs on a physical Open Duck Mini v2 with an emergency stop | not started |
