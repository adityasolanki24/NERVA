# NERVA state

*The single authoritative account of what NERVA is and does today (updated 2026-10-18). Everything is simulation
only. No hardware has been built or tested, and no human has evaluated NERVA's behaviour.*

## What NERVA is

A research biped built on **Open Duck Mini v2** (MuJoCo). It is used to develop:
1. a stable locomotion controller that accepts expressive modulation without losing stability;
2. a robot-side loop: perception → appraisal/memory → behaviour → locomotion, with a deterministic safety layer.

NERVA is also the first embodiment of the separate **APM** project (`affective-processing-module`). APM's
experience-learning and expression experiments run on NERVA's validated controller, read-only. Nothing in NERVA
depends on APM.

## Current components

| component | current choice | status | evidence |
|---|---|---|---|
| walking controller | **turn-translation neutral candidate** (frozen ONNX): 7 commands (rest, ±x, ±y at 0.074 m/s, turns ±0.6 rad/s), 0.54 s gait clock | **passes the neutral long motor gate** with and without action latency: 0 falls in 160 trials, every command 5/5, 80/80 pushes recovered | `experiments/locomotion_curriculum/results_turn_translation/`, `docs/neutral_motor_gate.md` |
| expressive posture on that controller | E1′ (torso pitch × crouch) | **fails**: pitch expressed at 9–30% of reference, crouch ignored; neutral skill retained | `experiments/locomotion_curriculum/results_e1prime/` |
| head channel | offsets added after the policy (APM measured envelope: ±0.4 rad at rest, none while walking) | measured, not trained | APM programme H (H0b) |
| reactive loop (scene, vision, tracking, appraisal, memory, behaviour) | profile "v2", affect Model Bv4 | all preregistered scenario criteria pass with the B2 policy; no falls in 15 runs | `experiments/reactive/`, `docs/context_facet_experiment.md` |
| safety | `nerva/safety.py`: stop on tilt ≥ 25° or stability risk ≥ 0.75, no affect inputs | a stop rule, not fall prevention | `docs/archive/architecture_refactor_2026-10-02.md` §3 |

## Demonstrated (simulation)

- Upstream Open Duck policy reproduced bit for bit.
- A neutral controller that passes a strict long motor gate.
- Style-conditioned policies can express tempo and torso pitch (S1), but not step height.
- The robot-side loop runs end to end with perception, memory and behaviour.
- Through APM: experience-dependent approach behaviour on the validated controller.

## Failed hypotheses (on record)

- Step height as a learnable style axis with the upstream reward (S1–S6).
- Body height above neutral (E1: infeasible references).
- Continuous crouch (E1′).
- Learned event prototypes beating hand-coded events (RQ9).
- From APM: an emotion-like PAD state does not improve behaviour (APM programme G).

## Active limitations

1. **Narrow motion vocabulary.** Seven discrete commands at one speed. Tempo can only be pauses. No validated
   continuous velocity or turning-rate control.
2. **Repeated sub-envelope pushes can topple NERVA.** The gate certifies recovery from single pushes. In APM
   programme J, 3–5 pushes of about 0.07 m/s in one encounter caused 2 falls.
3. **No exteroceptive sensing in the deployed loop.** APM's programme K shows NERVA's IMU cannot tell a changed
   entity apart at a safe distance; a camera (modelled, not built) can.
4. **Simulation only.** No hardware, no sim-to-real.

## Next

See [ROADMAP.md](ROADMAP.md). The motor-control milestone **M1** comes first: continuous velocity commands plus
robustness to repeated pushes. It is what richer, safe expression needs.
