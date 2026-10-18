# NERVA architecture (current)

The design history of the 2026-10-02 refactor (stages A–I, Model A/B, profile "v2") is in
[archive/architecture_refactor_2026-10-02.md](archive/architecture_refactor_2026-10-02.md). Code docstrings cite
its sections.

```
 MuJoCo scene (Open Duck Mini v2; people, ball; robot-eye camera)
   │ camera, IMU, touch, joint state
   ▼
 PERCEPTION      nerva/perception    colour+depth detection, multi-object tracker with IDs
   ▼
 WORLD           nerva/world         self state (tilt, stability risk), measured outcomes, scene graph
   ▼
 APPRAISAL       nerva/affect        frames over explicit outcome hypotheses, relative to goals and self state
 + MEMORY        nerva/memory        entity / episodic / place memory, outcome-grounded learning
   ▼
 AFFECT          nerva/affect        Model Bv4 (default) → PAD and target-conditioned action tendencies
   ▼
 BEHAVIOUR       nerva/behaviour     utility selection: what to do and how (reads tendencies, never emotion labels)
   ▼ locomotion command (one of 7) + optional style / head offsets
 SAFETY          nerva/safety.py     deterministic stop on instability; no affect input
   ▼
 CONTROLLER      nerva/sim           frozen ONNX policy (neutral candidate), NERVA's deployment contract
   ▼
 MuJoCo physics
```

## Boundaries

- **Open Duck** owns the robot model, physics, actuators and upstream policy. Its code is never edited. Only
  `nerva/sim/open_duck.py` and `nerva/training/` import it.
- **NERVA** owns everything above the locomotion command, plus its own trained policies.
- **APM** (separate repository) uses NERVA's simulator and frozen controller read-only, through its own adapter.
  NERVA has no APM dependency.

## Safety rules

- Physical safety never depends on appraisal, affect or any learned high-level model.
- Fall detection, joint and torque limits and stop rules are deterministic and sit below behaviour.
- PAD never drives joints. No LLM is in the control or safety loop.

## Training stack

`nerva/training/`:
- JAX/MJX environments: neutral joystick and style-conditioned;
- hash-verified references; auditable checkpoints; ONNX export.

Training runs on capped cloud GPUs (`cloud/`). Every candidate controller must pass the neutral long motor gate
(`docs/neutral_motor_gate.md`) before anything uses it.
