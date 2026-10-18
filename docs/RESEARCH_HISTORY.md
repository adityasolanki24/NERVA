# NERVA research history

Compact and chronological. The full evidence trail is [development_log.md](development_log.md), newest first. Every
preregistration and experiment record stays in `docs/` and `experiments/`, unchanged. Later corrective results govern
earlier ones (⟶).

| dates | phase | decisive result | record |
|---|---|---|---|
| 2026-09-28 | baseline | Open Duck Mini v2 policy reproduced bit for bit in a NERVA harness | `docs/open_duck_baseline.md` |
| 2026-09-28 | expressive locomotion on the pretrained policy (RQ1/1b/1c) | gait-clock style differences persist at matched speed; RQ1c head labels inverted | `experiments/expressive_locomotion/` |
| 2026-09-28 – 10-01 | style-conditioned policies S1–S6 (cloud L4) | S1 expresses tempo and torso pitch, **not step height** (preregistered fail). S2–S6 do not fix it; S5/S6 collapse to standing | `docs/style_policy_design.md` |
| 2026-10-01 | reactive loop, vision, tracking, memory M1–M4, Isaac Sim replay | full loop runs; memory 5/5 vs none 0/5 in the two-person scenario | `experiments/reactive/`, `docs/memory_design.md` |
| 2026-10-02 | architecture refactor A–I, Model B | typed contracts, outcome-grounded memory, profile v2 + Model B default; learned event prototypes worse than hand-coded (RQ9 falsified) | `docs/archive/architecture_refactor_2026-10-02.md` |
| 2026-10-02 – 10-09 | affect Model B v2–v4 | Bv2 removes PAD saturation; Bv4 (context facets) passes all criteria and becomes default ⟶ *APM programme G later showed a PAD-style state harmful for behaviour under equal tuning* | `docs/context_facet_experiment.md` |
| 2026-10-09 – 10-11 | neutral locomotion curriculum | B2 fails its gate; a series of diagnoses (references, normalisation, reward measurement point, MJX→native gap); the **turn-translation candidate passes the neutral long motor gate** | `docs/locomotion_curriculum.md`, `experiments/locomotion_curriculum/` |
| 2026-10-11 | expressive posture E1 / E1′ | E1 stopped (height above neutral infeasible); E1′ fails (pitch 9–30%, crouch ignored), neutral skill retained | `docs/expressive_posture_experiment.md`, `docs/expressive_posture_e1prime.md` |
| 2026-10-12 – 10-18 | used by APM (programmes H–K) | expression adapter on the frozen neutral controller; repeated sub-envelope pushes topple NERVA; IMU-only sensing cannot detect a changed entity at a safe distance | APM repository, `experiments/H_*` … `K_recovery` |
