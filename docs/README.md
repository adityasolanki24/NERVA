# NERVA documentation index

Start with the repository [README](../README.md), then:

| doc | read it for |
|---|---|
| [STATE.md](STATE.md) | what works, what does not, which controller and loop are current (**authoritative**) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the current loop, boundaries (Open Duck / NERVA / APM), safety rules |
| [ROADMAP.md](ROADMAP.md) | remaining milestones |
| [RESEARCH_HISTORY.md](RESEARCH_HISTORY.md) | the phases in one table, with links to their records |
| [development_log.md](development_log.md) | the full evidence trail for every number, newest first |

## Design references (describe code that exists)

| doc | what |
|---|---|
| [open_duck_baseline.md](open_duck_baseline.md) | how the Open Duck baseline works |
| [affect_model.md](affect_model.md) | affect Model A (legacy profile) |
| [memory_design.md](memory_design.md) | memory M1–M4 design |
| [reactive_behaviour_design.md](reactive_behaviour_design.md) | reactive behaviour stage 1 design |
| [style_policy_design.md](style_policy_design.md) | style-conditioned policy S1 design |
| [locomotion_curriculum.md](locomotion_curriculum.md) | neutral locomotion curriculum design |

## Experiment records (preregistrations and protocol reports; never edited)

| doc | title |
|---|---|
| [b2_long_turn_experiment.md](b2_long_turn_experiment.md) | B2 longer-turn support-region measurement |
| [b2_robustness_gate.md](b2_robustness_gate.md) | B2 neutral motor gate: preregistration (2026-10-09) |
| [b2_turn_diagnostic.md](b2_turn_diagnostic.md) | B2 turn-centre diagnostic |
| [base_origin_velocity_pilot.md](base_origin_velocity_pilot.md) | Base-origin reward velocity continuation (preregistration, 2026-10-10) |
| [context_facet_experiment.md](context_facet_experiment.md) | Same-source context facets: preregistration (2026-10-09) |
| [expressive_posture_e1prime.md](expressive_posture_e1prime.md) | E1′: continuous pitch × crouch conditioning over the admitted reference space (preregistration, 2026-10-11) |
| [expressive_posture_experiment.md](expressive_posture_experiment.md) | E1: continuous posture conditioning on the validated neutral candidate (preregistration, 2026-10-11) |
| [gait_averaged_tracking_pilot.md](gait_averaged_tracking_pilot.md) | Gait-averaged tracking continuation (preregistration, 2026-10-10) |
| [gpu_neutral_pilot.md](gpu_neutral_pilot.md) | Capped GPU neutral continuation (preregistration, 2026-10-10) |
| [identity_multibatch_stability.md](identity_multibatch_stability.md) | Bounded identity-preprocessing multi-batch update stability |
| [identity_preprocessing_smoke.md](identity_preprocessing_smoke.md) | On-policy identity-preprocessing numerical and restore screen |
| [neutral_contract_experiment.md](neutral_contract_experiment.md) | Shared neutral motor contract: preregistration |
| [neutral_learning_pilot.md](neutral_learning_pilot.md) | Balanced neutral learning pilot (preregistration, 2026-10-10) |
| [neutral_motor_audit.md](neutral_motor_audit.md) | Neutral motor-target audit |
| [neutral_motor_candidate.md](neutral_motor_candidate.md) | Neutral motor baseline candidate (opt-in environment) |
| [neutral_motor_gate.md](neutral_motor_gate.md) | Neutral long motor gate, with and without latency (preregistration, 2026-10-10) |
| [neutral_motor_smoke.md](neutral_motor_smoke.md) | Neutral candidate environment smoke |
| [neutral_normalization_audit.md](neutral_normalization_audit.md) | Saved neutral checkpoint normalization audit |
| [neutral_normalization_control.md](neutral_normalization_control.md) | Neutral normalization variance-floor control |
| [neutral_ppo_byte_retry.md](neutral_ppo_byte_retry.md) | Strict live-byte PPO smoke retry |
| [neutral_ppo_restore_smoke.md](neutral_ppo_restore_smoke.md) | Local neutral PPO and parameter-restore smoke |
| [neutral_reference_repair.md](neutral_reference_repair.md) | Neutral reference repair package |
| [normalization_schedule_comparison.md](normalization_schedule_comparison.md) | Offline normalization schedule and representation comparison |
| [normalization_timing_offline_retry.md](normalization_timing_offline_retry.md) | Offline retry of the preserved normalization-timing batch |
| [normalization_timing_replay.md](normalization_timing_replay.md) | Fixed-weight same-batch normalization-timing replay |
| [reference_derivative_alignment.md](reference_derivative_alignment.md) | Reference derivative time-alignment diagnostic: preregistration |
| [reference_subset_experiment.md](reference_subset_experiment.md) | Corrected neutral reference subset: preregistration |
| [turn_translation_pilot.md](turn_translation_pilot.md) | Turn-translation tracking continuation (preregistration, 2026-10-11) |

## Archive (superseded narrative, unmaintained)

| doc | what it was |
|---|---|
| [archive/architecture_refactor_2026-10-02.md](archive/architecture_refactor_2026-10-02.md) | current-vs-target architecture and refactor stages A–I (code docstrings cite its sections) |
| [archive/roadmap_2026-10.md](archive/roadmap_2026-10.md) | the long-term research-stage roadmap up to 2026-10-09 |
| [archive/research_questions.md](archive/research_questions.md) | research questions RQ1–RQ9 and their falsification criteria |
