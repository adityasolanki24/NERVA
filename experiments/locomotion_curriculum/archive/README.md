# Archived completed runners (moved 2026-10-10, content byte-identical)

These runners produced completed, preserved results. They are no longer part of the maintained neutral
motor workflow (see `../README.md`). Each file was moved with `git mv` and **no content change**, so its
SHA-256 matches any hash recorded at the time. Their results, preregistrations and raw artifacts are
untouched.

| runner | result directory | protocol |
|---|---|---|
| `contract_conformance.py` | `../results_contract/` | `docs/neutral_contract_experiment.md` |
| `neutral_smoke.py` | `../results_neutral_smoke/` | `docs/neutral_motor_smoke.md` |
| `long_turn.py` | `../results_long_turn/` | `docs/b2_long_turn_experiment.md` |
| `motor_audit.py` | `../results_motor_audit/` | `docs/neutral_motor_audit.md` |
| `turn_diagnostic.py` | `../results_turn/` | `docs/b2_turn_diagnostic.md` |
| `derivative_alignment.py` | `../results_derivative_alignment/` | `docs/reference_derivative_alignment.md` |
| `sim_gap_diagnostic.py` (moved 2026-10-11) | `../results_sim_gap/` | development log 2026-10-10 |
| `turn_pivot_diagnostic.py` (moved 2026-10-11) | `../results_turn_pivot/` | `docs/base_origin_velocity_pilot.md` |
| `turn_asymmetry_diagnostic.py` (moved 2026-10-11) | `../results_turn_asymmetry/` | `docs/turn_translation_pilot.md` |

**Reproducing a result exactly:** check out the implementation commit recorded in that result's
`protocol.json` (or the development log), where the runner is at its original path, and run it as
documented there, e.g. `python -m experiments.locomotion_curriculum.long_turn`. At the current commit
the same code is importable as `experiments.locomotion_curriculum.archive.<runner>`. Its shared imports
(`gate`, `normalization_timing`, `learning_support`) still resolve, because those modules now re-export
the maintained helpers unchanged.

Runners that other code still imports stay at their original paths, marked as completed in
`../README.md`. Moving them would require editing their imports and so changing their recorded hashes.
