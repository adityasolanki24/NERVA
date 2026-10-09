# Corrected reference subset: 0/7 pass; adoption blocked

Protocol `7800728`, implementation `1e0a42d`. Exactly seven recordings,
two workers, one attempt each, completed within caps in 92.81 s. No repair,
substitution, full-grid regeneration, training or upstream modification.
Raw recordings, effective presets, Fourier fits and generator logs are ignored.
Hashes and fixed criteria are in protocol.json / trials.json.

Known rotation and frame checks all pass (12/12). Reconstructing velocities
from actual timestamps fixes the demonstrated double-angle error and gives
explicit current-body velocity frames. Removing the constant yaw-step bias is
an isolated recording change, not a modification of historical artifacts.

All held-out joint-position, linear-velocity and angular-velocity fits pass.
However, **all six moving fits fail joint-velocity accuracy**. Analytic velocity
from a five-harmonic position fit is not sufficiently faithful to the recorded
backward differences at the fixed tolerance. Do not increase harmonics or relax
criteria after this outcome. The derivative convention and fitting strategy
need a new, separately preregistered comparison before reference adoption.

Static stand passes all criteria except contact fit (66% agreement versus
90% required). The upstream `--stand` branch freezes base/joints/feet while
support-phase labels continue to alternate. These labels cannot simply be
treated as measured static ground contact. A future recorder must validate
contact semantics against geometry; this experiment did not repair labels.

Left turn also fails positive-knee and achieved-command criteria. The other
six conditions pass knee sign checks; the other moving conditions pass their
mean command tracking checks. All joint-limit exceedances remain reported.
These kinematic results make no dynamic stability/readiness claim.

| Condition | Mean vx/vy/yaw | Max joint-velocity RMSE, rad/s |
|---|---|---:|
| stand | 0 / 0 / 0 | 0 |
| forward | +0.07288 / +0.01476 / 0 | 1.953 |
| backward | -0.07249 / +0.01477 / 0 | 1.954 |
| left | +0.00016 / +0.08779 / 0 | 1.974 |
| right | +0.00023 / -0.05829 / 0 | 1.960 |
| turn left | +0.00302 / +0.03600 / +0.60369 | 1.928 |
| turn right | -0.00282 / -0.00712 / -0.60368 | 1.929 |

Planar units are m/s and yaw rad/s. Left-turn lateral mean exceeds 0.02 m/s;
its right knee reaches -1.941 rad. All six moving fits report knee-limit
exceedances (informational, unchanged protocol).

The seven-point set is not a Cartesian training grid. The new schema rejects
unsupported command lookup and remains experimental. Existing B2's motor gate
is still failed; no references, rewards or policy defaults were adopted.

The subsequent [alignment diagnostic](../results_derivative_alignment/README.md)
reduces error by more than half but remains above the fixed absolute limit.
Shared rest/phase [conformance](../results_contract/README.md) passes independently.
Next: preregister derivative-consistent fitting, validated static contact
semantics and positive-knee initialization before any new generation. No motor
training or expressive objectives are justified by these results.

Final verification across these phases: 272 tests including slow tests passed;
Ruff clean. Two existing JAX cast warnings in unchanged training tests.
Public-content checks passed; raw artifacts remain ignored.
