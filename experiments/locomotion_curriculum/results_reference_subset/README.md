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

The seven-point set is not a Cartesian training grid. The new schema rejects
unsupported command lookup and remains experimental. Existing B2's motor gate
is still failed; no references, rewards or policy defaults were adopted.

Next: isolate derivative time alignment/fitting and static contact-label
semantics under a new protocol before generating anything more. Shared rest
and phase infrastructure may be tested independently, but no motor training
or expressive objectives are justified by these results.
