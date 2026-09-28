# Research questions

Only RQ1 is active. The rest are recorded so later work stays aimed, and are not being worked on.

## RQ1 (active): expressive conditioning of locomotion

**Status (2026-09-28), method A (gait-clock rate):**
- **RQ1, same command: yes.** There are measurable, consistent differences and no walking falls.
- **RQ1b, speed matched at 0.045 m/s: the differences remain.** The command confound is ruled out for the main effect (pitch).
- **But the effects are a coupled bundle, not independent.** Tempo goes up while stride, amplitude and forward pitch go down, all together along one axis.
- **Only a narrow speed band (about 0.045–0.056 m/s) can be matched at all,** because of the policy's command dead zone.
- Details: `experiments/expressive_locomotion/README.md`.
- **Open:** is one coupled scalar an adequate style representation? See RQ1c.

> Can the same biped locomotion controller produce **measurably different** movement styles, controlled by one variable `style ∈ [-1, 1]`, while remaining stable?

**Protocol outline (details fixed in Phase 5):**
- Same requested task for all conditions (e.g. walk forward, same command, same duration and initial state).
- Conditions: Style −1, Neutral (0), Style +1.
- Simulation first, raw accelerometer (see `open_duck_baseline.md` §10). The simulation is deterministic, so repeated runs need controlled perturbations (e.g. pushes or initial-state noise) to estimate variability.

**Candidate measurements.** We don't assume which of these will change:

| quantity | how |
|---|---|
| commanded vs **measured** velocity | base displacement in heading frame |
| step frequency, stride length | foot-contact events |
| step height | foot-site z during swing |
| torso posture | base pitch/roll mean and variance |
| head posture | neck/head joint angles |
| acceleration and smoothness | base acceleration; joint jerk |
| movement amplitude | joint-angle ranges |
| stability | falls, max tilt, recovery from pushes |
| effort | Σ τ², Σ abs(τ·q̇) (mechanical power) |

**Success means:** at least one quantity differs between Style −1 and Style +1 consistently and by more than run-to-run variation, *and* no condition falls or loses stability compared with Neutral.

**Not claimed by RQ1:** that any style *looks* hesitant or confident. That needs human evaluation (RQ4).

## RQ1c (next decision): what should the style representation be?

> Is one scalar adequate, or is a small style vector needed? If a vector, which dimensions, justified by measurement and by expressive-motion research?

- **Evidence so far (RQ1b):** the gait-clock scalar moves tempo, amplitude and posture together. It cannot set them independently, and it only covers a narrow speed band.
- **What the literature says observers use to read emotion from gait:**
  - overall speed, *and* posture/limb flexion plus dynamic cues, typically involving a few joints (Roether et al. 2009)
  - speed, stride length, heavy-footedness and arm swing (Montepare et al. 1987)
  - Laban Effort qualities: time, weight, space and flow (applied to robots by Knight & Simmons 2014)
- **Hypothesis to test next:** a small vector (tempo, step amplitude, torso posture, possibly smoothness) captures more of the expressive space than one scalar. Each component must be independently controllable, which the current policy cannot provide for posture or amplitude.

## Later questions (not active)

- **RQ2, affect → style.** Can a persistent PAD state, updated over time, drive `style` (and later more style dimensions) in a way that is stable and interpretable?
- **RQ3, appraisal → affect.** *(v0 prototype built; see `docs/affect_model.md`)* Does an EMA-inspired appraisal of synthetic events (successful walking, near fall, person approaching slowly or rapidly, obstacle blocking goal), mapped to PAD by an explicit, replaceable rule, produce sensible affect trajectories? The mapping is our hypothesis, not psychology.
- **RQ4, perception by people.** Do human observers perceive the style differences, and do they attribute the intended emotional qualities to them? This is the only route by which labels like "confident" could be justified.
- **RQ5, imitation of expressive references.** Does conditioning on expressive reference motions (animation, motion capture, designed motion) produce more natural-looking styles than parameter modulation, at acceptable stability cost?
- **RQ6, failure-aware behaviour.** Can appraisal of near-falls and failures change behaviour (e.g. more cautious gait) while safety remains deterministic?
