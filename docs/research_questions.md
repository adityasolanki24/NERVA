# Research questions

Status (2026-10-02): RQ1–RQ1c are answered for the pretrained policy; the style-conditioned policies
(S1–S6) continued RQ1 (`style_policy_design.md`). The **active questions are RQ7–RQ9** below, which tie
the architecture refactor (`architecture.md` §2) to falsifiable comparisons instead of feature
accumulation. Each criterion is fixed before the corresponding run and recorded in the development log.

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
- **Head-posture feasibility (2026-09-28): negative for runtime modulation.** (Labels corrected 2026-10-02: the study's "head up" is face down; see `experiments/expressive_locomotion/README.md`.)
  - Head posture changes as intended, but it reduces walking speed by up to 84% and reshapes the gait.
  - The effect comes from the physical head movement, not from the observed command.
  - So independent posture dimensions need a policy trained with them varying.
  - See `experiments/expressive_locomotion/README.md` (RQ1c).

## Later questions (not active)

- **RQ2, affect → style.** Can a persistent PAD state, updated over time, drive `style` (and later more style dimensions) in a way that is stable and interpretable?
- **RQ3, appraisal → affect.** *(Model A v0.2 built and used in the reactive loop; see `docs/affect_model.md`; continued as RQ8)* Does an EMA-inspired appraisal of synthetic events (successful walking, near fall, person approaching slowly or rapidly, obstacle blocking goal), mapped to PAD by an explicit, replaceable rule, produce sensible affect trajectories? The mapping is our hypothesis, not psychology.
- **RQ4, perception by people.** Do human observers perceive the style differences, and do they attribute the intended emotional qualities to them? This is the only route by which labels like "confident" could be justified.
- **RQ5, imitation of expressive references.** Does conditioning on expressive reference motions (animation, motion capture, designed motion) produce more natural-looking styles than parameter modulation, at acceptable stability cost?
- **RQ6, failure-aware behaviour.** Can appraisal of near-falls and failures change behaviour (e.g. more cautious gait) while safety remains deterministic?

## Architecture questions (active)

- **RQ7, grounded memory.** Does learning person and place associations from measurable outcomes
  (`OutcomeSignal`: near-collision, benign contact, loss of stability) instead of from the affect
  model's own emotion labels (a) keep the person-specific behaviour of the existing two-person
  ablations, and (b) remove self-reinforcement?
  - *Falsified if* the grounded path fails the existing, unchanged memory criteria where the legacy path
    passes, or if an entity's adverse association still grows across sightings with no new adverse
    outcome.
  - **Result (2026-10-02): partly falsified.**
    - Grounded passes all four two-person criteria (S1 and B2, 5/5).
    - A's threat never grows without a new adverse outcome.
    - It fails "engages B" in the together scenario (0/3 vs legacy 3/3). The likely cause, not tested, is
      the selector's global fear gate combined with grounded memory's larger threat.
- **RQ8, are discrete emotion categories necessary?** Model A (appraisal → labels → PAD) vs Model B
  (appraisal → PAD directly, tendencies from appraisal features), behind the same contract.
  - *Comparison:* fixed appraisal traces (boundedness, decay, sign agreement) and the reactive scenario
    criteria.
  - *Reading:* if Model B passes the same criteria, the categories are not necessary *for these
    behaviours*; it does not show either model matches human emotion.
  - **Result (2026-10-02):**
    - *Fixed trace:* both bounded and recovering; valence-sign agreement 5/7, so that criterion is not met.
    - *Default scenario:* Model B passes all four criteria (5/5).
    - *Two-person scenario:* Model B first made B2 fall in 5/5 seeds. Its withdraw tendency triggers a
      "look away" head posture (yaw −0.8) that B2 cannot hold (measured: 4/4 falls standing).
    - *With B2's measured-safe head envelope* (pitch ≥ −0.2, |yaw| ≤ 0.4), Model B passes all four memory
      criteria with no falls, and Model A still passes the default scenario.
    - *Reading:* for these behaviours the discrete categories were not necessary. Model B exposed a body
      limit that Model A never reached.
- **RQ9, learned events.** Do event prototypes found by prediction-error segmentation and online
  clustering of world/self state predict grounded outcomes at least as well as the hand-coded event
  labels, in the same simulated scenarios?
  - *Falsified if* the prototypes predict outcomes worse than the hand-coded labels on the predefined
    metric. Prototypes are not claimed to be emotions or human concepts.
  - **Result (2026-10-02): falsified for this design.**
    - Brier score, adverse within 3 s: prototypes 0.0249, hand-coded labels 0.0215, base rate 0.0246.
    - Brier score, benign within 3 s: prototypes 0.0575, hand-coded labels 0.0297, base rate 0.0583.
    - Most boundaries marked tracking/gait fluctuations. Other features or segmentation would be a new,
      separately preregistered experiment.
