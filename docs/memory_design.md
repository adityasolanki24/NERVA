# Design proposal: an evolving, brain-inspired memory for NERVA

Status: M1–M4 implemented 2026-10-01 (`nerva/memory.py`, `nerva/episodic.py`, `nerva/spatial.py`) (`nerva/memory.py`, `MemoryAppraiser`, `IdentityBinder` in `experiments/reactive/scenario.py`; results in the development log). M5 (face/body identity from real vision) proposed.
Claim labels: **[theory]** established finding or model from the literature; **[design]** NERVA choice;
**[hw]** hardware consideration.

## 1. What we want

A memory that lets the robot become familiar with places and things, form **relationships** with
people that change over time ("this person scared me once, but has been gentle since"), and use all of
that in appraisal and behaviour. It must run on robot hardware later: an embedded computer, bounded
RAM and storage, no network, no LLM in the control loop.

## 2. What we borrow from human memory

| Finding | What it tells us | Source |
|---|---|---|
| **Separate memory systems:** episodic (events), semantic (facts/knowledge), procedural (skills), working memory | Different stores with different update rules, not one database | Tulving; Soar splits memory the same way (Laird; Nuxoll & Laird) [theory] |
| **Complementary learning systems:** the hippocampus learns single episodes fast; the neocortex slowly extracts structure; **replay** transfers and generalises | A fast episodic buffer plus a slow semantic/relationship model, linked by offline consolidation | McClelland et al. 1995; Kumaran, Hassabis & McClelland 2016 [theory] |
| **Emotional modulation of encoding:** arousal (amygdala, stress hormones) strengthens consolidation | How strongly an event is stored scales with arousal and relevance; frightening events persist | McGaugh; LaLumiere, McGaugh & McIntyre 2017 [theory] |
| **Forgetting follows a power law** matched to how often information is needed | Memory activation from the history of use: A = ln Σ tⱼ^(−d); cheap to compute | Anderson & Schooler 1991; ACT-R [theory] |
| **Forgetting is adaptive:** memory exists to support decisions, not to be a perfect record | Bounded memory that prunes low-value detail and keeps gist is a feature, not a limitation | Richards & Frankland 2017 [theory] |
| **Reconsolidation:** a retrieved memory becomes changeable again, and **prediction error** is what triggers updating | Relationships evolve: meeting someone who behaves differently than expected updates the stored association | Nader, Schafe & LeDoux 2000; Sevenster, Beckers & Kindt 2013 [theory] |
| **Prioritised replay:** the brain replays the memories that most improve future decisions | Consolidate the most useful episodes first (salient, surprising, recent) | Mattar & Daw 2018 [theory] |
| **Somatic markers:** past emotional outcomes are attached to options and bias choices before deliberation | Every remembered entity/place carries an affective tag that feeds appraisal and action selection | Damasio; Bechara et al. [theory] |
| **Mere exposure:** repeated harmless exposure increases liking (an inverted U, with overexposure reducing it) | Familiarity grows liking up to a point | Zajonc 1968; later meta-analysis [theory] |
| **Relationships need shared history** | Long-term human–agent relationships depend on remembered shared episodes; memory-enabled robots are rated warmer and more trustworthy | Bickmore & Picard 2005; Ho et al. 2010 [theory] |

What we deliberately **don't** copy: the neural implementation. We copy the *functional principles*
(two learning speeds, emotional gating, decay by use, update on surprise, prioritised replay), because
they are cheap to compute and fit bounded hardware.

## 3. Proposed architecture (NERVA Memory, "NM") [design]

```
 PERCEPTION ──detections + identity embeddings──▶ WORKING MEMORY (now: ≤ ~8 active items, volatile)
                                                     │  salient events (arousal · surprise · relevance)
                                                     ▼
                                            EPISODIC MEMORY (fast, hippocampus-like)
                                            bounded buffer of events, each with an activation
                                                     │  consolidation / replay (idle time, prioritised)
                                                     ▼
          SEMANTIC & RELATIONSHIP MEMORY (slow, neocortex-like)          SPATIAL MEMORY (cognitive map)
          one record per entity (person / object) and per place        place graph + where things are
                                                     │
                     queries: "who is this, what do I expect, how do I feel about them/here?"
                                                     ▼
                         APPRAISAL (expectedness, desirability priors, controllability)
                         BEHAVIOUR (utility bonuses: approach the liked, avoid the feared)
```

### 3.1 Working memory
- The current tracks, the active goal and recent events. Bounded (about 8 items); nothing persists here.

### 3.2 Episodic memory (fast)
- **An episode is** (time, place node, entity IDs, event kind, appraisal vector, emotion snapshot, PAD,
  outcome, prediction error).
- **Encoding gate [design, from McGaugh]:** strength s₀ = relevance · (w_a·arousal + w_e·|prediction
  error| + w_n·novelty). Routine moments below a threshold are not stored individually; they only
  increment counters. So the robot remembers the lunge, not every second of watching a ball.
- **Activation [theory: ACT-R]:** A = ln(Σ tⱼ^(−d)) + s₀, where tⱼ is the time since each retrieval/
  rehearsal and d ≈ 0.5. Retrieval raises activation (use-dependent memory).
- **Retrieval by cue** (entity, place, situation). Score = activation + relevance (embedding / feature
  similarity) + emotional intensity: the same three signals as the Generative Agents memory stream
  (Park et al. 2023), computed numerically with no language model.
- **Bounded:** a fixed capacity (e.g. 5,000 episodes). Low-activation, already-consolidated episodes are
  pruned first.

### 3.3 Semantic and relationship memory (slow)
One record per **entity** (a person identity, or an object instance/category) and per **place**:
- **Identity:** a running mean of appearance embeddings. On hardware, a face embedding (MobileFaceNet:
  128-d, about 1M parameters, real-time on Jetson). In simulation, an appearance stand-in.
- **Familiarity** F: exposure with decay. It drives novelty (curiosity) and liking via mere exposure
  (inverted U).
- **Affective association (somatic marker):** expected valence and arousal/threat when this entity is
  present, learned by a prediction-error rule, V ← V + α·(experienced − V). α is larger for high
  arousal (emotional learning is faster) and larger when prediction error is large (reconsolidation).
- **Relationship state** for people, updated from episodes:
  - **familiarity:** how well known they are
  - **warmth/liking:** accumulated positive vs negative affect
  - **trust:** how predictable and benign their behaviour has been; this is the inverse of how often
    they surprised the robot negatively
  - **dominance/power:** do their actions leave the robot in control; from controllability
- **Slow drift** toward neutral in long absence, but strongly consolidated events decay far more slowly,
  so a frightening episode keeps influencing the robot, and later gentle episodes gradually outweigh it
  (extinction as new learning, not erasure).

### 3.4 Spatial memory
- A light place graph: nodes with positions, visited counts, place familiarity, place affect ("this
  corner is where I was frightened"), and last-seen object locations.
- This is a small version of a hierarchical 3D scene graph (Hydra; Hughes, Chang & Carlone 2022), and of
  the scene graph in the user's Rocky project.
- **SLAM:** in simulation the pose is known; on hardware, visual–inertial odometry/SLAM supplies the pose
  and the memory only stores the topological layer. So SLAM is a perception input, not part of the
  memory design.

### 3.5 Consolidation ("sleep")
- **When:** at idle times or on a schedule.
- **Replay [theory: CLS, Mattar & Daw]:** replay the top-k episodes by priority = emotional intensity ×
  prediction error × recency.
- **What replay does:**
  - updates entity/relationship records
  - merges repeated similar episodes into one gist episode with a count (compression)
  - marks episodes as consolidated (they can then be pruned without losing their effect on semantic
    memory)
  - this is also where reflection-like summaries would come from ("person A is usually gentle"), as
    numeric statistics, not text

### 3.7 Who did it? Identity binding and attribution (e.g. petting without seeing the face) [design]

Example: person A is recognised by face at a distance, walks up, and pets the robot while their face is
out of view. How does the positive experience end up attached to A?

1. **Track continuity.**
   - Identity is a belief attached to a *track*, not to the current face detection.
   - Once the face matches A, the track carries "A" with a confidence. Position/motion continuity, body
     and clothing appearance (and voice, on hardware) maintain it while the face is out of view.
   - The confidence decays slowly without confirming cues, and drops if the track is lost.
2. **Spatial attribution.**
   - A touch/contact event (head/back touch pads; on Open Duck possibly force/IMU disturbance; contact
     sensors in simulation) is attributed to the tracked person within reach.
   - With one person in reach, the attribution is near-certain; with several, it's split by distance.
3. **Touch appraisal.**
   - Slow, gentle, rhythmic stroking is appraised as positive. This parallels C-tactile afferents, which
     respond most strongly to gentle stroking at about 1–10 cm/s, the velocities people rate most pleasant (Löken et al. 2009) [theory].
   - A sharp impact is appraised as negative.
4. **Temporal credit.**
   - The affective outcome updates every entity active in working memory within an eligibility window
     of a few seconds (event binding into one episode).
   - Each update is weighted by recency × identity confidence: Δassociation = α · p(identity) ·
     (experienced − expected).
5. **Deferred binding.**
   - An unresolved episode is stored as "petted by unknown track #k".
   - If the same continuous track, or a later appearance re-identification, confirms A, the episode is
     resolved and applied to A's record at the next consolidation ("oh, it was you").
6. **Generalisation fallback.** Never-resolved outcomes update the "unknown person" category and the
   place's affect, so they weaken rather than vanish.

Test (added to §5): A approaches, pets the robot out of face view and leaves; B has never touched it.
Afterwards, warmth/approach toward A increases, but toward B it doesn't.

### 3.6 Interfaces to the rest of NERVA
- **Appraisal:**
  - expectedness = how well memory predicted the event (low for a known-gentle person suddenly lunging →
    surprise)
  - desirability prior = the entity's affective association
  - controllability = the relationship's dominance term
  - the current global "threat memory 45 s" and "novelty habituation" become per-entity memory
- **Behaviour** (the utility selector): approach utility gains + warmth · F-dependent liking; retreat/watch
  utility gains + threat expectation of that specific person; curiosity uses per-entity and per-place
  novelty.
- **Nothing in memory commands motion or bypasses the safety layer.**

## 4. Hardware plan [hw]

| Constraint | Choice |
|---|---|
| **Compute:** Jetson-class, shared with perception and the policy | All memory updates are O(1) arithmetic per event. Retrieval is a brute-force scan over ≤ 10⁴ small vectors (≈ ms). Consolidation runs only when idle, in bounded batches. |
| **RAM** | Working memory plus the recent episodic buffer in RAM (≈ a few MB). Semantic, relationship and spatial records are small (hundreds of entities × a few hundred bytes plus 128-d embeddings). |
| **Storage** | SQLite on flash (single file, transactional), written at consolidation. The model survives restarts. |
| **Latency** | Appraisal queries read cached entity records (no search in the 10 Hz loop). |
| **No network, no LLM in the loop** | Everything numeric and local. An optional language interface could later *read* memory offline, never write control. |
| **Privacy** | Identities are embeddings, not images. Local only, with a delete-person API. |
| **Determinism and debugging** | Every update is logged, and memory can be snapshotted and replayed in simulation. |

## 5. How we would test it (stated in advance)

1. **Person-specific wariness.** Two people, only one lunges. Afterwards the robot is wary of that
   person only: watch/retreat for A, approach for B.
2. **Relationships evolve.** After the lunge, repeated gentle encounters with A reduce wariness step by
   step (extinction). A second lunge after a long gentle period produces a large surprise (a prediction
   error against the now-trusted model).
3. **Familiarity.** A familiar room and objects elicit less orienting and inspection than a new object
   placed in the familiar room (novelty relative to memory, not to a timer).
4. **Adaptive forgetting.** After a simulated long absence, trivial encounters are forgotten but the
   frightening episode still biases appraisal.
5. **Budget.** Memory size and per-step compute stay within fixed limits over a long simulated run
   (e.g. 10⁵ steps).
6. **Ablation.** With memory disabled, behaviour differs measurably, so memory is causal, not decorative.

## 6. Implementation phases

| Phase | Content | Needs |
|---|---|---|
| **M1** | Entity memory: identity (simulated appearance), familiarity, affective association, relationship state; appraisal and behaviour read it | current sim |
| **M2** | Episodic store with emotional encoding gate, ACT-R activation, cue retrieval | M1 |
| **M3** | Consolidation/replay, reconsolidation on prediction error, pruning and compression | M2 |
| **M4** | Spatial place memory (place familiarity and affect) | M1 |
| **M5** | Real identity from vision: MuJoCo appearance features now; face embeddings in Isaac/hardware | real perception |

## Sources

- Löken, Wessberg, Morrison, McGlone & Olausson 2009, *Coding of pleasant touch by unmyelinated afferents in humans*, Nature Neuroscience 12(5): https://pubmed.ncbi.nlm.nih.gov/19363489/

- Kumaran, Hassabis & McClelland 2016, *What learning systems do intelligent agents need? CLS theory updated*: https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf
- Anderson & Schooler 1991 and ACT-R activation (overview): https://www.ai.rug.nl/~niels/publications/taatgenLebiereAnderson.pdf
- LaLumiere, McGaugh & McIntyre 2017, *Emotional modulation of learning and memory*: https://pmc.ncbi.nlm.nih.gov/articles/PMC5438110
- Nader, Schafe & LeDoux 2000 (reconsolidation), reference: https://www.nature.com/articles/nn1778
- Sevenster, Beckers & Kindt 2013, *Prediction error governs pharmacologically induced amnesia for learned fear*: https://research-portal.uu.nl/en/publications/prediction-error-demarcates-the-transition-from-retrieval-to-reco/
- Richards & Frankland 2017, *The persistence and transience of memory* (summary): https://neurosciencenews.com/forgetting-smarter-6947/
- Mattar & Daw 2018, *Prioritized memory access explains planning and hippocampal replay*: https://pmc.ncbi.nlm.nih.gov/articles/PMC6203620
- Somatic marker hypothesis (review): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7852379/
- Zajonc 1968, mere exposure: https://www.psy.lmu.de/allg2/download/audriemmo/ws1011/mere_exposure_effect.pdf
- Park et al. 2023, *Generative Agents*: https://arxiv.org/pdf/2304.03442v1
- Soar episodic memory (Nuxoll & Laird): https://users.cs.northwestern.edu/~mek802/papers/not-mine/cog-arch/nuxoll-2004-ICCM-epmem.pdf
- Hughes, Chang & Carlone 2022, *Hydra*: https://roboticsproceedings.org/rss18/p050.html
- Bickmore & Picard 2005, *Establishing and maintaining long-term human-computer relationships*: https://www.media.mit.edu/publications/establishing-and-maintaining-long-term-human-computer-relationships
- Ho, Dautenhahn, Lim & Du Casse 2010, *Modelling human memory in robotic companions*: https://researchprofiles.herts.ac.uk/en/publications/modelling-human-memory-in-robotic-companions-for-personalisation-/
- Long-term memory for social robots (RoboCup@Home proposal): https://arxiv.org/pdf/1811.10758
- MobileFaceNets (128-d face embeddings on edge devices): https://arxiv.org/abs/1804.07573v3
