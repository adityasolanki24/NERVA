# Overview

NERVA (Neural Expressive Robot with Variable Affect) is a research platform for an expressive biped. It investigates how **contextual appraisal** of events and a **persistent affective state** can influence physically grounded robot movement, while the robot's locomotion stays stable.

The idea in one line: the robot interprets what happens relative to its goals, expectations and capabilities (appraisal). That interpretation changes a slowly evolving internal state, represented as pleasure/valence, arousal and dominance (PAD). The state then shapes both *what* the robot does and *how* it moves. The internal state is an engineered representation for studying expressive behaviour. It is not a claim that the robot has emotions.

**Where it stands (September 2026):**
- **Platform:** the Open Duck Mini v2 biped in MuJoCo simulation, with its pretrained walking policy, used unmodified.
- **Expressive locomotion:** one style variable changes the policy's gait-clock rate and produces measurable differences that survive a speed-matched comparison. The effects remain coupled, so a three-dimensional style-conditioned policy is now being prepared.
- **Affect:** a simulation-only prototype maps synthetic events through EMA-inspired appraisal to a persistent PAD state. It is **not yet connected to movement**.
- **Nothing is evaluated by people yet.** No style has been shown to *look* like any emotion.

Architecture: `architecture.md`. Research questions: `research_questions.md`. Long-term trajectory: `roadmap.md`. History of what was done and measured: `development_log.md`.
