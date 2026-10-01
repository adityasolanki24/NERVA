# Learned event prototypes vs hand-coded events (RQ9)

Can events be discovered from experience instead of being defined by hand? `run.py` replays the reactive
scenarios, segments the robot's world/self feature stream at prediction-error peaks
(`nerva/events/segmentation.py`), clusters the segments into a bounded set of online prototypes
(`nerva/events/prototypes.py`), and compares how well the most recent prototype vs the most recent
hand-coded appraised event predicts measured outcomes (`nerva/world/outcomes.py`) in the next 3 s.
Protocol, metric and reading are fixed in `run.py`'s docstring.

Prototypes are not emotions and are not claimed to match human event categories. Simulation only; the
hand-coded event path remains the baseline that drives the robot. Results: `results/summary.json` and
`docs/development_log.md`.
