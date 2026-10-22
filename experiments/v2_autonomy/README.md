# V2 autonomy pilots

Local, non-confirmatory pilots for the new attention-to-behaviour connection.  They use NERVA's existing MuJoCo
simulation, rendered robot-eye RGB-D observations and the policy's measured capability envelope.  Output folders
are ignored because they contain timeline traces and rendered media.

`social_pilot.py` reuses the existing two-person scene: one person creates an observed near miss and another later
delivers a measured gentle contact.  It checks that only the latter produces a single safe greeting on re-encounter.
It is a component/safety pilot, not evidence of social generalisation or visual re-identification.

`render_milestones.py` produces the two review-first MuJoCo videos.  It writes the unlabelled comparisons,
separate labelled debug companions, every 20 ms control trace, the exact scripted world events, controller hash,
seed, render windows, and checksums into one ignored local output directory.  The robot runs from its RGB-D eye
camera in every condition; the scripts never provide a target to the controller.

```powershell
$env:OPEN_DUCK_ROOT = 'C:\path\to\open_duck'
python experiments/v2_autonomy/render_milestones.py --policy C:\path\to\b2.onnx --out experiments/v2_autonomy/results_review_seed62 --seed 62
```

`evaluate_milestones.py` is the CPU-only matched-seed check.  Each seed uses identical curiosity, positive-history,
and cautious-history schedules for the full controller, no-memory, no-attention, and fixed-rule baselines.  It
reports camera-track gaze error, inspection, greeting, approach/retreat/watch, encoder head ranges, tilt, risk
estimates and safety interventions, retaining a JSONL trace per run.

```powershell
python experiments/v2_autonomy/evaluate_milestones.py --policy C:\path\to\b2.onnx --out experiments/v2_autonomy/results_matched_seed --seeds 62,63
```

The unlabelled videos are the expression gate: numerical outputs only diagnose them.  A reviewer must inspect the
motion before treating an expression as successful.

```powershell
$env:OPEN_DUCK_ROOT = 'C:\path\to\open_duck'
python experiments/v2_autonomy/social_pilot.py --policy C:\path\to\b2.onnx --out experiments/v2_autonomy/results_social
```
