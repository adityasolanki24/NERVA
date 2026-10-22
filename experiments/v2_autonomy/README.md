# V2 autonomy pilots

Local, non-confirmatory pilots for the new attention-to-behaviour connection.  They use NERVA's existing MuJoCo
simulation, rendered robot-eye RGB-D observations and the policy's measured capability envelope.  Output folders
are ignored because they contain timeline traces and rendered media.

`social_pilot.py` reuses the existing two-person scene: one person creates an observed near miss and another later
delivers a measured gentle contact.  It checks that only the latter produces a single safe greeting on re-encounter.
It is a component/safety pilot, not evidence of social generalisation or visual re-identification.

```powershell
$env:OPEN_DUCK_ROOT = 'C:\path\to\open_duck'
python experiments/v2_autonomy/social_pilot.py --policy C:\path\to\b2.onnx --out experiments/v2_autonomy/results_social
```
