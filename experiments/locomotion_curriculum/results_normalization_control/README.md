# Variance-floor control: overall fail

Preregistered `1ee56c5`, implemented `d947120`. One epsilon=1e-4 intervention,
two 16-transition stages, seed 7, CPU, 225.75 s under a 900 s cap. Existing strict
epsilon-zero run is the frozen control; its protocol/core sources, versions,
references and checkpoint hashes all match. No repeated control training.

| Metric | Frozen fresh | Intervention fresh | Frozen warm | Intervention warm |
|---|---:|---:|---:|---:|
| Max normalized synthetic probe | 449070 | 81.202 | 449070 | 81.903 |
| Value loss | 884221440 | 10.4402 | 0.06124 | 0.05803 |
| KL | 155226423296 | **1489.4017** | 0.12137 | 0.13245 |

**Fresh KL fails the preregistered <=1.0 screen.** All other criteria pass;
overall success requires both stages. Variance-floor and amplification criteria
pass (std >=0.01 within float tolerance; inverse std <=100.0001; normalized peak
<=100 and >=100-fold reduction). Finite updates, count 16->32, live checkpoint
bytes, warm initialization and twenty restored-action probes all pass.

Read `comparison.json` for the overall study decision. `summary.json` records
the shared runner's plumbing decision only. Warm restore restarts Adam/RNG/counters.
Synthetic probes need not be physical states, and these 32 transitions do not
establish motor quality or general learning stability.

Preserve this negative result and all original artifacts. No epsilon tuning,
dependency change, default-policy promotion, deployment, paid work or expressive
training. Raw logs/checkpoints remain in ignored local storage. Next: preregister
a same-batch normalization-timing diagnostic at fixed weights before larger work.
