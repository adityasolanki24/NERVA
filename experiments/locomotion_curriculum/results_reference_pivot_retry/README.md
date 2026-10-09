# Corrected pure-turn references

Registration `f8728dc` plus preflight API repair `bc176bb`; implementation
`bc176bb`. Both eight-second motions pass every unchanged criterion in 18.31 s.
Installed planner translation/yaw geometry was verified through its support
polygon before generation. Pivot offset is derived once from initial geometry;
recorded poses are not shifted or clipped.

Body mean (vx, vy, yaw): left (0.005804, 0.016680, 0.603689), right
(-0.004149, 0.014414, -0.603689). Interval-velocity worst-component RMSE
0.226951 / 0.219495 rad/s. Knees and all joint limits pass. Preserve the earlier
repair 6/7 failure and the preflight abort. No trained-policy result.
