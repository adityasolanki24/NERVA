# Saved neutral checkpoint normalization audit

Preregistered 2026-10-09 after the fresh PPO smoke reported finite but very large
value loss/KL, before inspecting saved normalization values. This is a diagnostic
of existing artifacts, not an optimizer experiment or a changed smoke criterion.

Hypothesis: the tiny observation sample leaves some slots at the pinned Brax
standard-deviation floor (1e-6 with variance epsilon 0), amplifying observations
outside that sample. Such amplification is a possible contributor to the loss
scale; this audit cannot establish causation or motor quality.

Inspect the preserved fresh and warm checkpoints only after the smoke stops.
No dynamics, optimizer updates, parameter edits, new sampling or cloud work.
One native CPU process, 60 s parent watchdog, never overwrite output directories.
Verify the existing contract, configuration and every published checkpoint hash
before loading. Record normalization count, finite/positive standard deviations,
min/median/max and indices at <=1.0001e-6, separately for 101 policy and 212
privileged observation slots. Report inverse-standard-deviation maxima.

Reuse the smoke's five fixed probes (zeros, +/-0.1, linspace +/-0.2 and saved
mean). Record maximum absolute normalized observation, policy action and value
prediction. Check NumPy's (x-mean)/std against the pinned Brax normalizer, with
absolute/relative tolerance 1e-6. Mean probe must normalize to zero. These probes
need not be physically achievable and cannot diagnose rollout distribution shift.

Also resave/reload each existing parameter tree in new ignored storage and
compare every leaf's literal bytes, shape, dtype and named structure. This
strengthens the smoke's numeric-equality check to distinguish signed zero;
it does not retrospectively observe discarded in-memory training arrays.

All-required diagnostic integrity: input hashes/contracts match, both trees and
all computed outputs finite, standard deviations positive, manual/Brax
normalization agreement, zero mean-probe normalization and literal-byte
serialization roundtrip. Hypothesis support is descriptive: floor slots and
large probe amplification together are consistent with susceptibility, not proof
that normalization caused the PPO metrics. No loss/KL acceptance threshold is
invented after the smoke. Keep the smoke result and all original metrics intact.

Stop on any mismatch, nonfinite output, timeout or API error; preserve partial
reports/logs. Results may justify a separately preregistered normalization-control
smoke before larger neutral training. They do not authorize paid work, new reward
weights or expressive objectives.
