# Saved-checkpoint audit

Preregistered `7a3b7c5`, implemented `2218811`; all integrity criteria pass in
2.88 s with no rollouts or optimizer updates. Both checkpoints have six policy
and seventeen privileged slots at the 1e-6 standard-deviation floor. The fixed
synthetic probes reach normalized magnitude 449070, saturate some policy actions
and produce value predictions up to magnitude 12401.

This demonstrates susceptibility to amplification outside the tiny saved sample;
probes are not necessarily physical states and do not establish loss causation.
NumPy/Brax normalization agree. Saved-tree reserialization preserves literal
bytes across all 21 leaves; it does not retrospectively observe original live
arrays. Original smoke metrics and parameters are preserved.
