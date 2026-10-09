# Derivative alignment: timing helps but is insufficient

Protocol `b24aed2`, implementation `0d42391`. Original raw/fit hashes verified;
all endpoint metrics reproduced to 1e-8. Same recordings and five-harmonic
coefficients, no generation or refitting. Four known-signal checks pass.

Maximum per-joint held-out RMSE (rad/s):

| Condition | Endpoint derivative | Midpoint derivative | Interval average |
|---|---:|---:|---:|
| stand | 0 | 0 | 0 |
| forward | 1.953 | 0.827 | 0.814 |
| backward | 1.954 | 0.828 | 0.814 |
| left | 1.974 | 0.848 | 0.835 |
| right | 1.960 | 0.857 | 0.843 |
| turn left | 1.928 | 0.841 | 0.828 |
| turn right | 1.929 | 0.837 | 0.824 |

Both alternatives reduce error by more than half in all six moving conditions,
but neither meets the preregistered absolute 0.5 rad/s limit. Aggregate
**timing alone insufficient**. The original reference subset remains 0/7 pass;
contact labels and the invalid left-turn knee remain unrepaired. No adoption,
training or dynamic feasibility claim follows from these results.

Next: a separately preregistered derivative-consistent fitting comparison,
with fixed model complexity and held-out cycles, plus validated static contact
semantics and a deterministic positive-knee initialization check. Do not tune
the completed experiment or regenerate the full grid.

Final verification across these phases: 272 tests including slow tests passed;
Ruff clean. Two existing JAX cast warnings in unchanged training tests.
Public-content checks passed; raw artifacts remain ignored.
