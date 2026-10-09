# Shared neutral motor contract: conformance passes

Protocol `009bb5c`, implementation `88c6074`. Seed 123, 1,090 commands each
at periods 27 and 20; 2,180 ticks total. All fixed criteria pass in 1.55 s
under the 120-second cap. Maximum NumPy/JAX phase discrepancy 3.07e-7;
canonical command discrepancy zero; indices and rest modes match exactly.
Rest freezes at zero and movement onset restarts at zero; all phase norms,
finite values and index ranges pass. Symmetric planar reward at vx or vy
0.05 is 0.778801, below exact-rest reward 1.0.

`nerva.motor_contract` supplies a shared backend-parametric clock, NumPy
inference adapter and JAX training adapter. It is not wired to B2, any historical
policy, existing reward or deterministic safety. No rollout, PPO or motor
readiness evaluation took place. The failed reference subset still blocks
candidate environment training and adoption. These are infrastructure results.
