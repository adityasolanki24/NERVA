# Neutral candidate CPU smoke

Registration `f92e19e`, implementation `25510c8`; protocol
`docs/neutral_motor_smoke.md`. **Pass: 80/80 steps**, 74.88 s under a 600 s
parent watchdog, no optimisation, cloud work, terminations or zero reward clipping.

Seven separate eight-step commands plus stand-forward-stop (24 steps), seed 7,
flat terrain, zero actions. All states, observations and raw/scaled rewards
finite; exact reference selection and shared clock/mode parity pass. NumPy/JAX
parity over 189 reference phases: worst position error 4.63e-7 rad, analytic
velocity error 5.51e-6 rad/s. Policy/privileged observations 101/212 slots.

The manifest records five passing repair targets plus two geometric-turn
targets with report/recording/fit hashes. Original 0/7 and subsequent 6/7
negatives and the preflight binding abort remain intact. Raw motions, fits
and diagnostic logs remain ignored local artifacts.

Weighted imitation range -9.711 to +2.348, rest pose/velocity -3.533 to 0,
alive +20. Reward 0.2143–0.5582 per tick. Weights were not tuned. This checks
environment plumbing only; no trained candidate policy or motor-readiness
result, and B2's failed gate is unchanged. Next: capped PPO/restore smoke,
followed by a neutral learned-policy comparison and long readiness gate.
