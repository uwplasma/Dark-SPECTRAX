"""picconv follow-up A (declared 2026-10-26 BEFORE running; conv_gates.py is untouched), prompted by relhhs: the
deterministic relativistic continuum gives dE_1D,e = 0.056 E_th at t = 1000, ~400x below the PIC dW_th ~ 21.5 E_th, so
the PIC heating may be seeded by PIC shot noise.  Question: does dW_th(1000) scale with the initial noise level or tend to
a noise-independent value?

Two independent noise knobs:
  (i)  ppc ladder (conv_gates.py: 103, 206, 412, 824, 1648): shot-noise power ~ 1/ppc.  Prediction-to-test: dW = a + b/ppc.
       If a ~ 0 the heating is noise-limited (b/ppc decreasing with ppc); if the heating rises or stays with ppc it is not
       a pure shot-noise effect.  (Existing data: ppc 103 -> 412 raised dW_th by 19 %.)
  (ii) mirrored velocity loading (relpic_run.py --mirror): velocities drawn in exactly cancelling (v, -v) pairs on adjacent
       quiet-start positions, same variance and zero mean: low-k density/current noise is suppressed at fixed ppc.
       Runs (relativistic, dx 0.04, dt 0.02, quadratic, T 1000): M103_s0, M103_s1, M206_s0.
Reported outcome (no pass/fail threshold): dW_th(1000) of M runs against the random-load runs at the same ppc, in units of the
random-load seed std; Fit coefficient a, b (E_th) from (i).  Scaling statement: 'noise-independent' if |M - random| < 2 sigma
and the (i) fit gives a within 2 sigma of the finest value; 'noise-seeded' if M differs by > 20 % from random loading or a
is consistent with zero.
"""
FOLLOWUP = {
    "M103_s0": ("--T 1000 --ppc 103 --seed 0 --mirror", 1), "M103_s1": ("--T 1000 --ppc 103 --seed 1 --mirror", 1),
    "M206_s0": ("--T 1000 --ppc 206 --seed 0 --mirror", 1),
}
