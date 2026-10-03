# B01 nonlinear Landau
k=0.5, eps=0.05 (gamma_L tau_B=4.3): grid shows monotone (slowing) decay to 2e-4|E0| by t=100, no envelope rebound -> not a bounce test; all SPECTRAX nu=0 runs recur (N=256/512/1024 deviate >10% in log-envelope at t=40/58/75).
k=0.3 (gamma_L tau_B = 0.79, 0.56, 0.35 for eps=0.01,0.02,0.05): grid first envelope min t=72.1, 50.4, 31.45; first max 155.0, 106.25, 65.6 (identical at Nx32/Nv1024/dt.025 and Nx64/Nv2048/dt.0125).
t_min ∝ eps^-0.515 (O'Neil: -0.5); t_min/tau_B = 1.15, 1.13, 1.12.
SPECTRAX nu=0 N=1024 reproduces the first minimum exactly (72.1, 50.4, 31.45) and stays within 10% log-envelope until t=200, 100.7, 101; N=512 until 140, 126, 85; N=256 until 79, 65, 49.
Parent hypercollision nu=1 (N=256/512): no envelope extrema found; it suppresses the trapping rebound (dashed curve fig3) -> a numerical closure, not collisionless physics.
Grid min f: -5e-4 .. -3e-3 (spectral interpolation of filamented f). Envelope = running max over ~1 plasma period.
