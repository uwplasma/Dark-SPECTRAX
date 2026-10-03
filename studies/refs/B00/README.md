# B00 linear Landau, k lambda_D = 0.5, 0.3, seed 1e-4
Observable: complex E_k(t); fit E_k = e^{gt}(A e^{-iwt}+B e^{iwt}), 5 window variants. Base windows [10,40] (k=0.5), [16,150] (k=0.3);
SPECTRAX window end = min(base, 0.9 t_c), t_c = first time |E_sx - E_grid(dt=0.0125)| > 1e-3|E(0)|.
| k | wofz root | grid SL (Richardson dt) | Gkeyll p2 Nv=64/128 | SPECTRAX N=128 | SPECTRAX N=256 |
|---|---|---|---|---|---|
| 0.5 | 1.4156619 - 0.1533595i | 1.415662(15) - 0.1533591(68)i | 1.415661 - 0.1533593i (Nv=64) | 1.4156617 - 0.1533591i | 1.4156617 - 0.1533591i |
| 0.3 | 1.1598465 - 0.0126204i | 1.159844(16) - 0.0126180(16)i | 1.1598446 - 0.0126200i (Nv=128) | 1.1598463 - 0.0126203i | 1.1598459 - 0.0126200i |
SPECTRAX N=32 cannot be fit (t_c 15.2 / 24.9 leaves <8 time units). t_c(N=32,64,128,256): k=0.5: 15.2, 24.4, 37.5, >45; k=0.3: 24.9, 40.3, 62.2, 93.3.
Window-variation std of SPECTRAX fits: <=2e-6 (w), <=1.2e-6 (g). beta=v_t/c 0.1 vs 1.0 max |dE|/eps = 2.9e-5 (k=0.5), 1.1e-4 (k=0.3): tolerance-level, not physics.
