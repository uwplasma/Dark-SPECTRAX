"""rclosure: resolution-consistent field-scaled closure for the asymmetric-Hermite (AW) truncation instability.

DECLARED BEFORE RUNNING (never amended after results; a wrong gate is recorded as a failure and a follow-up is declared).

Stage M (frozen field, this script `measure`, `fit`): the truncated AW Fourier-Hermite operator of studies/highk_eig.py,
df/dt = -v df/dx + (q/m) E(x) df/dv (electrons, q/m = -1, u = 0), kept modes |k| <= K (k = j k1, k1 = 2 pi/40), Hermite order
Nn, width a, with the PR #66 hypercollision shape s(n) = n(n-1)(n-2)/((N-1)(N-2)(N-3)) damping the diagonal at rate
nu s(n).  Measured: (i) gamma_AW = max Re lambda at nu = 0; (ii) nu_req = the smallest nu (bisection in log nu on
[1e-3, 1e2] x base, 8 steps) for which max Re lambda <= 1e-8 omega_pe, base = sqrt(2 Nn) peak / a.  Expressed as the
dimensionless required strength c_req = nu_req / (|q/m| base).
 Grid (training): (K, Nn) in {2,5,10} x {32,64,128,256} U {21} x {32,64,128} U {42} x {32,64}; peak = max|E - <E>| in
 {1e-3, 4e-3, 1e-2, 3e-2}; a = sqrt(2e-3); field shapes: "single" cos(k1 x); "flat" sum_{j<=K} cos(j k1 x + phi_j);
 "inv" sum_{j<=K} cos(j k1 x + phi_j)/j; phi_j from numpy default_rng(20261019); each normalised to the peak.
 Held out (never used in the fit): (K, Nn) = (7, 96), (15, 48), (30, 48) at peak 2e-3 and 6e-3 for all three shapes, and
 (K, Nn) = (5, 64), peak 4e-3, "single", with a in {0.02, 0.1}.
 Report: the fitted exponents of c_req over (K, Nn, peak) (log-linear least squares on the worst shape at each (K, Nn,
 peak)), and the formula.
Gates for the closure formula c(K, Nn) (declared form: nu = c(K, Nn) |q/m| sqrt(2 N) peak / a, c(K, Nn) = c0 (K/5)^p (Nn/64)^q
 with p, q, c0 from the fit of the upper envelope, c0 raised so that Q1 holds with the safety factor below):
 Q1 (sufficiency)  formula nu >= 1.25 nu_req at every training AND held-out point, all shapes (so max Re <= 1e-8 with margin);
 Q2 (not wasteful) formula nu <= 8 nu_req at every training point of shape "single" and "inv" (a closure 8x above the
     requirement everywhere is rejected as non-informative);
 Q3 (baseline, informational) the fixed c = 4 closure of #66: fraction of training points with max Re > 1e-8, reported.
 A direct re-evaluation of max Re with the formula nu at all points (not only the bisected nu_req) is the Q1 record.

Stage D (decisive, `studies/nxconv_run.py --closure-kmax`, GPU allowed), v_q/v_te = 0.1 H05 (pump frame + remap, noise_floor
 1e-14, rtol 1e-10, atol 1e-14, T = 1000, chunked) with the scaled closure, runs (Nx, Nn) in {(32,64), (64,64), (64,128),
 (64,256), (128,128), (128,256)}; analysis `studies/rclosure_analysis.py` (same t_res/t_pos/ledger definitions as
 studies/gpures_gates.py, 10% of the running max of the first run):
 D1 pairs: (32,64)v(64,128), (64,128)v(128,256), (64,64)v(64,128), (64,128)v(64,256), (64,128)v(128,128), (64,256)v(128,256):
    t_res(dK_e) >= 1000 and t_res(W_ext) >= 1000 for every pair;
 D2 positivity per run: K_s > 0 and k=0 T_x > 0 through T;
 D3 ledger per run: max_t |(K - K(0)) - sum_j W_j| <= 1e-6 max_t |W_ext|;
 D4 per run: status success and t_end = T;
 D5 fidelity: for t <= 500 each scaled-closure run agrees with the c = 4 #66 record of the same (Nx, Nn) (studies/gpures/records,
    studies/nxconv) within 1e-3 of the running max on dK_e and W_ext (closure must not change resolved physics); missing
    c = 4 records are reported, not substituted.
 Verdict PASS only if D1-D5 all hold; otherwise the first failing gate and the certified window are recorded and stop.
Stage B (B10, ordinary mi/me = 1836, j = 8 point, Nn in {64, 128, 256}, scaled closure instead of fixed nu): B1 no blow-up at
 any Nn, B2 gamma(Nn = 64, 256) within 3% of gamma(Nn = 128), B3 gamma at a closure strength x3 within 3% of baseline (the
 closure sensitivity: the fixed-nu 100 -> 300 move was 5.3%).  Reported whatever the outcome.

Run: python studies/rclosure_gates.py measure|fit  -> studies/rclosure/measure.json, fit.json
"""
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

D = Path(__file__).resolve().parent / "rclosure"
k1 = 2 * np.pi / 40.0
A0 = np.sqrt(2e-3)
TOL = 1e-8
GRID = [(K, N) for K in (2, 5, 10) for N in (32, 64, 128, 256)] + [(21, N) for N in (32, 64, 128)] + [(42, 32), (42, 64)]
PEAKS = (1e-3, 4e-3, 1e-2, 3e-2)
SHAPES = ("single", "flat", "inv")
HELD = [(K, N, p, s, A0) for (K, N) in ((7, 96), (15, 48), (30, 48)) for p in (2e-3, 6e-3) for s in SHAPES] \
    + [(5, 64, 4e-3, "single", a) for a in (0.02, 0.1)]


def field(K, peak, shape):
    """E_j (complex amplitude of e^{+i j k1 x}), j = -K..K, normalised to max|E - <E>| = peak."""
    if shape == "single":
        amp = {1: 0.5, -1: 0.5}
    else:
        ph = np.random.default_rng(20261019).uniform(0, 2 * np.pi, 64)
        w = np.ones(K) if shape == "flat" else 1.0 / np.arange(1, K + 1)
        amp = {}
        for j in range(1, K + 1):
            amp[j] = 0.5 * w[j - 1] * np.exp(1j * ph[j - 1])
            amp[-j] = np.conj(amp[j])
    x = np.linspace(0, 40, 4096, endpoint=False)
    E = sum(v * np.exp(1j * j * k1 * x) for j, v in amp.items()).real
    s = peak / np.abs(E).max()
    return {j: v * s for j, v in amp.items()}


def shape_s(Nn):
    n = np.arange(Nn, dtype=float)
    return n * (n - 1) * (n - 2) / ((Nn - 1) * (Nn - 2) * (Nn - 3))


def operator(K, Nn, amp, a):
    """Sparse AW operator (identical to studies/highk_eig.operator for a single mode, checked in tests)."""
    nk = 2 * K + 1
    m = np.arange(Nn)
    rows, cols, vals = [], [], []
    for ik in range(nk):
        k = (ik - K) * k1
        base = ik * Nn
        rows += [base + m[:-1], base + m[1:]]
        cols += [base + m[:-1] + 1, base + m[1:] - 1]
        vals += [-1j * k * a * np.sqrt((m[:-1] + 1) / 2), -1j * k * a * np.sqrt(m[1:] / 2)]
        for dk, Ek in amp.items():
            jk = ik - dk
            if 0 <= jk < nk:
                rows.append(base + m[1:])
                cols.append(jk * Nn + m[1:] - 1)
                vals.append(Ek * np.sqrt(2 * m[1:]) / a)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nk * Nn, nk * Nn))


def maxre(L, nu, Nn, nk):
    M = L.toarray()
    if nu:
        M = M - np.diag(np.tile(nu * shape_s(Nn), nk))
    return float(np.linalg.eigvals(M).real.max())


def point(args):
    K, Nn, peak, shape, a = args
    amp = field(K, peak, shape)
    L = operator(K, Nn, amp, a)
    nk = 2 * K + 1
    base = np.sqrt(2 * Nn) * peak / a
    g0 = maxre(L, 0.0, Nn, nk)
    g4 = maxre(L, 4.0 * base, Nn, nk)
    lo, hi = np.log(1e-3), np.log(1e2)
    if g0 <= TOL:
        creq = 0.0
    else:
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if maxre(L, np.exp(mid) * base, Nn, nk) <= TOL:
                hi = mid
            else:
                lo = mid
        creq = float(np.exp(hi))
    return {"K": K, "Nn": Nn, "peak": peak, "shape": shape, "a": a, "gamma_AW": g0, "maxre_c4": g4, "c_req": creq}


def measure(workers=4):
    import multiprocessing as mp
    pts = [(K, N, p, s, A0) for (K, N) in GRID for p in PEAKS for s in SHAPES] + HELD
    pts.sort(key=lambda t: -t[0] * t[1])
    D.mkdir(exist_ok=True)
    out = []
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(point, pts):
            out.append(r)
            print(json.dumps(r), flush=True)
    out.sort(key=lambda r: (r["K"], r["Nn"], r["peak"], r["shape"], r["a"]))
    (D / "measure.json").write_text(json.dumps(out, indent=0) + "\n")


if __name__ == "__main__":
    {"measure": measure}[sys.argv[1]]()
