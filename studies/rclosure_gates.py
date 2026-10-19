"""rclosure: resolution-consistent field-scaled closure for the asymmetric-Hermite (AW) truncation instability.

DECLARED BEFORE RUNNING (never amended after results; a wrong gate is recorded as a failure and a follow-up is declared).

REVISION v2 (declared before any measurement result existed; the v1 measure run was killed with no output). A pre-run probe
(E_k1 = 1e-2, K = 5, Nn = 64, single mode, nu = 0) gave max Re lambda = 0.756 (dense LAPACK), 0.846 (ARPACK) and an asymptotic
time-domain growth rate of 0.702 (exp(L t) applied to a random vector, t = 200): at these amplitudes the operator is so
non-normal that eigenvalues are round-off sensitive, and the simulation sees the time-domain growth.  v2 therefore defines
gamma as the asymptotic growth rate of ||exp((L - nu diag s) t) v0||, v0 a fixed random vector (rng 1), integrated in
chunks of 5 omega_pe^-1 to T_int = 240 and averaged over the last 40 (log-norm slope); "stable" means gamma <= 1e-3 (gamma_tol).
Everything else (grid, shapes, gates Q1-Q3 with gamma_tol in place of 1e-8) is unchanged; the fixed-c = 4 baseline (Q3) uses
the same gamma.

REVISION v3 (declared after a pre-run probe on K = 5, Nn = 64, single mode, T_int = 100, no measurement-grid result yet).  The probe
gave, for peak E = 1e-3 / 4e-3 / 1e-2 and c = 0, 1, 4, 16, 64, 256 (nu = c |q/m| sqrt(2 Nn) peak / a): gamma = 0.134 0.063 0.033
0.029 0.032 0.028 / 0.468 0.309 0.190 0.108 0.058 0.041 / 0.635 0.383 0.225 0.164 0.136 0.112.  Raising c does NOT bring
gamma to <= 0: it leaves a floor.  The v2 bisection for c_req (bracket up to 1e3) is therefore replaced: v3 measures gamma
on the fixed ladder c in {0, 4, 16, 64, 256, 1024} and defines c_req = the smallest ladder value with gamma <= gamma_tol
(null if none).  Declared hypothesis H*: for the order-2 hypercollision shape s(n), NO ladder c reaches gamma <= gamma_tol at
peak >= 1e-3 at any (K, Nn) of the grid, i.e. the premise "a closure strength c(K, Nn) keeps gamma_AW <= 0" is false for this
closure family, and the AW floor gamma_floor = gamma(c = 1024) is the quantity to fit.  Declared consequences, fixed now:
 (a) if c_req is non-null at every training point, the original Q1-Q3 and stage D apply unchanged;
 (b) if H* holds (c_req null at > 90% of the points with peak >= 1e-3), stage D is NOT run as specified; the result is recorded as
     a rejection of the closure-strength route, with the fitted floor law gamma_floor(K, Nn, peak) (log-linear, exponents
     reported) and the check whether gamma_floor(c) falls below the growth the H05 run can tolerate (1/(T=1000) times
     ln(1e-14 noise floor -> 1e-3 field) ~ 0.02 for peak 1e-3.. ) is NOT made a gate; a follow-up study is then declared in a new
     commit before anything else is run.
Grid v3 (replaces the v1/v2 grid): (K, Nn) in {2, 5, 10, 21} x {32, 64, 128} U {(5, 256), (10, 256)}; peak in {1e-3, 4e-3};
shapes "single" and "inv" (the "flat" shape only at K <= 10); a = sqrt(2e-3); T_int = 150, mean of the last 30; gamma_tol 1e-3.
Held-out: the (K, Nn) = (7, 96), (15, 48) points at peak 2e-3, shapes single and inv, ladder c = 0, 4, 1024 (used only to check
the fitted floor law within 30%).  The K = 42 and peak >= 1e-2 points of v1/v2 are dropped for cost; stated, not hidden.

Stage M (frozen field, this script `measure`, `fit`): the truncated AW Fourier-Hermite operator of studies/highk_eig.py,
df/dt = -v df/dx + (q/m) E(x) df/dv (electrons, q/m = -1, u = 0), kept modes |k| <= K (k = j k1, k1 = 2 pi/40), Hermite order
Nn, width a, with the PR #66 hypercollision shape s(n) = n(n-1)(n-2)/((N-1)(N-2)(N-3)) damping the diagonal at rate
nu s(n).  Measured: (i) gamma_AW = max Re lambda at nu = 0; (ii) nu_req = the smallest nu (bisection in log nu on
[5e-2, 1e3] x base, 7 steps) for which gamma <= gamma_tol = 1e-3 omega_pe, base = sqrt(2 Nn) peak / a.  Expressed as the
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
 Q1 (sufficiency)  formula nu >= 1.25 nu_req at every training AND held-out point, all shapes (so gamma <= gamma_tol with margin);
 Q2 (not wasteful) formula nu <= 8 nu_req at every training point of shape "single" and "inv" (a closure 8x above the
     requirement everywhere is rejected as non-informative);
 Q3 (baseline, informational) the fixed c = 4 closure of #66: fraction of training points with gamma > gamma_tol, reported.
 A direct re-evaluation of gamma with the formula nu at all points (not only the bisected nu_req) is the Q1 record.

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

RESULT OF STAGE M (v3, studies/rclosure/measure.json, 82 points; recorded here by the commit that declares stage F, before F runs).
c_req is null (no ladder c up to 1024 reaches gamma <= 1e-3) at 39/39 points with peak 4e-3 and 23/39 with peak 1e-3 (62/78 =
79% overall, below the 90% bar of consequence (b) when pooled; 100% at 4e-3, the amplitude of the k1 field at t ~ 200-450 of the
v_q 0.1 runs).  Neither (a) nor (b) is met pooled; the declared reading for the gap: Q1 FAILS (not non-null everywhere), so stage D
as specified, which presupposes a closure that keeps gamma <= 0 at every refinement, is NOT run, and no SPECTRAX option is added
(a closure whose premise fails would be a number without a derivation).  Fits (log-linear, 66-77 points, rms residual 0.4-0.5 in
ln gamma, so factors of ~1.6, not tight): gamma(c = 0) ~ 32 Nn^0.57 peak^1.2 (K exponent 0.00: the AW growth in a frozen field
does not rise with the Fourier cut-off, contrary to the single-mode eigenvalue table of highk; it rises as ~sqrt(Nn) and
~linearly in the field), gamma(c = 4) ~ 2.4 Nn^0.78 peak^1.1, floor gamma(c = 1024) ~ 11 K^0.15 Nn^0.30 peak^1.26.  At
peak 4e-3 the c = 4 growth is 0.04-0.46 and the c = 1024 floor 0.02-0.14, i.e. e^{20-100} over T = 1000 from the 1e-14 noise.
The closure-strength route therefore cannot be made resolution-consistent; held-out points (7,96) and (15,48) at peak 2e-3
(c = 1024 gamma 0.018-0.038 and 0.003-0.023) sit within the scatter of the floor fit (not a pass/fail gate).

STAGE F (declared before running; follow-up to the Q1 failure): does a much stronger fixed c still extend the real H05 window,
and at what cost to resolved physics?  v_q/v_te = 0.1, Nx 64, Nn 128, pump frame + remap, noise_floor 1e-14, rtol 1e-10,
atol 1e-14, T = 1000, chunked, c in {16, 64, 256} (nxconv_run.py --field-nu c --tag _rcf, GPU allowed); reference c = 4 record:
studies/gpures/records (t_end 595, K_s <= 0 at 581).  Window(c) = min(t_pos, t_reached).
 F1 (extension) window(c = 256) >= 744 (1.25 x 595); reported for every c.
 F2 (fidelity) for t <= 500, dK_e and W_ext of each run agree with the c = 4 record within 1e-3 of the running max.
 F3 (ledger/status) ledger <= 1e-6 of max|W_ext| and solver status success through the window.
 Verdict: the strong-closure route is "useful" only if F1 and F2 and F3 hold at one c; otherwise recorded as failed (an extension
 with broken fidelity is not a result).  No B10 rerun (it needed the scaled closure).

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
TOL = 1e-3
TINT, DT = 150.0, 5.0
GRID = [(K, N) for K in (2, 5, 10, 21) for N in (32, 64, 128)] + [(5, 256), (10, 256)]
PEAKS = (1e-3, 4e-3)
SHAPES = ("single", "flat", "inv")
LADDER = (0.0, 4.0, 16.0, 64.0, 256.0, 1024.0)
HELD = [(K, N, 2e-3, sh, A0) for (K, N) in ((7, 96), (15, 48)) for sh in ("single", "inv")]


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
    """Asymptotic growth rate (v2 definition) of exp((L - nu diag s) t); name kept from v1."""
    import scipy.sparse.linalg as sla
    M = L.tocsc()
    if nu:
        M = M - sp.diags(np.tile(nu * shape_s(Nn), nk))
    v = np.random.default_rng(1).standard_normal(M.shape[0]) + 0j
    v /= np.linalg.norm(v)
    logs = []
    for _ in range(int(round(TINT / DT))):
        v = sla.expm_multiply(M * DT, v)
        nrm = np.linalg.norm(v)
        logs.append(np.log(nrm))
        v /= nrm
    return float(np.mean(logs[-6:]) / DT)


def point(args):
    K, Nn, peak, shape, a = args
    L = operator(K, Nn, field(K, peak, shape), a)
    base = np.sqrt(2 * Nn) * peak / a
    ladder = (0.0, 4.0, 1024.0) if (K, Nn) in ((7, 96), (15, 48)) else LADDER
    g = {c: maxre(L, c * base, Nn, 2 * K + 1) for c in ladder}
    ok = [c for c in ladder if g[c] <= TOL]
    return {"K": K, "Nn": Nn, "peak": peak, "shape": shape, "a": float(a), "gamma": {str(c): v for c, v in g.items()},
            "c_req": ok[0] if ok else None}


def measure(workers=4):
    import multiprocessing as mp
    pts = [(K, N, p, s, A0) for (K, N) in GRID for p in PEAKS for s in SHAPES if s != "flat" or K <= 10] + HELD
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
