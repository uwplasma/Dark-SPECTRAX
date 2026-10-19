"""stiffdiag: what limits the adaptive step at the failure of the H05 v_q/v_te = 0.1 Hermite runs?  Stiffness or instability?

DECLARED BEFORE RUNNING (never amended after results; a wrong gate is recorded as a failure and a follow-up is declared).

Question.  The H05 runs (v_q/v_te 0.1, Nx 64, Nn 128, c = 4 ends at t ~ 595; c = 64 ends at ~ 760) end when the Dopri8 step
collapses (step budget) just after the positivity check fails.  Is the step limited by
 (a) STIFFNESS: eigenvalues of the RHS Jacobian J = d rhs/d y so large (|Im lambda| from streaming k v_max, or Re lambda << 0
     from the closure) that the explicit Dopri8 stability region, not accuracy, sets the step; the solution may stay smooth;
     curable by an exponential / implicit treatment of the linear block; or
 (b) TRUE INSTABILITY: eigenvalues with Re lambda > 0 (AW truncation) giving genuine exponential growth of the top Hermite
     modes, which the accuracy controller must follow; not curable by an integrator.

Method (script `measure`, one process per case, run on checkpoints of a production run made with the same drivers as
studies/gpures_*.py, noise_floor 1e-14, rtol 1e-10, atol 1e-14, 20-unit segments, part files = checkpoints):
 * State vector v = real unknowns of the dealiased (2/3) Fourier-Hermite state: Re/Im of Ck, Fk at the kept kx (Im at k = 0
   dropped), plus the 6 basis velocities u_s of the pump frame (alpha_s, the ledger W and the unused Dk have zero RHS or no
   feedback: they add only zero eigenvalues and are dropped).  J = jax.linearize of the full RHS (including the closure
   wrapper of nxconv_run.install_closure) at the checkpoint; dense J by JVPs on unit vectors (n ~ 11k).
 * Linear part J_L = J linearized about the spatially homogeneous part of the state (all k != 0 content of Ck and Fk set to
   zero; k = 0 content, basis and drive kept): streaming + closure + Maxwell + current + force from the k = 0 field.  It is
   block diagonal in k (off-block leakage measured); each k block (524 columns) is diagonalised exactly (dense).
   Field-nonlinear part J_NL = J - J_L (mode coupling E_k' d/dv f_k-k' and the dependence of the closure on the field);
   its spectral radius and max Re lambda are from ARPACK on the operator v -> J v - J_L v (LM and LR), operator 2-norm by
   svds at the dense checkpoints.
 * Spectra of J: dense eigvals at the dense checkpoints (t = 300, the second to last and the last), ARPACK (k = 12, LM and LR)
   at every checkpoint, compared where both exist.
 * Dopri8 stability function R(z) = 1 + z b^T (I - zA)^-1 1 from the diffrax tableau; h_stab(lambda) = the largest h with
   |R(s lambda)| <= 1 for all s in (0, h] (eigenvalues with Re > 0 are put on the imaginary-axis direction: their growth is
   physical, only their oscillation limits the explicit step); h_stab_min = min over the eigenvalues of J.
 * Step history: accepted steps and rejections per 20-unit segment from the run records (dt_acc = segment length / accepted
   steps).  Tolerance test from each checkpoint: 1 time unit with rtol in {1e-10, 1e-8, 1e-6, 1e-4}, same atol and noise
   floor: a step set by accuracy scales like rtol^(-1/8) (steps(1e-10)/steps(1e-4) = 10^(6/8) = 5.6); a step set by the
   explicit stability bound does not depend on the tolerance (ratio 1).
 * Growth: top-third electron Hermite energy E3(t) (n >= 2 Nn/3 summed over k) from the run, amplitude rate
   0.5 d ln E3/dt over the 10 units before the checkpoint; predicted: max Re lambda(J) and the finite-time tangent growth
   (exp(5 J) power iteration, 4 repeats, dense checkpoints) of the top-third part.

Checkpoints.  Healthy reference t_ref = 300; failure point t_f = the end of the run (c = 4: ~595; c = 64: ~760); near-failure =
t >= t_f - 40.

Hypotheses (decided by the gates below, at every near-failure checkpoint of a case):
 HA (stiffness-limited)    dt_acc / h_stab_min >= 0.3 AND steps(1e-10) / steps(1e-4) <= 1.5.
 HB (instability present)  max Re lambda(J) >= 0.01 omega_pe AND the measured top-third amplitude rate is within a factor 2 of
                           the finite-time tangent growth rate at the dense checkpoints.
 HC (accuracy-limited)     dt_acc / h_stab_min <= 0.05 AND steps(1e-10) / steps(1e-4) >= 3.
Decomposition of the collapse C = dt_acc(t_ref) / dt_acc(t_f):
 f_stab = clip( ln[h_stab_min(t_ref) / h_stab_min(t_f)] / ln C, 0, 1 )   (share of the step collapse explained by the shrinking
                                                                       stability bound of the Jacobian spectrum)
 f_tol  = 1 - clip( ln[steps(1e-10)/steps(1e-4)] / (0.75 ln 10), 0, 1 )   (share of the step at t_f that does not respond to the
                                                                       tolerance, i.e. stiffness-type)
 Both are reported; the curable fraction is stated as the interval [min, max] of the two, the rest is not curable by an integrator
 (accuracy needs the full resolution of a rough or growing solution).
Gates:
 S0 (consistency of the numerics)  JVP vs central finite difference on a random direction rel. error <= 1e-5; J_L off-block
    leakage <= 1e-10 of ||J_L||_F; where dense and ARPACK both exist: |rho_dense - rho_arpack| <= 1e-3 rho_dense and
    |abscissa_dense - abscissa_arpack| <= 1e-3 + 1e-3 |abscissa_dense|.  A failure is recorded and the affected quantity is
    reported as unverified.
 S1 verdict per case: "STIFFNESS-LIMITED" if HA holds at every near-failure checkpoint; "INSTABILITY/ACCURACY-LIMITED" if HC holds
    at every near-failure checkpoint and max Re lambda(J) >= 0.01 at the last one; otherwise "MIXED" with f_stab and f_tol reported.
 S2 (pass criterion for the question) the two cases give the same S1 verdict; otherwise the verdict is per case.
Eigenvalues of the strongly non-normal operator are round-off sensitive (rclosure: 0.756 dense vs 0.846 ARPACK vs 0.702
time-domain); hence the time-domain tangent growth and the tolerance test are the primary evidence and the eigenvalue bounds
secondary, and the report states this.

Runs: studies/stiffdiag_batch.sh CASE (production to failure, checkpoints), then
  python studies/stiffdiag_gates.py measure CASE  -> studies/stiffdiag/<case>.json   (CASE in c4, c64)
  python studies/stiffdiag_gates.py analyze       -> studies/stiffdiag/summary.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
RUNS = D / "nxconv" / "runs"
OUT = D / "stiffdiag"
CASES = {"c4": ("vq0.1_Nx64_Nn128_c4_nf1e-14_sd", 4.0, (100, 300, 500, 540, 560, 580), (300, 560)),
         "c64": ("vq0.1_Nx64_Nn128_c64_nf1e-14_sd", 64.0, (100, 300, 500, 700, 720, 740), (300, 720))}
SEG = 20.0
RTOLS = (1e-10, 1e-8, 1e-6, 1e-4)


def parts_of(case):
    ps = sorted(RUNS.glob(f"{case}.part*.json"), key=lambda p: int(p.suffixes[-2][5:]))
    return [(json.loads(p.read_text()), RUNS / (p.name[:-5] + ".npz")) for p in ps]


def dopri8_stability():
    """Return (coeffs of R, table of critical |z| versus direction angle theta in [pi/2, 3pi/2])."""
    import diffrax
    tb = diffrax.Dopri8().tableau
    s = len(tb.b_sol)
    A = np.zeros((s, s))
    for i, row in enumerate(tb.a_lower):
        A[i + 1, :len(row)] = np.asarray(row)
    b = np.asarray(tb.b_sol, float)
    one, r, M = np.ones(s), [1.0], np.eye(s)
    for _ in range(s + 1):
        r.append(float(b @ M @ one))
        M = A @ M
    r = np.array(r)  # R(z) = sum r_j z^j
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 721)
    zs = np.linspace(1e-3, 30.0, 30000)
    crit = np.empty_like(th)
    for i, t in enumerate(th):
        z = zs * np.exp(1j * t)
        R = np.abs(np.polyval(r[::-1], z))
        bad = np.nonzero(R > 1.0)[0]
        crit[i] = zs[bad[0]] if bad.size else zs[-1]
    return r, th, crit


def h_stab(lam, th, crit):
    """Explicit stability step bound for each eigenvalue (Re > 0 put on the imaginary-axis direction)."""
    lam = np.asarray(lam, complex)
    mag = np.abs(lam)
    ang = np.where(lam.real > 0, np.pi / 2 * np.sign(lam.imag + 1e-300) % (2 * np.pi), np.angle(lam) % (2 * np.pi))
    ang = np.where(ang < np.pi / 2, np.pi - ang, ang)  # conjugate symmetry: map into [pi/2, 3pi/2]
    ang = np.where(ang > 3 * np.pi / 2, 3 * np.pi - ang, ang)
    c = np.interp(ang, th, crit)
    with np.errstate(divide="ignore"):
        return np.where(mag > 1e-12, c / np.maximum(mag, 1e-300), np.inf)


class System:
    """Real-vector view of the dealiased state and the RHS of the (closure-wrapped) Dark model at a fixed time."""

    def __init__(self, m, y, t):
        import jax.numpy as jnp
        self.m, self.t, self.y = m, float(t), {k: jnp.asarray(v) for k, v in y.items()}
        self.kk = np.nonzero(np.asarray(m.p["mask23"])[0, :, 0] > 0)[0]
        assert self.kk[0] == 0
        self.R, self.K = self.y["Ck"].shape[0], len(self.kk)
        R, K = self.R, self.K
        self.sl = {"reC": (0, R * K), "imC": (R * K, R * K + R * (K - 1))}
        o = self.sl["imC"][1]
        self.sl["reF"] = (o, o + 6 * K)
        o += 6 * K
        self.sl["imF"] = (o, o + 6 * (K - 1))
        o += 6 * (K - 1)
        self.sl["u"] = (o, o + 6)
        self.n = o + 6

    def _cx(self, re, im, rows):
        import jax.numpy as jnp
        im = jnp.concatenate([jnp.zeros((rows, 1)), im.reshape(rows, self.K - 1)], axis=1)
        return re.reshape(rows, self.K) + 1j * im

    def unpack(self, v, base):
        s = self.sl
        g = lambda k: v[s[k][0]:s[k][1]]  # noqa: E731
        C = self._cx(g("reC"), g("imC"), self.R)
        F = self._cx(g("reF"), g("imF"), 6)
        kk = self.kk
        out = dict(base)
        out["Ck"] = base["Ck"].at[:, 0, kk, 0].set(C)
        out["Fk"] = base["Fk"].at[:, 0, kk, 0].set(F)
        out["B"] = base["B"].at[0].set(g("u").astype(base["B"].dtype))
        return out

    def pack(self, y):
        import jax.numpy as jnp
        kk = self.kk
        C, F = y["Ck"][:, 0, kk, 0], y["Fk"][:, 0, kk, 0]
        return jnp.concatenate([C.real.ravel(), C.imag[:, 1:].ravel(), F.real.ravel(), F.imag[:, 1:].ravel(),
                                jnp.real(y["B"][0])])

    def func(self, base):
        from darkspectrax import _simulation as _sim

        def f(v):
            return self.pack(_sim.rhs(self.t, self.unpack(v, base), self.m))
        return f

    def background(self):
        y = dict(self.y)
        y["Ck"] = y["Ck"].at[:, 0, 1:, 0].set(0.0)
        y["Fk"] = y["Fk"].at[:, 0, 1:, 0].set(0.0)
        return y

    def dense(self, base, chunk=384):
        import jax
        import jax.numpy as jnp
        f = self.func(base)
        v0 = self.pack(base)
        _, lin = jax.linearize(f, v0)
        vl = jax.jit(jax.vmap(lin))
        cols = []
        for a in range(0, self.n, chunk):
            idx = jnp.arange(a, min(a + chunk, self.n))
            cols.append(np.asarray(vl(jnp.eye(self.n)[idx])))
        return np.concatenate(cols, 0).T  # J[:, j] = lin(e_j)

    def operator(self, base):
        import jax
        f = self.func(base)
        _, lin = jax.linearize(f, self.pack(base))
        lj = jax.jit(lin)
        return lambda x: np.asarray(lj(np.asarray(x, float)))


def arnoldi(mv, n, which, k=12, ncv=80):
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs
    op = LinearOperator((n, n), matvec=mv, dtype=float)
    try:
        return eigs(op, k=k, which=which, ncv=ncv, tol=1e-8, maxiter=3000, return_eigenvectors=False), True
    except ArpackNoConvergence as e:
        return e.eigenvalues, False


def spec_summary(lam):
    lam = np.asarray(lam, complex)
    if lam.size == 0:
        return {}
    return {"rho": float(np.abs(lam).max()), "abscissa": float(lam.real.max()), "min_re": float(lam.real.min()),
            "max_abs_im": float(np.abs(lam.imag).max())}


def block_spectra(J, sysm):
    """Exact per-k eigenvalues of J_L (blocks: k = 0 holds Ck0, Fk0, u; k > 0 holds Ck_k, Fk_k)."""
    from scipy.linalg import eigvals
    R, K, s = sysm.R, sysm.K, sysm.sl
    idx = []
    for ik in range(K):
        ii = list(range(s["reC"][0] + ik * 0, s["reC"][0]))  # placeholder, replaced below
        ii = [s["reC"][0] + r * K + ik for r in range(R)] + [s["reF"][0] + r * K + ik for r in range(6)]
        if ik > 0:
            ii += [s["imC"][0] + r * (K - 1) + ik - 1 for r in range(R)] + [s["imF"][0] + r * (K - 1) + ik - 1 for r in range(6)]
        else:
            ii += list(range(s["u"][0], s["u"][1]))
        idx.append(np.array(ii))
    inblock = np.zeros(J.shape, bool)
    eigs_all, per_k = [], []
    for ik, ii in enumerate(idx):
        inblock[np.ix_(ii, ii)] = True
        lam = eigvals(J[np.ix_(ii, ii)])
        eigs_all.append(lam)
        per_k.append({"k_index": int(sysm.kk[ik]), **spec_summary(lam)})
    leak = float(np.linalg.norm(np.where(inblock, 0.0, J)) / max(np.linalg.norm(J), 1e-300))
    return np.concatenate(eigs_all), per_k, leak


def tangent_growth(J, sysm, nhe, tau=5.0, reps=4, seed=1):
    """Power iteration of exp(tau J): norm and top-third electron Hermite amplitude growth rates (last reps-1 repeats)."""
    from scipy.sparse.linalg import expm_multiply
    R, K, s = sysm.R, sysm.K, sysm.sl
    top = np.arange(R)
    top = top[(top < nhe) & (top >= (2 * nhe) // 3)]  # electron rows, n >= 2Nn/3

    def e3(v):
        re = v[s["reC"][0]:s["reC"][1]].reshape(R, K)[top]
        im = np.concatenate([np.zeros((R, 1)), v[s["imC"][0]:s["imC"][1]].reshape(R, K - 1)], 1)[top]
        return float((re ** 2 + im ** 2).sum())
    v = np.random.default_rng(seed).standard_normal(sysm.n)
    v /= np.linalg.norm(v)
    rec, A = [], J * tau
    for _ in range(reps):
        e0 = e3(v)
        w = expm_multiply(A, v)
        rec.append((float(np.log(np.linalg.norm(w)) / tau), 0.5 * float(np.log(max(e3(w), 1e-300) / max(e0, 1e-300))) / tau))
        v = w / np.linalg.norm(w)
    return {"norm_rate": float(np.mean([r[0] for r in rec[1:]])), "top3_amp_rate": float(np.mean([r[1] for r in rec[1:]])),
            "per_repeat": rec}


def measured_top3_rate(pr_list, t_c, nhe, win=10.0):
    ts, es = [], []
    for info, npz in pr_list:
        if info["t_end"] < t_c - win - 1 or info["t_start"] > t_c + 1e-9:
            continue
        z = np.load(npz)
        sel = (z["t"] >= t_c - win - 1e-9) & (z["t"] <= t_c + 1e-9)
        ts += list(z["t"][sel])
        es += list(z["herm_n"][sel, 0, (2 * nhe) // 3:].sum(-1))
    if len(ts) < 4:
        return None
    ts, es = np.array(ts), np.array(es)
    return float(0.5 * np.polyfit(ts, np.log(np.maximum(es, 1e-300)), 1)[0])


def segment_steps(info, npz, t_c):
    """(dt_acc, accepted, rejected, length) of the 20-unit segment of this part that ends at t_c."""
    st = np.load(npz)["seg_stats"]
    j = len(st) - 1
    t0 = info["t_start"] + SEG * j
    L = t_c - t0
    acc = st[j, 0] - st[j, 1]
    return {"dt_acc": float(L / acc), "accepted": int(acc), "rejected": int(st[j, 1]), "length": float(L)}


def measure(key):
    import jax
    import jax.numpy as jnp
    from scipy.linalg import eigvals
    from scipy.sparse.linalg import svds

    import darkspectrax as ds
    from lane_c_run import model
    from nxconv_run import install_closure
    case, c, want, dense_at = CASES[key]
    OUT.mkdir(exist_ok=True)
    pr = parts_of(case)
    assert pr, f"no parts for {case}"
    t_f = pr[-1][0]["t_end"]
    sel = [(i, info, npz) for i, (info, npz) in enumerate(pr)
           if any(abs(info["t_end"] - w) < 1e-6 for w in want) or i == len(pr) - 1]
    m = model(0.1, 128, 64)
    install_closure(m, c)
    _, th, crit = dopri8_stability()
    res = {"case": case, "c": c, "t_f": t_f, "dopri8_imag_limit": float(np.interp(np.pi / 2, th, crit)),
           "dopri8_real_limit": float(np.interp(np.pi, th, crit)), "checkpoints": []}
    for ip, info, npz in sel:
        tic = time.perf_counter()
        t_c, z = info["t_end"], np.load(npz)
        y = {k: jnp.asarray(z["y_" + k]) for k in ("Ck", "Fk", "Dk", "W", "B")}
        S = System(m, y, t_c)
        rec = {"t": t_c, "n": S.n, "dense": any(abs(t_c - d) < 1e-6 for d in dense_at) or ip == len(pr) - 1,
               **segment_steps(info, npz, t_c)}
        base = S.y
        f = S.func(base)
        v0 = S.pack(base)
        w = np.random.default_rng(2).standard_normal(S.n)
        w *= np.linalg.norm(v0) * 1e-3 / np.linalg.norm(w)
        mvf = S.operator(base)
        eps = 1e-3
        fd = (np.asarray(f(v0 + eps * w)) - np.asarray(f(v0 - eps * w))) / (2 * eps)
        rec["fd_rel_err"] = float(np.linalg.norm(mvf(w) - fd) / max(np.linalg.norm(fd), 1e-300))
        mv_full = mvf
        mv_bg = S.operator(S.background())
        for which in ("LM", "LR"):
            lam, ok = arnoldi(mv_full, S.n, which)
            rec[f"arpack_{which}"] = {**spec_summary(lam), "converged": ok}
            lam, ok = arnoldi(lambda x: mv_full(x) - mv_bg(x), S.n, which)
            rec[f"arpack_NL_{which}"] = {**spec_summary(lam), "converged": ok}
        if rec["dense"]:
            J = S.dense(base)
            JL = S.dense(S.background())
            lamJ = eigvals(J, check_finite=False)
            lamL, per_k, leak = block_spectra(JL, S)
            rec["dense_J"] = spec_summary(lamJ)
            rec["J_L"] = {**spec_summary(lamL), "leakage": leak, "worst_k_abscissa": max(per_k, key=lambda d: d["abscissa"]),
                          "worst_k_rho": max(per_k, key=lambda d: d["rho"])}
            JN = J - JL
            rec["J_NL"] = {"fro": float(np.linalg.norm(JN)), "op2": float(svds(JN, k=1, return_singular_vectors=False)[0]),
                           "fro_J": float(np.linalg.norm(J)), "fro_J_L": float(np.linalg.norm(JL))}
            hJ, hL = h_stab(lamJ, th, crit), h_stab(lamL, th, crit)
            rec["h_stab_min_J"], rec["h_stab_min_J_L"] = float(hJ.min()), float(hL.min())
            rec["n_unstable_J"] = int((lamJ.real > 1e-6).sum())
            rec["n_unstable_J_L"] = int((lamL.real > 1e-6).sum())
            rec["tangent"] = tangent_growth(J, S, m.Nn)
            del J, JL, JN
        else:
            lam = np.asarray(arnoldi(mv_full, S.n, "LM")[0])
            rec["h_stab_min_J_arpack_LM"] = float(h_stab(lam, th, crit).min())
        # tolerance test
        rec["tol_test"] = []
        for r in RTOLS:
            o = ds.run(m, y, 1.0, n_save=2, rtol=r, atol=1e-14, noise_floor=1e-14, t0=t_c, max_steps=300000)
            rec["tol_test"].append({"rtol": r, "steps": o["num_steps"], "accepted": o["num_accepted"], "rejected": o["num_rejected"],
                                    "status": o["status"], "run_s": o["run_time"]})
        rec["top3_amp_rate_measured"] = measured_top3_rate(pr, t_c, m.Nn)
        rec["wall_s"] = time.perf_counter() - tic
        res["checkpoints"].append(rec)
        (OUT / f"{key}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps({k: rec[k] for k in ("t", "dt_acc", "fd_rel_err", "wall_s")}, default=float), flush=True)
    return 0


def analyze():
    out = {}
    for key in CASES:
        p = OUT / f"{key}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        cps, t_f = r["checkpoints"], r["t_f"]
        near = [c for c in cps if c["t"] >= t_f - 40]
        ref = next((c for c in cps if abs(c["t"] - 300) < 1e-6), None)
        last = cps[-1]

        def ratio(c):
            return c["dt_acc"] / c["h_stab_min_J"] if "h_stab_min_J" in c else c["dt_acc"] / c["h_stab_min_J_arpack_LM"]

        def sratio(c):
            s = {x["rtol"]: x["steps"] for x in c["tol_test"]}
            return s[1e-10] / s[1e-4]
        rows = [{"t": c["t"], "dt_acc": c["dt_acc"], "dt_over_hstab": ratio(c), "steps_ratio_1e-10_over_1e-4": sratio(c),
                 "abscissa_arpack": c["arpack_LR"].get("abscissa"), "abscissa_dense": c.get("dense_J", {}).get("abscissa"),
                 "rho_dense": c.get("dense_J", {}).get("rho")} for c in cps]
        HA = all(ratio(c) >= 0.3 and sratio(c) <= 1.5 for c in near)
        HC = all(ratio(c) <= 0.05 and sratio(c) >= 3 for c in near)
        ab = last.get("dense_J", {}).get("abscissa", last["arpack_LR"].get("abscissa"))
        growth = [(c["t"], c["top3_amp_rate_measured"], c["tangent"]["top3_amp_rate"]) for c in cps if "tangent" in c]
        HB = bool(ab is not None and ab >= 0.01 and all(m is not None and 0.5 <= m / g <= 2 for _, m, g in growth if g > 0))
        verdict = "STIFFNESS-LIMITED" if HA else ("INSTABILITY/ACCURACY-LIMITED" if (HC and ab is not None and ab >= 0.01) else "MIXED")
        d = {"t_f": t_f, "rows": rows, "HA": HA, "HB": HB, "HC": HC, "verdict": verdict, "growth_measured_vs_tangent": growth}
        if ref is not None and "h_stab_min_J" in ref and "h_stab_min_J" in last:
            C = ref["dt_acc"] / last["dt_acc"]
            d["collapse_C"] = C
            d["f_stab"] = float(np.clip(np.log(ref["h_stab_min_J"] / last["h_stab_min_J"]) / np.log(C), 0, 1)) if C > 1 else None
        s = sratio(last)
        d["f_tol"] = float(1 - np.clip(np.log(s) / (0.75 * np.log(10)), 0, 1))
        d["S0"] = {"fd_rel_err_max": max(c["fd_rel_err"] for c in cps),
                   "leak_max": max((c["J_L"]["leakage"] for c in cps if "J_L" in c), default=None),
                   "dense_vs_arpack": [{"t": c["t"], "rho": (c["dense_J"]["rho"], c["arpack_LM"].get("rho")),
                                        "abscissa": (c["dense_J"]["abscissa"], c["arpack_LR"].get("abscissa"))}
                                       for c in cps if "dense_J" in c]}
        out[key] = d
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        sys.exit(measure(sys.argv[2]))
    analyze()
