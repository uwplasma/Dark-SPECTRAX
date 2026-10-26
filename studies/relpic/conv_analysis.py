"""picconv analysis: evaluates the gates declared in conv_gates.py (thresholds read from there).
    python studies/relpic/conv_analysis.py  ->  studies/relpic/conv_summary.json
Units: E_th = 5e-4.  Missing records are skipped and listed."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conv_gates as C  # noqa: E402
from analysis import dW, load, trailing  # noqa: E402

ETH = 5e-4


def vals(tags, at, fn=dW):
    out = []
    for t in tags:
        h = load(t)
        if h is not None and h["t"][-1] >= at - 1e-6:
            out.append(fn(h, at) / ETH)
    return np.array(out)


def wek(h, at):
    return trailing(h["t"], h["nonzero_electric"], at)


def we0(h, at):
    return trailing(h["t"], h["mean_E"] ** 2 / 2, at)


def stat(tags, at, fn=dW):
    v = vals(tags, at, fn)
    return dict(n=len(v), mean=float(v.mean()) if len(v) else None,
                std=float(v.std(ddof=1)) if len(v) > 1 else None,
                se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None, values=v.tolist())


def peak(tags, key):
    v = [float(load(t)[key].max()) / ETH if key != "mean_E" else float((load(t)[key] ** 2 / 2).max()) / ETH
         for t in tags if load(t) is not None]
    return float(np.mean(v)) if v else None


def fit_inf(x, y, p):
    """y = y_inf + c x^-p (ppc) or c x^p (dx): linear least squares in (1, x^-p); returns (y_inf, y at x_ref coefficient)."""
    A = np.vstack([np.ones_like(x), x ** (-p)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def main():
    out = dict(gates_declared_in="studies/relpic/conv_gates.py", E_th=ETH, HHS=C.HHS)
    F = [f"F412_s{i}" for i in C.SEEDS_FINE]
    R = [f"R_s{i}" for i in range(3)]
    Nn = ["N412_s0", "N412_s1"]
    groups = dict(R103=R, P206=["P206"], F412=F, P824=["P824"], P1648=["P1648"], X2=["X2"], X1=["X1"], Q412=["Q412"],
                  Q103=["Q_s0", "Q_s1"], D=["D"], N103=["N_s0", "N_s1", "N_s2"], N412=Nn, P4old=["P4"])
    obs = {}
    for g, tg in groups.items():
        obs[g] = {str(int(at)): dict(dW_th=stat(tg, at), W_E0=stat(tg, at, we0), W_Ek=stat(tg, at, wek)) for at in (1000.0, 2000.0)}
        obs[g]["peak_W_Ek"] = peak(tg, "nonzero_electric")
        obs[g]["peak_W_E0"] = peak(tg, "mean_E")
    out["observables"] = obs
    g = {}
    for at in ("1000", "2000"):
        f = obs["F412"][at]["dW_th"]
        sig = f["std"] if f["n"] > 1 else None
        rs = obs["R103"][at]["dW_th"]["std"]

        def cmp(name, a, b, rel, sg):
            if a is None or b is None:
                return dict(status="missing")
            tol = max(rel * abs(a), C.SIGMA * np.sqrt(2) * (sg or 0.0))
            return dict(a=a, b=b, diff=a - b, tol=float(tol), ok=bool(abs(a - b) <= tol))

        m = lambda k: obs[k][at]["dW_th"]["mean"]
        if at == "1000":
            g["C1a_412_to_824"] = cmp("c1a", m("P824"), f["mean"], C.C1_REL, sig)
            g["C1b_824_to_1648"] = cmp("c1b", m("P1648"), m("P824"), C.C1_REL, sig)
            g["C2_dx_0.02_to_0.01"] = cmp("c2", m("X1"), m("X2"), C.C2_REL, rs)
            g["C3_shape_Q412"] = cmp("c3", m("Q412"), f["mean"], C.C3_REL, sig)
            g["C4_dt_D"] = cmp("c4", m("D"), obs["R103"][at]["dW_th"]["mean"], C.C4_REL, rs)
    out["gates"] = g
    # C5: extrapolation at t = 1000
    ext = {}
    at = "1000"
    pp = [(103, obs["R103"][at]["dW_th"]["mean"]), (206, obs["P206"][at]["dW_th"]["mean"]), (412, obs["F412"][at]["dW_th"]["mean"]),
          (824, obs["P824"][at]["dW_th"]["mean"]), (1648, obs["P1648"][at]["dW_th"]["mean"])]
    pp = [(x, y) for x, y in pp if y is not None]
    dd = [(0.04, obs["R103"][at]["dW_th"]["mean"]), (0.02, obs["X2"][at]["dW_th"]["mean"]), (0.01, obs["X1"][at]["dW_th"]["mean"])]
    dd = [(x, y) for x, y in dd if y is not None]
    if len(pp) >= 3:
        x, y = map(np.array, zip(*pp))
        for p in C.FIT_P:
            c = fit_inf(x, y, p)
            ext[f"ppc_p{p}"] = dict(W_inf=float(c[0]), c=float(c[1]), at412=float(c[0] + c[1] * 412 ** (-p)),
                                    resid_rms=float(np.sqrt(np.mean((c[0] + c[1] * x ** (-p) - y) ** 2))))
    if len(dd) >= 3:
        x, y = map(np.array, zip(*dd))
        A = np.vstack([np.ones_like(x), x ** 2]).T
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        ext["dx_p2"] = dict(W_inf=float(c[0]), c=float(c[1]), at_0p04=float(c[0] + c[1] * 0.04 ** 2))
    f = obs["F412"][at]["dW_th"]
    if f["mean"] is not None and "ppc_p1.0" in ext and "ppc_p0.5" in ext:
        ra = [ext[f"ppc_p{p}"]["W_inf"] / ext[f"ppc_p{p}"]["at412"] for p in C.FIT_P]
        fac_ppc = float(np.mean(ra)); half = abs(ra[0] - ra[1]) / 2
        fac_dx = ext["dx_p2"]["W_inf"] / ext["dx_p2"]["at_0p04"] if "dx_p2" in ext else None
        W = f["mean"] * fac_ppc * (fac_dx if fac_dx else 1.0)
        unc = np.sqrt((f["se"] or 0) ** 2 + (f["mean"] * half) ** 2 + (f["mean"] * abs(fac_dx - 1) if fac_dx else 0) ** 2)
        ext["C5"] = dict(W_inf=float(W), uncertainty=float(unc), factor_ppc=fac_ppc, factor_dx=fac_dx, seed_se=f["se"],
                         ppc_model_half_spread_rel=float(half))
        ext["C6_vs_HHS_t1000"] = dict(W_inf=float(W), HHS=C.HHS["t1000"], ratio=float(W / C.HHS["t1000"]),
                                      ok=bool(abs(W / C.HHS["t1000"] - 1) <= C.HHS["tolerance"]))
    n = obs["N412"]
    for t in ("1000", "2000"):
        if n[t]["dW_th"]["mean"] is not None:
            ext[f"C6_newtonian_t{t}"] = dict(N412=n[t]["dW_th"]["mean"], HHS=C.HHS[f"t{t}"], ratio=n[t]["dW_th"]["mean"] / C.HHS[f"t{t}"],
                                               ok=bool(abs(n[t]["dW_th"]["mean"] / C.HHS[f"t{t}"] - 1) <= C.HHS["tolerance"]))
            fr = obs["F412"][t]["dW_th"]
            if fr["mean"] is not None and n[t]["dW_th"]["se"] is not None:
                ext[f"C7_ratio_RoverN_t{t}"] = dict(ratio=fr["mean"] / n[t]["dW_th"]["mean"], R=fr["mean"], N=n[t]["dW_th"]["mean"])
    out["extrapolation"] = ext
    (HERE / "conv_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:7000])


if __name__ == "__main__":
    main()
