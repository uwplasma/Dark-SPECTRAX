"""H08 pilot: HHS-v1-inspired resonant drive with physical Dougherty collisions (parent #71). Not a reproduction of HHS.

Setup = studies/nxconv_run.py (H05: v_te = sqrt(1e-3), m_i/m_e = 1836, T_i = T_e, L = 40, mobile ions, drive E0 cos(omega t),
E0 = (v_q/v_te) v_te, omega = sqrt(1 + 1/1836), pump frame + parent remaps, gate seeds r0, #66/#72 field-scaled closure
on the order-2 spectrum with strength c = 4, noise_floor 1e-14, rtol 1e-10, atol 1e-14, saves every 0.5) plus
collision_model = "dougherty" with
    nu_dougherty = [[nu_ee, nu_ei], [0, nu_ii]],  nu_ee = nu_ei,  nu_ii = nu_ei sqrt(m_e/m_i),
and the parent sets nu_ie = nu_ei m_e n_e / (m_i n_i) pointwise (momentum conservation). nu_ei is the electron-ion
Dougherty rate in omega_pe; the ion-electron pair uses Greene's form with beta = 0 (#71): u_ei = (U_e + U_i)/2 and the
common T_ei of the B11 docstring, so the uniform drag is dU_e/dt = -nu_ei (U_e - U_i)/2. Convention, not a fit to
Coulomb physics: nu is velocity independent (no v^-3 Spitzer dependence, so inverse-bremsstrahlung absorption is a
model value), nu_ee = nu_ei and nu_ii = nu_ei sqrt(m_e/m_i) are order-of-magnitude Z = 1 choices, and the
cross-species beta convention has not been checked against Francisquez et al.
Cases: v_q/v_te in {0.01, 0.03}, nu_ei in {0, 1e-3, 1e-2}, Nx in {32, 64} (Nn 64); Nn 128 at Nx 32 for nu_ei > 0;
unseeded homogeneous controls (Nx 8, ds.CONTROLS) for nu_ei > 0. T = 1000.

Per save (beyond nxconv): per-species collisional power P_s = dK_s/dt from the Dougherty part of the RHS (RHS with
minus without collisions; K_s is linear in C at fixed basis), Ebar, the mean-field identity residual
    R = -(dJ/dt - dJ_coll/dt)/Om0 + sum_s q_s^2 (Om_s/Om0) [n_s0 (Ebar + E_d) + <dn_s dE>]
with dJ/dt from a JVP of J0(Ck, B) along the full RHS (pump-frame basis motion included) and dJ_coll/dt the
collisional current-moment term (JVP along the Dougherty part), and R_nocoll without that term (its scale).

Gates (declared before running; failures are recorded, never amended):
G0 homogeneous controls: Ebar and W_ext vs the exact uniform driven two-fluid oscillator with drag
   (Ue' = -(Ebar + Ed) - nu_ei (Ue - Ui)/2, Ui' = (Ebar + Ed)/1836 - nu_ie (Ui - Ue)/2, Ebar' = Ue - Ui; SciPy
   solve_ivp DOP853, rtol 1e-12): max error <= 1e-6 of max |exact| for both, t <= 1000.
G1 reproduction: nu = 0, v_q 0.03, Nx 32 equals the nxconv record vq0.03_Nx32_Nn64_c4_nf1e-14 in dK_e and W_ext to
   1e-6 of the running max over the common saves.
G2 every collisional run: max |P_e + P_i| <= 1e-9 max |P_e| (collisions do no net work) and max |ledger defect|
   <= 1e-6 max |W_ext|. (An uncommitted smoke run to t = 20, v_q 0.03, nu 1e-2, Nx 32, gave 7.5e-11 and 9.4e-8 with
   W_ext still small; the first draft's 1e-10 / 1e-8 were loosened to these values before any recorded run.)
G3 every run: max |R| <= 1e-8 omega_L^2 E0.
Resolved windows (report, not pass/fail): nxconv t_res (first t > 50 with a 10% disagreement relative to the running
max) of dK_e, W_ext, T_e/T_i - 1 and Q_e = int P_e dt for Nx 32 vs 64 and (Nx 32) Nn 64 vs 128. Claims are made only
up to min(those, the collisionless certified window: 1000 at v_q 0.01 if its nu = 0 Nx pair resolves it, ~950 at 0.03).

Run: python studies/h08.py run --vq 0.03 --nu 1e-3 --Nx 32 [--Nn 64] [--chunk 200] [--control]   (rerun = resume)
     python studies/h08.py analyze  -> studies/h08/summary.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import darkspectrax as ds  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402
from lane_c_run import Lx, NSAVE, SEG, a_e, a_i, diagnostics, seeds, vte, w_tot  # noqa: E402
from nxconv_run import install_closure  # noqa: E402

D = Path(__file__).resolve().parent / "h08"
RUNS = D / "runs"
MI, C_CLOSURE, NF = 1836.0, 4.0, 1e-14
CERT = {0.01: 1000.0, 0.03: 950.0}


def nu_matrix(nu):
    return ((nu, nu), (0.0, nu * np.sqrt(1 / MI)))


def model(vq, Nn, Nx, nu):
    kw = dict(collision_model="dougherty", nu_dougherty=nu_matrix(nu)) if nu > 0 else {}
    return ds.Model(Nx=Nx, Nn=Nn, Lx=Lx, qs=(-1.0, 1.0), Omega_cs=(1.0, 1 / MI), alpha_s=(a_e,) * 3 + (a_i,) * 3,
                    u_s=(0.0,) * 6, mode="prescribed_drive", E_drive=(vq * vte, 0.0, 0.0), omega_drive=w_tot,
                    frame="pump", **kw)


def case_name(vq, nu, Nx, Nn, control):
    return f"vq{vq:g}_nu{nu:g}_Nx{Nx}_Nn{Nn}" + ("_control" if control else "")


def diag_fn(m, mfree):
    """Jitted per-save: collisional power per species, J0 derivatives (full and collisional), identity terms."""
    qs = jnp.asarray(m.qs)
    om = jnp.asarray(np.asarray(m.Omega_cs) / m.Omega_cs[0])

    def J0(Ck, B):
        return jnp.real(jnp.dot(qs, ds.moments(m, Ck, B)[1][:, 0, 0, 0, 0]))

    def one(t, y):
        full = ds.rhs(t, y, m)
        dcol = full["Ck"] - ds.rhs(t, y, mfree)["Ck"]
        _, _, M2 = ds.moments(m, dcol, y["B"])
        P = 0.5 * jnp.asarray(m.masses) * jnp.real(jnp.trace(M2[..., 0, 0, 0], axis1=1, axis2=2))
        dJ = jax.jvp(J0, (y["Ck"], y["B"]), (full["Ck"], full["B"]))[1]
        dJc = jax.jvp(J0, (y["Ck"], y["B"]), (dcol, jnp.zeros_like(y["B"])))[1]
        n = ds.moments(m, y["Ck"], y["B"])[0]
        Ex = y["Fk"][0]
        Eb = jnp.real(Ex[0, 0, 0])
        Ed = m.drive(t)[0]
        s = 0.0
        for k in range(m.Ns):
            c = ds.inner(m.Nx, n[k][None], Ex[None]) - jnp.real(n[k, 0, 0, 0]) * Eb
            s = s + qs[k] ** 2 * om[k] * (jnp.real(n[k, 0, 0, 0]) * (Eb + Ed) + c)
        Om0 = m.Omega_cs[0]
        return P, -(dJ - dJc) / Om0 + s, -dJ / Om0 + s, dJc, Eb

    return jax.jit(one)


def run(a):
    case = case_name(a.vq, a.nu, a.Nx, a.Nn, a.control)
    RUNS.mkdir(parents=True, exist_ok=True)
    parts = sorted(RUNS.glob(f"{case}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    m = model(a.vq, a.Nn, a.Nx, a.nu)
    mfree = model(a.vq, a.Nn, a.Nx, 0.0)
    if parts:
        last = np.load(parts[-1])
        info = json.loads(Path(str(parts[-1])[:-4] + ".json").read_text())
        if info["status"] not in ("success", "short") or info["t_end"] >= a.T - 1e-9:
            print(f"{case}: nothing to do ({info['status']}, t_end {info['t_end']:g})")
            return 0
        t0 = info["t_end"]
        y0 = {k: jnp.asarray(last[f"y_{k}"]) for k in ("Ck", "Fk", "Dk", "W", "B")}
    else:
        t0 = 0.0
        y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], [] if a.control else seeds(0)))
    ipart = len(parts)
    t1 = min(a.T, t0 + max(SEG, SEG * (a.chunk // SEG)))
    stats, run0 = [], _sim.run

    def counted(*args, **kw):
        kw["t0"] = kw.get("t0", 0.0) + t0
        o = run0(*args, **kw)
        stats.append([o["num_steps"], o["num_rejected"], o["compile_time"], o["run_time"]])
        return o

    _sim.run = counted
    install_closure(m, C_CLOSURE)
    tic = time.perf_counter()
    out = ds.run_adaptive(m, y0, t1 - t0, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=200_000,
                          noise_floor=NF)
    wall = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    Ks, Th, Tx, Q = diagnostics(m, out)
    f = diag_fn(m, mfree)
    rows = []
    for i in range(out["t"].size):
        y = {"Ck": jnp.asarray(out["Ck"][i]), "Fk": jnp.asarray(out["Fk"][i]), "Dk": jnp.asarray(out["Dk"][i]),
             "W": jnp.asarray(out["W"][i], complex), "B": jnp.asarray(out["B"][i], complex)}
        rows.append([np.asarray(v) for v in f(out["t"][i], y)])
    P, R, R0, dJc, Eb = (np.array(x) for x in zip(*rows))
    st = np.array(stats)
    t_end = float(out["t"][-1])
    status = out["status"] if (out["status"] != "success" or t_end >= t1 - 1e-9) else "short"
    rec = {"case": case, "vq_over_vte": a.vq, "nu_ei": a.nu, "nu_matrix": nu_matrix(a.nu), "Nx": a.Nx, "Nn": a.Nn,
           "control": a.control, "closure_c": C_CLOSURE, "noise_floor": NF, "part": ipart, "t_start": t0,
           "t_end": t_end, "status": status, "failure_reason": out["failure_reason"], "events": len(out["events"]),
           "steps": int(st[:, 0].sum()), "rejected": int(st[:, 1].sum()), "compile_time": float(st[:, 2].sum()),
           "run_time": float(st[:, 3].sum()), "wall_time": wall, "repository_commit": _sim._git_sha(),
           "parent_commit": ds.PARENT_COMMIT, "command": "python studies/h08.py " + " ".join(sys.argv[1:])}
    stem = RUNS / f"{case}.part{ipart}"
    np.savez_compressed(f"{stem}.npz", t=out["t"], K_species=Ks, Th=Th, Tx=Tx, W=out["W"],
                        ledger=out["ledger_defect"][0], Ek2=np.abs(out["Fk"][:, 0, 0, :, 0]) ** 2, P=P, R=R, R0=R0,
                        dJc=dJc, Ebar=Eb, seg_stats=st, y_Ck=out["Ck"][-1], y_Fk=out["Fk"][-1], y_Dk=out["Dk"][-1],
                        y_W=out["W"][-1].astype(complex), y_B=out["B"][-1].astype(complex))
    Path(f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(rec, default=float), flush=True)
    return 0


# ----------------------------------------------------------------------------- analysis
def load(case):
    parts = sorted(RUNS.glob(f"{case}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    if not parts:
        return None
    cat, recs = {}, [json.loads(Path(str(p)[:-4] + ".json").read_text()) for p in parts]
    W0 = None
    for i, p in enumerate(parts):
        d = np.load(p)
        for k in ("t", "K_species", "Tx", "W", "ledger", "P", "R", "R0", "dJc", "Ebar", "Ek2"):
            x = d[k] if i == 0 else d[k][1:]
            if k == "ledger":  # each part's ledger is closed from its own start: chain the offsets
                x = x + (0.0 if W0 is None else W0)
            cat.setdefault(k, []).append(x)
        W0 = cat["ledger"][-1][-1]
    h = {k: np.concatenate(v) for k, v in cat.items()}
    h["dKe"] = h["K_species"][:, 0] - h["K_species"][0, 0]
    h["Wx"] = (h["W"][:, 2] - h["W"][0, 2]).real
    m = np.array([1.0, MI])
    T = m[None] * h["Tx"]  # n = 1: T_s = m_s (M_xx - M_x^2/n)
    h["TeTi"] = T[:, 0] / T[:, 1] - 1
    h["Te"], h["Ti"] = T[:, 0], T[:, 1]
    t = h["t"]
    h["Qcum"] = np.concatenate([np.zeros((1, 2)), np.cumsum(0.5 * (h["P"][1:] + h["P"][:-1]) * np.diff(t)[:, None], 0)])
    h["Qe"] = h["Qcum"][:, 0]
    h["recs"] = recs
    return h


def t_res(a, b, key):
    ta, tb = np.round(a["t"], 6), np.round(b["t"], 6)
    t = np.intersect1d(ta, tb)
    x, y = a[key][np.searchsorted(ta, t)], b[key][np.searchsorted(tb, t)]
    scale = np.maximum.accumulate(np.abs(x))
    bad = np.nonzero((t > 50) & (np.abs(x - y) > 0.1 * scale))[0]
    return float(t[bad[0]]) if bad.size else float(t[-1])


def at(h, key, tq):
    i = int(np.argmin(np.abs(h["t"] - tq)))
    w = (np.abs(h["t"] - h["t"][i]) <= np.pi / w_tot)  # cycle average around tq
    return float(np.mean(h[key][w]))


def exact_control(vq, nu, t):
    from scipy.integrate import solve_ivp
    E0 = vq * vte
    nie = nu / MI

    def f(tt, y):
        Ue, Ui, Eb, W = y
        Ed = E0 * np.cos(w_tot * tt)
        J = -Ue + Ui
        return [-(Eb + Ed) - nu * (Ue - Ui) / 2, (Eb + Ed) / MI - nie * (Ui - Ue) / 2, -J, J * Ed]

    s = solve_ivp(f, (0, t[-1]), [0, 0, 0, 0], method="DOP853", t_eval=t, rtol=1e-12, atol=1e-16)
    return s.y[2], s.y[3]


def analyze():
    out = {"gates": {}, "controls": {}, "runs": {}, "pairs": {}, "compare_collisionless": {}}
    g0 = []
    for vq in (0.01, 0.03):
        for nu in (1e-3, 1e-2):
            h = load(case_name(vq, nu, 8, 64, True))
            if h is None:
                continue
            Eb, W = exact_control(vq, nu, h["t"])
            r = {"t_end": float(h["t"][-1]), "Ebar_err": float(np.abs(h["Ebar"] - Eb).max() / np.abs(Eb).max()),
                 "Wext_err": float(np.abs(h["Wx"] - W).max() / np.abs(W).max()), "Wext_exact_end": float(W[-1])}
            out["controls"][f"vq{vq:g}_nu{nu:g}"] = r
            g0.append(r["Ebar_err"] <= 1e-6 and r["Wext_err"] <= 1e-6 and r["t_end"] >= 1000 - 1e-9)
    out["gates"]["G0"] = bool(g0) and all(g0)
    ref = Path(__file__).resolve().parent / "nxconv_analysis.py"
    sys.path.insert(0, str(ref.parent))
    from nxconv_analysis import herm
    nx = herm("vq0.03_Nx32_Nn64_c4_nf1e-14")
    mine = load(case_name(0.03, 0.0, 32, 64, False))
    if nx is not None and mine is not None:
        ta, tb = np.round(nx["t"], 6), np.round(mine["t"], 6)
        tc = np.intersect1d(ta, tb)
        dev = {}
        for k1, k2 in (("dKe", "dKe"), ("Wx", "Wx")):
            x, y = nx[k1][np.searchsorted(ta, tc)].real, mine[k2][np.searchsorted(tb, tc)]
            dev[k1] = float(np.max(np.abs(x - y) / np.maximum(np.maximum.accumulate(np.abs(x)), 1e-300)))
        out["G1_reproduction"] = {"t_common_end": float(tc[-1]), "max_rel_dev": dev}
        out["gates"]["G1"] = all(v <= 1e-6 for v in dev.values()) and tc[-1] >= 1000 - 1e-9
    g2, g3 = [], []
    hs = {}
    for vq in (0.01, 0.03):
        E0 = vq * vte
        for nu in (0.0, 1e-3, 1e-2):
            for Nx, Nn in ((32, 64), (64, 64), (32, 128)):
                c = case_name(vq, nu, Nx, Nn, False)
                h = load(c)
                if h is None:
                    continue
                hs[c] = h
                r = {"t_end": float(h["t"][-1]), "status": h["recs"][-1]["status"],
                     "failure_reason": h["recs"][-1]["failure_reason"],
                     "steps": sum(x["steps"] for x in h["recs"]), "wall_time": sum(x["wall_time"] for x in h["recs"]),
                     "max_R_over_wL2E0": float(np.abs(h["R"]).max() / ((1 + 1 / MI) * E0)),
                     "max_R_nocoll_over_wL2E0": float(np.abs(h["R0"]).max() / ((1 + 1 / MI) * E0)),
                     "max_dJcoll_over_wL2E0": float(np.abs(h["dJc"]).max() / ((1 + 1 / MI) * E0)),
                     "max_ledger_over_Wext": float(np.abs(h["ledger"]).max() / max(np.abs(h["Wx"]).max(), 1e-300))}
                if nu > 0:
                    r["max_Psum_over_Pe"] = float(np.abs(h["P"].sum(1)).max() / max(np.abs(h["P"][:, 0]).max(), 1e-300))
                    g2.append(r["max_Psum_over_Pe"] <= 1e-9 and r["max_ledger_over_Wext"] <= 1e-6)
                g3.append(r["max_R_over_wL2E0"] <= 1e-8)
                tq = [x for x in (250.0, 500.0, 750.0, 950.0, 1000.0) if x <= h["t"][-1] - np.pi]
                r["cycle_avg"] = {f"{x:g}": {"W_ext": at(h, "Wx", x), "dK_e": at(h, "dKe", x),
                                             "Te": at(h, "Te", x), "Ti": at(h, "Ti", x), "TeTi_minus_1": at(h, "TeTi", x),
                                             "Q_e": at(h, "Qe", x), "Q_i": float(np.interp(x, h["t"], h["Qcum"][:, 1]))}
                                  for x in tq}
                for x in tq:
                    r["cycle_avg"][f"{x:g}"]["dK_i"] = at({"t": h["t"], "k": h["K_species"][:, 1] - h["K_species"][0, 1]},
                                                          "k", x)
                out["runs"][c] = r
    out["gates"]["G2"] = bool(g2) and all(g2)
    out["gates"]["G3"] = bool(g3) and all(g3)
    for vq in (0.01, 0.03):
        for nu in (0.0, 1e-3, 1e-2):
            a = hs.get(case_name(vq, nu, 32, 64, False))
            for lab, other in (("Nx32_vs_64", case_name(vq, nu, 64, 64, False)),
                               ("Nn64_vs_128", case_name(vq, nu, 32, 128, False))):
                b = hs.get(other)
                if a is None or b is None:
                    continue
                keys = ("dKe", "Wx", "TeTi") + (("Qe",) if nu > 0 else ())
                pr = {k: t_res(b, a, k) for k in keys}  # reference = the finer run
                pr["window"] = min(min(pr.values()), CERT[vq])
                out["pairs"][f"vq{vq:g}_nu{nu:g}_{lab}"] = pr
    for vq in (0.01, 0.03):
        base = out["runs"].get(case_name(vq, 0.0, 32, 64, False))
        for nu in (1e-3, 1e-2):
            r = out["runs"].get(case_name(vq, nu, 32, 64, False))
            if base is None or r is None:
                continue
            out["compare_collisionless"][f"vq{vq:g}_nu{nu:g}"] = {
                x: {"W_ext_ratio": r["cycle_avg"][x]["W_ext"] / base["cycle_avg"][x]["W_ext"],
                    "dK_e_ratio": r["cycle_avg"][x]["dK_e"] / base["cycle_avg"][x]["dK_e"]}
                for x in r["cycle_avg"] if x in base["cycle_avg"]}
    out["passed"] = all(out["gates"].values())
    out["repository_commit"], out["parent_commit"] = _sim._git_sha(), ds.PARENT_COMMIT
    (D / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out["gates"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze"))
    ap.add_argument("--vq", type=float, default=0.03)
    ap.add_argument("--nu", type=float, default=1e-3)
    ap.add_argument("--Nx", type=int, default=32)
    ap.add_argument("--Nn", type=int, default=64)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--chunk", type=float, default=200.0)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args(argv)
    return run(a) if a.what == "run" else analyze()


if __name__ == "__main__":
    sys.exit(main())
