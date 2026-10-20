"""R3 kinetic runs with Dark-SPECTRAX (gates and channel definitions: r3_gates.py, declared before any run).

Usage: python studies/r3_kin.py eta w theta pol [--Nx 512] [--Nn 4] [--amp 1] [--nopacket] [--rtol 1e-10] [--tag name]
theta is 'none' (B0 = 0) or degrees. Writes studies/r3/kin_<name>.json (slim record).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import darkspectrax as ds
from darkspectrax._diagnostics import moments
from darkspectrax._model import rhs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_gates as g  # noqa: E402
import r3_ref as ref  # noqa: E402

OUT = Path(__file__).resolve().parent / "r3"
M_FINE = 4096


def make_model(case, Nx, Nn, vte=None):
    a = np.sqrt(2) * (g.VTE if vte is None else vte)
    return ds.Model(Nx=Nx, Nn=Nn, Nm=Nn, Np=Nn, Lx=g.LB, qs=(-1.0,), Omega_cs=(1.0,), alpha_s=(a, a, a), u_s=(0.0, 0.0, 0.0),
                    rho_background=0.0, mode="self_consistent", eta=case.eta, Omega_D=g.OMEGA_D)


def kept(Nx):
    return (Nx - 1) // 3  # parent strict 2/3 mask keeps 3 j < Nx


def background(model, case, vte=None):
    """Electron Maxwellian n(x) in the static force-balance field of the code (solved on the discrete C_1 row), residual
    charge dE0/dx and its dark near field. Returns (state, info)."""
    Nx, J = model.Nx, kept(model.Nx)
    x = np.arange(Nx) * (g.LB / Nx)
    n = g.density(x, case.w)
    A = float(np.prod(np.asarray(model.alpha_s)[:3]))
    y = model.zeros()
    nk = np.fft.rfft(n) / Nx
    nk[J + 1:] = 0.0  # modes outside the 2/3 mask are not evolved consistently by the parent: keep the profile band-limited
    Ck = y["Ck"].at[0, 0, :, 0].set(jnp.asarray(nk / A))
    y = {**y, "Ck": Ck}
    f = jax.jit(lambda yy: rhs(0.0, yy, model)["Ck"][1, 0, :J + 1, 0])
    base = np.asarray(f(y))
    cols = []
    for j in range(1, J + 1):
        for ph in (1.0, 1j):
            yy = {**y, "Fk": y["Fk"].at[0, 0, j, 0].set(ph)}
            cols.append(np.asarray(f(yy)) - base)
    Mx = np.stack([np.concatenate([c.real, c.imag]) for c in cols], axis=1)
    rhsv = -np.concatenate([base.real, base.imag])
    sol, res, rank, sv = np.linalg.lstsq(Mx, rhsv, rcond=None)
    Ebal = np.zeros(Nx // 2 + 1, complex)
    Ebal[1:J + 1] = sol[0::2] + 1j * sol[1::2]
    kj = 2 * np.pi * np.arange(Nx // 2 + 1) / g.LB
    E0 = Ebal / (1 + case.eta ** 2 * kj ** 2 / (kj ** 2 + g.OMEGA_D ** 2))
    rho = 1j * kj * E0
    phi = case.eta * rho / (kj ** 2 + g.OMEGA_D ** 2)
    Fk = y["Fk"].at[0, 0, :, 0].set(jnp.asarray(E0)).at[3:, 0, 0, 0].set(jnp.asarray(case.b0, complex))
    Dk = y["Dk"].at[0, 0, :, 0].set(jnp.asarray(-1j * kj * phi)).at[9, 0, :, 0].set(jnp.asarray(phi))
    y = {**y, "Fk": Fk, "Dk": Dk}
    r = np.asarray(f(y))
    info = {"balance_residual_max": float(np.abs(r).max()), "balance_base_max": float(np.abs(base).max()),
            "E0_max": float(np.abs(np.fft.irfft(E0 * Nx, n=Nx)).max()),
            "dark_near_Ex_max": float(np.abs(np.fft.irfft(-1j * kj * phi * Nx, n=Nx)).max()),
            "lstsq_rank": int(rank), "lstsq_cond": float(sv[0] / sv[-1])}
    return y, info


def add_packet(model, y, case):
    F = ref.packet_coeffs(case)
    Nk = model.Nx // 2 + 1
    m = min(Nk, F.shape[1])
    Dk = y["Dk"].at[:, 0, :m, 0].add(jnp.asarray(F[:, :m]))
    return {**y, "Dk": Dk}


def real_fields(model, Fk, Dk, Ck, M):
    """19 rows (v, E, B, E_D, B_D, A_D, phi_D) on the M-point grid from the stored coefficients; also n(x)."""
    n_k, M_k, _ = moments(model, Ck)

    def to_x(c):
        c = np.asarray(c)[0, :, 0] if c.ndim == 3 else np.asarray(c)
        out = np.zeros(M // 2 + 1, complex)
        m = min(model.Nx // 2, M // 2)
        out[:m] = c[:m]
        return np.fft.irfft(out, n=M) * M
    ne = to_x(n_k[0])
    Mx = np.stack([to_x(M_k[0, i]) for i in range(3)])
    X = np.zeros((19, M))
    X[ref.IV] = Mx / ne
    for i in range(3):
        X[ref.IE.start + i] = to_x(Fk[i])
        X[ref.IB.start + i] = to_x(Fk[3 + i])
        X[ref.IED.start + i] = to_x(Dk[i])
        X[ref.IBD.start + i] = to_x(Dk[3 + i])
        X[ref.IAD.start + i] = to_x(Dk[6 + i])
    X[ref.IPH] = to_x(Dk[9])
    return X, ne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eta", type=float)
    ap.add_argument("w", type=float)
    ap.add_argument("theta")
    ap.add_argument("pol")
    ap.add_argument("--Nx", type=int, default=512)
    ap.add_argument("--Nn", type=int, default=4)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--nopacket", action="store_true")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seg", type=int, default=3)
    ap.add_argument("--vte", type=float, default=g.VTE)
    a = ap.parse_args()
    theta = None if a.theta == "none" else int(a.theta)
    case = ref.Case(a.eta, a.w, theta, a.pol, g.AMP * a.amp)
    name = f"kin_{case.name}_Nx{a.Nx}_Nn{a.Nn}" + ("_amp%g" % a.amp if a.amp != 1 else "") + ("_nopacket" if a.nopacket else "") + ("_vte%g" % a.vte if a.vte != g.VTE else "") + a.tag
    model = make_model(case, a.Nx, a.Nn, a.vte)
    tic = time.perf_counter()
    yb, binfo = background(model, case)
    t_bg = time.perf_counter() - tic
    y0 = yb if a.nopacket else add_packet(model, yb, case)
    Xeq, ne_eq = real_fields(model, yb["Fk"], yb["Dk"], yb["Ck"], M_FINE)
    Ethr = lambda out, i: out["K"][i] * g.LB  # noqa: E731
    xf = np.arange(M_FINE) * (g.LB / M_FINE)
    ia, ib = int(round((g.XU - 7 * case.w) / (g.LB / M_FINE))), int(round((g.XU + 7 * case.w) / (g.LB / M_FINE)))

    def pert(Fk, Dk, Ck):
        X, ne = real_fields(model, Fk, Dk, Ck, M_FINE)
        X[1:] -= Xeq[1:]
        return X, ne
    X0, ne0 = pert(y0["Fk"], y0["Dk"], y0["Ck"])
    uf = lambda X, ne: 0.5 * ((X[ref.IE] ** 2).sum(0) + (X[ref.IB] ** 2).sum(0) + (X[ref.IED] ** 2).sum(0) + (X[ref.IBD] ** 2).sum(0) + g.OMEGA_D ** 2 * ((X[ref.IAD] ** 2).sum(0) + X[ref.IPH] ** 2) + ne * (X[ref.IV] ** 2).sum(0))  # noqa: E731
    dxf = g.LB / M_FINE
    E_in = float(np.sum(uf(X0, ne0)) * dxf) if not a.nopacket else 1.0
    zone_E = lambda X, ne: float((np.sum(uf(X, ne)[ia:ib + 1]) - 0.5 * (uf(X, ne)[ia] + uf(X, ne)[ib])) * dxf)  # noqa: E731
    Ez0 = zone_E(X0, ne0)
    Kb0 = float(np.sum(0.5 * ne0 * (X0[ref.IV] ** 2).sum(0)) * dxf)
    seg_len = g.T_END / a.seg
    nsave = int(round(seg_len / 0.5)) + 1
    t_all, Fa, Fb, Ehist, steps, led, status = [], [], [], [], 0, 0.0, "success"
    y = y0
    cache = {}
    t0 = 0.0
    Ksave = []
    ledger_max = np.zeros(3)
    run_time = 0.0
    n_dev = []
    for s in range(a.seg):
        out = ds.run(model, y, seg_len, n_save=nsave, rtol=a.rtol, atol=1e-14, t0=t0, cache=cache)
        run_time += out["run_time"]
        steps += out["num_steps"]
        ledger_max = np.maximum(ledger_max, np.abs(out["ledger_defect"]).max(axis=1))
        if out["status"] != "success":
            status = out["failure_reason"]
        for i in range(out["t"].size):
            if s > 0 and i == 0:
                continue
            X, ne = pert(out["Fk"][i], out["Dk"][i], out["Ck"][i])
            S, SD, SDp = ref.fluxes(X[:, [ia, ib]])
            t_all.append(float(out["t"][i]))
            Fa.append([S[0], SD[0], SDp[0]])
            Fb.append([S[1], SD[1], SDp[1]])
            Ksave.append(float(out["K"][i]) * g.LB - float(np.sum(0.5 * ne * (X[ref.IV] ** 2).sum(0)) * dxf))
            n_dev.append(float(np.abs(ne - ne_eq).max()))
        last = {k: out[k][-1] for k in ("Fk", "Dk", "Ck", "W")}
        y = {k: jnp.asarray(v).astype(complex) for k, v in last.items()}
        t0 = float(out["t"][-1])
        XT, neT = pert(out["Fk"][-1], out["Dk"][-1], out["Ck"][-1])
        print(f"{name} seg {s} done t={t0} steps={out['num_steps']} status={out['status']} run={out['run_time']:.0f}s", flush=True)
        if out["status"] != "success":
            break
    t_all, Fa, Fb = np.array(t_all), np.array(Fa), np.array(Fb)
    rec = {"name": name, "case": {**case.__dict__, "name": case.name}, "Nx": a.Nx, "Nn": a.Nn, "amp_scale": a.amp,
           "nopacket": a.nopacket, "vte": a.vte, "rtol": a.rtol, "background": binfo, "status": status, "steps": steps,
           "run_time": run_time, "bg_time": t_bg,
           "ledger_max_abs": ledger_max.tolist(), "parent": ds.PARENT_COMMIT, "repository_commit": ds._simulation._git_sha(),
           "command": " ".join(["python", "studies/r3_kin.py"] + sys.argv[1:]), "max_dn_over_nhi": float(max(n_dev) / g.N_HI),
           "max_abs_E_dark_final": float(max(np.abs(XT[ref.IE]).max(), np.abs(XT[ref.IED]).max()))}
    if not a.nopacket:
        n_final = neT
        ch, res = ref.channels(XT, case, M_FINE, n_final, E_in)
        Kth_T = Ksave[-1]
        ch["absorbed"] = (Kth_T - Ksave[0]) / E_in
        ch["closure"] = ref.closure_of(ch)
        EzT = zone_E(XT, neT)
        flux = np.trapezoid(Fa.sum(1) - Fb.sum(1), t_all)
        flux_no = np.trapezoid(Fa[:, :2].sum(1) - Fb[:, :2].sum(1), t_all)
        rec["flux_series"] = {"t": t_all[::1].tolist(), "Fa": (Fa / E_in).tolist(), "Fb": (Fb / E_in).tolist()}
        rec.update({"E_in": E_in, "channels": ch, "branch_windows": res,
                    "ledger_rel_E_in": (ledger_max / E_in).tolist(),
                    "budget": {"zone_dE": (EzT - Ez0) / E_in, "flux_in": flux / E_in, "residual": (EzT - Ez0 - flux) / E_in,
                               "residual_without_phiA": (EzT - Ez0 - flux_no) / E_in}})
        del Kb0
    (OUT).mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("name", "status", "channels", "budget", "ledger_rel_E_in", "max_dn_over_nhi") if k in rec}, indent=1), flush=True)


if __name__ == "__main__":
    main()
