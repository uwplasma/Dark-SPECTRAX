"""r1nl analysis: implements the rules declared in studies/r1nl_gates.py (no rule is changed here).

python studies/r1nl_analysis.py stage1   -> studies/r1nl/stage1_summary.json   (certified windows, v*)
python studies/r1nl_analysis.py stage2   -> studies/r1nl/stage2_summary.json   (detuning, reservoir, coherence at v*)
python studies/r1nl_analysis.py slim     -> drop the restart states (y_*) from the run records (they are not analysis data)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1_common as c1  # noqa: E402
import r1nl_gates as G  # noqa: E402

D = Path(__file__).resolve().parent / "r1nl"
RUNS = D / "runs"
GRID = Path(__file__).resolve().parent / "nxconv" / "grid"
VT2 = c1.vte ** 2
TS = ("t", "K_species", "Th", "Tx", "Q", "W", "U_gamma", "U_D", "Ebar", "EDbar", "Ek2")


def name(kind, vq, mem, **kw):
    m = G.MEMBERS[mem]
    s = f"{kind}_vq{vq:g}_Nx{m['Nx']}_Nn{m['Nn']}_c{m['c']:g}"
    if kind == "phase":
        s += f"_tau{kw['tau']:g}_r{kw['real']}"
    if kind == "sc":
        s += f"_u{kw['u']:g}"
    return s


def load(case):
    parts = sorted(RUNS.glob(f"{case}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    if not parts:
        return None
    h, recs = {}, []
    for i, p in enumerate(parts):
        d = np.load(p)
        recs.append(json.loads(Path(str(p)[:-4] + ".json").read_text()))
        for k in TS:
            h.setdefault(k, []).append(d[k] if i == 0 else d[k][1:])
    h = {k: np.concatenate(v) for k, v in h.items()}
    h["recs"], h["case"] = recs, case
    h["t_reached"] = float(h["t"][-1])
    h["status"], h["failure"] = recs[-1]["status"], recs[-1]["failure_reason"]
    h["dKe"] = h["K_species"][:, 0] - h["K_species"][0, 0]
    h["dKi"] = h["K_species"][:, 1] - h["K_species"][0, 1]
    h["Wext"] = h["W"][:, 2]
    scale = max(np.abs(h["W"][:, 2]).max(), np.abs(h["W"][:, 1]).max(), 1e-300)
    h["ledger"] = float(np.abs(h["K_species"].sum(1) - h["K_species"][0].sum() - h["W"].sum(1)).max() / scale)
    h["first_nonpositive"] = c1.first_nonpositive(h)
    h["steps"] = int(sum(r["steps"] for r in recs))
    h["wall"] = float(sum(r["wall_time"] for r in recs))
    return h


def cap(h):
    caps = [h["t_reached"]] + ([h["first_nonpositive"]] if h["first_nonpositive"] is not None else [])
    return min(caps)


def pair_time(a, b, keys):
    n = min(a["t"].size, b["t"].size)
    t = a["t"][:n]
    res = []
    for k in keys:
        x = c1.disagreement(t, a[k][:n], b[k][:n])
        res.append(x if x is not None else float(t[-1]))
    return min(res)


def ds_t(x):
    return None if x is None else float(x)


def stage1():
    S = {"vqs": {}, "gates_doc": "studies/r1nl_gates.py"}
    for vq in G.V_LIST:
        H = {m: load(name("pre", vq, m)) for m in G.MEMBERS}
        if any(h is None for h in H.values()):
            S["vqs"][f"{vq:g}"] = {"missing": [m for m, h in H.items() if h is None]}
            continue
        pairs = {f"{a}-{b}": pair_time(H[a], H[b], ("dKe", "Wext")) for a, b in G.PAIRS}
        pairs_by_obs = {f"{a}-{b}": {k: ds_t(c1.disagreement(H[a]["t"][:min(H[a]['t'].size, H[b]['t'].size)],
                                                             H[a][k][:min(H[a]['t'].size, H[b]['t'].size)],
                                                             H[b][k][:min(H[a]['t'].size, H[b]['t'].size)])) for k in ("dKe", "Wext")}
                        for a, b in G.PAIRS}
        ledger = {m: h["ledger"] for m, h in H.items()}
        win_parts = {"cap": 1000.0, **{f"pair {k}": v for k, v in pairs.items()},
                     **{f"cap {m}": cap(h) for m, h in H.items()}}
        window = min(win_parts.values())
        limiter = min(win_parts, key=win_parts.get)
        cert = bool(max(ledger.values()) <= G.LEDGER_TOL and window > 50)
        grid = {}
        for nx in (32, 64):
            p = GRID / f"g_vq{vq:g}_Nx{nx}_Nv4096_dt0.02_osc_v32_spectral_xmask23.npz"
            if p.exists():
                g = np.load(p)
                gd = {"t": g["t"], "dKe": g["dK_e"], "Wext": g["W_ext"]}
                n = min(gd["t"].size, H["B"]["t"].size)
                grid[f"grid_Nx{nx}_vs_B"] = {"t_reached_grid": float(g["t"][-1]),
                                             "agree_until": min(ds_t(c1.disagreement(H["B"]["t"][:n], H["B"][k][:n], gd[k][:n])) or float(H["B"]["t"][n - 1]) for k in ("dKe", "Wext"))}
        S["vqs"][f"{vq:g}"] = {
            "window": window, "limiter": limiter, "certified": cert, "ledger": ledger, "pairs": pairs, "pairs_by_observable": pairs_by_obs,
            "t_reached": {m: h["t_reached"] for m, h in H.items()}, "first_nonpositive": {m: h["first_nonpositive"] for m, h in H.items()},
            "status": {m: h["status"] for m, h in H.items()}, "failure": {m: h["failure"] for m, h in H.items()},
            "steps": {m: h["steps"] for m, h in H.items()}, "wall_s": {m: h["wall"] for m, h in H.items()}, "grid": grid}
        print(vq, "window", window, "limiter", limiter, "cert", cert, pairs, flush=True)
    ok = {v: s for v, s in S["vqs"].items() if s.get("certified")}
    strong = [v for v, s in ok.items() if s["window"] >= G.W_MIN_STRONG]
    if strong:
        S["vstar"] = max(strong, key=float)
    elif ok:
        S["vstar"] = max(ok, key=lambda v: ok[v]["window"])
    S["vstar_rule"] = "largest certified v_q with Window >= 300; else the largest Window"
    (D / "stage1_summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
    print("v*", S.get("vstar"))


# ---------------------------------------------------------------- stage 2 helpers
def lin_fields(t, E0, w0=c1.w_L):
    """Homogeneous mean-field ODE: Ebar, W_ext for the drive E0 cos(w0 t) (the exact linear law)."""
    from scipy.integrate import solve_ivp
    wL2 = c1.w_L ** 2
    s = solve_ivp(lambda tt, x: [-x[1], wL2 * (x[0] + E0 * np.cos(w0 * tt)), x[1] * E0 * np.cos(w0 * tt)], (0, t[-1]), [0, 0, 0],
                  t_eval=t, rtol=1e-12, atol=1e-18, method="DOP853")
    return s.y[0], s.y[2]


def lockin_phase(t, E, w=c1.w_L, P=5 * 2 * np.pi / c1.w_L):
    z = E * np.exp(-1j * w * t)
    I = np.concatenate([[0], np.cumsum(0.5 * (z[1:] + z[:-1]) * np.diff(t))])
    Ir = np.interp(t - P, t, I.real) + 1j * np.interp(t - P, t, I.imag)
    ph = np.angle(I - Ir)
    ph[t < P] = np.nan
    return ph


def slope(t, y, lo, hi):
    s = (t >= lo) & (t <= hi) & np.isfinite(y)
    return float(np.polyfit(t[s], y[s], 1)[0]) if s.sum() > 3 else None


def departure(t, Dm, win, thr_b=G.TOL10, thr_r=G.TOL5, mask=None):
    """N1 rule. Dm: dict member -> D(t). Departure iff some t: |D_B| > thr_b and |D_m| > thr_r for all refinement members."""
    sel = (t > 50) & (t <= win)
    if mask is not None:
        sel &= mask
    refs = [k for k in Dm if k != "B"]
    cond = sel & (np.abs(Dm["B"]) > thr_b)
    for k in refs:
        cond &= np.abs(Dm[k]) > thr_r
    bound = float(max(np.abs(Dm[k][sel]).max() for k in Dm)) if sel.any() else None
    return {"departure": bool(cond.any()), "t_first": float(t[np.argmax(cond)]) if cond.any() else None, "bound_max_abs_D": bound,
            "max_abs_D_by_member": {k: (float(np.abs(Dm[k][sel]).max()) if sel.any() else None) for k in Dm},
            "D_at_window_end": {k: float(np.interp(win, t, Dm[k])) for k in Dm}}


def otsi(h, Nx):
    E = h["Ek2"]
    kmax = Nx // 3 - 1
    den = E[:, 1:kmax + 1].sum(1)
    num = E[:, 4:kmax + 1].sum(1)
    t = h["t"]
    frac = num / np.maximum(den, 1e-300)
    bad = (frac >= 0.1) & (t > 100)
    to = float(t[np.argmax(bad)]) if bad.any() else None
    gr = None
    if to is not None:
        s = (t >= to - 50) & (t <= to + 50)
        gr = float(np.polyfit(t[s], np.log(np.maximum(num[s], 1e-300)), 1)[0])
    return {"t_otsi": to, "growth_rate_energy_of_k>=4_sum_within_50": gr, "k_range": [1, kmax]}


def stage2():
    S1 = json.loads((D / "stage1_summary.json").read_text())
    v = float(S1["vstar"])
    win = S1["vqs"][f"{v:g}"]["window"]
    E0 = v * c1.vte
    S = {"vstar": v, "window": win, "E0": E0, "N1": {}, "N2": {}, "N3": {}, "N4": {}, "N5": {}}
    H = {m: load(name("pre", v, m)) for m in ("B", "X", "N", "XN")}
    t = H["B"]["t"]
    nmin = min(h["t"].size for h in H.values())
    t = t[:nmin]
    Eb_lin, W_lin = lin_fields(t, E0)
    sel = t > 50
    Dm = {m: np.where(sel, H[m]["Wext"][:nmin] / np.where(np.abs(W_lin) > 0, W_lin, np.nan) - 1, 0.0) for m in H}
    S["N1"] = departure(t, Dm, win)
    S["N1"]["Wext_over_nTe_at_window"] = {m: float(np.interp(win, t, H[m]["Wext"][:nmin]) / VT2) for m in H}
    S["N1"]["Wext_lin_over_nTe_at_window"] = float(np.interp(win, t, W_lin) / VT2)
    # N2 frequency shift
    ph_lin = lockin_phase(t, Eb_lin)
    dom, lo = {}, min(0.8 * win, win - 4 * 2 * np.pi / c1.w_L)
    for m in H:
        dph = np.unwrap(np.nan_to_num(lockin_phase(t, H[m]["Ebar"][:nmin]) - ph_lin, nan=0.0))
        dph[~np.isfinite(ph_lin)] = np.nan
        dom[m] = slope(t, dph, lo, win)
    spread = max(abs(dom[m] - dom["B"]) for m in H if m != "B")
    S["N2"] = {"delta_omega_by_member": dom, "refinement_spread": spread, "lo": lo, "delta_omega_over_wL_B": dom["B"] / c1.w_L,
               "resolved": bool(abs(dom["B"]) / c1.w_L > 1e-3 and abs(dom["B"]) > 3 * spread),
               "bound": abs(dom["B"]) + spread, "Ebar_amp_ratio_kin_over_lin_at_window":
                   {m: float(np.abs(H[m]["Ebar"][:nmin][(t > win - 31) & (t <= win)]).max() / np.abs(Eb_lin[(t > win - 31) & (t <= win)]).max()) for m in H}}
    # N3 partition
    def at(h, key):
        return float(np.interp(win, h["t"], key) / VT2)
    part = {}
    for m, h in H.items():
        Nx = G.MEMBERS[m]["Nx"]
        part[m] = {"dKe": at(h, h["dKe"]), "dKi": at(h, h["dKi"]), "Wext": at(h, h["Wext"]),
                   "dU_gamma": at(h, h["U_gamma"] - h["U_gamma"][0]), "dTh_e": at(h, h["Th"][:, 0] - h["Th"][0, 0]),
                   "dTh_i": at(h, h["Th"][:, 1] - h["Th"][0, 1]),
                   "ion_share": float(np.interp(win, h["t"], h["dKi"]) / np.interp(win, h["t"], h["Wext"])),
                   "Qi_max_abs": float(np.abs(h["Q"][h["t"] <= win, 1]).max()),
                   "Qi_rms_last20pct": float(np.sqrt(np.mean(h["Q"][(h["t"] > 0.8 * win) & (h["t"] <= win), 1] ** 2))),
                   "Qi_corr_with_Ebar_last20pct": float(np.corrcoef(h["Q"][(h["t"] > 0.8 * win) & (h["t"] <= win), 1],
                                                                    h["Ebar"][(h["t"] > 0.8 * win) & (h["t"] <= win)])[0, 1]),
                   "Ek_fluct_max_ratio_to_E0sq": float(h["Ek2"][h["t"] <= win, 1:].sum(1).max() / E0 ** 2), **otsi(h, Nx)}
    S["N3"] = part
    # N4 reservoir
    for u in G.U_LIST:
        Sb, Sx = load(name("sc", v, "B", u=u)), load(name("sc", v, "XN", u=u))
        if Sb is None or Sx is None:
            S["N4"][f"u{u:g}"] = {"missing": True}
            continue
        eta, U0 = Sb["recs"][0]["eta"], float(Sb["U_D"][0])
        n = min(Sb["t"].size, Sx["t"].size)
        ts = Sb["t"][:n]
        wsc = min(win, cap(Sb), cap(Sx), pair_time(Sb, Sx, ("dKe",)), pair_time({"t": Sb["t"], "WD": Sb["W"][:, 1]}, {"t": Sx["t"], "WD": Sx["W"][:, 1]}, ("WD",)))
        X, UDlin = c1.coupled_solution(ts, eta, E0 / eta)
        dev = {m: np.abs(h["U_D"][:n] - UDlin) / U0 for m, h in (("B", Sb), ("XN", Sx))}
        s_ = (ts <= wsc)
        t_dev = float(ts[np.argmax(s_ & (dev["B"] > 0.1))]) if (s_ & (dev["B"] > 0.1)).any() else None
        WDlin = U0 - UDlin
        mask = WDlin > 0.2 * np.maximum.accumulate(WDlin)
        Dres = {m: np.where(mask & (ts > 50), h["W"][:n, 1] / np.where(np.abs(WDlin) > 0, WDlin, np.nan) - 1, 0.0) for m, h in (("B", Sb), ("XN", Sx))}
        dp = departure(ts, Dres, wsc, mask=mask)
        # t_split vs matched coherent drive
        tw = min(wsc, cap(H["B"]))
        tt = np.array([x for x in H["B"]["t"] if x <= tw])
        Wp = H["B"]["Wext"][:tt.size]
        WDk = np.interp(tt, Sb["t"], Sb["W"][:, 1])
        bad = (np.abs(WDk - Wp) > 0.1 * np.abs(Wp)) & (tt > 50)
        t_sp = float(tt[np.argmax(bad)]) if bad.any() else None
        _, Wl = lin_fields(tt, E0)
        badl = (np.abs(np.interp(tt, ts, WDlin) - Wl) > 0.1 * np.abs(Wl)) & (tt > 50)
        t_spl = float(tt[np.argmax(badl)]) if badl.any() else None
        S["N4"][f"u{u:g}"] = {
            "eta": eta, "U_D0_over_nTe": U0 / VT2, "window_sc": wsc, "t_reached": [Sb["t_reached"], Sx["t_reached"]],
            "first_nonpositive": [Sb["first_nonpositive"], Sx["first_nonpositive"]], "ledger": [Sb["ledger"], Sx["ledger"]],
            "max_dev_UD_over_UD0": {m: float(d[s_].max()) for m, d in dev.items()}, "t_dev_10pct": t_dev, "G5like_pass": bool(dev["B"][s_].max() <= 0.1),
            "WD_departure": dp, "WD_over_nTe_at_window": float(np.interp(wsc, Sb["t"], Sb["W"][:, 1]) / VT2),
            "WD_over_UD0_at_window_kin_lin": [float(np.interp(wsc, ts, Sb["W"][:n, 1]) / U0), float(np.interp(wsc, ts, WDlin) / U0)],
            "min_UD_over_UD0_kin": float(Sb["U_D"][:n][s_].min() / U0), "min_UD_over_UD0_lin": float(UDlin[s_].min() / U0),
            "t_split_kin": t_sp, "t_split_lin": t_spl,
            "t_split_pass": bool((t_sp is None and t_spl is None) or (t_sp is not None and t_spl is not None and 1 / 1.5 <= t_sp / t_spl <= 1.5)),
            "ion_share_at_window": float(np.interp(wsc, Sb["t"], Sb["dKi"]) / np.interp(wsc, Sb["t"], Sb["W"][:, 1]))}
    # N5 coherence
    for tau in G.TAU_LIST:
        R = [load(name("phase", v, "B", tau=tau, real=r)) for r in range(G.NREAL)]
        Rx = load(name("phase", v, "XN", tau=tau, real=0))
        if any(r is None for r in R) or Rx is None:
            S["N5"][f"tau{tau:g}"] = {"missing": [i for i, r in enumerate(R) if r is None] + (["XN"] if Rx is None else [])}
            continue
        wc = min([win] + [cap(r) for r in R] + [cap(Rx)])
        tt = R[0]["t"]
        nm = min(r["t"].size for r in R)
        tt = tt[:nm]
        ratios, Ds, Dxx = [], [], None
        late = (tt >= 0.8 * wc) & (tt <= wc)
        for r, h in enumerate(R):
            z = c1.noise_knots(tau, r, fac=2.0)
            Wl = c1.linear_work_realization(tt, z, E0)
            Dr = np.where(tt > 50, h["Wext"][:nm] / np.where(np.abs(Wl) > 0, Wl, np.nan) - 1, 0.0)
            Ds.append(float(np.nanmax(np.abs(Dr[(tt > 50) & (tt <= wc)]))))
            ratios.append(float(h["Wext"][:nm][late].mean() / Wl[late].mean()))
            if r == 0:
                Wl0 = Wl
        nx = min(nm, Rx["t"].size)
        Dx0 = float(np.interp(wc, tt[:nx], Rx["Wext"][:nx] / np.where(np.abs(Wl0[:nx]) > 0, Wl0[:nx], np.nan)) - 1)
        ratios = np.array(ratios)
        Rm, se = float(ratios.mean()), float(ratios.std(ddof=1) / np.sqrt(ratios.size))
        dep = bool(abs(Rm - 1) > 0.05 and abs(Rm - 1) > 3 * se and np.sign(Dx0) == np.sign(Rm - 1) and abs(Dx0) > 0.025)
        S["N5"][f"tau{tau:g}"] = {
            "window_coh": wc, "R_by_realization": ratios.tolist(), "R": Rm, "se": se, "departure": dep, "XN_r0_D_at_window": Dx0,
            "bound_if_none": abs(Rm - 1) + 2 * se, "max_abs_D_by_realization": Ds, "ledger_max": max(r["ledger"] for r in R + [Rx]),
            "late_slope_over_nTe": float(np.polyfit(tt[late], np.mean([h["Wext"][:nm][late] for h in R], axis=0), 1)[0] / VT2),
            "mean_Wext_over_nTe_at_window": float(np.mean([np.interp(wc, tt, h["Wext"][:nm]) for h in R]) / VT2)}
    # N6 ledger of every stage-2 run
    led = {}
    for p in sorted(RUNS.glob("*.part0.npz")):
        h = load(p.name[:-len(".part0.npz")])
        led[h["case"]] = h["ledger"]
    S["N6"] = {"ledger_max": max(led.values()), "pass": bool(max(led.values()) <= G.LEDGER_TOL), "n_runs": len(led)}
    (D / "stage2_summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
    print(json.dumps({k: S[k] for k in ("N1", "N2")}, indent=1, default=float)[:3000])


def slim():
    for p in RUNS.glob("*.npz"):
        d = np.load(p)
        if any(k.startswith("y_") for k in d.files):
            keep = {k: d[k] for k in d.files if not k.startswith("y_")}
            np.savez_compressed(p, **keep)


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2, "slim": slim}[sys.argv[1]]()
