"""Lane C analysis of studies/lane_c/runs (criteria fixed before reading the records).

- t_res(pair) = first t > 50 where the Nn and 2Nn runs (same realization) differ by > 10% of the running max on
  dK_e or W_ext (the gate's `disagreement`). If none, t_res is censored at the common end (reported as ">=").
- t_pos = first save where any species K_s or k=0 T_x is <= 0.
- Linear reference: undamped uniform two-fluid oscillator driven exactly at resonance from rest,
  W_lin(t) = mu/2 (r'^2 + w^2 r^2), r = -(A/2w) t sin(wt), A = E0(1+eps), mu = 1/(1+eps), eps = m_e/m_i;
  its cycle average is (1+eps) E0^2 [t^2 + 1/w^2] / 8 ~ (1+eps) E0^2 t^2/8 (n = m_e = 1).
- Certified window for a realization/drive = [0, min(t_res(dK_e), t_res(W_ext), t_pos of the finer run)] of the
  finest available pair. Physics numbers are quoted only inside it.
- Q_s = <dn_s dE_x> fit in 60-wide windows to c0 + sum_{h=1,2} (a_h cos h w t + b_h sin h w t); phase relative to
  the pump cos(w t).

Run: python studies/lane_c_analysis.py -> studies/lane_c/summary.json
"""

import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "lane_c"
vte2, eps = 1e-3, 1 / 1836
w = np.sqrt(1 + eps)
VQ = (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1)


def load(vq, Nn, r):
    p = D / "runs" / f"vq{vq:g}_Nn{Nn}_r{r}"
    j, z = Path(f"{p}.json"), Path(f"{p}.npz")
    if not j.exists():
        return None
    return json.loads(j.read_text()), dict(np.load(z))


def W_lin(t, vq):
    A = vq * np.sqrt(vte2) * (1 + eps)
    r, rd = -(A / (2 * w)) * t * np.sin(w * t), -(A / (2 * w)) * (np.sin(w * t) + w * t * np.cos(w * t))
    return 0.5 / (1 + eps) * (rd ** 2 + w ** 2 * r ** 2)


def disagreement(t, a, b, tol=0.1):
    scale = np.maximum(np.maximum.accumulate(np.maximum(np.abs(a), np.abs(b))), 1e-12)
    bad = (np.abs(a - b) > tol * scale) & (t > 50)
    return float(t[np.argmax(bad)]) if bad.any() else None


def t_pos(d):
    bad = (d["K"] <= 0).any(1) | (d["Tx"] <= 0).any(1)
    return float(d["t"][np.argmax(bad)]) if bad.any() else None


def pair(c, f):
    (jc, dc), (jf, df) = c, f
    n = min(dc["t"].size, df["t"].size)
    t = dc["t"][:n]
    out = {"common_end": float(t[-1])}
    for nm, a, b in (("dK_e", dc["K"][:n, 0] - dc["K"][0, 0], df["K"][:n, 0] - df["K"][0, 0]),
                     ("W_ext", dc["W"][:n, 2], df["W"][:n, 2])):
        x = disagreement(t, a, b)
        out[nm] = x if x is not None else float(t[-1])
        out[nm + "_censored"] = x is None
    return out


def at(d, t0, key):
    i = np.searchsorted(d["t"], t0 - 1e-9)
    return d[key][min(i, d["t"].size - 1)]


def qfit(t, q, t1, t0):
    s = (t >= t0) & (t <= t1)
    tt = t[s]
    X = np.stack([np.ones_like(tt)] + [f(h * w * tt) for h in (1, 2) for f in (np.cos, np.sin)], 1)
    c = np.linalg.lstsq(X, q[s], rcond=None)[0]
    return {"dc": float(c[0]), "amp_w": float(np.hypot(c[1], c[2])), "phase_w": float(np.arctan2(-c[2], c[1])),
            "amp_2w": float(np.hypot(c[3], c[4])), "window": [float(t0), float(t1)]}


S = {"scan": {}, "realizations": {}}
for vq in VQ:
    runs = {Nn: load(vq, Nn, 0) for Nn in (64, 128, 256)}
    e = {"runs": {Nn: {k: r[0][k] for k in ("status", "failure_reason", "t_reached", "events", "steps", "rejected",
                                            "compile_time", "run_time", "wall_time", "max_ledger_over_W_ext")}
                       | {"t_pos": t_pos(r[1])} for Nn, r in runs.items() if r}}
    pairs = {f"{a}v{b}": pair(runs[a], runs[b]) for a, b in ((64, 128), (128, 256)) if runs[a] and runs[b]}
    e["pairs"] = pairs
    if pairs:
        key = list(pairs)[-1]
        fine = runs[int(key.split("v")[1])]
        tp = t_pos(fine[1])
        p = pairs[key]
        tc = min(p["dK_e"], p["W_ext"], tp if tp is not None else np.inf)
        d = fine[1]
        e["certified"] = {"pair": key, "t_cert": tc, "censored": bool(p["dK_e_censored"] and p["W_ext_censored"]
                                                                       and tc == p["common_end"])}
        e["heating_at_t_res"] = {nm: float(np.interp(p[nm], d["t"], d["W"][:, 2]) / vte2) for nm in ("dK_e", "W_ext")}
        e["vosc_over_vte_at_t_res"] = {nm: vq * p[nm] / 2 for nm in ("dK_e", "W_ext")}
        sel = d["t"] <= tc
        t = d["t"][sel]
        ratio = d["W"][sel, 2] / np.maximum(W_lin(t, vq), 1e-300)
        late = t > 20
        below = late & (ratio < 0.9)
        e["W_ext_over_W_lin"] = {"at_t_cert": float(ratio[-1]), "min_t_gt_20": float(ratio[late].min()),
                                 "first_below_0.9": float(t[np.argmax(below)]) if below.any() else None}
        P = 2 * np.pi / w
        cyc = sel & (d["t"] > tc - 3 * P)  # last three drive periods inside the certified window
        Wm = d["W"][cyc, 2].mean()
        e["W_ext_over_W_lin"]["cycle_avg_at_t_cert"] = float(Wm / W_lin(d["t"][cyc], vq).mean())
        fit = late & (t > 100)
        e["W_ext_over_W_lin"]["detuning_fit_dw"] = float(np.sqrt(max(np.polyfit(t[fit] ** 2, 1 - ratio[fit], 1)[0], 0) * 12))
        e["partition_at_t_cert"] = {
            "window": [float(d["t"][cyc][0]), float(d["t"][cyc][-1])], "W_ext_over_nTe": float(Wm / vte2),
            **{f"{nm}/W_ext": float(np.mean(x[cyc] - x[0]) / Wm) for nm, x in (
                ("dK_e", d["K"][:, 0]), ("dK_i", d["K"][:, 1]), ("dU_E", d["U_gamma"]),
                ("dThermal_e(k=0)", d["Th"][:, 0]), ("dThermal_i(k=0)", d["Th"][:, 1]))},
            "dThermal_e(k=0) peak-to-peak/W_ext": float(np.ptp(d["Th"][cyc, 0]) / Wm)}
        if tc > 80:
            e["Q_i_last_window"] = qfit(d["t"], d["Q"][:, 1], tc, tc - 60)
            s60 = (d["t"] >= tc - 60) & (d["t"] <= tc)
            e["Q_i_last_window"]["E_mean_amp"] = float(np.sqrt(2 * d["U_gamma"][s60].max()))
            e["Q_i_last_window"]["amp_w_over_E_mean_amp"] = e["Q_i_last_window"]["amp_w"] / e["Q_i_last_window"]["E_mean_amp"]
            e["Q_e_minus_Q_i_max"] = float(np.abs(d["Q"][:, 0] - d["Q"][:, 1]).max())
    S["scan"][f"{vq:g}"] = e
    print(vq, json.dumps({k: e.get(k) for k in ("pairs", "certified", "heating_at_t_res")}, default=float), flush=True)

# seed realizations: spread of the observables at common certified times
for vq in (0.01, 0.03, 0.1):
    R = {r: (load(vq, 64, r), load(vq, 128, r)) for r in (0, 1, 2)}
    R = {r: v for r, v in R.items() if v[0] and v[1]}
    if len(R) < 2:
        continue
    tres = {r: pair(*v) for r, v in R.items()}
    tps = {r: t_pos(v[1][1]) for r, v in R.items()}
    tc = min(min(p["dK_e"], p["W_ext"]) for p in tres.values())
    tc = min([tc] + [x for x in tps.values() if x is not None])
    obs = {}
    for r, (_, (_, d)) in R.items():
        i = np.searchsorted(d["t"], tc - 1e-9)
        o = {"W_ext": d["W"][i, 2], "dK_e": d["K"][i, 0] - d["K"][0, 0], "dK_i": d["K"][i, 1] - d["K"][0, 1],
             "dTh_i": d["Th"][i, 1] - d["Th"][0, 1]}
        o["W_ext/W_lin"] = o["W_ext"] / W_lin(d["t"][i], vq)
        if tc > 80:
            q = qfit(d["t"], d["Q"][:, 1], tc, tc - 60)
            o.update({"Q_i_amp_w": q["amp_w"], "Q_i_phase_w": q["phase_w"], "Q_i_dc": q["dc"]})
        obs[r] = o
    # refinement uncertainty at the same time: |Nn64 - Nn128| for realization 0
    (_, c), (_, f) = R[0]
    i, j = np.searchsorted(c["t"], tc - 1e-9), np.searchsorted(f["t"], tc - 1e-9)
    ref = {"W_ext": abs(c["W"][i, 2] - f["W"][j, 2]), "dK_e": abs((c["K"][i, 0] - c["K"][0, 0]) - (f["K"][j, 0] - f["K"][0, 0])),
           "dK_i": abs((c["K"][i, 1] - c["K"][0, 1]) - (f["K"][j, 1] - f["K"][0, 1]))}
    S["realizations"][f"{vq:g}"] = {
        "t_res_pairs": tres, "t_pos_Nn128": tps, "t_common_cert": tc,
        "observables_Nn128": {r: {k: float(v) for k, v in o.items()} for r, o in obs.items()},
        "seed_mean": {k: float(np.mean([o[k] for o in obs.values()])) for k in obs[0]},
        "seed_std": {k: float(np.std([o[k] for o in obs.values()], ddof=1)) for k in obs[0]},
        "refinement_abs_diff_r0": {k: float(v) for k, v in ref.items()}}
    print(vq, "realizations", json.dumps(S["realizations"][f"{vq:g}"], default=float), flush=True)

# boundary fit: t_res = C vq^-p over censored-free points (min of the two observables)
pts = [(float(k), min(e["pairs"][list(e["pairs"])[-1]]["dK_e"], e["pairs"][list(e["pairs"])[-1]]["W_ext"]))
       for k, e in S["scan"].items() if e.get("pairs") and not e["certified"]["censored"]]
if len(pts) >= 2:
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    p, c = np.polyfit(x, y, 1)
    S["boundary_fit"] = {"points": pts, "t_res ~ C vq^p": {"p": float(p), "C": float(np.exp(c))},
                         "heating_at_t_res_range": [min(S["scan"][f"{v:g}"]["heating_at_t_res"]["dK_e"] for v, _ in pts),
                                                    max(S["scan"][f"{v:g}"]["heating_at_t_res"]["dK_e"] for v, _ in pts)]}
# unseeded controls: W_ext against the exact uniform oscillator
S["unseeded"] = {}
for vq in (0.01, 0.1):
    r = load(vq, 64, -1)
    if r:
        d = r[1]
        m = d["t"] > 20
        S["unseeded"][f"{vq:g}"] = {"status": r[0]["status"], "t_reached": r[0]["t_reached"],
                                    "max_rel_dev_from_W_lin": float(np.abs(d["W"][m, 2] / W_lin(d["t"][m], vq) - 1).max())}
# mobile-ion grid vs finest Hermite (r0)
S["grid_mobile"] = {}
for g in sorted(D.glob("gridm_*.npz")):
    G = dict(np.load(g))
    vq = float(g.stem.split("_")[1][2:])
    for Nn in (256, 128):
        r = load(vq, Nn, 0)
        if r:
            break
    h = r[1]
    th, tg = np.round(h["t"], 6), np.round(G["t"], 6)
    a, b = np.isin(th, tg), np.isin(tg, th)
    tt = h["t"][a]
    S["grid_mobile"][g.stem] = {"hermite_Nn": Nn, "common_end": float(tt[-1]),
                                "dK_e": disagreement(tt, h["K"][a, 0] - h["K"][0, 0], G["dK_e"][b]),
                                "W_ext": disagreement(tt, h["W"][a, 2], G["W_ext"][b]),
                                "grid_first_fmin_below_-1e-3": float(G["t"][np.argmax(G["fmin_rel"] < -1e-3)])
                                if (G["fmin_rel"] < -1e-3).any() else None,
                                "grid_first_edge_above_1e-6": float(G["t"][np.argmax(G["edge"] > 1e-6)])
                                if (G["edge"] > 1e-6).any() else None,
                                "U_k_at": {f"{T}": float(np.interp(T, G["t"], G["U_k"])) for T in (100, 300, 400, 500, 600, 700, 900)
                                           if T <= G["t"][-1]}}
(D / "summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
