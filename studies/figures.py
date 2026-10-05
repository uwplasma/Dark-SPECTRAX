"""README figures and movies. One command per family: ``python studies/figures.py NAME``.

Record families only read committed records (``docs/_static``, ``studies/*``) and redraw them with the shared
helpers below. ``phase`` runs new Hermite simulations: it first writes its own record
(``docs/_static/phase_space/run.json`` + ``data.npz``) and then draws movies and a still panel from that record;
``python studies/figures.py phase --draw`` redraws from the saved record without rerunning.
``djic_phase`` reads the Dark-JAX-in-Cell particle snapshots written by ``studies/djic_phase_space.py``.

Every output directory ``docs/_static/<name>/`` holds the image(s), ``run.json`` (command, commits, sources,
numbers quoted in the README) and, for figures that compute new curves, ``data.npz``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"
STUDIES = ROOT / "studies"
COLOR = {"ordinary": "#2166ac", "dark": "#b2182b", "root": "0.15", "grid": "#1b7837", "pic": "#762a83",
         "lane_b": "#e08214", "fixed": "0.55"}
ETA, OMEGA_D, BETA = 0.3, 1.0, 0.1  # gallery coupling and v_t/c mapping used by B00-B06


# ----------------------------------------------------------------------------------------------- shared helpers
def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    p.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5, "legend.fontsize": 8.5,
                       "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.spines.top": False,
                       "axes.spines.right": False, "savefig.dpi": 130, "figure.dpi": 100})
    return p


def figure(ncols, nrows=1, w=4.3, h=3.4):
    fig, axes = plt().subplots(nrows, ncols, figsize=(w * ncols, h * nrows), constrained_layout=True, squeeze=False)
    return fig, axes.ravel()


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_json(rel):
    return json.loads((ROOT / rel).read_text())


def load_npz(rel):
    with np.load(ROOT / rel, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def finish(name, fig=None, record=None, data=None, filename="figure.png"):
    """Write docs/_static/<name>/{filename, run.json, data.npz}."""
    out = STATIC / name
    out.mkdir(parents=True, exist_ok=True)
    if fig is not None:
        fig.savefig(out / filename)
        plt().close(fig)
    rec = {"figure": name, "command": f"python studies/figures.py {name}", "repository_commit": git_sha(),
           **(record or {})}
    (out / "run.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
    if data:
        np.savez_compressed(out / "data.npz", **data)
    print(f"wrote {out.relative_to(ROOT)}: {sorted(p.name for p in out.iterdir())}")
    return rec


def time_axis(ax, label=r"$\omega_{pe}t$"):
    ax.set_xlabel(label)
    ax.grid(alpha=0.25, lw=0.5)


def shade_uncertified(ax, t0, t1, label="not certified"):
    if t1 > t0:
        ax.axvspan(t0, t1, color="0.88", zorder=-2, lw=0, label=label)


def chi(w, k, pops):
    """Electron susceptibility of drifting Maxwellians [(fraction, v_t, u)] (standard deviation v_t)."""
    from scipy.special import wofz
    tot = 0.0
    for fr, vt, u in pops:
        z = (w - k * u) / (np.sqrt(2) * k * vt)
        tot = tot + fr / (k * vt) ** 2 * (1 + z * 1j * np.sqrt(np.pi) * wofz(z))
    return tot


def D_L(w, k, pops, eta):
    """Longitudinal Maxwell-Proca dispersion (c = 1): (Q - Omega_D^2)(1 + chi) + eta^2 Q chi, Q = w^2 - k^2."""
    Q = w ** 2 - k ** 2
    c = chi(w, k, pops)
    return (Q - OMEGA_D ** 2) * (1 + c) + eta ** 2 * Q * c


def root(k, pops, eta, guess):
    from scipy.optimize import root as _root
    f = lambda x: [D_L(x[0] + 1j * x[1], k, pops, eta).real, D_L(x[0] + 1j * x[1], k, pops, eta).imag]  # noqa: E731
    s = _root(f, [guess.real, guess.imag], tol=1e-14)
    w = s.x[0] + 1j * s.x[1]
    return w, abs(D_L(w, k, pops, eta))


def hermite_functions(N, xi):
    """psi_n(xi) = H_n(xi) exp(-xi^2) / sqrt(pi 2^n n!) for n < N (stable three-term recurrence; parent basis)."""
    psi = np.zeros((N,) + np.shape(xi))
    psi[0] = np.exp(-xi ** 2) / np.sqrt(np.pi)
    if N > 1:
        psi[1] = np.sqrt(2.0) * xi * psi[0]
    for n in range(1, N - 1):
        psi[n + 1] = np.sqrt(2.0 / (n + 1)) * xi * psi[n] - np.sqrt(n / (n + 1)) * psi[n - 1]
    return psi


def reconstruct(Ck, Nn, Nx, alpha, u, v, x_points=None):
    """Reduced f_s(x, v_x) = a_y a_z sum_n C_n(x) psi_n((v - u)/a_x) of one species (Nm = Np = 1).

    ``Ck`` has shape ``(Nn, 1, Nx//2+1, 1)`` (parent layout); ``alpha``/``u`` are the 3 widths/drifts.
    Returns ``f`` of shape ``(len(x), len(v))``; ``x_points`` (default Nx) is the inverse-FFT length.
    """
    nx = Nx if x_points is None else x_points
    full = np.zeros((Nn, nx // 2 + 1), complex)
    full[:, :Ck.shape[2]] = Ck[:, 0, :, 0]
    C = np.fft.irfft(full, n=nx, axis=1, norm="forward")  # C_n(x), same convention as the parent
    psi = hermite_functions(Nn, (v - u[0]) / alpha[0])
    return alpha[1] * alpha[2] * C.T @ psi


# ----------------------------------------------------------------------------------------------- landau
def fig_landau():
    """Dispersion diagram (both longitudinal branches, avoided crossing) + B00 runs against the roots."""
    rec = load_json("docs/_static/b00/run.json")
    arr = load_npz("docs/_static/b00/run.npz")
    vte = rec["settings"]["vte"]
    pops = [(1.0, vte, 0.0)]
    kl = np.linspace(0.7, 0.01, 140)
    branches = {}
    for name, eta, g0 in (("ordinary", 0.0, 1.674 - 0.392j), ("dark Langmuir", ETA, 1.708 - 0.369j)):
        w, ws = g0, []
        for x in kl:
            w, _ = root(x / vte, pops, eta, w)
            ws.append(w)
        branches[name] = np.array(ws)
    k = kl / vte
    a, b = k ** 2 + OMEGA_D ** 2 + 1 + ETA ** 2, k ** 2 + OMEGA_D ** 2 + ETA ** 2 * k ** 2
    cold = np.sqrt(0.5 * (a + np.sqrt(a ** 2 - 4 * b)))  # cold upper root: (w^2-k^2-W^2)(w^2-1) = eta^2(w^2-k^2)
    branches["dark Proca"] = np.array([root(kk, pops, ETA, g + 0j)[0] for kk, g in zip(k, cold)])
    branches["uncoupled Proca"] = np.sqrt(k ** 2 + OMEGA_D ** 2)
    meas = [r for r in rec["results"] if "omega_fit" in r and r["Nn"] == 128 and r["Nx"] == 8]

    fig, ax = figure(3, w=4.0, h=3.4)
    ax[0].plot(kl, branches["ordinary"].real, color=COLOR["ordinary"], label="ordinary Langmuir root")
    ax[0].plot(kl, branches["dark Langmuir"].real, color=COLOR["dark"], label="dark: Langmuir-like root")
    ax[0].plot(kl, branches["dark Proca"].real, color=COLOR["dark"], ls="--", label="dark: Proca-like root")
    ax[0].plot(kl, branches["uncoupled Proca"], color="0.6", ls=":", label=r"$\omega^2=k^2c^2+\Omega_D^2$ ($\eta=0$)")
    for r in meas:
        ax[0].plot(r["k_lambda_De"], r["omega_fit"][0], "o", mfc="none", color=COLOR[r["model"]], ms=6)
    ax[0].set(xlim=(0, 0.7), ylim=(0.8, 2.6), xlabel=r"$k\lambda_{De}$", ylabel=r"Re $\omega/\omega_{pe}$",
              title="Longitudinal branches (circles: runs)")
    ax[0].legend(loc="upper left", fontsize=7.5)
    ax[0].grid(alpha=0.25, lw=0.5)
    ax[1].semilogy(kl, -branches["ordinary"].imag, color=COLOR["ordinary"], label="ordinary root")
    ax[1].semilogy(kl, -branches["dark Langmuir"].imag, color=COLOR["dark"], label="dark root")
    for r in meas:
        if r["k_lambda_De"] >= 0.3:  # k lambda = 0.2: damping not resolved inside the Hermite window
            ax[1].plot(r["k_lambda_De"], -r["omega_fit"][1], "o", mfc="none", color=COLOR[r["model"]], ms=6)
    ax[1].set(xlim=(0.15, 0.7), ylim=(1e-6, 1), xlabel=r"$k\lambda_{De}$", ylabel=r"$-$Im $\omega/\omega_{pe}$",
              title="Landau damping rate")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.25, lw=0.5)
    kk = 0.5
    root_row = [r for r in rec["results"] if r.get("roots") and r["k_lambda_De"] == kk][0]["roots"]
    for mdl in ("ordinary", "dark"):
        tag = f"k{kk}_{mdl}_Nn128_Nx8"
        t, z = arr[f"t_{tag}"], arr[f"Ek_{tag}"]
        ax[2].semilogy(t, np.abs(z.imag) + 1e-30, color=COLOR[mdl], lw=1, label=f"{mdl} run (Nn = 128)")
        g = root_row[mdl][1]
        ax[2].semilogy(t, np.abs(z[0]) * np.exp(g * t), "--", color=COLOR[mdl], lw=1.2,
                       label=rf"root decay, $\gamma$ = {g:.4f}")
    ax[2].axvspan(*rec["settings"]["fit_windows"][str(kk)], color="0.9", zorder=-2, label="fit window")
    ax[2].set(ylabel=r"|Im $E_{x,k}|$ (parent units)", title=rf"Landau damping at $k\lambda_{{De}}$ = {kk}")
    time_axis(ax[2])
    ax[2].legend(loc="lower left")
    fig.suptitle(rf"Electrons $v_{{te}}=0.1c$, fixed ions; dark: $\eta$ = {ETA}, $\Omega_D=\omega_{{pe}}$",
                 fontsize=10.5)
    finish("landau", fig, {"sources": ["docs/_static/b00/run.json", "docs/_static/b00/run.npz"],
                           "measured_points": [{k: r[k] for k in ("k_lambda_De", "model", "omega_fit", "omega_root")}
                                               for r in meas]},
           {"k_lambda_De": kl, **{n.replace(" ", "_"): v for n, v in branches.items()}})


# ----------------------------------------------------------------------------------------------- growth
def fig_growth():
    """B02 two-stream and B03 bump-on-tail: |E_k| ordinary and dark against root growth."""
    rec = load_json("docs/_static/b02_b03/run.json")
    arr = load_npz("docs/_static/b02_b03/run.npz")
    rows = rec["results"]
    cases = list(dict.fromkeys(r["case"] for r in rows))
    titles = {"B02_two_stream_vt0.1_k0.6": r"two-stream $v_t$ = 0.1, $k\lambda$ = 0.6",
              "B02_two_stream_vt0.3_k0.4": r"two-stream $v_t$ = 0.3, $k\lambda$ = 0.4",
              "B03_bump_on_tail_k0.3": r"bump-on-tail, $k\lambda_{De}$ = 0.3"}
    fig, ax = figure(len(cases), w=4.2, h=3.4)
    for a, name in zip(ax, cases):
        for mdl in ("ordinary", "dark"):
            r = [x for x in rows if x["case"] == name and x["model"] == mdl][0]
            t, A = arr[f"{name}_{mdl}_t"], arr[f"{name}_{mdl}_A"]
            a.semilogy(t, A / BETA, color=COLOR[mdl], lw=1, label=f"{mdl}: fit {r['growth_fit']:.4f}")
            t0, t1 = r["window"]
            tt = np.linspace(t0, t1, 2)
            i = np.argmin(np.abs(t - t1))
            a.semilogy(tt, 3 * A[i] / BETA * np.exp(r["root"][1] * (tt - t1)), "--", color=COLOR[mdl], lw=1.4,
                       label=rf"root $\gamma$ = {r['root'][1]:.4f} (offset x3)")
        a.axvspan(t0, t1, color="0.9", zorder=-2)
        a.set(title=titles.get(name, name), ylabel=r"$|E_{x,k}|$ (electrostatic units)")
        time_axis(a)
        a.legend(loc="lower right")
    fig.suptitle(rf"Growth: ordinary and dark ($\eta$ = {ETA}, $\Omega_D=\omega_{{pe}}$), shaded = fit window",
                 fontsize=10.5)
    finish("growth", fig, {"sources": ["docs/_static/b02_b03/run.json", "docs/_static/b02_b03/run.npz"]})


# ----------------------------------------------------------------------------------------------- grid codes
def fig_grid():
    """B01 nonlinear Landau envelope and B06 echo: Hermite against independent semi-Lagrangian grid solvers."""
    h = load_npz("studies/b01_b06/run.npz")
    g1 = load_npz("studies/refs/B01/grid_k0.3_eps0.05.npz")
    g6 = load_npz("studies/refs/B06/grid_Nv1024_dt0.0125.npz")
    dg = load_npz("studies/dark_grid/run.npz")
    rb = load_json("studies/b01_b06/run.json")
    rd = load_json("studies/dark_grid/run.json")
    from scipy.ndimage import maximum_filter1d

    def env(t, E):  # running max over one plasma period (the B01 envelope rule)
        return maximum_filter1d(np.abs(E), size=int(round(2 * np.pi / 1.1 / (t[1] - t[0]))), mode="nearest")

    fig, ax = figure(2, w=5.2, h=3.6)
    a = ax[0]
    a.semilogy(g1["t"], env(g1["t"], g1["E"]), color=COLOR["grid"], lw=2.4, alpha=0.45, label="ordinary grid (Vlasov-Poisson)")
    a.semilogy(h["B01_ordinary_N1024_nu0.0_t"], env(h["B01_ordinary_N1024_nu0.0_t"], h["B01_ordinary_N1024_nu0.0_E"]), color=COLOR["ordinary"],
               lw=0.8, label="ordinary Hermite N = 1024")
    a.semilogy(dg["B01_grid_dt0.0125_t"], env(dg["B01_grid_dt0.0125_t"], dg["B01_grid_dt0.0125_E"]), color="#d6604d", lw=2.4, alpha=0.45,
               label="dark grid (Vlasov-Ampere-Proca)")
    a.semilogy(h["B01_dark_N1024_nu0.0_t"], env(h["B01_dark_N1024_nu0.0_t"], h["B01_dark_N1024_nu0.0_E"]), color=COLOR["dark"], lw=0.8,
               label="dark Hermite N = 1024")
    a.set(xlim=(0, 100), ylim=(0.035, 0.1), ylabel=r"envelope of $|E_{x,k}|$ (electrostatic units)",
          title=r"Nonlinear Landau, $k$ = 0.3, $\epsilon$ = 0.05")
    time_axis(a)
    a.legend(loc="lower right", fontsize=7.5)
    a = ax[1]
    a.plot(g6["t"], np.abs(g6["E"][:, 2]), color=COLOR["grid"], lw=2.6, alpha=0.45, label="ordinary grid")
    a.plot(h["B06_ordinary_N512_nu0.0_t"], np.abs(h["B06_ordinary_N512_nu0.0_E3"]), color=COLOR["ordinary"], lw=0.9,
           label="ordinary Hermite N = 512")
    a.plot(dg["B06_grid_dt0.0125_t"], np.abs(dg["B06_grid_dt0.0125_E3"]), color="#d6604d", lw=2.6, alpha=0.45,
           label="dark grid")
    a.plot(h["B06_dark_N512_nu0.0_t"], np.abs(h["B06_dark_N512_nu0.0_E3"]), color=COLOR["dark"], lw=0.9,
           label="dark Hermite N = 512")
    a.set(xlim=(15, 45), ylabel=r"$|E_{x,k_3}|$, $k_3 = k_2-k_1$ = 0.5", title="Two-pulse echo (k = 1, then 1.5)")
    time_axis(a)
    a.legend(loc="upper left", fontsize=7.5)
    o, d = rb["B06"]["ordinary_N512_nu0.0"], rd["B06_dark"]
    rec = {"sources": ["studies/b01_b06/run.npz", "studies/refs/B01/grid_k0.3_eps0.05.npz",
                       "studies/refs/B06/grid_Nv1024_dt0.0125.npz", "studies/dark_grid/run.npz"],
           "echo_ordinary_amp_rel_to_grid": o["amp_rel_to_grid"],
           "echo_dark_amp_rel_to_dark_grid": d["hermite_dark_N512_nu0.0"]["amp_rel_to_grid"],
           "echo_dark_over_ordinary": d["hermite_dark_N512_nu0.0"]["amp"] / o["amp"] - 1}
    finish("grid", fig, rec)


# ----------------------------------------------------------------------------------------------- Dark-JAX-in-Cell
def fig_djic():
    """C05: matched Landau case, Hermite against Dark-JAX-in-Cell PIC reruns (field history and fitted rates)."""
    rec = load_json("studies/c05/run.json")
    h = load_npz("studies/c05/run.npz")
    pic = load_npz("studies/c05_pic/cells128_particles320000.npz")
    fig, ax = figure(3, w=4.1, h=3.4)
    a = ax[0]
    for mdl, hk, pk in (("ordinary", "ordinary_Nn256", "parent_Ek"), ("dark", "self_consistent_Nn256", "dark_run_Ek")):
        a.semilogy(pic["t"], np.abs(pic[pk] / pic[pk][0]), color=COLOR[mdl], lw=2.2, alpha=0.35, label=f"{mdl} PIC")
        a.semilogy(h[f"{hk}_t"], np.abs(h[f"{hk}_E"] / h[f"{hk}_E"][0]), color=COLOR[mdl], lw=0.9, label=f"{mdl} Hermite")
    a.set(xlim=(0, 16), ylabel=r"$|E_{x,k}(t)/E_{x,k}(0)|$",
          title="PIC 128 cells, 320k particles vs Hermite")
    time_axis(a)
    a.legend(loc="lower left", fontsize=7.5)
    rows = []
    for f in sorted((STUDIES / "c05_pic").glob("*.json")):
        j = json.loads(f.read_text())
        rows.append((j["cells"], j["particles"], j["ordinary"], j["dark"], j["dark_slope_stderr"],
                     j.get("ordinary_slope_stderr", j["dark_slope_stderr"]), j["wall_dark_incl_compile"]))
    rows.sort(key=lambda r: (r[1], r[0]))
    for col, (mdl, key) in enumerate((("ordinary", "ordinary_Nn256"), ("dark", "self_consistent_Nn256"))):
        a = ax[1 + col]
        markers = {32: "s", 64: "^", 128: "o"}
        for cells in (32, 64, 128):
            sel = [r for r in rows if r[0] == cells]
            a.errorbar([r[1] for r in sel], [-r[2 + col][1] for r in sel], yerr=[r[5 - col] for r in sel],
                       marker=markers[cells], color=COLOR["pic"], ls="-", lw=0.8, ms=5, capsize=2,
                       alpha=0.4 + 0.2 * (cells // 64), label=f"PIC, {cells} cells")
        a.axhline(-rec["spectrax"][key]["maxima_fit"][1], color=COLOR[mdl], lw=1.4,
                  label="Hermite Nn = 256 (same fit rule)")
        a.axhline(-rec["roots"][mdl][1], color=COLOR["root"], lw=1, ls="--", label="linear root")
        a.set_xscale("log")
        a.set_xticks([2e4, 4e4, 8e4, 1.6e5, 3.2e5], ["20k", "40k", "80k", "160k", "320k"])
        a.minorticks_off()
        a.set(xlabel="particles", ylabel=r"fitted damping $-\gamma/\omega_{pe}$", title=f"{mdl} damping rate")
        a.grid(alpha=0.25, lw=0.5)
        a.legend(loc="upper right", fontsize=7.5)
    p = rec["physical_inputs"]
    fig.suptitle(rf"Same inputs in both codes: $k\lambda_{{De}}$ = {p['k_lambda_De']}, $v_{{te}}$ = "
                 rf"{p['sigma_over_c']}c, seed {p['density_seed']}; dark: $\eta$ = {p['eta']}, "
                 rf"$\Omega_D$ = {p['Omega_D_over_wp']:g}$\omega_{{pe}}$", fontsize=10.5)
    finish("djic", fig, {"sources": ["studies/c05/run.json", "studies/c05/run.npz", "studies/c05_pic/*.json",
                                     "studies/c05_pic/cells128_particles320000.npz"],
                         "pic_rows": [dict(zip(("cells", "particles", "ordinary", "dark", "dark_stderr",
                                                "ordinary_stderr", "wall_s"), r)) for r in rows]})


# ----------------------------------------------------------------------------------------------- conversion
VTE2, EPS = 1e-3, 1 / 1836
W_DRIVE = np.sqrt(1 + EPS)


def W_lin(t, vq):
    """Exact linear resonant work on the uniform two-fluid oscillator (studies/lane_c_analysis.py)."""
    A = vq * np.sqrt(VTE2) * (1 + EPS)
    r = -(A / (2 * W_DRIVE)) * t * np.sin(W_DRIVE * t)
    rd = -(A / (2 * W_DRIVE)) * (np.sin(W_DRIVE * t) + W_DRIVE * t * np.cos(W_DRIVE * t))
    return 0.5 / (1 + EPS) * (rd ** 2 + W_DRIVE ** 2 * r ** 2)


def cycle_mean(t, y, period=2 * np.pi / W_DRIVE):
    """Running mean over exactly one drive period (linear interpolation onto period/64 spacing)."""
    tf = np.arange(t[0], t[-1], period / 64)
    yf = np.interp(tf, t, y)
    k = np.ones(64) / 64
    return tf[63:] - 0.5 * period, np.convolve(yf, k, mode="valid")


def fig_conversion():
    """H05 resonant drive (W_ext vs the exact linear law, partition, certified times by lane) and H07 reservoir."""
    s = load_json("studies/lane_c/summary.json")
    lb = load_json("studies/lane_b/summary.json")
    hh = load_npz("studies/hhs/run.npz")
    fixed = load_json("studies/h05_pump/run.json")["fixed_basis_resolved_until"]
    T = 1000.0
    fig, ax = figure(2, 2, w=5.0, h=3.4)
    vqs = ("0.001", "0.01", "0.03", "0.1")
    cmap = plt().get_cmap("viridis")
    data, quoted = {}, {}
    for i, vq in enumerate(vqs):
        e = s["scan"][vq]
        tc = e["certified"]["t_cert"]
        d = load_npz(f"studies/lane_c/runs/vq{vq}_Nn128_r0.npz")
        m = (d["t"] <= tc) & (d["t"] >= 20)
        t = d["t"][m]
        tt, ratio = cycle_mean(t, d["W"][m, 2] / np.maximum(W_lin(t, float(vq)), 1e-300))
        c = cmap(i / (len(vqs) - 0.5))
        ax[0].plot(tt, ratio, color=c, lw=1.1, label=rf"$v_q/v_{{te}}$ = {vq}, certified to {tc:g}")
        if tc < T:
            ax[0].plot([tc], [ratio[-1]], "|", color=c, ms=12, mew=2)
        data[f"vq{vq}_t"], data[f"vq{vq}_ratio"] = tt, ratio
        quoted[vq] = {"t_cert": tc, "W_ext_over_W_lin_cycle_avg": e["W_ext_over_W_lin"]["cycle_avg_at_t_cert"]}
    ax[0].set(xlim=(0, T), ylim=(0.985, 1.002), ylabel=r"$W_{\rm ext}/W_{\rm lin}$ (cycle mean)",
              title="Work against the exact linear resonant law (Nn = 128)")
    time_axis(ax[0])
    ax[0].legend(loc="lower left", fontsize=7.5)
    vq = "0.03"
    tc = s["scan"][vq]["certified"]["t_cert"]
    d = load_npz(f"studies/lane_c/runs/vq{vq}_Nn128_r0.npz")
    m = (d["t"] <= tc) & (d["t"] > 20)
    W = d["W"][m, 2]
    for lab, y, c in ((r"$\Delta K_e/W_{\rm ext}$", d["K"][m, 0] - d["K"][0, 0], COLOR["ordinary"]),
                      (r"$\Delta U_E/W_{\rm ext}$", d["U_gamma"][m] - d["U_gamma"][0], COLOR["lane_b"]),
                      (r"$10^3\,\Delta K_i/W_{\rm ext}$", 1e3 * (d["K"][m, 1] - d["K"][0, 1]), COLOR["grid"])):
        tt, yy = cycle_mean(d["t"][m], y)
        yy = yy / cycle_mean(d["t"][m], W)[1]
        ax[1].plot(tt, yy, color=c, lw=1.2, label=lab)
    shade_uncertified(ax[1], tc, T)
    ax[1].set(xlim=(0, T), ylim=(0, 0.8), ylabel="fraction of external work (cycle mean)",
              title=rf"Energy partition, $v_q/v_{{te}}$ = {vq} (Nn = 128)")
    time_axis(ax[1])
    ax[1].legend(loc="upper right", ncol=2, fontsize=7.5)
    a = ax[2]
    v = np.array([float(x) for x in s["scan"]])
    lanes = {"Lane C, nu = 0 (Nn 128 vs 256 or 64 vs 128)": ([s["scan"][x]["certified"]["t_cert"] for x in s["scan"]],
                                                              [s["scan"][x]["certified"]["censored"] for x in s["scan"]],
                                                              COLOR["ordinary"], "o")}
    for nu, mk in ((1, "s"), (2, "D")):
        ks = [x for x in s["scan"] if f"vq{x}_o2_Nn64_nu{nu}" in lb]
        tt = [min(lb[f"vq{x}_o2_Nn64_nu{nu}"]["Nn_pair"][q] for q in ("dK", "W")) for x in ks]
        cen = [lb[f"vq{x}_o2_Nn64_nu{nu}"]["Nn_pair"]["dK_censored"] for x in ks]
        lanes[f"Lane B, order-2 closure nu = {nu} (Nn 64 vs 128)"] = (tt, cen, COLOR["lane_b"], mk, ks)
    for lab, val in lanes.items():
        tt, cen, c, mk = val[:4]
        vv = np.array([float(x) for x in val[4]]) if len(val) > 4 else v
        a.plot(vv, tt, mk, color=c, mfc="none" if "nu = 2" in lab else c, ms=6, label=lab)
        for x, y, cc in zip(vv, tt, cen):
            if cc:
                a.annotate("", (x, y * 1.12), (x, y), arrowprops={"arrowstyle": "->", "color": c, "lw": 0.8})
    a.plot([float(x) for x in fixed], list(fixed.values()), "x", color=COLOR["fixed"], ms=7,
           label="fixed basis (no pump frame)")
    bf = s["boundary_fit"]["t_res ~ C vq^p"]
    vv = np.geomspace(0.02, 0.1, 20)
    a.plot(vv, bf["C"] * vv ** bf["p"], color="0.3", lw=0.8, ls="--",
           label=rf"fit $t_{{res}}$ = {bf['C']:.0f} $(v_q/v_{{te}})^{{{bf['p']:.2f}}}$")
    a.set(xscale="log", xlabel=r"$v_q/v_{te}$", ylabel=r"resolved time $\omega_{pe}t_{res}$", ylim=(0, 1150),
          title="Resolved time by lane (arrow: run ended, still agreeing)")
    a.grid(alpha=0.25, lw=0.5)
    a.legend(loc="lower left", fontsize=7)
    a = ax[3]
    h7 = load_json("studies/hhs/run.json")["H07"]
    for eta, ls in (("0.03", "-"), ("0.001", "--")):
        t = hh[f"H07_eta{eta}_t"]
        a.semilogy(t[1:], hh[f"H07_eta{eta}_W_ext"][1:], color=COLOR["ordinary"], ls=ls, lw=1.1,
                   label=rf"prescribed drive $W_{{\rm ext}}$, $\eta$ = {eta}")
        a.semilogy(t[1:], np.abs(hh[f"H07_eta{eta}_W_D"][1:]), color=COLOR["dark"], ls=ls, lw=1.1,
                   label=rf"finite reservoir $W_D$, $\eta$ = {eta}")
        a.axhline(h7[f"eta{eta}"]["U_D0"], color="0.4", ls=ls, lw=0.7)
        a.text(990, h7[f"eta{eta}"]["U_D0"] * 1.25, rf"$U_D(0)$, $\eta$ = {eta}", ha="right", fontsize=8,
               color="0.3")
    a.set(ylabel="work on the plasma (parent units)", ylim=(1e-9, 2e-3),
          title=r"Homogeneous drive vs finite reservoir ($\eta E_D(0)=E_0$)")
    time_axis(a)
    a.legend(loc="lower right", fontsize=7)
    fig.suptitle("Resonant dark-photon drive (HHS-v1-inspired inputs, nonrelativistic, mobile ions, Nx = 8)",
                 fontsize=10.5)
    finish("conversion", fig, {"sources": ["studies/lane_c/summary.json", "studies/lane_c/runs/*_Nn128_r0.npz",
                                           "studies/lane_b/summary.json", "studies/h05_pump/run.json",
                                           "studies/hhs/run.npz"],
                               "certified": quoted, "boundary_fit": bf}, data)


# ----------------------------------------------------------------------------------------------- conservation
def fig_conservation():
    """Work ledger (relative to W_ext) and Hermite-pair disagreement that defines t_res, v_q/v_te = 0.1."""
    s = load_json("studies/lane_c/summary.json")
    vq = "0.1"
    fig, ax = figure(2, w=5.0, h=3.4)
    runs = {}
    for Nn, c in ((64, "#92c5de"), (128, "#4393c3"), (256, "#053061")):
        d = load_npz(f"studies/lane_c/runs/vq{vq}_Nn{Nn}_r0.npz")
        tpos = s["scan"][vq]["runs"][str(Nn)]["t_pos"]
        m = d["t"] <= (tpos if tpos else d["t"][-1])
        t = d["t"][m]
        led = (d["K"][m].sum(1) - d["K"][0].sum()) + (d["U_gamma"][m] - d["U_gamma"][0]) - d["W"][m, 2]
        W = np.maximum.accumulate(np.abs(d["W"][m, 2]))
        ax[0].semilogy(t[t > 5], np.abs(led[t > 5]) / W[t > 5] + 1e-18, color=c, lw=0.9,
                       label=f"Nn = {Nn} (to first non-positive K or T, t = {tpos:g})")
        runs[Nn] = (d["t"], d["K"][:, 0] - d["K"][0, 0])
    ax[0].set(ylabel=r"$|\Delta K+\Delta U_E-W_{\rm ext}|\,/\,\max W_{\rm ext}$",
              title=rf"Energy ledger, $v_q/v_{{te}}$ = {vq}")
    time_axis(ax[0])
    ax[0].legend(loc="upper left", fontsize=7.5)
    tc = s["scan"][vq]["certified"]["t_cert"]
    for (a, b), c in (((64, 128), "#4393c3"), ((128, 256), "#053061")):
        n = min(runs[a][0].size, runs[b][0].size)
        t = runs[a][0][:n]
        scale = np.maximum.accumulate(np.maximum(np.abs(runs[a][1][:n]), np.abs(runs[b][1][:n])))
        ax[1].semilogy(t, np.abs(runs[a][1][:n] - runs[b][1][:n]) / np.maximum(scale, 1e-30) + 1e-16, color=c, lw=0.9,
                       label=f"Nn {a} vs {b}")
    ax[1].axhline(0.1, color="0.3", ls="--", lw=0.8, label="10% criterion")
    ax[1].axvline(tc, color=COLOR["dark"], lw=1, label=rf"$t_{{res}}$ = {tc:g}")
    shade_uncertified(ax[1], tc, 1000)
    ax[1].set(xlim=(0, 700), ylim=(1e-12, 10), ylabel=r"$|\Delta K_e^{(a)}-\Delta K_e^{(b)}|$ / running max",
              title="Hermite-order agreement defines the resolved time")
    time_axis(ax[1])
    ax[1].legend(loc="upper left", fontsize=7.5)
    finish("conservation", fig, {"sources": [f"studies/lane_c/runs/vq{vq}_Nn*_r0.npz", "studies/lane_c/summary.json"],
                                 "t_res": tc})


# ----------------------------------------------------------------------------------------------- performance
def fig_performance():
    """Wall time against damping-rate error for C05 (PIC reruns and Hermite), and H05 wall time per Hermite order."""
    rec = load_json("studies/c05/run.json")
    fig, ax = figure(2, w=5.0, h=3.4)
    a = ax[0]
    floor = 1e-5
    for mdl, key in (("ordinary", "ordinary"), ("dark", "dark")):
        hk = "ordinary_Nn" if mdl == "ordinary" else "self_consistent_Nn"
        ref = rec["spectrax"][hk + "256"]["maxima_fit"][1]  # converged Hermite value of the same fit rule
        pts = []
        for f in sorted((STUDIES / "c05_pic").glob("*.json")):
            j = json.loads(f.read_text())
            wall = j["wall_parent_incl_compile"] if mdl == "ordinary" else j["wall_dark_incl_compile"]
            pts.append((wall, abs(j[key][1] / ref - 1)))
        a.loglog([p[0] for p in pts], [p[1] for p in pts], "o", color=COLOR[mdl], mfc="none",
                 label=f"{mdl}: Dark-JAX-in-Cell PIC (cell/particle pairs)")
        hs = rec["spectrax"][hk + "128"]
        a.loglog([hs["run_time"] + hs["compile_time"]], [max(abs(hs["maxima_fit"][1] / ref - 1), floor)], "*",
                 color=COLOR[mdl], ms=11, label=f"{mdl}: Hermite Nn = 128 (difference < {floor:g}, drawn at it)")
    a.set(xlabel="wall time incl. compile (s, CPU)", ylabel=r"$|\gamma_{\rm fit}/\gamma_{\rm Hermite\,256}-1|$",
          ylim=(5e-6, 1), title="C05 Landau: cost vs distance from converged value")
    a.grid(alpha=0.25, lw=0.5, which="both")
    a.legend(loc="lower right", fontsize=7)
    a = ax[1]
    s = load_json("studies/lane_c/summary.json")
    for vq, c in (("0.01", COLOR["ordinary"]), ("0.1", COLOR["dark"])):
        rr = s["scan"][vq]["runs"]
        Nn = sorted(int(n) for n in rr)
        a.plot(Nn, [rr[str(n)]["run_time"] / rr[str(n)]["t_reached"] * 100 for n in Nn], "o-", color=c,
               label=rf"$v_q/v_{{te}}$ = {vq}: run time per 100 $\omega_{{pe}}^{{-1}}$")
        a.plot(Nn, [rr[str(n)]["compile_time"] for n in Nn], "s:", color=c, mfc="none", label="compile time")
    a.set(xscale="log", yscale="log", xlabel="Hermite order Nn", ylabel="seconds", ylim=(0.3, 600),
          title="H05 pump frame, Nx = 8, two species")
    a.set_xticks([64, 128, 256], ["64", "128", "256"])
    a.minorticks_off()
    a.grid(alpha=0.25, lw=0.5, which="both")
    a.legend(loc="upper left", fontsize=7.5)
    finish("performance", fig, {"sources": ["studies/c05/run.json", "studies/c05_pic/*.json",
                                            "studies/lane_c/summary.json"],
                                "note": "timings from the records' own machines; not a controlled benchmark"})


# ----------------------------------------------------------------------------------------------- phase space
# Electrostatic inputs (omega_pe = 1, v_te = 1 of the reference population) mapped with v_t/c = BETA as in
# examples/instabilities.py. Populations: (fraction, v_t, u). Seeds are density perturbations at the box mode.
PHASE_CASES = {
    "landau": {"title": r"nonlinear Landau damping, $k\lambda_{De}$ = 0.3, $\delta n/n$ = 0.05",
               "pops": [(1.0, 1.0, 0.0)], "k": 0.3, "seed": 0.05, "Nx": 16, "Nn": 512, "nu": 0.0, "T": 100.0,
               "v": (-5.0, 5.0), "units": ("v_{te}", r"\lambda_{De}"), "delta": True},
    "two_stream": {"title": r"two-stream, $u=\pm 1$, $v_t$ = 0.3, $k$ = 0.4",
                   "pops": [(0.5, 0.3, 1.0), (0.5, 0.3, -1.0)], "k": 0.4, "seed": 1e-3, "Nx": 16, "Nn": 128,
                   "nu": 1.0, "T": 60.0, "v": (-2.6, 2.6), "units": ("v_0", r"v_0/\omega_{pe}")},
    "bump_on_tail": {"title": r"bump-on-tail, $k\lambda_{De}$ = 0.3",
                     "pops": [(0.9, 1.0, -0.45), (0.1, 0.5, 4.05)], "k": 0.3, "seed": 1e-3, "Nx": 16, "Nn": 128,
                     "nu": 1.0, "T": 100.0, "v": (-4.0, 7.5), "units": ("v_{te}", r"\lambda_{De}")},
}
PHASE_FRAMES = 81
CACHE = ROOT / "artifacts" / "phase_space"  # full coefficient histories (not committed; rebuilt by the run)


def phase_model(case, dark, Nn):
    import darkspectrax as ds
    c = PHASE_CASES[case]
    pops = [(fr, BETA * vt, BETA * u) for fr, vt, u in c["pops"]]
    return ds.Model(Nx=c["Nx"], Nn=Nn, Lx=2 * np.pi / c["k"] * BETA, qs=(-1.0,) * len(pops),
                    Omega_cs=(1.0,) * len(pops), alpha_s=tuple(np.repeat([np.sqrt(2) * vt for _, vt, _ in pops], 3)),
                    u_s=tuple(v for _, _, u in pops for v in (u, 0.0, 0.0)), rho_background=1.0, nu=c["nu"],
                    mode="self_consistent" if dark else "ordinary", eta=ETA if dark else 0.0, Omega_D=OMEGA_D), pops


def phase_simulate(case, dark, Nn):
    """One Hermite run; returns E_k1(t) (electrostatic units), U_E(t), the ledger and Ck at the movie frames."""
    import darkspectrax as ds
    c = PHASE_CASES[case]
    m, pops = phase_model(case, dark, Nn)
    pert = [(s, (1, 0, 0), 0.5 * fr * c["seed"]) for s, (fr, _, _) in enumerate(pops)]
    y = ds.consistent_fields(m, ds.maxwellian(m, [fr for fr, _, _ in pops], pert))
    stride = 5
    out = ds.run(m, y, c["T"], n_save=stride * (PHASE_FRAMES - 1) + 1, rtol=1e-8, atol=1e-14, max_steps=200_000,
                 dtmin=1e-6)
    t = out["t"]
    ok = np.isfinite(t) & np.isfinite(out["K"])
    return {"t": t, "E1": out["Fk"][:, 0, 0, 1, 0] / BETA, "U_E": out["U_gamma"], "K": out["K"], "W": out["W"],
            "ledger": np.nanmax(np.abs(out["ledger_defect"][:, :int(ok.sum())]), axis=1), "Ck": out["Ck"][::stride], "t_frames": t[::stride],
            "status": out["status"], "failure_reason": out["failure_reason"], "valid": int(ok.sum()),
            "run_time": out["run_time"], "compile_time": out["compile_time"], "steps": out["num_steps"]}


def disagreement_time(t, a, b, tol=0.1, t_min=1.0):
    """First t where |a - b| exceeds tol x running max(|a|, |b|) (the H05 t_res rule), or None."""
    scale = np.maximum.accumulate(np.maximum(np.abs(a), np.abs(b)))
    bad = (np.abs(a - b) > tol * np.maximum(scale, 1e-300)) & (t > t_min)
    return float(t[np.argmax(bad)]) if bad.any() else None


def phase_f(case, dark, Nn, Ck_frames, nx=96, nv=160):
    """Total reduced f(x, v_x) (electrostatic velocity units) on the display grid for each frame."""
    c = PHASE_CASES[case]
    m, pops = phase_model(case, dark, Nn)
    v = np.linspace(*c["v"], nv) * BETA
    a = np.asarray(m.alpha_s).reshape(-1, 3)
    u = np.asarray(m.u_s).reshape(-1, 3)
    H = Nn
    f = np.zeros((len(Ck_frames), nx, nv))
    for i, Ck in enumerate(Ck_frames):
        for s in range(len(pops)):
            f[i] += reconstruct(Ck[s * H:(s + 1) * H], Nn, c["Nx"], a[s], u[s], v, x_points=nx)
    return np.linspace(0, 2 * np.pi / c["k"], nx, endpoint=False), v / BETA, f * BETA


def phase_run(cases):
    """Run ordinary and dark at Nn and 2Nn for each case; write the record and the coefficient cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    rec_path = STATIC / "phase_space" / "run.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"cases": {}}
    data = dict(load_npz("docs/_static/phase_space/data.npz")) if (STATIC / "phase_space" / "data.npz").exists() else {}
    for case in cases:
        c = PHASE_CASES[case]
        entry = {"inputs": {k: v for k, v in c.items() if k not in ("title", "units", "delta")}, "eta": ETA, "Omega_D": OMEGA_D,
                 "beta": BETA, "rtol": 1e-8, "atol": 1e-14, "solver": "Dopri8", "runs": {}}
        res = {}
        for dark in (False, True):
            for Nn in (c["Nn"], 2 * c["Nn"]):
                tag = f"{'dark' if dark else 'ordinary'}_Nn{Nn}"
                r = phase_simulate(case, dark, Nn)
                res[tag] = r
                print(case, tag, r["status"], r["failure_reason"], f"{r['run_time']:.1f}s", flush=True)
                np.savez_compressed(CACHE / f"{case}_{tag}.npz", Ck=r["Ck"], t_frames=r["t_frames"])
                entry["runs"][tag] = {k: r[k] for k in ("status", "failure_reason", "valid", "run_time",
                                                       "compile_time", "steps")}
                entry["runs"][tag]["max_ledger_defect_over_K0"] = float(r["ledger"].max() / abs(r["K"][0]))
                data[f"{case}_{tag}_t"], data[f"{case}_{tag}_E1"] = r["t"], r["E1"]
                data[f"{case}_{tag}_U_E"] = r["U_E"]
        for mdl in ("ordinary", "dark"):
            a, b = res[f"{mdl}_Nn{c['Nn']}"], res[f"{mdl}_Nn{2 * c['Nn']}"]
            n = min(a["valid"], b["valid"])
            tr = disagreement_time(a["t"][:n], a["U_E"][:n], b["U_E"][:n])
            entry[f"t_res_{mdl}"] = tr if tr is not None else float(a["t"][n - 1])
            entry[f"t_res_{mdl}_censored"] = tr is None
        rec["cases"][case] = entry
    rec.update({"command": "python studies/figures.py phase", "repository_commit": git_sha(),
                "t_res_rule": "first t > 1 where |U_E(Nn) - U_E(2Nn)| > 0.1 running max (t_res_*), and first frame where max|f(Nn) - f(2Nn)| > 0.1 max f (t_res_f_*); frames at or after the earliest of the four are not drawn (t_drawn)",
                "reconstruction": "f(x, v_x) = a_y a_z sum_n C_n(x) psi_n((v - u)/a_x) per population, Nn-run"})
    (STATIC / "phase_space").mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=2, default=float) + "\n")
    np.savez_compressed(STATIC / "phase_space" / "data.npz", **data)


def phase_draw(cases):
    """Movies (animated WebP) and a still panel from the record and the coefficient cache."""
    from PIL import Image
    p = plt()
    rec = load_json("docs/_static/phase_space/run.json")
    data = load_npz("docs/_static/phase_space/data.npz")
    for case in cases:
        c, e = PHASE_CASES[case], rec["cases"][case]
        fs = {}
        for mdl in ("ordinary", "dark"):
            z1, z2 = (np.load(CACHE / f"{case}_{mdl}_Nn{n}.npz") for n in (c["Nn"], 2 * c["Nn"]))
            n = int(min(np.isfinite(z1["t_frames"]).sum(), np.isfinite(z2["t_frames"]).sum()))
            x, v, f1 = phase_f(case, mdl == "dark", c["Nn"], z1["Ck"][:n])
            f2 = phase_f(case, mdl == "dark", 2 * c["Nn"], z2["Ck"][:n])[2]
            diff = np.abs(f1 - f2).max(axis=(1, 2)) / f2.max(axis=(1, 2))  # Nn vs 2Nn, L-inf / max f
            bad = np.nonzero(diff > 0.1)[0]
            t_f = float(z1["t_frames"][bad[0]]) if bad.size else float(z1["t_frames"][n - 1])
            e[f"t_res_f_{mdl}"], e[f"t_res_f_{mdl}_censored"] = t_f, not bad.size
            e[f"f_difference_{mdl}"] = diff.tolist()
            fs[mdl] = (z1["t_frames"][:n], f1, diff)
        # Frames drawn: both models resolved by both rules (field energy and f itself, Nn vs 2Nn, 10%).
        t_res = min(e["t_res_ordinary"], e["t_res_dark"], e["t_res_f_ordinary"], e["t_res_f_dark"])
        keep = {m: fs[m][0] < t_res for m in fs}
        nd = min(keep["ordinary"].sum(), keep["dark"].sum())
        tf = fs["ordinary"][0][:nd]
        fs = {m: (fs[m][0][:nd], fs[m][1][:nd], fs[m][2][:nd]) for m in fs}
        fmax = max(fs[m][1].max() for m in fs)
        neg = {m: [float(fr.min() / fr.max()) for fr in fs[m][1]] for m in fs}  # min f / max f per frame
        dmax = max(np.abs(fs[m][1] - fs[m][1].mean(axis=1, keepdims=True)).max() for m in fs)
        e["t_drawn"] = t_res
        frames = []
        for i in range(len(tf)):
            fig = p.figure(figsize=(9.0, 5.0), dpi=88, constrained_layout=True)
            gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])
            for j, mdl in enumerate(("ordinary", "dark")):
                ax = fig.add_subplot(gs[0, j])
                fr = fs[mdl][1][i]
                ext = (x[0], x[-1] + x[1], v[0], v[-1])
                if c.get("delta"):  # small perturbation: show f - <f>_x so trapping is visible
                    im = ax.imshow((fr - fr.mean(axis=0)).T, origin="lower", aspect="auto", cmap="RdBu_r",
                                   vmin=-dmax, vmax=dmax, extent=ext)
                else:
                    from matplotlib.colors import PowerNorm
                    im = ax.imshow(np.maximum(fr, 0).T, origin="lower", aspect="auto", cmap="magma", extent=ext,
                                   norm=PowerNorm(0.5, vmin=0, vmax=fmax))  # sqrt scale shows weak beams
                    if fr.min() < -0.01 * fmax:  # mark negative f honestly instead of clipping it to black
                        ax.contourf(x + 0.5 * x[1], v, fr.T, levels=[fr.min() - 1, -0.01 * fmax], colors=["#4fc3f7"])
                    ax.text(0.98, 0.03, "cyan: f < -1% max f", transform=ax.transAxes, color="#4fc3f7",
                            fontsize=7.5, ha="right")
                label = "ordinary" if mdl == "ordinary" else rf"dark ($\eta$ = {ETA}, $\Omega_D=\omega_{{pe}}$)"
                vu, xu = c["units"]
                ax.set(title=f"{label}, t = {tf[i]:.1f}", xlabel=rf"$x\,/\,({xu})$",
                       ylabel=rf"$v_x/{vu}$" if j == 0 else None)
                ax.text(0.02, 0.03, f"min f / max f = {neg[mdl][i]:+.1e}", transform=ax.transAxes, color="w",
                        fontsize=8)
            fig.colorbar(im, ax=fig.axes, shrink=0.8,
                         label=r"$f-\langle f\rangle_x$" if c.get("delta") else r"$f(x,v_x)$ (square-root scale)")
            ax = fig.add_subplot(gs[1, :])
            for mdl in ("ordinary", "dark"):
                key = f"{case}_{mdl}_Nn{c['Nn']}"
                tt, E = data[f"{key}_t"], data[f"{key}_E1"]
                m = np.isfinite(tt) & (tt <= t_res)
                ax.semilogy(tt[m], np.abs(E[m]), color=COLOR[mdl], lw=1, label=mdl)
            ax.axvline(tf[i], color="0.3", lw=0.8)
            shade_uncertified(ax, t_res, c["T"], label="not resolved (Nn vs 2Nn: field or f differ > 10%)")
            ax.set(xlim=(0, c["T"]), ylabel=r"$|E_{x,k}|$")
            time_axis(ax)
            ax.legend(loc="lower right", fontsize=7.5, ncol=3)
            fig.suptitle(f"{c['title']}  (Hermite Nn = {c['Nn']}, Nx = {c['Nx']}, nu = {c['nu']:g})", fontsize=10)
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            p.close(fig)
        out = STATIC / "phase_space" / f"{case}.webp"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0, quality=70, method=6)
        e["movie"] = {"file": out.name, "frames": len(frames), "t_last_frame": float(tf[-1]),
                      "bytes": out.stat().st_size, "min_f_over_max_ordinary": min(neg["ordinary"]),
                      "min_f_over_max_dark": min(neg["dark"]),
                      "max_f_difference_drawn": max(float(fs[m][2].max()) for m in fs)}
        print(f"wrote {out.name} ({out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")
    (STATIC / "phase_space" / "run.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")


def fig_phase(*args):
    cases = [a for a in args if a in PHASE_CASES] or list(PHASE_CASES)
    if "--draw" not in args:
        phase_run(cases)
    phase_draw(cases)


def fig_djic_phase(*args):
    """Two-stream phase space: Hermite reconstruction against Dark-JAX-in-Cell markers, ordinary and dark."""
    from PIL import Image
    p = plt()
    case, c = "two_stream", PHASE_CASES["two_stream"]
    rec = load_json("docs/_static/phase_space/run.json")["cases"][case]
    pic_rec = load_json("studies/djic_phase/run.json")
    pic = load_npz("artifacts/djic_phase/two_stream.npz")
    herm = load_npz("docs/_static/phase_space/data.npz")
    t_res = min(rec["t_res_ordinary"], rec["t_res_dark"])
    fs = {}
    for mdl in ("ordinary", "dark"):
        z = np.load(CACHE / f"{case}_{mdl}_Nn{c['Nn']}.npz")
        keep = np.isfinite(z["t_frames"]) & (z["t_frames"] <= t_res) & (z["t_frames"] > 0)
        x, v, f = phase_f(case, mdl == "dark", c["Nn"], z["Ck"][keep], nx=64, nv=80)
        fs[mdl] = (z["t_frames"][keep], f)
    n = min(len(fs["ordinary"][0]), len(fs["dark"][0]))
    tf = fs["ordinary"][0][:n]
    fmax = max(fs[m][1].max() for m in fs)
    xe, ve = np.linspace(0, 1, 65), np.linspace(*c["v"], 81)
    norm = 1.0 / (2 * pic_rec["particles_per_beam"] * (xe[1] - xe[0]) * (ve[1] - ve[0]))  # mean density 1
    tp = pic["ordinary_t"]
    idx = [int(np.argmin(np.abs(tp - t))) for t in tf]
    assert np.allclose(tp[idx], tf, atol=1e-6), "PIC and Hermite frames must coincide"
    frames = []
    for i in range(n):
        fig = p.figure(figsize=(9.0, 6.6), dpi=80, constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.6, 1])
        for j, mdl in enumerate(("ordinary", "dark")):
            for row, (src, img) in enumerate((("Hermite", fs[mdl][1][i].T),
                                               ("PIC", pic[f"{mdl}_H"][idx[i]].T * norm))):
                ax = fig.add_subplot(gs[row, j])
                im = ax.imshow(img, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=fmax,
                               extent=(0, 1, c["v"][0], c["v"][1]))
                lab = "ordinary" if mdl == "ordinary" else rf"dark ($\eta$ = {ETA})"
                ax.set(title=f"{src}, {lab}, t = {tf[i]:.1f}", xlabel=r"$x/L$" if row else None,
                       ylabel=r"$v_x/v_0$" if j == 0 else None)
        fig.colorbar(im, ax=fig.axes, shrink=0.7, label=r"$f(x,v_x)$")
        ax = fig.add_subplot(gs[2, :])
        for mdl in ("ordinary", "dark"):
            key = f"{case}_{mdl}_Nn{c['Nn']}"
            th, Eh = herm[f"{key}_t"], herm[f"{key}_E1"]
            m = np.isfinite(th) & (th <= t_res)
            i0 = int(np.argmin(np.abs(th - tp[0])))
            ax.semilogy(th[m], np.abs(Eh[m] / Eh[i0]), color=COLOR[mdl], lw=1.2, label=f"Hermite {mdl}")
            ax.semilogy(tp, np.abs(pic[f"{mdl}_E1"] / pic[f"{mdl}_E1"][0]), color=COLOR[mdl], lw=2.4, alpha=0.35,
                        label=f"PIC {mdl}")
        ax.axvline(tf[i], color="0.3", lw=0.8)
        shade_uncertified(ax, t_res, c["T"], label="Hermite not resolved")
        ax.set(xlim=(0, c["T"]), ylabel=r"$|E_{x,k}(t)/E_{x,k}(0.75)|$")
        time_axis(ax)
        ax.legend(loc="lower right", fontsize=7, ncol=3)
        fig.suptitle(f"{c['title']}: Hermite Nn = {c['Nn']} vs Dark-JAX-in-Cell {pic_rec['cells']} cells, "
                     f"{2 * pic_rec['particles_per_beam']:,} markers", fontsize=10)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        p.close(fig)
    out = STATIC / "djic_phase"
    out.mkdir(parents=True, exist_ok=True)
    frames[0].save(out / "two_stream.webp", save_all=True, append_images=frames[1:], duration=120, loop=0,
                   quality=70, method=6)
    frames[min(n - 1, int(np.searchsorted(tf, 0.8 * tf[-1])))].save(out / "figure.png")
    gr = {}
    for mdl in ("ordinary", "dark"):  # growth of |E_k1| over the same window, both codes
        w = (1 / 3 * tf[-1], 2 / 3 * tf[-1])
        key = f"{case}_{mdl}_Nn{c['Nn']}"
        th, Eh = herm[f"{key}_t"], np.abs(herm[f"{key}_E1"])
        mh, mp = (th >= w[0]) & (th <= w[1]), (tp >= w[0]) & (tp <= w[1])
        gr[mdl] = {"window": w, "hermite": float(np.polyfit(th[mh], np.log(Eh[mh]), 1)[0]),
                   "pic": float(np.polyfit(tp[mp], np.log(np.abs(pic[f"{mdl}_E1"][mp])), 1)[0])}
    finish("djic_phase", None, {"sources": ["docs/_static/phase_space/run.json", "studies/djic_phase/run.json",
                                            "artifacts/djic_phase/two_stream.npz (python studies/djic_phase_space.py)"],
                                "frames": n, "t_last_frame": float(tf[-1]),
                                "webp_bytes": (out / "two_stream.webp").stat().st_size, "growth_fits": gr})


FAMILIES = {"landau": fig_landau, "growth": fig_growth, "grid": fig_grid, "djic": fig_djic,
            "conversion": fig_conversion, "conservation": fig_conservation, "performance": fig_performance,
            "phase": fig_phase, "djic_phase": fig_djic_phase}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    names = [n for n in FAMILIES if "phase" not in n] if argv[:1] == ["all"] else argv[:1]
    if not names or names[0] not in FAMILIES:
        raise SystemExit(f"usage: python studies/figures.py {{{'|'.join(FAMILIES)}|all}}")
    for n in names:
        FAMILIES[n](*(argv[1:] if len(names) == 1 else []))


if __name__ == "__main__":
    main()
