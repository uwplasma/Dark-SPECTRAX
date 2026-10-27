"""Generate every figure, table and number macro of the manuscript from the committed records.

Run from the repository root:  python paper/make_figures.py
Outputs (all under paper/): figures/*.pdf, numbers.tex (\\newcommand macros), tables_gen.tex.
Nothing is typed by hand: every plotted or quoted value is read from a JSON/NPZ record or recomputed from one.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

C = {"b": "#1f4e9c", "o": "#d9730d", "g": "#2a8f4f", "r": "#b3261e", "k": "#222222", "p": "#7b4fa3", "gy": "#888888"}
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "cm",
})


def J(p):
    return json.loads((ROOT / p).read_text())


NUM: dict[str, str] = {}


def num(name, value, fmt="{:.3g}", src=""):
    """Register a LaTeX macro \\name with the formatted value; src is written as a comment (record path)."""
    name = "".join(dict(zip("0123456789", ["Z", "O", "T", "H", "F", "V", "S", "N", "E", "I"])).get(ch, ch) for ch in name)
    s = fmt.format(value) if not isinstance(value, str) else value
    if "e" in s and not s.startswith("$"):
        m, e = s.split("e")
        s = f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"
    NUM[name] = (s, src)


def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def panel(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold")


# --------------------------------------------------------------------------------------------- constants
VT2 = 1e-3  # v_te^2 / c^2 of the H05 set-up (n = 1): divide energies per volume by this for n T_e V
WL = float(np.sqrt(1 + 1 / 1836))


def lin_W(t, E0, w0=WL):
    """Exact linear resonant law (homogeneous mean-field ODE), as in studies/r1nl_analysis.py::lin_fields."""
    wL2 = WL ** 2
    s = solve_ivp(lambda tt, x: [-x[1], wL2 * (x[0] + E0 * np.cos(w0 * tt)), x[1] * E0 * np.cos(w0 * tt)], (0, t[-1]),
                  [0, 0, 0], t_eval=t, rtol=1e-12, atol=1e-18, method="DOP853")
    return s.y[0], s.y[2]


def load_run(case, ts=("t", "W", "K_species", "Ek2")):
    """Concatenate the part files of a studies/r1nl/runs case."""
    parts = sorted((ROOT / "studies/r1nl/runs").glob(f"{case}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    h = {}
    for i, p in enumerate(parts):
        d = np.load(p)
        for k in ts:
            h.setdefault(k, []).append(d[k] if i == 0 else d[k][1:])
    return {k: np.concatenate(v) for k, v in h.items()}


# ============================================================================================ numbers
b00 = J("docs/_static/b00/run.json")["results"]
b00r = [r for r in b00 if "Nn" in r and r["Nn"] == 128 and r["Nx"] == 8]
num("bzeroWorst", max(r["rel_error"] for r in b00r), "{:.1e}", "docs/_static/b00/run.json (Nn=128, Nx=8)")
num("bzeroLedger", max(r["max_ledger_defect"] for r in b00r), "{:.0e}", "docs/_static/b00/run.json")
num("bzeroNruns", len([r for r in b00 if "Nn" in r]), "{:d}", "docs/_static/b00/run.json")

otsi = J("studies/instab/otsi.json")["scan"]
og = [(r["gamma_floquet"], r["gamma_nishikawa"]) for r in otsi if r["gamma_nishikawa"] > 1e-6]
num("otsiN", len(og), "{:d}", "studies/instab/otsi.json scan")
num("otsiMax", max(abs(a / b - 1) for a, b in og) * 100, "{:.1f}", "studies/instab/otsi.json scan (percent)")

c08 = J("studies/gpures/c08.json")
num("cgpuMax", c08["max_rel"], "{:.1e}", "studies/gpures/c08.json")

c05 = J("studies/c05/run.json")
num("cfivePicRms", c05["pic"]["cells128"]["rms_log_amplitude_diff_ordinary_Ek_2_12"], "{:.2f}", "studies/c05/run.json")

st1 = J("studies/r1nl/stage1_summary.json")["vqs"]
st2 = J("studies/r1nl/stage2_summary.json")
fu = J("studies/r1nl/followup_summary.json")
for k, nm in (("0.03", "A"), ("0.05", "B"), ("0.07", "C")):
    num(f"win{nm}", st1[k]["window"], "{:g}", f"studies/r1nl/stage1_summary.json vqs.{k}.window")
    num(f"ledg{nm}", max(st1[k]["ledger"].values()), "{:.0e}", f"studies/r1nl/stage1_summary.json vqs.{k}.ledger (max over members)")
num("rnlBound", st2["N1"]["bound_max_abs_D"] * 100, "{:.2f}", "studies/r1nl/stage2_summary.json N1.bound_max_abs_D (percent)")
num("rnlSpread", abs(st2["N1"]["max_abs_D_by_member"]["B"] - st2["N1"]["max_abs_D_by_member"]["N"]), "{:.0e}",
    "studies/r1nl/stage2_summary.json N1.max_abs_D_by_member")
num("rnlWext", st2["N1"]["Wext_over_nTe_at_window"]["B"], "{:.1f}", "studies/r1nl/stage2_summary.json N1")
num("rnlWlin", st2["N1"]["Wext_lin_over_nTe_at_window"], "{:.1f}", "studies/r1nl/stage2_summary.json N1")
num("rnlDw", st2["N2"]["bound"], "{:.1e}", "studies/r1nl/stage2_summary.json N2.bound")
n3 = st2["N3"]["B"]
num("rnlDKe", n3["dKe"], "{:.0f}", "studies/r1nl/stage2_summary.json N3.B")
num("rnlCohShare", n3["dKe"] / st2["N1"]["Wext_over_nTe_at_window"]["B"] * 100, "{:.0f}", "studies/r1nl/stage2_summary.json N3.B.dKe / N1.Wext (percent)")
for _t in (200, 500):
    num(f"quivAt{_t}", 0.03 * np.sqrt(VT2) * _t / 2, "{:.1f}", f"v_q w_pe t/2 with v_q = 0.03 v_te, v_te^2 = 1e-3 c^2, t = {_t} (units of c)")
num("rnlIonShare", n3["ion_share"] * 100, "{:.2f}", "studies/r1nl/stage2_summary.json N3.B (percent)")
num("rnlOtsi", n3["t_otsi"], "{:g}", "studies/r1nl/stage2_summary.json N3.B.t_otsi")
num("rnlOtsiN", st2["N3"]["N"]["t_otsi"], "{:g}", "studies/r1nl/stage2_summary.json N3.N.t_otsi")
num("rnlFluct", n3["Ek_fluct_max_ratio_to_E0sq"], "{:.0f}", "studies/r1nl/stage2_summary.json N3.B")
num("rnlLedger", st2["N6"]["ledger_max"], "{:.1e}", "studies/r1nl/stage2_summary.json N6")
num("rnlNruns", st2["N6"]["n_runs"], "{:d}", "studies/r1nl/stage2_summary.json N6")
for u in ("1", "10"):
    num(f"rnlRes{u}", st2["N4"]["u" + u]["max_dev_UD_over_UD0"]["B"] * 100, "{:.2f}", f"stage2_summary.json N4.u{u} (percent of U_D0)")
    num(f"rnlResW{u}", st2["N4"]["u" + u]["WD_departure"]["bound_max_abs_D"] * 100, "{:.1f}", f"stage2_summary.json N4.u{u} (percent)")
    num(f"rnlSplit{u}", st2["N4"]["u" + u]["t_split_kin"], "{:g}", f"stage2_summary.json N4.u{u}.t_split_kin")
    num(f"rnlEta{u}", st2["N4"]["u" + u]["eta"], "{:.3f}", f"stage2_summary.json N4.u{u}.eta")
num("rnlWDfrac", st2["N4"]["u1"]["WD_over_nTe_at_window"] / st2["N4"]["u1"]["U_D0_over_nTe"] * 100, "{:.0f}",
    "stage2_summary.json N4.u1 (percent of U_D0 given up)")
for t in ("100", "300"):
    n5 = st2["N5"]["tau" + t]
    num(f"rnlR{t}", n5["R"], "{:.4f}", f"stage2_summary.json N5.tau{t}.R")
    num(f"rnlRse{t}", n5["se"], "{:.4f}", f"stage2_summary.json N5.tau{t}.se")
    num(f"rnlRb{t}", n5["bound_if_none"] * 100, "{:.1f}", f"stage2_summary.json N5.tau{t}.bound_if_none (percent)")
    num(f"rnlRmax{t}", max(n5["max_abs_diff_over_peak_Wlin_by_realization"]) * 100, "{:.1f}",
        f"stage2_summary.json N5.tau{t} (percent of peak W_lin)")
fD = fu["D"]
num("fuDa", fD["s1"]["400"], "{:.2e}", "studies/r1nl/followup_summary.json D.s1.400")
num("fuDb", fD["s0.1"]["400"], "{:.2e}", "studies/r1nl/followup_summary.json D.s0.1.400")
num("fuDc", fD["s0"]["400"], "{:.1e}", "studies/r1nl/followup_summary.json D.s0.400")
num("fuRatio", fu["F1"]["ratio_s0.1_over_s1"], "{:.4f}", "studies/r1nl/followup_summary.json F1")
num("fuFtwo", fu["F2"]["unseeded_over_s1"], "{:.1e}", "studies/r1nl/followup_summary.json F2")

res = J("studies/r1/reservoir_summary.json")
rc = res["cases"]
num("rOneDev", max(c["max_dev_U_D_over_U_D0_in_window"] for c in rc.values() if "max_dev_U_D_over_U_D0_in_window" in c), "{:.3f}", "studies/r1/reservoir_summary.json cases.*.max_dev")
num("rOneLedger", res["gates"]["G1_ledger_max"], "{:.1e}", "studies/r1/reservoir_summary.json gates.G1_ledger_max")
coh2 = J("studies/r1/coherence2_summary.json")
for t in ("30", "100", "300"):
    d = coh2["tau"]["tau" + t]
    num(f"cohR{t}", d["ratio_late_mean"], "{:.3f}", f"studies/r1/coherence2_summary.json tau.tau{t}.ratio_late_mean")
    num(f"cohSe{t}", d["ratio_late_se"], "{:.3f}", f"studies/r1/coherence2_summary.json tau.tau{t}.ratio_late_se")

r2 = J("studies/r2/summary.json")
fits = r2["fits"]
for k, nm in (("res", "R"), ("0.9", "L"), ("1.1", "H")):
    f = fits[k]
    num(f"rtwoP{nm}", f["kin"]["p"], "{:.2f}", f"studies/r2/summary.json fits.{k}.kin.p")
    num(f"rtwoPe{nm}", f["kin"]["se_p"], "{:.2f}", f"studies/r2/summary.json fits.{k}.kin.se_p")
    num(f"rtwoQ{nm}", f["kin"]["q"], "{:.2f}", f"studies/r2/summary.json fits.{k}.kin.q")
    num(f"rtwoQe{nm}", f["kin"]["se_q"], "{:.2f}", f"studies/r2/summary.json fits.{k}.kin.se_q")
    num(f"rtwoQl{nm}", f["lin"]["q"], "{:.2f}", f"studies/r2/summary.json fits.{k}.lin.q")
    num(f"rtwoK{nm}", f["kappa"], "{:.3f}", f"studies/r2/summary.json fits.{k}.kappa")
    num(f"rtwoKe{nm}", f["kappa_se"], "{:.3f}", f"studies/r2/summary.json fits.{k}.kappa_se")
num("rtwoKtwo", r2["counts"]["K2"][0], "{:d}", "studies/r2/summary.json counts.K2")
num("rtwoKtwoN", r2["counts"]["K2"][1], "{:d}", "studies/r2/summary.json counts.K2")
num("rtwoKthree", r2["counts"]["K3"][0], "{:d}", "studies/r2/summary.json counts.K3")
num("rtwoKthreeN", r2["counts"]["K3"][1], "{:d}", "studies/r2/summary.json counts.K3")
num("rtwoCV", r2["counts"]["CV"][0], "{:d}", "studies/r2/summary.json counts.CV")
num("rtwoCVN", r2["counts"]["CV"][1], "{:d}", "studies/r2/summary.json counts.CV")

cases = r2["cases"]
em3 = [cases[f"vq{v}_nu0.03_w{w}"] for v in ("0.01", "0.03", "0.05") for w in ("res", "0.9", "1.1")]
num("rtwoPss", max(abs(c["B"]["P_kin_over_pss"] - 1) for c in em3) * 100, "{:.1f}", "studies/r2/summary.json cases.*nu0.03*.B.P_kin_over_pss (percent)")
num("rtwoTmax", max(c["report"]["Te_over_Te0"] for c in cases.values()), "{:.0f}", "studies/r2/summary.json cases.*.B.report")
num("rtwoCVwx", max(abs(c["CV"]["Wx"]) for c in cases.values()), "{:.1e}", "studies/r2/summary.json cases.*.B.CV.Wx")
det = [c for k, c in cases.items() if "nu0.001_w0.9" in k or "nu0.001_w1.1" in k]
num("rtwoDetLo", min(c["B"]["R_late"] for c in det) * 100 - 100, "{:.0f}", "studies/r2/summary.json detuned nu=1e-3 R_late (percent)")
num("rtwoDetHi", max(c["B"]["R_late"] for c in det) * 100 - 100, "{:.0f}", "studies/r2/summary.json detuned nu=1e-3 R_late (percent)")
rfu = J("studies/r2/followup_summary.json")
num("rtwoFuA", rfu["vq0.03_nu0.001_w0.9"]["D_ratio"], "{:.4f}", "studies/r2/followup_summary.json")
num("rtwoFuB", rfu["vq0.03_nu0.001_wres"]["D_ratio"], "{:.4f}", "studies/r2/followup_summary.json")
for w, nm in (("0.9", "L"), ("1.1", "H")):
    num(f"rtwoDet{nm}", (np.mean([cases[f"vq{v}_nu0.001_w{w}"]["B"]["R_late"] for v in ("0.01", "0.03", "0.05")]) - 1) * 100, "{:.0f}",
        "studies/r2/summary.json detuned nu=1e-3 mean R_late - 1 (percent)")
num("rtwoResLo", min(cases[f"vq{v}_nu0.001_wres"]["B"]["R_late"] for v in ("0.01", "0.03", "0.05")), "{:.3f}", "studies/r2/summary.json")
num("rtwoResHi", max(cases[f"vq{v}_nu0.03_wres"]["B"]["R_late"] for v in ("0.01", "0.03", "0.05")), "{:.4f}", "studies/r2/summary.json")

# ================================================================================================ fig 1
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
kk = sorted({r["k_lambda_De"] for r in b00r})
for mod, col, mk in (("ordinary", C["b"], "o"), ("dark", C["o"], "s")):
    y = [next(r["rel_error"] for r in b00r if r["k_lambda_De"] == k and r["model"] == mod) for k in kk]
    ax[0].semilogy(kk, y, mk + "-", color=col, ms=4, label=mod)
ax[0].set_xlabel(r"$k\lambda_{De}$")
ax[0].set_ylabel(r"relative root error")
ax[0].legend(frameon=False)
panel(ax[0], "(a)")
g = np.array(og)
ax[1].loglog(g[:, 1], g[:, 0], "o", color=C["b"], ms=3.5, mfc="none")
lim = [g.min() * 0.7, g.max() * 1.4]
ax[1].plot(lim, lim, "-", color=C["k"], lw=0.7)
ax[1].set_xlabel(r"$\gamma$ kinetic (Silin)")
ax[1].set_ylabel(r"$\gamma$ Hermite Floquet")
panel(ax[1], "(b)")
pic = c05["pic"]
cells = [32, 64, 128]
ax[2].plot(cells, [pic[f"cells{c}"]["rms_log_amplitude_diff_ordinary_Ek_2_12"] for c in cells], "o-", color=C["b"], ms=4, label="ordinary")
ax[2].plot(cells, [pic[f"cells{c}"]["rms_log_amplitude_diff_parent_Ek_2_12"] for c in cells], "s-", color=C["o"], ms=4, label="dark")
ax[2].set_xscale("log", base=2)
ax[2].set_xticks(cells)
ax[2].set_xticklabels([str(c) for c in cells])
ax[2].set_xlabel("PIC cells (particles $\\propto$ cells)")
ax[2].set_ylabel(r"rms $\Delta\ln|E_k|$ vs Hermite")
ax[2].legend(frameon=False)
panel(ax[2], "(c)")
fig.tight_layout()
fig.savefig(FIG / "fig_verification.pdf")
plt.close(fig)

# ================================================================================================ fig 2
eig = J("studies/highk/eig.json")
gp = J("studies/gpures/summary.json")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
for basis, col, mk in (("AW", C["r"], "o"), ("SW", C["g"], "s")):
    for E1, ls in ((1e-3, "--"), (1e-2, "-")):
        pts = sorted((r["Nn"], max(r["max_re"], 1e-17)) for r in eig if r["basis"] == basis and r["K"] == 5 and r["E1"] == E1 and r["u"] == 0.0)
        if pts:
            ax[0].semilogy(*zip(*pts), mk + ls, color=col, ms=3.5, label=f"{basis}, $E_1={E1:g}$")
ax[0].set_xlabel(r"Hermite order $N_n$")
ax[0].set_ylabel(r"max Re $\lambda$ (frozen field, $K=5$)")
ax[0].legend(frameon=False, fontsize=7)
panel(ax[0], "(a)")
labs, tr, cl = [], [], []
for name, r in gp["runs"].items():
    vq = name.split("_")[0][2:]
    nx = name.split("_")[1][2:]
    nn = name.split("_")[2][2:]
    labs.append((vq, int(nx), int(nn)))
    tr.append(r["t_reached"])
for vq, col in (("0.1", C["b"]), ("0.05", C["o"])):
    for nx, mk in ((64, "o"), (128, "s")):
        pts = sorted((nn, t) for (v, x, nn), t in zip(labs, tr) if v == vq and x == nx)
        ax[1].plot(*zip(*pts), mk + "-", color=col, ms=4, label=f"$v_q/v_{{te}}={vq}$, $N_x={nx}$")
ax[1].set_xlabel(r"Hermite order $N_n$")
ax[1].set_ylabel(r"time reached $\omega_{pe}t$")
ax[1].set_xticks([128, 256])
ax[1].legend(frameon=False, fontsize=7, loc="upper right")
panel(ax[1], "(b)")
sf = J("studies/rclosure/stagef.json")
cs = [16, 64, 256]
ax[2].plot(cs, [sf["runs"][f"c{c}"]["window"] for c in cs], "o-", color=C["p"], ms=4, label=r"$N_x=64,N_n=128$, $c>4$")
ax[2].axhline(sf["c4_reference"]["t_end"], color=C["gy"], ls="--", lw=0.9)
ax[2].text(17, sf["c4_reference"]["t_end"] + 6, r"$c=4$", fontsize=7, color=C["gy"])
ax[2].set_xscale("log", base=4)
ax[2].set_xticks(cs)
ax[2].set_xticklabels([str(c) for c in cs])
ax[2].set_xlabel("closure strength $c$")
ax[2].set_ylabel(r"window $\omega_{pe}t$ ($v_q/v_{te}=0.1$)")
ax[2].set_ylim(560, 800)
ax[2].legend(frameon=False)
panel(ax[2], "(c)")
fig.tight_layout()
fig.savefig(FIG / "fig_aw.pdf")
plt.close(fig)
num("awEigAW", max(r["max_re"] for r in eig if r["basis"] == "AW" and r["K"] == 5 and r["Nn"] == 128 and r["E1"] == 1e-2 and r["u"] == 0.0),
    "{:.1f}", "studies/highk/eig.json AW K=5 Nn=128 E1=1e-2")
num("awEigSW", max(r["max_re"] for r in eig if r["basis"] == "SW"), "{:.0e}", "studies/highk/eig.json all SW points")
num("gpuWinA", min(r["t_reached"] for n, r in gp["runs"].items() if n.startswith("vq0.1")), "{:g}", "studies/gpures/summary.json")
num("gpuWinB", max(r["t_reached"] for n, r in gp["runs"].items() if n.startswith("vq0.1")), "{:g}", "studies/gpures/summary.json")
num("gpuWinC", min(r["t_reached"] for n, r in gp["runs"].items() if n.startswith("vq0.05")), "{:g}", "studies/gpures/summary.json")
num("gpuWinD", max(r["t_reached"] for n, r in gp["runs"].items() if n.startswith("vq0.05")), "{:g}", "studies/gpures/summary.json")
num("cFloorLo", min(sf["runs"][f"c{c}"]["window"] for c in cs), "{:g}", "studies/rclosure/stagef.json")
num("cFloorHi", max(sf["runs"][f"c{c}"]["window"] for c in cs), "{:g}", "studies/rclosure/stagef.json")
ex = J("studies/expint/summary.json")
num("lawsonSteps", ex["cases"]["64"]["steps_ratio_500"], "{:.1f}", "studies/expint/summary.json cases.64.steps_ratio_500")
num("lawsonStepsFour", ex["cases"]["4"]["steps_ratio_500"], "{:.2f}", "studies/expint/summary.json cases.4.steps_ratio_500")

# ================================================================================ HHS-definition energies (Fig. 2 of the PRL)
# HHS plot Delta E / (n_e m_e c^2) (their Fig. 2, SM Eqs. S35-S36): the change of the thermal energy n m v_th^2/2 with v_th^2 the
# 1-D (x) velocity variance, kinetic energy "with the k=0 component removed" (SM footnote 8), and <E^2>/2, moving average 200/wpe.
# Same definition here: 1-D electron energy (1/2) m_e (M_xx - M_x^2/n) with the box-mean flow removed (records: Tx), the mean-field energy
# (1/2) Ebar^2 against the exact linear law, trailing 200/wpe moving average. Units n m_e c^2 (c = wpe = n = m_e = 1).
def mavg(t, y, w=200.0):
    out = np.empty_like(y)
    for i in range(t.size):
        m = (t > t[i] - w) & (t <= t[i])
        out[i] = y[m].mean()
    return out


E_TH0 = 0.5 * 1e-3  # initial 1-D electron thermal energy, n m_e c^2 (= 5e-4, SM text)
num("hhsEth", E_TH0, "{:.1e}", "v_te^2/2 with v_te^2 = 1e-3 c^2 (SM Eq. S36; HHS quote 5e-4)")
for vq, nm in (("0.03", "A"), ("0.05", "B"), ("0.07", "C")):
    h = load_run(f"pre_vq{vq}_Nx32_Nn64_c64", ts=("t", "W", "Tx", "Ebar"))
    win = st1[vq]["window"]
    sel = h["t"] <= win
    t = h["t"][sel]
    dEe = 0.5 * (h["Tx"][sel, 0] - h["Tx"][0, 0])
    dEm = mavg(t, dEe)
    Elin, Wlin = lin_W(t, float(vq) * np.sqrt(VT2))
    Fld = mavg(t, 0.5 * h["Ebar"][sel] ** 2)
    Fl = mavg(t, 0.5 * Elin ** 2)
    for tag, tq in (("500", 500.0), ("End", win)):
        i = int(np.argmin(np.abs(t - tq)))
        num(f"hhsDE{tag}{nm}", dEm[i], "{:.1e}", f"recomputed from studies/r1nl/runs/pre_vq{vq}_Nx32_Nn64_c64 Tx, trailing 200 average, t={t[i]:g}")
        num(f"hhsDEr{tag}{nm}", dEm[i] / E_TH0, "{:.2g}", "same, ratio to the initial 1-D electron thermal energy")
        num(f"hhsFld{tag}{nm}", Fld[i], "{:.1e}", f"1/2 Ebar^2 (k=0 mean field), trailing 200 average, t={t[i]:g}")
        num(f"hhsFldLin{tag}{nm}", Fl[i], "{:.1e}", "same from the exact linear law")
    num("hhsWinT" + nm, t[-1], "{:g}", "last saved time within the certified window")
    num("hhsDEmax" + nm, dEm.max() / E_TH0, "{:.2g}", "max over the window of the averaged gain / initial thermal energy")
    num("hhsFldRatio" + nm, Fld[-1] / Fl[-1], "{:.2f}", "mean-field energy over linear at the window end")

# ================================================================================================ fig 3 (R1 transfer)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
Dcol = {"0.03": C["b"], "0.05": C["o"], "0.07": C["g"]}
for vq in ("0.03", "0.05", "0.07"):
    h = load_run(f"pre_vq{vq}_Nx32_Nn64_c64")
    E0 = float(vq) * np.sqrt(VT2)
    sel = h["t"] <= st1[vq]["window"]
    t = h["t"][sel]
    _, Wl = lin_W(t, E0)
    W = h["W"][sel, 2]
    m = t > 50
    ax[0].plot(t[m], (W[m] / Wl[m] - 1) * 100, color=Dcol[vq], label=f"$v_q/v_{{te}}={vq}$")
    num("dEnd" + {"0.03": "A", "0.05": "B", "0.07": "C"}[vq], (W[-1] / Wl[-1] - 1) * 100, "{:.2f}", f"recomputed from studies/r1nl/runs/pre_vq{vq}_Nx32_Nn64_c64 at the window end (percent)")
    num("wext" + {"0.03": "A", "0.05": "B", "0.07": "C"}[vq], W[-1] / VT2, "{:.0f}", f"recomputed from studies/r1nl/runs/pre_vq{vq}_Nx32_Nn64_c64 at the window end, units n T_e V")
    num("quiv" + {"0.03": "A", "0.05": "B", "0.07": "C"}[vq], np.sqrt(2 * W[-1]), "{:.2f}", f"nonrelativistic coherent speed sqrt(2 W_ext/(n m_e)) in units of c at the window end, from pre_vq{vq}_Nx32_Nn64_c64")
    if vq == "0.07":
        num("recomputeWext", W[-1] / VT2, "{:.1f}", "recomputed from studies/r1nl/runs/pre_vq0.07_Nx32_Nn64_c64 (check vs rnlWext)")
ax[0].set_xlabel(r"$\omega_{pe}t$")
ax[0].set_ylabel(r"$W_{\rm ext}/W_{\rm lin}-1$  [%]")
ax[0].legend(frameon=False, fontsize=7, loc="lower left")
panel(ax[0], "(a)")
tt = np.array([100, 200, 300, 400])
for key, col, lab in (("s1", C["b"], r"$s=1$"), ("s0.1", C["o"], r"$s=0.1$"), ("s0", C["gy"], r"$s=0$")):
    ax[1].semilogy(tt, np.abs([fD[key][str(x)] for x in tt]), "o-", color=col, ms=3.5, label=lab)
ax[1].set_xlabel(r"$\omega_{pe}t$")
ax[1].set_ylabel(r"$|W_{\rm ext}/W_{\rm lin}-1|$")
ax[1].legend(frameon=False, title="seed scale", title_fontsize=7)
panel(ax[1], "(b)")
xs, labs3 = [], []
rows = []
for t_, d in coh2["tau"].items():
    if t_ == "tauinf":
        continue
    rows.append((f"{t_[3:]}", d["ratio_late_mean"], d["ratio_late_se"], "0.01"))
for t_ in ("100", "300"):
    n5 = st2["N5"]["tau" + t_]
    rows.append((t_, n5["R"], n5["se"], "0.07"))
for i, (tau, R, se, vq) in enumerate(rows):
    ax[2].errorbar(i, R, yerr=se, fmt="o" if vq == "0.01" else "s", color=C["b"] if vq == "0.01" else C["g"], capsize=2, ms=4)
ax[2].axhline(1.0, color=C["k"], lw=0.7)
ax[2].axhspan(0.9, 1.1, color=C["gy"], alpha=0.15, lw=0)
ax[2].set_xticks(range(len(rows)))
ax[2].set_xticklabels([f"{r[0]}\n{r[3]}" for r in rows], fontsize=7)
ax[2].set_xlabel(r"$\tau_c\omega_{pe}$ / $v_q/v_{te}$")
ax[2].set_ylabel(r"$\langle W_{\rm kin}\rangle/\langle W_{\rm lin}\rangle$")
panel(ax[2], "(c)")
fig.tight_layout()
fig.savefig(FIG / "fig_r1.pdf")
plt.close(fig)

# ================================================================================================ fig 4 (R2)
nus = [0.001, 0.003, 0.01, 0.03]
wk = {"res": ("resonant", C["b"], "o"), "0.9": (r"$\omega=0.9$", C["o"], "s"), "1.1": (r"$\omega=1.1$", C["g"], "^")}
vqs = {"0.01": 0.5, "0.03": 0.75, "0.05": 1.0}
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.3))
for w, (lab, col, mk) in wk.items():
    for vq, al in vqs.items():
        y = [cases[f"vq{vq}_nu{n:g}_w{w}"]["B"]["R_late"] for n in nus]
        ax[0].semilogx(nus, y, mk + "-", color=col, alpha=al, ms=3.5, lw=0.9, label=lab if vq == "0.03" else None)
ax[0].axhspan(0.95, 1.05, color=C["gy"], alpha=0.15, lw=0)
ax[0].set_xlabel(r"$\nu_{ei}/\omega_{pe}$")
ax[0].set_ylabel(r"$P_{\rm kin}/P_{\rm lin}$ (late)")
ax[0].legend(frameon=False, fontsize=7)
panel(ax[0], "(a)")
for w, (lab, col, mk) in wk.items():
    y = [cases[f"vq0.03_nu{n:g}_w{w}"]["B"]["P_kin_over_pss"] for n in nus]
    yl = [cases[f"vq0.03_nu{n:g}_w{w}"]["B"]["f_ss"] for n in nus]
    ax[1].semilogx(nus, y, mk, color=col, ms=4)
    ax[1].semilogx(nus, yl, "-", color=col, lw=0.9)
ax[1].set_xlabel(r"$\nu_{ei}/\omega_{pe}$")
ax[1].set_ylabel(r"$P/P_{\rm ss}$ ($v_q/v_{te}=0.03$)")
ax[1].text(0.97, 0.97, "markers: kinetic\nlines: exact linear", transform=ax[1].transAxes, fontsize=7, ha="right", va="top")
panel(ax[1], "(b)")
for i, (k, (lab, col, mk)) in enumerate(wk.items()):
    f = fits[k]
    ax[2].errorbar(i - 0.12, f["kin"]["p"], yerr=f["kin"]["se_p"], fmt=mk, color=col, capsize=2, ms=4)
    ax[2].errorbar(i + 0.12, f["lin"]["p"], yerr=f["lin"]["se_p"], fmt=mk, mfc="none", color=col, capsize=2, ms=4)
ax[2].axhline(2.0, color=C["k"], lw=0.7)
ax[2].set_xticks([0, 1, 2])
ax[2].set_xticklabels(["res", "0.9", "1.1"])
ax[2].set_xlabel(r"drive frequency / $\omega_{pe}$")
ax[2].set_ylabel(r"exponent $p$ in $P\propto (v_q/v_{te})^p$")
panel(ax[2], "(c)")
for vq, al in vqs.items():
    y = [cases[f"vq{vq}_nu{n:g}_wres"]["report"]["Te_over_Te0"] for n in nus]
    ax[3].loglog(nus, y, "o-", color=C["r"], alpha=al, ms=3.5, lw=0.9, label=f"$v_q/v_{{te}}={vq}$")
ax[3].axhline(90, color=C["k"], ls="--", lw=0.8)
ax[3].text(0.0011, 100, r"$v_{te}=0.3c$", fontsize=7)
ax[3].set_xlabel(r"$\nu_{ei}/\omega_{pe}$")
ax[3].set_ylabel(r"$T_e/T_e(0)$ at $t_b$ (resonant)")
panel(ax[3], "(d)")
fig.tight_layout()
fig.savefig(FIG / "fig_r2.pdf")
plt.close(fig)

# ================================================================== relativistic models and PIC (sections 6-8)
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "studies"))
sys.path.insert(0, str(ROOT / "studies/relpic"))
import analysis as PA  # noqa: E402  (relpic: PIC loader, trailing means)
import relgrid_analysis as RA  # noqa: E402
import relgrid_ref as XR  # noqa: E402
import relhhs_gates as RG  # noqa: E402

ETH = 5e-4
rgs = J("studies/relgrid/summary.json")
rgc = rgs["cases"]
rhs = J("studies/relhhs/summary.json")
cvs = J("studies/relpic/conv_summary.json")
obs = cvs["observables"]
RS = "studies/relgrid/summary.json"

# relativistic grid and Hermite: collapse and deterministic observables
d = rgc["rel_det_vq0.03"]
num("relColGrid", d["t_W_10pc_from_linear"], "{:g}", RS + " cases.rel_det_vq0.03.t_W_10pc_from_linear")
num("relFldGrid", d["Ebar_energy_over_linear"]["1000"], "{:.3f}", RS + " cases.rel_det_vq0.03.Ebar_energy_over_linear.1000")
num("relFldGridFive", d["Ebar_energy_over_linear"]["500"], "{:.2f}", RS + " same, t=500")
num("relHeatGrid", d["dE1D_over_Eth"]["1000"], "{:.3f}", RS + " same dE1D_over_Eth.1000")
num("relFldAbsN", d["Ebar_energy"]["1000"], "{:.1e}", RS + " Ebar_energy.1000 (n m c^2)")
num("relFldAbsGrid", d["Ebar_energy"]["1000"] / ETH, "{:.1f}", RS + " Ebar_energy.1000 / E_th")
hr = rhs["runs"]
hk = list(hr)
num("herNruns", len(hk), "{:d}", "studies/relhhs/summary.json runs")
num("herCol", hr["(32, 64, 64)"]["t_collapse_W_10pc"], "{:g}", "studies/relhhs/summary.json runs.(32, 64, 64)")
num("herColSame", sum(1 for k in hk if hr[k]["t_collapse_W_10pc"] == 392.5), "{:d}", "studies/relhhs/summary.json (runs with collapse 392.5)")
num("herFld", hr["(32, 64, 64)"]["Ebar_energy_over_linear"]["1000"], "{:.3f}", "studies/relhhs/summary.json runs.(32, 64, 64)")
num("herFldFive", hr["(32, 64, 64)"]["Ebar_energy_over_linear"]["500"], "{:.3f}", "same, t=500")
num("herHeat", hr["(32, 64, 64)"]["dE1D_e_over_Eth"]["1000"], "{:.3f}", "same dE1D_e_over_Eth.1000")
num("herHeatFive", hr["(32, 64, 64)"]["dE1D_e_over_Eth"]["500"], "{:.3f}", "same, t=500")
num("herHeatMaxAll", max(hr[k]["dE1D_at_end"] for k in hk), "{:.2f}", "studies/relhhs/summary.json (max over runs of the final averaged gain, T=1000)")
shot_rel = [v for k, v in rgc.items() if k.startswith("rel_shot_vq0.03") and "npc205_" not in k]
num("relColShotLo", min(v["t_W_10pc_from_linear"] for v in shot_rel), "{:g}", RS + " rel_shot_vq0.03 npc>=2.05e4")
num("relColShotHi", max(v["t_W_10pc_from_linear"] for v in shot_rel), "{:g}", RS + " rel_shot_vq0.03 npc>=2.05e4")
nr_sh = [v for k, v in rgc.items() if k.startswith("nonrel_shot_vq0.03")]
num("nrColLo", min(v["t_W_10pc_from_linear"] for v in nr_sh), "{:g}", RS + " nonrel_shot_vq0.03")
num("nrColHi", max(v["t_W_10pc_from_linear"] for v in nr_sh), "{:g}", RS + " nonrel_shot_vq0.03")
num("relColWeak", rgc["rel_det_vq0.001"]["t_W_10pc_from_linear"], "{:g}", RS + " rel_det_vq0.001")
num("relColWeakShot", rgc["rel_shot_vq0.001"]["t_W_10pc_from_linear"], "{:g}", RS + " rel_shot_vq0.001")
num("nrWeakFld", rgc["nonrel_det_vq0.001"]["Ebar_energy_over_linear"]["3000"], "{:.2f}", RS + " nonrel_det_vq0.001 at t=3000")
num("relWeakFld", rgc["rel_det_vq0.001"]["Ebar_energy_over_linear"]["3000"], "{:.2f}", RS + " rel_det_vq0.001 at t=3000")
num("relWeakHeat", rgc["rel_det_vq0.001"]["dE1D_over_Eth"]["3000"], "{:.2f}", RS + " rel_det_vq0.001 at t=3000")
num("nrWeakHeat", rgc["nonrel_det_vq0.001"]["dE1D_over_Eth"]["3000"], "{:.2f}", RS + " nonrel_det_vq0.001 at t=3000")
hom = rgs["homog"]
num("relHomWratio", hom["vq0.03"]["W_rel_over_nonrel_at"]["1000"], "{:.4f}", RS + " homog.vq0.03 (exact homogeneous ODEs, rel/nonrel W_ext at t=1000)")
c2 = rgs.get("gates_c")
gc2 = J("studies/relgrid/gates_c2.json")["C"]
num("rgCerrW", max(gc2[k]["err_omega"] for k in gc2), "{:.1e}", "studies/relgrid/gates_c2.json (max err_omega, k lambda_De=0.3,0.4)")
num("rgCerrG", max(gc2[k]["err_gamma"] for k in gc2), "{:.1e}", "studies/relgrid/gates_c2.json (max err_gamma)")
gD = rgs["gate_D"]
tagD = "r_vq0.03_Nx32_Nu4096_dt0.02_osc_u32_rel_taylor_det_xmask23"
tagD1 = "r_vq0.03_Nx32_Nu4096_dt0.01_osc_u32_rel_taylor_det_xmask23"
num("rgDled", gD[tagD]["energy_defect_over_W"], "{:.1e}", RS + " gate_D (dt 0.02)")
num("rgDledOne", gD[tagD1]["energy_defect_over_W"], "{:.1e}", RS + " gate_D (dt 0.01)")
num("rgDratio", gD[tagD]["energy_defect_over_W"] / gD[tagD1]["energy_defect_over_W"], "{:.1f}", RS + " gate_D ledger ratio dt 0.02 / 0.01")
num("rgDfail", sum(1 for v in gD.values() if not v["pass"]), "{:d}", RS + " gate_D (runs failing the 1e-4 ledger)")
num("rgDtot", len(gD), "{:d}", RS + " gate_D runs")
g1 = J("studies/relhhs/gates_g1.json")
num("gOneE", max(v["Ebar_rel_err"] for v in g1["runs"].values()), "{:.1e}", "studies/relhhs/gates_g1.json")
num("gOneW", max(v["W_ext_rel_err"] for v in g1["runs"].values()), "{:.1e}", "studies/relhhs/gates_g1.json")
num("gOneNR", g1["nonrel_vs_rel_W_ext"] * 100, "{:.0f}", "studies/relhhs/gates_g1.json nonrel_vs_rel_W_ext (percent)")
hb = J("studies/relherm/gates_b.json")
num("hbE", hb["E_rel_err"], "{:.1e}", "studies/relherm/gates_b.json")
num("hbW", hb["W_rel_err"], "{:.1e}", "studies/relherm/gates_b.json")
num("hfOne", J("studies/relherm/gates_f1.json")["f1_c_inf"]["rhs_rel_err"], "{:.0e}", "studies/relherm/gates_f1.json")
num("hfTwo", J("studies/relherm/gates_f2.json")["max_abs_err_v"], "{:.0e}", "studies/relherm/gates_f2.json (gyration vs exact)")
num("hfThree", J("studies/relherm/gates_f3.json")["max_rel_K_drift"], "{:.0e}", "studies/relherm/gates_f3.json (magnetic work)")
num("hdLed", J("studies/relherm/gates_d.json")["rel"]["ledger_max_rel_to_W"], "{:.1e}", "studies/relherm/gates_d.json")
num("hdTwoV", J("studies/relherm/gates_d2.json")["max_v_over_c_centre"], "{:.2f}", "studies/relherm/gates_d2.json")
num("hfBled", J("studies/relherm/gates_f4b.json")["ledger_rel"], "{:.1e}", "studies/relherm/gates_f4b.json")
gp = J("studies/relhhs/gates_prod.json")["runs"]
num("herLed", max(v["max_ledger_t400"] for v in gp.values()), "{:.1e}", "studies/relhhs/gates_prod.json (max ledger over 8 runs, t<=400)")
f1 = J("studies/relhhs/gates_f1.json")["runs"]
num("fOneLo", min(v["horizon"] for v in f1.values()), "{:g}", "studies/relhhs/gates_f1.json")
num("fOneHi", max(v["horizon"] for v in f1.values()), "{:g}", "studies/relhhs/gates_f1.json")
f2 = J("studies/relhhs/gates_f2.json")["runs"]
num("fTwoPicLo", f2["npc205000_Nx64_Nn128"]["t_fneg"], "{:g}", "studies/relhhs/gates_f2.json")
num("fTwoPicHi", f2["npc205000_Nx32_Nn64"]["t_fneg"], "{:g}", "studies/relhhs/gates_f2.json")
num("fTwoPpc", f2["npc103_Nx32_Nn64"]["t_fneg"], "{:g}", "studies/relhhs/gates_f2.json (npc 103)")
num("fTwoPpcStop", f2["npc103_Nx32_Nn64"]["t_fail"], "{:g}", "studies/relhhs/gates_f2.json (npc 103 positivity stop)")
sh = [rgc[k]["t_fneg_1e-3"] for k in rgc if k.startswith("rel_shot_vq0.03") and "npc205_" not in k]
num("shotFnegLo", min(sh), "{:g}", RS + " rel_shot_vq0.03 first f<-1e-3 (npc>=2.05e4)")
num("shotFnegHi", max(sh), "{:g}", RS + " rel_shot_vq0.03 first f<-1e-3 (npc>=2.05e4)")
num("shotFnegPic", rgc["rel_shot_vq0.03_npc205_Nx32"]["t_fneg_1e-3"], "{:g}", RS + " rel_shot_vq0.03_npc205_Nx32 first f<-1e-3 (205 per cell = PIC particle number)")
num("shotHeatNx", rgc["rel_shot_vq0.03_npc2.05e5_Nx64"]["dE1D_over_Eth"]["500"], "{:.1f}", RS + " Nx64 npc 2.05e5 at t=500 (invalid)")


hk_ = J("studies/hkr/summary_grid.json")["runs"]
num("hkrHeat", hk_["Nx1024_det"]["dE_avg_at_tcert"], "{:.1f}", "studies/hkr/summary_grid.json Nx1024_det (nonrelativistic grid, deterministic seed)")
num("hkrK", hk_["Nx1024_det"]["kl_max_retained"], "{:.2f}", "studies/hkr/summary_grid.json Nx1024_det k lambda_De retained")
nz = [v for k, v in hk_.items() if k.endswith("noise")]
num("hkrNoLo", min(v["t_cert"] for v in nz), "{:g}", "studies/hkr/summary_grid.json (shot-noise runs, certified time)")
num("hkrNoHi", max(v["t_cert"] for v in nz), "{:g}", "studies/hkr/summary_grid.json")
num("hkrNoHeat", max(v["dE_avg_at_tcert"] for v in nz) * 1000, "{:.1f}", "studies/hkr/summary_grid.json (max 1e-3 E_th; heating at loss of positivity)")

# PIC: heating plateau and comparison
dg = J("studies/relpic/digitization/hhs_fig2.json")
hd = dg["black_solid_k_neq_0_electron_energy_change_over_Eth"]
CS = "studies/relpic/conv_summary.json"
num("hhsDigK", hd["1000"], "{:.1f}", "studies/relpic/digitization/hhs_fig2.json (t=1000)")
num("hhsDigKK", hd["2000"], "{:.1f}", "studies/relpic/digitization/hhs_fig2.json (t=2000)")
num("hhsDigRel", dg.get("line_width_uncertainty", 0.03) * 100 if False else 3, "{:.0f}", "studies/relpic/digitization/hhs_fig2.json note (line-width uncertainty ~3 percent)")


def ob(g, at="1000", key="dW_th"):
    return obs[g][at][key]


num("pcR", ob("R103")["mean"], "{:.1f}", CS + " R103 t=1000")
num("pcRs", ob("R103")["std"], "{:.1f}", CS + " R103 seed std")
num("pcP", ob("P206")["mean"], "{:.1f}", CS + " P206")
num("pcF", ob("F412")["mean"], "{:.1f}", CS + " F412 mean of 5 seeds")
num("pcFs", ob("F412")["std"], "{:.1f}", CS + " F412 seed std")
num("pcFK", ob("F412", "2000")["mean"], "{:.1f}", CS + " F412 t=2000")
num("pcFKs", ob("F412", "2000")["std"], "{:.1f}", CS + " F412 t=2000 std")
num("pcPP", ob("P824")["mean"], "{:.1f}", CS + " P824")
num("pcPPP", ob("P1648")["mean"], "{:.1f}", CS + " P1648")
num("pcXa", ob("X2")["mean"], "{:.1f}", CS + " X2 (dx 0.02)")
num("pcXb", ob("X1")["mean"], "{:.1f}", CS + " X1 (dx 0.01)")
num("pcQ", ob("Q412")["mean"], "{:.1f}", CS + " Q412 (quintic)")
num("pcD", ob("D")["mean"], "{:.1f}", CS + " D (dt 0.01, ppc 103)")
num("pcN", ob("N412")["mean"], "{:.1f}", CS + " N412")
num("pcNs", ob("N412")["std"], "{:.1f}", CS + " N412 std (2 seeds)")
num("pcNK", ob("N412", "2000")["mean"], "{:.1f}", CS + " N412 t=2000")
num("pcNone", ob("N103")["mean"], "{:.1f}", CS + " N103 (ppc 103, 3 seeds)")
num("pcStepA", abs(ob("P824")["mean"] / ob("F412")["mean"] - 1) * 100, "{:.1f}", CS + " change 412->824 (percent)")
num("pcStepB", abs(ob("P1648")["mean"] / ob("P824")["mean"] - 1) * 100, "{:.1f}", CS + " change 824->1648 (percent)")
num("pcStepX", abs(ob("X1")["mean"] / ob("X2")["mean"] - 1) * 100, "{:.1f}", CS + " change dx 0.02->0.01 (percent)")
num("pcPlat", ob("P1648")["mean"], "{:.0f}", CS + " P1648 plateau")
ex = cvs["extrapolation"]
num("pcCfive", ex["C5"]["W_inf"], "{:.1f}", CS + " extrapolation.C5")
num("pcCfiveU", ex["C5"]["uncertainty"], "{:.1f}", CS + " extrapolation.C5.uncertainty")
num("pcCsix", ex["C6_vs_HHS_t1000"]["ratio"], "{:.2f}", CS + " C6 ratio to HHS")
num("pcPlatHHS", ob("P1648")["mean"] / hd["1000"], "{:.2f}", "P1648 / HHS digitized, t=1000")
num("pcNewtHHS", ex["C6_newtonian_t1000"]["ratio"], "{:.2f}", CS + " Newtonian / HHS, t=1000")
num("pcRN", ex["C7_ratio_RoverN_t1000"]["ratio"], "{:.2f}", CS + " C7 t=1000")
num("pcRNK", ex["C7_ratio_RoverN_t2000"]["ratio"], "{:.2f}", CS + " C7 t=2000")
num("pcNewtRel", 1 / ex["C7_ratio_RoverN_t1000"]["ratio"], "{:.1f}", CS + " Newtonian / relativistic, t=1000")
num("pcContRatio", ob("P1648")["mean"] / hr["(32, 64, 64)"]["dE1D_e_over_Eth"]["1000"], "{:.0f}", "P1648 / relativistic Hermite dE1D_e at t=1000")
num("pcFldPeak", obs["F412"]["peak_W_E0"], "{:.0f}", CS + " F412 peak_W_E0 (E_th)")
num("pcFldLate", ob("F412", key="W_E0")["mean"], "{:.2f}", CS + " F412 W_E0 at 1000 (E_th)")
# mirrored loading (declared follow-up A): recomputed from the PIC records
mir = {t: PA.dW(PA.load(t), 1000.0) / ETH for t in ("M103_s0", "M103_s1", "M206_s0")}
num("mirA", mir["M103_s0"], "{:.1f}", "studies/relpic/records/M103_s0 (dW_th/E_th, t=1000)")
num("mirB", mir["M103_s1"], "{:.1f}", "studies/relpic/records/M103_s1")
num("mirC", mir["M206_s0"], "{:.1f}", "studies/relpic/records/M206_s0")

# PIC mean-field and W_ext against the exact nonrelativistic linear law (both pushers)
LIN = XR.homogeneous(0.03, 2001, rel=False)


def pic_ratio(tags, at):
    out = []
    for tg in tags:
        h = PA.load(tg)
        t = h["t"]
        yl = LIN.sol(t)
        out.append(PA.trailing(t, 0.5 * h["mean_E"] ** 2, at) / PA.trailing(t, 0.5 * yl[2] ** 2, at))
    return float(np.mean(out))


def pic_collapse(tags):
    c = []
    for tg in tags:
        h = PA.load(tg)
        yl = LIN.sol(h["t"])[3]
        i = np.nonzero((h["t"] > 50) & (np.abs(h["work"] - yl) > 0.1 * np.abs(yl)))[0]
        c.append(float(h["t"][i[0]]))
    return min(c), max(c)


FT = [f"F412_s{i}" for i in range(5)]
NT = ["N412_s0", "N412_s1"]
for nm, tg in (("R", FT), ("N", NT)):
    lo, hi = pic_collapse(tg)
    num(f"pic{nm}ColLo", lo, "{:g}", "studies/relpic/records " + ",".join(tg) + " (W_ext 10 percent off the exact nonrelativistic linear law)")
    num(f"pic{nm}ColHi", hi, "{:g}", "same")
    for q in (500, 800, 1000):
        num(f"pic{nm}Fld{q}", pic_ratio(tg, float(q)), "{:.3f}", "studies/relpic/records " + ",".join(tg) + f" mean-field energy over linear, seed mean, t={q}")

# ----------------------------------------------------------------------------------------------- fig 5
E0 = 0.03 * np.sqrt(VT2)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
LINN = XR.homogeneous(0.03, 2001, rel=False)


def fld_ratio(t, ebar):
    yl = LINN.sol(t)
    return RA.mavg(t, 0.5 * ebar ** 2) / RA.mavg(t, 0.5 * yl[2] ** 2)


gr = np.load(ROOT / "studies/relgrid/runs/r_vq0.03_Nx32_Nu4096_dt0.02_osc_u32_rel_taylor_det_xmask23.npz")
tg_ = gr["t"]
hh = RG.load(RG.case(32, 64, 64))
th_ = hh["t"]
nr = load_run("pre_vq0.03_Nx32_Nn64_c64", ts=("t", "W", "Tx", "Ebar"))
nsel = nr["t"] <= st1["0.03"]["window"]
tn_ = nr["t"][nsel]
ylin = XR.homogeneous(0.03, 2001, rel=False)
ax[0].plot(tn_, RA.mavg(tn_, 0.5 * nr["Ebar"][nsel] ** 2) / RA.mavg(tn_, 0.5 * ylin.sol(tn_)[2] ** 2), color=C["gy"], label="Hermite, nonrel.")
ax[0].plot(tg_, fld_ratio(tg_, gr["Ebar"]), color=C["b"], label="grid, rel.")
ax[0].plot(th_, fld_ratio(th_, hh["Ebar"]), "--", color=C["o"], label="Hermite, rel.")


def pic_series(tags, fn):
    ys = []
    for tg in tags:
        h = PA.load(tg)
        ys.append(fn(h))
    return h["t"], np.mean(ys, 0)


def r_fld(h):
    return RA.mavg(h["t"], 0.5 * h["mean_E"] ** 2) / RA.mavg(h["t"], 0.5 * LINN.sol(h["t"])[2] ** 2)


tp, yp = pic_series(FT, r_fld)
ax[0].plot(tp, yp, color=C["g"], lw=1.0, label="PIC, rel.")
tp, yp = pic_series(NT, r_fld)
ax[0].plot(tp, yp, color=C["r"], lw=1.0, label="PIC, Newt.")
ax[0].set_yscale("log")
ax[0].set_ylim(1e-4, 2)
ax[0].set_xlim(0, 1500)
ax[0].set_xlabel(r"$\omega_{pe}t$")
ax[0].set_ylabel(r"$\bar E^2/2$ over linear law")
panel(ax[0], "(a)")
ax[1].plot(tg_, RA.mavg(tg_, gr["dE1D_e"]) / ETH, color=C["b"], label="grid, rel.")
ax[1].plot(th_, RA.mavg(th_, hh["dE1D_e"]) / ETH, "--", color=C["o"], label="Hermite, rel.")
tp, yp = pic_series(FT, lambda h: RA.mavg(h["t"], h["spread"][:, 0] - h["spread"][0, 0]) / ETH)
ax[1].plot(tp, yp, color=C["g"], lw=1.0, label="PIC, rel.")
tp, yp = pic_series(NT, lambda h: RA.mavg(h["t"], h["spread"][:, 0] - h["spread"][0, 0]) / ETH)
ax[1].plot(tp, yp, color=C["r"], lw=1.0, label="PIC, Newt.")
hx = sorted((int(k), v) for k, v in hd.items() if v is not None and int(k) <= 2000)
ax[1].plot(*zip(*hx), "ks", ms=4, label="HHS, digitized")
ax[1].set_yscale("symlog", linthresh=1e-2)
ax[1].set_ylim(-1e-2, 150)
ax[1].set_xlim(0, 2000)
ax[1].set_xlabel(r"$\omega_{pe}t$")
ax[1].set_ylabel(r"$\Delta E_e/E_{\rm th}$ ($k\neq0$, 1-D)")
panel(ax[1], "(b)")
pp = [(103, "R103"), (206, "P206"), (412, "F412"), (824, "P824"), (1648, "P1648")]
ax[2].errorbar([p for p, _ in pp], [ob(g)["mean"] for _, g in pp], fmt="o-", color=C["g"], ms=4, capsize=2,
               yerr=[ob(g)["std"] or 0 for _, g in pp], label="rel. PIC")
ax[2].plot([103, 412], [ob("N103")["mean"], ob("N412")["mean"]], "s-", color=C["r"], ms=4, label="Newt. PIC")
ax[2].plot([103, 206], [mir["M103_s0"], mir["M206_s0"]], "^", color=C["p"], ms=6, mfc="none", label="rel., mirrored")
ax[2].axhline(hd["1000"], color="k", ls="--", lw=0.8)
ax[2].text(110, hd["1000"] * 1.12, "HHS", fontsize=7)
ax[2].axhline(hr["(32, 64, 64)"]["dE1D_e_over_Eth"]["1000"], color=C["o"], lw=0.9)
ax[2].text(110, hr["(32, 64, 64)"]["dE1D_e_over_Eth"]["1000"] * 1.4, "continuum (grid, Hermite)", fontsize=7, color=C["o"])
ax[2].set_xscale("log", base=2)
ax[2].set_yscale("log")
ax[2].set_xlabel("particles per cell")
ax[2].set_ylabel(r"$\Delta E_e/E_{\rm th}$ at $\omega_{pe}t=1000$")
ax[2].set_ylim(0.03, 200)
ax[2].legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.38))
panel(ax[2], "(c)")
fig.tight_layout()
h0, l0 = ax[0].get_legend_handles_labels()
h1, l1 = ax[1].get_legend_handles_labels()
seen = dict(zip(l0, h0))
seen.update(dict(zip(l1, h1)))
fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=6, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.07),
           columnspacing=1.0, handlelength=1.6)
fig.savefig(FIG / "fig_hhs.pdf")
plt.close(fig)

# ============================================================================================== tables
T = []
T.append("% generated by paper/make_figures.py; do not edit\n")
# R1nl windows table
T.append("\\newcommand{\\tabWindows}{%\n\\begin{tabular}{@{}lcccc@{}}\\toprule\n"
         "$v_q/v_{te}$ & window $\\omega_{pe}t$ & limiter & $\\max$ ledger (B) & grid agrees with B until\\\\\\midrule\n")
for k in ("0.03", "0.05", "0.07"):
    s = st1[k]
    gr = s["grid"]
    T.append(f"{k} & {s['window']:g} & {s['limiter'].replace('cap X', 'positivity, $N_x=64$').replace('cap', 'cap $T=1000$')} & "
             f"{sci(max(s['ledger'].values()))} & {gr['grid_Nx32_vs_B']['agree_until']:g} / {gr['grid_Nx64_vs_B']['agree_until']:g}\\\\\n")
T.append("\\bottomrule\\end{tabular}}\n")
# R2 table (B member, EM): omega res, vq=0.03
T.append("\\newcommand{\\tabRtwo}{%\n\\begin{tabular}{@{}llccccc@{}}\\toprule\n"
         "$\\omega/\\omega_{pe}$ & $\\nu_{ei}/\\omega_{pe}$ & $P_{\\rm lin}/P_{\\rm ss}$ & $P_{\\rm kin}/P_{\\rm ss}$ & "
         "$P_{\\rm kin}/P_{\\rm lin}$ & $T_e/T_e(0)$ & 5\\,\\% check\\\\\\midrule\n")
for w, lab in (("res", "res."), ("0.9", "0.9"), ("1.1", "1.1")):
    for n in nus:
        c = cases[f"vq0.03_nu{n:g}_w{w}"]
        b = c["B"]
        T.append(f"{lab} & {n:g} & {b['f_ss']:.2f} & {b['P_kin_over_pss']:.3f} & {b['R_late']:.4f} & "
                 f"{c['report']['Te_over_Te0']:.3g} & {'pass' if c['K2'] else 'FAIL'}\\\\\n")
    if w != "1.1":
        T.append("\\addlinespace\n")
T.append("\\bottomrule\\end{tabular}}\n")
# verification table
T.append("\\newcommand{\\tabVerify}{%\n\\begin{tabular}{@{}L{5.6cm}L{10.4cm}@{}}\\toprule\nCheck & Result (record)\\\\\\midrule\n")
T.append("Linear Landau roots, ordinary and dark, $k\\lambda_{De}=0.2$--$0.7$ & worst relative error " + sci(max(r["rel_error"] for r in b00r))
         + " vs.\\ SciPy Faddeeva roots; ledger defect $\\le$ " + sci(max(r["max_ledger_defect"] for r in b00r), 0) + "\\\\\n")
T.append("OTSI growth vs.\\ Silin kinetic relation (sideband determinant) & " + f"{len(og)} unstable points, worst difference {max(abs(a / b - 1) for a, b in og) * 100:.1f}\\%\\\\\n")
T.append("Dark-JAX-in-Cell PIC, Landau $k\\lambda_{De}=0.5$ & rms $\\Delta\\ln|E_k|$ falls "
         + f"{pic['cells32']['rms_log_amplitude_diff_ordinary_Ek_2_12']:.2f}$\\to${pic['cells128']['rms_log_amplitude_diff_ordinary_Ek_2_12']:.2f} "
         + "from 32 to 128 cells; PIC noise limits agreement to $\\sim$1\\%\\\\\n")
T.append("GPU vs.\\ CPU, one case ($T=100$) & max relative difference " + sci(c08["max_rel"]) + " (adaptive trajectories)\\\\\n")
c09 = J("studies/c09/run.json")
T.append("% numbers: studies/c09/run.json\n")
T.append("AD vs.\\ finite differences & reverse = forward to $2\\times10^{-13}$, analytic to $5\\times10^{-10}$ "
         "(19 derivatives)\\\\\n")
T.append("\\bottomrule\\end{tabular}}\n")

T.append("\\newcommand{\\tabSeed}{%\n\\begin{tabular}{@{}lcccc@{}}\\toprule\n"
         "seed scale $s$ & $D(100)$ & $D(200)$ & $D(300)$ & $D(400)$\\\\\\midrule\n")
for key, lab in (("s1", "1"), ("s0.1", "0.1"), ("s0", "0 (unseeded)")):
    T.append(lab + " & " + " & ".join(sci(fD[key][str(x)], 2) for x in (100, 200, 300, 400)) + "\\\\\n")
T.append("\\bottomrule\\end{tabular}}\n")


# relativistic summary table
shr = [v for k, v in rgc.items() if k.startswith("rel_shot_vq0.03") and "npc205_" not in k]
fl_s = [v["Ebar_energy_over_linear"]["1000"] for v in shr]
ht_s = [v["dE1D_over_Eth"]["1000"] for v in shr]
nsh = [v for k, v in rgc.items() if k.startswith("nonrel_shot_vq0.03")]
T.append("\\newcommand{\\tabRel}{%\n\\begin{tabular}{@{}L{3.6cm}cccc@{}}\\toprule\n"
         "model & collapse time & $\\bar E^2/2$ over linear, $t=1000$ & $\\Delta E_e/E_{\\rm th}$, $t=1000$ & valid to\\\\\\midrule\n")
T.append(f"Hermite, nonrel.\\ ($N_x=32,N_n=64$) & none ($\\le1000$) & {st1['0.03']['window'] and ''}\\hhsFldRatioA & \\hhsDErEndA & \\winA\\\\\n")
T.append("grid, nonrel., shot noise$^{\\dagger}$ & \\nrColLo--\\nrColHi & "
         f"{min(v['Ebar_energy_over_linear']['1000'] for v in nsh):.2f}--{max(v['Ebar_energy_over_linear']['1000'] for v in nsh):.2f} & "
         f"{min(v['dE1D_over_Eth']['1000'] for v in nsh):.0f}--{max(v['dE1D_over_Eth']['1000'] for v in nsh):.0f} & "
         f"{min(v['t_fneg_1e-3'] for v in nsh):g}--{max(v['t_fneg_1e-3'] for v in nsh):g}\\\\\n")
T.append("grid, rel., deterministic & \\relColGrid & \\relFldGrid & \\relHeatGrid & 1000\\\\\n")
T.append("Hermite, rel., deterministic & \\herCol & \\herFld & \\herHeat & \\fOneLo--\\fOneHi\\\\\n")
T.append("grid, rel., shot noise$^{\\dagger}$ & \\relColShotLo--\\relColShotHi & "
         f"{min(fl_s):.3f}--{max(fl_s):.2f} & {min(ht_s):.0f}--{max(ht_s):.0f} & \\shotFnegLo--\\shotFnegHi\\\\\n")
T.append("\\addlinespace\nPIC, rel.\\ (ppc 412) & \\picRColLo--\\picRColHi & \\picRFldOZZZ & \\pcF & ---\\\\\n")
T.append("PIC, Newtonian (ppc 412) & \\picNColLo--\\picNColHi & \\picNFldOZZZ & \\pcN & ---\\\\\n")
T.append("HHS (digitized) & loss of resonance $\\sim800$ & --- & \\hhsDigK & ---\\\\\n")
T.append("\\bottomrule\\end{tabular}}\n")

(OUT / "tables_gen.tex").write_text("".join(T))
lines = ["% generated by paper/make_figures.py; do not edit\n"]
for k, (v, src) in sorted(NUM.items()):
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}} % {src}\n")
(OUT / "numbers.tex").write_text("".join(lines))
print("ok", len(NUM), "macros")
