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
         "$P_{\\rm kin}/P_{\\rm lin}$ & $T_e/T_e(0)$ & K2\\\\\\midrule\n")
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
T.append("\\newcommand{\\tabVerify}{%\n\\begin{tabular}{@{}p{5.6cm}p{10.4cm}@{}}\\toprule\nCheck & Result (record)\\\\\\midrule\n")
T.append("Linear Landau roots, ordinary and dark, $k\\lambda_{De}=0.2$--$0.7$ & worst relative error " + sci(max(r["rel_error"] for r in b00r))
         + " vs.\\ SciPy Faddeeva roots; ledger defect $\\le$ " + sci(max(r["max_ledger_defect"] for r in b00r), 0) + "\\\\\n")
T.append("OTSI growth vs.\\ Silin kinetic relation (sideband determinant) & " + f"{len(og)} unstable points, worst difference {max(abs(a / b - 1) for a, b in og) * 100:.1f}\\%\\\\\n")
T.append("Dark-JAX-in-Cell PIC, Landau $k\\lambda_{De}=0.5$ & rms $\\Delta\\ln|E_k|$ falls "
         + f"{pic['cells32']['rms_log_amplitude_diff_ordinary_Ek_2_12']:.2f}$\\to${pic['cells128']['rms_log_amplitude_diff_ordinary_Ek_2_12']:.2f} "
         + "from 32 to 128 cells; PIC noise limits agreement to $\\sim$1\\%\\\\\n")
T.append("GPU vs.\\ CPU, one case ($T=100$) & max relative difference " + sci(c08["max_rel"]) + " (adaptive trajectories)\\\\\n")
c09 = J("studies/c09/run.json")
T.append("% C09 numbers: docs/results.md section C09 and docs/status.md C09 row\n")
T.append("AD vs.\\ finite differences & reverse = forward to $2\\times10^{-13}$, analytic to $5\\times10^{-10}$ "
         "(\\texttt{studies/c09/run.json}, 19 derivatives)\\\\\n")
T.append("\\bottomrule\\end{tabular}}\n")

T.append("\\newcommand{\\tabSeed}{%\n\\begin{tabular}{@{}lcccc@{}}\\toprule\n"
         "seed scale $s$ & $D(100)$ & $D(200)$ & $D(300)$ & $D(400)$\\\\\\midrule\n")
for key, lab in (("s1", "1"), ("s0.1", "0.1"), ("s0", "0 (unseeded)")):
    T.append(lab + " & " + " & ".join(sci(fD[key][str(x)], 2) for x in (100, 200, 300, 400)) + "\\\\\n")
T.append("\\bottomrule\\end{tabular}}\n")

(OUT / "tables_gen.tex").write_text("".join(T))
lines = ["% generated by paper/make_figures.py; do not edit\n"]
for k, (v, src) in sorted(NUM.items()):
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}} % {src}\n")
(OUT / "numbers.tex").write_text("".join(lines))
print("ok", len(NUM), "macros")
