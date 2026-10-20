"""R3 figures from the records (docs/_static/r3/*.png). Run: python studies/r3_figures.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_ref as ref  # noqa: E402
from r3_analysis import kin_name, ref_name  # noqa: E402

root = Path(__file__).resolve().parents[1]
D = root / "studies/r3"
OUT = root / "docs/_static/r3"
OUT.mkdir(parents=True, exist_ok=True)


def L(name):
    p = D / name
    return json.loads(p.read_text()) if p.exists() else None


# 1: transverse conversion vs w (eta 0.15, 0.3): reference, simple LZ, refined, kinetic Nx 512
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ws = np.linspace(2, 12, 21)
for ax, eta in zip(axs, (0.15, 0.3)):
    wl = (2, 3, 4, 6, 8, 12)
    for key, col, lab in (("trans_dark", "C0", "transmitted dark"), ("refl_photon", "C1", "reflected photon"), ("refl_dark", "C2", "reflected dark")):
        pts = []
        for w in wl:
            r = L(ref_name(eta, float(w), None, "y"))
            if r:
                c = r["channels"]
                v = c[key] + (c["wrap_" + key] if "wrap_" + key in c else 0.0)
                pts.append((w, v))
        ax.plot(*zip(*pts), "o", color=col, label=lab + " (cold ref)")
        lzs = [ref.lz_prediction(ref.Case(eta, w, None, "y"))[key] for w in ws]
        rfs = [ref.refined_prediction(ref.Case(eta, w, None, "y"))[key] for w in ws[::4]]
        ax.plot(ws, lzs, "--", color=col, lw=1)
        ax.plot(ws[::4], rfs, "x", color=col, ms=7)
        kin = []
        for w in wl:
            k = L(kin_name(eta, float(w), None, "y"))
            if k:
                c = k["channels"]
                kin.append((w, c[key] + (c["wrap_" + key] if "wrap_" + key in c else 0.0)))
        if kin:
            ax.plot(*zip(*kin), "s", mfc="none", color=col, ms=9)
    ax.set_title(f"transverse, B0 = 0, eta = {eta}")
    ax.set_xlabel("ramp width w  (L_c = 0.77 w at B0 = 0)")
axs[0].set_ylabel("energy fraction")
axs[0].legend(fontsize=7)
fig.text(0.5, -0.02, "dots: cold reference; dashed: Landau-Zener law; crosses: refined (complex turning point) law; open squares: Dark-SPECTRAX Nx 512", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "transverse_conversion.png", dpi=130, bbox_inches="tight")

# 2: exponent ratio vs eta
f = json.loads((D / "followup_summary.json").read_text())
fig, ax = plt.subplots(figsize=(5, 3.6))
rows = [r for r in f["FU4"]["rows"] if r["gated"] and r["theta"] is None]
ax.plot([r["eta_fnc_over_kbar2"] for r in rows], [r["exponent_ratio"] for r in rows], "o")
s = np.linspace(0, 0.4, 50)
ax.plot(s, 1 - 4.5 * s ** 2, "k:", label="1 - 4.5 s^2 (fit)")
ax.set_xlabel("s = eta f n_c / kbar^2")
ax.set_ylabel("ln T_ref / ln P_LZ")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "lz_exponent_ratio.png", dpi=130)

# 3: longitudinal
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, eta in zip(axs, (0.15, 0.3)):
    wl = (2, 3, 4, 6, 8, 12)
    for key, col in (("trans_dark", "C0"), ("refl_dark", "C2"), ("stored", "C3")):
        pts = [(w, L(ref_name(eta, float(w), None, "x"))["channels"][key]) for w in wl if L(ref_name(eta, float(w), None, "x"))]
        ax.plot(*zip(*pts), "o-", color=col, label=key + " (cold ref)")
    pts = [(w, L(ref_name(eta, float(w), None, "x"))["analytic"]["wkb"]["T_wkb"]) for w in wl if L(ref_name(eta, float(w), None, "x"))]
    ax.plot(*zip(*pts), "k--", label="WKB tunnelling exp(-2 int kappa)")
    ax.set_title(f"longitudinal, B0 = 0, eta = {eta}")
    ax.set_xlabel("ramp width w")
k = L(kin_name(0.3, 4.0, None, "x", Nx=3072, Nn=4))
if k:
    c = k["channels"]
    axs[1].plot([4], [c["trans_dark"]], "s", mfc="none", color="C0", ms=10)
    axs[1].plot([4], [c["refl_dark"] + c["wrap_refl_dark"]], "s", mfc="none", color="C2", ms=10, label="Dark-SPECTRAX Nx 3072")
axs[0].set_ylabel("energy fraction")
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "longitudinal_conversion.png", dpi=130)

# 4: B0 effect at w = 4, eta 0.3 (cold reference)
fig, ax = plt.subplots(figsize=(10, 3.8))
labels, data = [], []
for th, pol in ((None, "y"), (None, "x"), (0, "y"), (0, "x"), (45, "x"), (45, "y"), (45, "z"), (90, "x"), (90, "y"), (90, "z")):
    r = L(ref_name(0.3, 4.0, th, pol))
    c = r["channels"]
    labels.append(("B0=0" if th is None else f"th={th}") + f" {pol}")
    data.append([c["trans_dark"], c["refl_dark"] + c["wrap_refl_dark"], c["refl_photon"] + c["wrap_refl_photon"], c["stored"], c["trans_photon"]])
data = np.array(data)
bottom = np.zeros(len(labels))
for j, (nm, col) in enumerate((("transmitted dark", "C0"), ("reflected dark", "C2"), ("reflected photon", "C1"), ("stored at layer", "C3"), ("transmitted photon", "C4"))):
    ax.bar(labels, data[:, j], bottom=bottom, color=col, label=nm)
    bottom += data[:, j]
ax.set_ylabel("energy fraction")
ax.legend(fontsize=7, ncol=5, loc="lower center")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(OUT / "b0_effect.png", dpi=130)
print("figures written")
