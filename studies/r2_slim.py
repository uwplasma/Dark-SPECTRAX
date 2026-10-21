"""R2: write slim records studies/r2/records/<case>.{npz,json} from the full run directory, and the figure docs/_static/r2/r2_absorption.png.
Run (where the full runs are): python studies/r2_slim.py [runs_dir]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_analysis as an  # noqa: E402
import r2_gates as g  # noqa: E402
import r2_run  # noqa: E402
import r2_theory as th  # noqa: E402


def main(argv):
    if argv:
        an.RUNS = Path(argv[0])
    rec = HERE / "r2" / "records"
    rec.mkdir(parents=True, exist_ok=True)
    names = sorted({p.name.split(".part")[0] for p in an.RUNS.glob("*.part0.json")})
    for n in names:
        h = an.load(n)
        f4 = lambda x: np.asarray(x, np.float32)  # noqa: E731  (slim records are float32; the analysis used the full float64 runs)
        if "_control" not in n:
            full = n.startswith("G_") or ("Nx32_Nn64" in n)
            arr = dict(t=f4(h["t"]), W_ext=f4(h["Wx"]), dK_e=f4(h["dKe"]))
            if full:
                arr.update(dK_i=f4(h["K_species"][:, 1] - h["K_species"][0, 1]), T_e=f4(h["Te"]), T_i=f4(h["Ti"]),
                           Q_i_cum=f4(h["Qcum"][:, 1]))
            np.savez_compressed(rec / f"{n}.npz", **arr)
        parts = [{k: v for k, v in r.items() if k != "failure_reason" or v is not None} for r in h["recs"]]
        (rec / f"{n}.json").write_text(json.dumps(parts, indent=1, default=float) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ = json.loads((HERE / "r2" / "summary.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    col = {"res": "C3", "0.9": "C0", "1.1": "C2"}
    for tag in g.OMEGA:
        nus = g.NU
        rk = [summ["cases"][f"vq0.03_nu{n:g}_w{tag}"]["B"] for n in nus]
        ax[0].loglog(nus, [r["P_kin"] for r in rk], "o", color=col[tag], label=f"kinetic, omega = {tag}")
        ax[0].loglog(nus, [r["P_lin"] for r in rk], "-", color=col[tag], lw=1)
        ax[0].loglog(nus, [r["p_ss"] for r in rk], "--", color=col[tag], lw=1)
        ax[1].semilogx(nus, [r["R_late"] for r in rk], "o-", color=col[tag], label=tag)
    ax[0].set_xlabel("nu_ei"), ax[0].set_ylabel("late rate (code units)"), ax[0].legend(fontsize=7)
    ax[0].set_title("v_q/v_te = 0.03: kinetic (dots), exact linear (solid), steady state (dashed)", fontsize=8)
    ax[1].axhspan(0.95, 1.05, color="0.9"), ax[1].set_xlabel("nu_ei"), ax[1].set_ylabel("P_kin / P_lin"), ax[1].legend(title="omega")
    ax[1].set_title("late-rate ratio, v_q/v_te = 0.03", fontsize=8)
    for lab, c, tag, nu, ls in (("res, nu 1e-2", "C3", "res", 1e-2, "-"), ("res, nu 3e-2", "C1", "res", 3e-2, "-"),
                                 ("0.9, nu 1e-3", "C0", "0.9", 1e-3, "-")):
        h = an.load(r2_run.case_name("EM", 0.03, nu, tag, 32, 64, False))
        ax[2].plot(h["t"], h["Wx"], color=c, ls=ls, label=lab + " kinetic")
        ax[2].plot(h["t"], th.exact_linear(0.03, nu, r2_run.omega_of(tag), h["t"])[1], "k:", lw=1)
    ax[2].set_xlabel("t omega_pe"), ax[2].set_ylabel("W_ext"), ax[2].set_yscale("log"), ax[2].set_ylim(1e-6, 1), ax[2].legend(fontsize=7)
    ax[2].set_title("W_ext(t): kinetic vs exact linear (dotted)", fontsize=8)
    out = HERE.parent / "docs" / "_static" / "r2"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "r2_absorption.png", dpi=110)


if __name__ == "__main__":
    main(sys.argv[1:])
