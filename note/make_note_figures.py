"""The note's own calculation of relativistic detuning at k = 0 (not a study record).

Integrates the homogeneous collisionless relativistic Vlasov equation (48 Gauss-Hermite momentum nodes, cold ions,
m_i/m_e = 1836, resonant drive at omega_L) and the slow-flow model of Chapter 3, and writes
figures/k0_relativistic.json and figures/fig_cold_relativistic.pdf.  The records quoted for comparison are the
relativistic Hermite collapse time (paper/numbers.tex, herCol) and the deterministic relativistic grid
(studies/relgrid/summary.json).  Chapters 5 and 10 refer back to this calculation; check_relosc.py is a companion check
of the 3/16 shift of a cold oscillator.

Run from the repository root:  python note/make_note_figures.py
"""
import json
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- homogeneous (k = 0) relativistic plasma driven at resonance (units c = omega_pe = n = 1) ----
VTE2 = 1e-3
MIME = 1 / 1836.0
WL = np.sqrt(1 + MIME)
NQ = 48  # Gauss-Hermite nodes for the initial electron momentum distribution (Maxwellian, variance VTE2)
_x, _w = np.polynomial.hermite.hermgauss(NQ)
P0, WQ = np.sqrt(2 * VTE2) * _x, _w / np.sqrt(np.pi)


def hom_run(vq, rel=True, thermal=True, tmax=800.0, nt=8001):
    """Mean-field oscillator with a momentum ensemble: dp/dt = -(E + E_drive), dE/dt = -J,
    J = -<p/gamma> + ion current.  Exact for the collisionless homogeneous Vlasov problem.
    Returns t and the external work W_ext = int J E_drive dt."""
    E0 = vq * np.sqrt(VTE2)
    p0 = P0 if thermal else np.zeros(1)
    wq = WQ if thermal else np.ones(1)
    n = len(p0)

    def vel(p):
        return p / np.sqrt(1 + p * p) if rel else p

    def rhs(t, y):
        p, ui, E = y[:n], y[n], y[n + 1]
        F = E + E0 * np.cos(WL * t)
        return np.concatenate([-F * np.ones(n), [MIME * F, vel(p) @ wq - vel(ui)]])

    ts = np.linspace(0, tmax, nt)
    s = solve_ivp(rhs, [0, tmax], np.concatenate([p0, [0, 0]]), t_eval=ts, rtol=1e-10, atol=1e-13)
    p, ui = s.y[:n], s.y[n]
    J = -(vel(p).T @ wq) + vel(ui)
    P = J * E0 * np.cos(WL * ts)
    W = np.concatenate([[0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(ts))])
    return ts, W


def reduced_run(vq, tmax=800.0, nt=8001, thermal=True):
    """Slow-flow model: r' = E0 cos psi, psi' = d(r) - E0 sin(psi)/r with detuning
    d = 3 theta/4 (hot plasma, <gamma^-3>) + (3/64) r^2 (finite-amplitude shift); W/W_lin = (r/(E0 t))^2."""
    E0 = vq * np.sqrt(VTE2)
    d0 = 0.75 * VTE2 if thermal else 0.0

    def rhs(t, y):
        r, psi = y
        return [E0 * np.cos(psi), d0 + 3 / 64 * r**2 - E0 * np.sin(psi) / r]

    ts = np.linspace(1e-6, tmax, nt)
    s = solve_ivp(rhs, [1e-6, tmax], [E0 * 1e-6, 0.0], t_eval=ts, rtol=1e-10, atol=1e-14)
    return ts, s.y[0] ** 2 / (E0 * ts) ** 2


def first_below(ts, ratio, level=0.9, tmin=20.0):
    i = np.argmax((ts > tmin) & (ratio < level))
    return float(ts[i])


def main():
    out = {}
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    cases = ((0.03, 1500, "C0", 392.5), (0.001, 2500, "C3", 1309.5))
    for vq, tmax, c, rec in cases:
        nt = int(tmax * 10) + 1
        ts, Wr = hom_run(vq, True, True, tmax, nt)
        _, Wn = hom_run(vq, False, True, tmax, nt)
        ratio = np.ones_like(Wn)
        m = np.abs(Wn) > 1e-30
        ratio[m] = Wr[m] / Wn[m]
        tr, rr = reduced_run(vq, tmax, nt)
        _, rc = reduced_run(vq, tmax, nt, thermal=False)
        out[f"vq{vq}"] = {"kinetic_k0_10pct": first_below(ts, ratio), "slow_flow_10pct": first_below(tr, rr),
                          "slow_flow_cold_10pct": first_below(tr, rc) if rc.min() < 0.9 else None, "record": rec}
        ax.plot(ts, ratio, c, label=rf"$k=0$ kinetic, $v_q/v_{{te}}={vq}$")
        ax.plot(tr, rr, "k--", lw=0.9, label="slow-flow model" if vq == 0.03 else None)
    ax.set_xlabel(r"$\omega_{pe}t$")
    ax.set_ylabel(r"$W_{\rm ext}^{\rm rel}/W_{\rm ext}^{\rm nonrel}$")
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_cold_relativistic.pdf"))
    out["_label"] = "calculation made for the explanatory note, not a study record; times are wpe*t of the 10% departure of W_ext from the nonrelativistic law"
    with open(os.path.join(HERE, "figures", "k0_relativistic.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
