"""hkr gates, declared before any run of studies/hkr_grid.py beyond a 2-time-unit smoke test (cost benchmark).

Study: does the Fourier range k lambda_De <= 0.045 (Nx 32) / 0.1 (Nx 64) of the Hermite runs control the HHS energy
measure at v_q/v_te = 0.03 (nonrelativistic)?  Grid reference with the 2/3 mask:  Nx 64 (k lambda_De <= 0.10),
256 (0.42), 640 (1.06), 1024 (1.69); the mask keeps |n| <= (Nx-1)//3 modes of k = 2 pi n / 40, lambda_De = v_te = 0.0316.
Common numerics: Nv 2048, vmax 16 v_te (dv = 0.0156 v_te as nxconv Nv 4096 / vmax 32), dt 0.02, to omega_pe t = 1000.
Seeds: det (nxconv deterministic modes 1, 2) and noise (det + white shot-noise density, Np = 1e7, rng 1).
"""
import numpy as np

E_TH = 5e-4  # initial electron thermal energy density, v_te^2 / 2 in n m c^2
AVG = 200.0  # trailing average window, HHS filter
RULE = 0.10  # two runs agree where |a - b| <= RULE * max(|a|, |b|) (first departure, as docs/results.md)
FLOOR_E = 0.02  # energy-measure comparison ignores |dE| below FLOOR_E * E_TH (noise/initial transients)
SPEC_FLOOR = 1e-6  # field spectrum: modes with |E_k|^2 >= SPEC_FLOOR * running max |E_1|^2 of the reference run

GATES = {
    # V1 script equivalence: hkr_grid.py (Nx 64 det, Nv 4096, vmax 32, dt 0.02) reproduces the committed nxconv record
    # g_vq0.03_Nx64_Nv4096_dt0.02_osc_v32_spectral_xmask23 in dK_e for t <= 100
    "V1_max_rel_dKe": 1e-6,
    # V2 linear Landau damping at k lambda_De = 0.5 and 1.0 against the kinetic dispersion root (electrons only, fit of
    # the |E_1| envelope peaks over t <= 40 (0.5) and t <= 8 (1.0), well before the v-grid recurrence 2 pi/(k dv))
    "V2_rate_rel": 0.03,
    "V2_freq_rel": 0.01,
    # C per-run certification time = min(first min f/max f < -1e-3, first edge > 1e-6, first |energy defect|/W > 1e-6)
    "C_fneg": -1e-3, "C_edge": 1e-6, "C_defect": 1e-6,
    # Q1 verdict on the main question, evaluated at omega_pe t = 500 and at the end of the common certified window:
    #   'cutoff matters'   : dE_e(Nx 640 or 1024, noise seed) / dE_e(Nx 64) >= 2 on the 200-averaged measure
    #   'HHS reached'      : dE_e >= 25 E_th (half of the ~50 E_th read from the HHS plot)
    #   otherwise 'k-range does not explain the difference' (within the certified window)
    "Q1_ratio": 2.0, "Q1_hhs": 25.0, "Q1_t": 500.0,
    # H Hermite-vs-grid: Hermite run counts as followed to t_res = last t with 10% agreement on dE_e (200-avg) with the
    # grid at the same Nx; certified window = min(Hermite positivity/diagnostics, t_res)
}


def lan_root(kl):
    """Kinetic Langmuir root omega/omega_pe of 1 + (1/(kl^2)) (1 + z Z(z)) = 0 with lambda_De = 1, z = omega/(sqrt2 k)."""
    from scipy.optimize import fsolve
    from scipy.special import wofz

    def f(p):
        z = (p[0] + 1j * p[1]) / (np.sqrt(2) * kl)
        r = 1 + (1 + 1j * np.sqrt(np.pi) * z * wofz(z)) / kl ** 2
        return [r.real, r.imag]
    g0 = {0.5: (1.4156, -0.1533), 1.0: (2.0, -0.9)}.get(kl, (1.5, -0.2))
    s = fsolve(f, g0, xtol=1e-12)
    return s[0], s[1]


def trailing(t, y, w=AVG):
    out = np.empty_like(y)
    for i in range(len(t)):
        m = (t >= t[i] - w) & (t <= t[i])
        out[i] = y[m].mean()
    return out


def agree_time(t, a, b, floor, rule=RULE):
    """Last time up to which a and b agree continuously (first departure); '>=' if never."""
    d = np.abs(a - b) > rule * np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)
    i = np.nonzero(d)[0]
    return (float(t[i[0]]), False) if i.size else (float(t[-1]), True)
