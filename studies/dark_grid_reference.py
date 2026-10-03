"""Independent dark reference: grid Vlasov-Ampere-Proca (studies/refs/code/slv_proca.py) verified alone, then
compared with the Hermite dark B01 and B06 runs (studies/b01_b06/run.npz).

Units: electrostatic (omega_pe = 1, v_te = 1), c = 1/beta = 10, eta = 0.3, Omega_D = omega_pe, Yukawa start.
Run: python studies/dark_grid_reference.py -> studies/dark_grid/{run.json, run.npz}
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.special import wofz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "refs" / "code"))
from fit import fit_pair  # noqa: E402
from slv import maxwellian  # noqa: E402
from slv_proca import VAP  # noqa: E402

from scipy.ndimage import maximum_filter1d  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402


def envelope_extrema(t, E, prominence=0.3):
    """Same rule as studies/b01_b06.py and studies/refs/code/b01.py (prominence 0.3 there)."""
    dt = t[1] - t[0]
    w = int(round(2 * np.pi / 1.1 / dt))
    env = maximum_filter1d(np.abs(E), size=w, mode="nearest")
    le = np.log(env)
    imax, _ = find_peaks(le, prominence=prominence, distance=w)
    imin, _ = find_peaks(-le, prominence=prominence, distance=w)
    return env, [float(t[i]) for i in imin if t[i] > 5], [float(t[i]) for i in imax if t[i] > 5]

c, eta, Om = 10.0, 0.3, 1.0
rec = {"case_id": "dark_grid", "command": "python studies/dark_grid_reference.py", "c": c, "eta": eta, "Omega_D": Om,
       "solver": "NumPy semi-Lagrangian Vlasov-Ampere + exact per-k Proca propagator, Strang splitting"}
arrays = {}


def D_L(w, k):
    z = w / (np.sqrt(2) * k)
    chi = (1 + z * 1j * np.sqrt(np.pi) * wofz(z)) / k ** 2
    Q = w ** 2 - c ** 2 * k ** 2
    return (Q - Om ** 2) * (1 + chi) + eta ** 2 * Q * chi


# ---------- verification alone: linear Landau roots and ledger, two timesteps + Richardson ----------
lin = {}
for k, win in ((0.5, (10.0, 36.0)), (0.3, (16.0, 40.0))):
    r = root(lambda x: [D_L(x[0] + 1j * x[1], k).real, D_L(x[0] + 1j * x[1], k).imag], [1.4, -0.1], tol=1e-14).x
    fits = {}
    for dt in (0.05, 0.025, 0.0125):
        s = VAP(2 * np.pi / k, 16, 8.0, 256, eta, Om, c)
        f = (1 + 1e-4 * np.cos(k * s.x))[:, None] * maxwellian(s.v)[None, :]
        tic = time.perf_counter()
        f, t, E, D, d = s.run_proca(f, dt, 40.0, save_every=int(round(0.1 / dt)))
        w, g, res = fit_pair(t, E[:, 0], r[0], r[1], win)
        fits[dt] = {"w": w, "g": g, "fit_resid": res, "wall": time.perf_counter() - tic,
                    "dark_ledger_defect_over_U_E0": float(np.abs(d[:, 3] + d[:, 2] - d[0, 2]).max() / d[0, 1]),
                    "max_gauss": float(d[:, 4].max()), "max_dark_gauss": float(d[:, 5].max())}
    a, b = fits[0.025], fits[0.0125]
    lin[f"k{k}"] = {"root": [r[0], r[1]], "fits": {str(k_): v for k_, v in fits.items()},
                    "richardson": [b["w"] + (b["w"] - a["w"]) / 3, b["g"] + (b["g"] - a["g"]) / 3]}
    lin[f"k{k}"]["richardson_minus_root"] = [lin[f"k{k}"]["richardson"][0] - r[0], lin[f"k{k}"]["richardson"][1] - r[1]]
    print("linear", k, lin[f"k{k}"], flush=True)
rec["linear_verification"] = lin

H = np.load(HERE / "b01_b06" / "run.npz")

# ---------- B01 dark: k = 0.3, eps = 0.05 ----------
B01 = {}
k, eps = 0.3, 0.05
for dt in (0.025, 0.0125):
    s = VAP(2 * np.pi / k, 32, 8.0, 1024, eta, Om, c)
    f = (1 + eps * np.cos(k * s.x))[:, None] * maxwellian(s.v)[None, :]
    tic = time.perf_counter()
    f, t, E, D, d = s.run_proca(f, dt, 100.0, save_every=int(round(0.05 / dt)))
    env, mins, maxs = envelope_extrema(t, E[:, 0])
    B01[f"grid_dt{dt}"] = {"env_min": mins, "env_max": maxs, "wall": time.perf_counter() - tic,
                           "dark_ledger_defect_over_U_E0": float(np.abs(d[:, 3] + d[:, 2] - d[0, 2]).max() / d[0, 1])}
    arrays[f"B01_grid_dt{dt}_t"], arrays[f"B01_grid_dt{dt}_E"] = t, E[:, 0]
    print("B01 grid", dt, B01[f"grid_dt{dt}"], flush=True)
tg, Eg = arrays["B01_grid_dt0.0125_t"], arrays["B01_grid_dt0.0125_E"]
envg = envelope_extrema(tg, Eg)[0]
for key in ("dark_N1024_nu0.0", "dark_N512_nu0.0", "dark_N512_nu1.0"):
    th, Eh = H[f"B01_{key}_t"], H[f"B01_{key}_E"]
    envh = envelope_extrema(th, Eh)[0]
    dev = np.abs(np.log(envh / np.interp(th, tg, envg)))
    B01[f"hermite_{key}"] = {"env_min_prominence_0p2": envelope_extrema(th, Eh, 0.2)[1],"first_t_logenv_dev_gt_0p1": float(th[np.argmax(dev > 0.1)]) if dev.max() > 0.1 else None,
                             "max_logenv_dev_t_le_80": float(dev[th <= 80].max())}
B01["grid_dt0.0125"]["env_min_prominence_0p2"] = envelope_extrema(tg, Eg, 0.2)[1]
envg_c = envelope_extrema(arrays["B01_grid_dt0.025_t"], arrays["B01_grid_dt0.025_E"])[0]
B01["grid_dt_halving_max_logenv_diff"] = float(np.abs(np.log(envg / np.interp(tg, arrays["B01_grid_dt0.025_t"], envg_c))).max())
rec["B01_dark"] = B01
print("B01", B01, flush=True)

# ---------- B06 dark echo ----------
B06 = {}
k0, m1, m2, tau, eps1, d2 = 0.25, 4, 6, 10.0, 0.01, 0.05
for dt in (0.025, 0.0125):
    s = VAP(2 * np.pi / k0, 64, 8.0, 512, eta, Om, c)
    f = (1 + eps1 * np.cos(m1 * k0 * s.x))[:, None] * maxwellian(s.v)[None, :]
    tic = time.perf_counter()
    f, t, E, D, d = s.run_proca(f, dt, 45.0, modes=(m1, m2, m2 - m1), save_every=int(round(0.05 / dt)),
                                kicks=[(tau, lambda x: d2 * np.cos(m2 * k0 * x))])
    A = np.abs(E[:, 2])
    i = np.argmax(A * (t > tau + 5))
    B06[f"grid_dt{dt}"] = {"t_echo": float(t[i]), "amp": float(A[i]), "wall": time.perf_counter() - tic}
    arrays[f"B06_grid_dt{dt}_t"], arrays[f"B06_grid_dt{dt}_E3"] = t, A
    print("B06 grid", dt, B06[f"grid_dt{dt}"], flush=True)
ref = B06["grid_dt0.0125"]
for key in ("dark_N256_nu0.0", "dark_N512_nu0.0", "dark_N512_nu1.0"):
    th, Ah = H[f"B06_{key}_t"], H[f"B06_{key}_E3"]
    i = np.argmax(Ah * (th > tau + 5))
    B06[f"hermite_{key}"] = {"t_echo": float(th[i]), "amp": float(Ah[i]), "amp_rel_to_grid": float(Ah[i] / ref["amp"] - 1),
                             "max_abs_dev_over_grid_echo": float(np.abs(Ah - np.interp(th, arrays["B06_grid_dt0.0125_t"],
                                                                                        arrays["B06_grid_dt0.0125_E3"])).max()
                                                                 / ref["amp"])}
rec["B06_dark"] = B06
print("B06", B06, flush=True)

out = HERE / "dark_grid"
out.mkdir(exist_ok=True)
(out / "run.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
np.savez_compressed(out / "run.npz", **arrays)
