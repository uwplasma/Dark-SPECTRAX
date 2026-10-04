"""B01 nonlinear Landau and B06 two-pulse echo: ordinary and dark, against the imported grid references.

Electrostatic inputs (omega_pe = 1, v_te = 1) mapped with v_t/c = beta = 0.1 as in studies/compare_refs.py.
Dark runs: eta = 0.3, Omega_D = omega_pe, Yukawa-consistent start (no grid reference exists for them; the
grid is ordinary Vlasov-Poisson). Comparison windows end at the measured contamination times of the
imported records (B01: first time |log env_sx/env_grid| > 0.1 at N = 512/1024: 85/101; B06: k1 Hermite
recurrence at t ~ 27 (N = 256) and 38 (N = 512)). Closure nu = 0 and the parent hypercollision nu = 1 are
run side by side; nu is numerical, not a physical collision rate.

Run: python studies/b01_b06.py -> studies/b01_b06/{run.json, run.npz}
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks

import darkspectrax as ds

HERE = Path(__file__).resolve().parent
beta, eta, Omega_D = 0.1, 0.3, 1.0
rec = {"case_id": "B01_B06", "parent_commit": ds.PARENT_COMMIT, "repository_commit": ds._simulation._git_sha(),
       "command": "python studies/b01_b06.py", "beta": beta, "eta": eta, "Omega_D": Omega_D}
arrays = {}


def envelope_extrema(t, E):
    """Same rule as studies/refs/code/b01.py: running max over ~1 period, find_peaks prominence 0.3."""
    dt = t[1] - t[0]
    w = int(round(2 * np.pi / 1.1 / dt))
    env = maximum_filter1d(np.abs(E), size=w, mode="nearest")
    le = np.log(env)
    imax, _ = find_peaks(le, prominence=0.3, distance=w)
    imin, _ = find_peaks(-le, prominence=0.3, distance=w)
    return env, [float(t[i]) for i in imin if t[i] > 5], [float(t[i]) for i in imax if t[i] > 5]


def model(k0, Nx, Nn, nu, dark):
    return ds.Model(Nx=Nx, Nn=Nn, Lx=2 * np.pi / k0 * beta, alpha_s=(np.sqrt(2) * beta,) * 3, rho_background=1.0,
                    nu=nu, mode="self_consistent" if dark else "ordinary", eta=eta if dark else 0.0, Omega_D=Omega_D)


# ---------------- B01: k = 0.3, eps = 0.05 -------------------------------------------------------
g = np.load(HERE / "refs/B01/grid_k0.3_eps0.05.npz")
tg, Eg = g["t"], g["E"]
envg, gmin, gmax = envelope_extrema(tg, Eg)
B01 = {"grid": {"env_min": gmin, "env_max": gmax}}
tmax = 100.0
for Nn, nu in ((512, 0.0), (1024, 0.0), (512, 1.0)):
    for dark in (False, True):
        m = model(0.3, 16, Nn, nu, dark)
        y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 0.025)]))
        out = ds.run(m, y, tmax, n_save=int(tmax / 0.05) + 1, rtol=1e-10, atol=1e-14)
        t, E = out["t"], out["Fk"][:, 0, 0, 1, 0] / beta
        env, mins, maxs = envelope_extrema(t, E)
        key = f"{'dark' if dark else 'ordinary'}_N{Nn}_nu{nu}"
        r = {"status": out["status"], "env_min": mins, "env_max": maxs,
             "ledger_defect": float(np.abs(out["ledger_defect"]).max()),
             "compile_time": out["compile_time"], "run_time": out["run_time"], "steps": out["num_steps"]}
        if not dark:
            eg = np.interp(t, tg, envg)
            dev = np.abs(np.log(env / eg))
            r["first_t_logenv_dev_gt_0p1"] = float(t[np.argmax(dev > 0.1)]) if dev.max() > 0.1 else None
        B01[key] = r
        arrays[f"B01_{key}_t"], arrays[f"B01_{key}_E"] = t, E
        print("B01", key, r, flush=True)
for Nn, nu in ((512, 0.0), (1024, 0.0), (512, 1.0)):
    o, d = arrays[f"B01_ordinary_N{Nn}_nu{nu}_E"], arrays[f"B01_dark_N{Nn}_nu{nu}_E"]
    t = arrays[f"B01_ordinary_N{Nn}_nu{nu}_t"]
    sel = t <= 60.0
    B01[f"dark_vs_ordinary_N{Nn}_nu{nu}_rms_log_env_diff_t_le_60"] = float(np.sqrt(np.mean(
        np.log(envelope_extrema(t, d)[0][sel] / envelope_extrema(t, o)[0][sel]) ** 2)))
rec["B01"] = B01

# ---------------- B06: echo k1 = 1.0 (m=4), k2 = 1.5 (m=6) kick at tau = 10, echo at k3 = 0.5 -------
k0, m1, m2, tau, eps1, d2, tmax6 = 0.25, 4, 6, 10.0, 0.01, 0.05, 45.0
m3 = m2 - m1
ge = np.load(HERE / "refs/B06/grid_Nv1024_dt0.0125.npz")
tg6, Eg6 = ge["t"], np.abs(ge["E"][:, 2])
ig = np.argmax(Eg6 * (tg6 > tau + 5))
B06 = {"grid": {"t_echo": float(tg6[ig]), "amp": float(Eg6[ig])}, "t_echo_ballistic": m2 * tau / m3}


def kick(Ck, Nn, Nx, L, a):
    """f(x, v) -> f(x, v - d2 cos(k2 x)) exactly in the truncated basis: C(x) <- expm(s(x) R) C(x)."""
    Nf = 128
    Cf = np.zeros((Nn, Nf // 2 + 1), complex)
    Cf[:, :Nx // 2 + 1] = Ck[:, 0, :, 0]
    Cx = np.fft.irfft(Cf, n=Nf, axis=1, norm="forward")
    xf = np.arange(Nf) * L / Nf
    R = np.diag(np.sqrt(2 * np.arange(1, Nn)), -1)
    s = beta * d2 * np.cos(m2 * 2 * np.pi * xf / L) / a
    for j in range(Nf):
        Cx[:, j] = expm(s[j] * R) @ Cx[:, j]
    out = Ck.copy()
    out[:, 0, :, 0] = np.fft.rfft(Cx, axis=1, norm="forward")[:, :Nx // 2 + 1]
    return out


for Nn, nu in ((256, 0.0), (512, 0.0), (512, 1.0)):
    for dark in (False, True):
        m = model(k0, 32, Nn, nu, dark)  # Nx = 32 keeps |m| <= 10 under the strict mask
        y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (m1, 0, 0), eps1 / 2)]))
        a1 = ds.run(m, y, tau, n_save=int(tau / 0.05) + 1, rtol=1e-10, atol=1e-14)
        yk = {"Ck": kick(a1["Ck"][-1], Nn, 32, m.Lx, m.alpha_s[0]), "Fk": a1["Fk"][-1], "Dk": a1["Dk"][-1],
              "W": a1["W"][-1].astype(complex)}
        # the external kick changes the kinetic energy; it is reported as a separate ledger term
        dK_kick = float(ds.energies(m, yk)["K"]) - float(a1["K"][-1])
        a2 = ds.run(m, yk, tmax6 - tau, n_save=int((tmax6 - tau) / 0.05) + 1, rtol=1e-10, atol=1e-14, t0=tau)
        t = np.concatenate([a1["t"], a2["t"][1:]])
        E3 = np.abs(np.concatenate([a1["Fk"][:, 0, 0, m3, 0], a2["Fk"][1:, 0, 0, m3, 0]])) / beta
        E1 = np.abs(np.concatenate([a1["Fk"][:, 0, 0, m1, 0], a2["Fk"][1:, 0, 0, m1, 0]])) / beta
        i = np.argmax(E3 * (t > tau + 5))
        key = f"{'dark' if dark else 'ordinary'}_N{Nn}_nu{nu}"
        r = {"status": [a1["status"], a2["status"]], "t_echo": float(t[i]), "amp": float(E3[i]),
             "kinetic_energy_from_kick": dK_kick,
             "ledger_defect_after_kick": float(np.abs(a2["ledger_defect"]).max()),
             "run_time": a1["run_time"] + a2["run_time"], "compile_time": a1["compile_time"] + a2["compile_time"]}
        if not dark:
            r["amp_rel_to_grid"] = r["amp"] / B06["grid"]["amp"] - 1
            r["max_abs_dev_k3_over_grid_echo"] = float(np.abs(E3 - np.interp(t, tg6, Eg6)).max() / B06["grid"]["amp"])
            r["max_k1_after_t20"] = float(E1[t > 20].max())
        B06[key] = r
        arrays[f"B06_{key}_t"], arrays[f"B06_{key}_E3"], arrays[f"B06_{key}_E1"] = t, E3, E1
        print("B06", key, r, flush=True)
rec["B06"] = B06

d = HERE / "b01_b06"
d.mkdir(exist_ok=True)
(d / "run.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
np.savez_compressed(d / "run.npz", **arrays)
