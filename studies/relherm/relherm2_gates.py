"""Declared gates for the 1D2V extension (transverse B). Thresholds fixed BEFORE running.

 f1  inv_c2 = 0: kinetic RHS (Ex, Ey, Bz all nonzero, x-dependent, u0 = (0.3,-0.2), ax != ay) and currents (Jx, Jy) equal
     spectrax Hermite_Fourier_system / plasma_current (b668d87), max rel err <= 1e-12; inv_c2 = 1e-8: <= 1e-6.
 f2  gyration of a drifting Maxwellian (mean u = (0.3, 0), sigma_u = 0.1, q/m = 1, uniform Bz0 = 1, E = 0, frozen fields), inv_c2 = 1,
     N = 32 per axis, ax = ay = 0.2, RK4 dt = 0.01, T = 10: <v_x>(t), <v_y>(t) against the exact relativistic solution (every particle
     rotates by angle (q/m) Bz0 t / gamma(u0), tensor 100x100 Gauss-Hermite quadrature of the initial Maxwellian):
     max abs error <= 1e-5. The nonrelativistic (rigid rotation by (q/m)Bz0 t) answer must differ by > 1e-3 (discrimination).
 f3  magnetic work zero: max_t |K - K0| / K0 <= 1e-6 in the f2 run (K with gamma - 1).
 f4  energy ledger K - K0 + U - U0 - W_ext, U = (Ex^2 + Ey^2 + Bz^2)/2, 1D2V electron-ion run with uniform Ey drive
     E0 cos(3 t), E0 = 1.5 (quiver u about 0.5), density seed 1e-3 at k = 3, Lx = 2 pi / 3, Nx = 16, N = 24, dt = 0.01, T = 30:
     max |ledger| <= 1e-5 max|K - K0|; also reports the peak mean electron speed.
"""
import json, os, sys
import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import relherm as R
import relherm2 as R2


def out(name, d):
    json.dump(d, open(os.path.join(HERE, "gates_%s.json" % name), "w"), indent=1, default=float)
    print(json.dumps(d, indent=1, default=float))


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def gate_f1(spx):
    sys.path.insert(0, spx)
    from spectrax import _model, _simulation
    Ng, Nx, Lx = 8, 8, 5.0
    nk = Nx // 2 + 1
    ax, ay, u0 = 0.7, 0.45, (0.3, -0.2)
    qm = -1.3
    rng = np.random.default_rng(2)
    dec = np.exp(-0.15 * (np.arange(Ng)[:, None] + np.arange(Ng)[None, :]))
    C = (rng.normal(size=(Ng, Ng, nk)) + 1j * rng.normal(size=(Ng, Ng, nk))) * dec[:, :, None]
    C[:, :, 0] = C[:, :, 0].real
    fld = []
    for _ in range(3):
        f = rng.normal(size=nk) + 1j * rng.normal(size=nk)
        f[0] = f[0].real
        fld.append(f)
    Ex, Ey, Bz = fld
    # SPECTRAX layout: index (m, n) flattened as m*Ng + n, n = x-axis, m = y-axis; mine C[nx, ny]
    Ck = jnp.asarray(C.transpose(1, 0, 2).reshape(Ng * Ng, 1, nk, 1))
    Ny = Nz = 1
    mask = jnp.asarray(_simulation._twothirds_mask(Ny, Nx, Nz))
    Cx = jnp.fft.irfftn(Ck * mask, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
    Fk = jnp.zeros((6, Ny, nk, Nz), complex)
    for i, f in ((0, Ex), (1, Ey), (5, Bz)):
        Fk = Fk.at[i, 0, :, 0].set(jnp.asarray(f))
    F = jnp.fft.irfftn(Fk * mask, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
    ky, kx, kz = jnp.meshgrid(jnp.zeros(1), 2 * jnp.pi * jnp.arange(nk), jnp.zeros(1), indexing="ij")
    k2 = kx ** 2
    n = jnp.arange(Ng)[None, None, None, :, None, None, None]
    m = jnp.arange(Ng)[None, None, :, None, None, None, None]
    p = jnp.zeros((1, 1, 1, 1, 1, 1, 1))
    tabs = (jnp.sqrt(n + 1.0), jnp.sqrt(n * 1.0), jnp.sqrt(m + 1.0), jnp.sqrt(m * 1.0), jnp.sqrt(p + 1.0), jnp.sqrt(p))
    col = jnp.zeros((1, Ng, Ng))
    alpha, us = jnp.array([ax, ay, 1.0]), jnp.array([u0[0], u0[1], 0.0])
    ref = _model.Hermite_Fourier_system(Ck, Cx, F, kx, ky, kz, k2, col, *tabs, Lx, 1.0, 1.0, 0.0, 0.0, alpha, us,
                                        jnp.array([qm]), jnp.array([1.0]), Ng, Ng, 1, 1, mask23=mask)
    ref = np.asarray(ref).reshape(Ng, Ng, nk).transpose(1, 0, 2)  # -> [nx, ny, k]
    Jr = np.asarray(_model.plasma_current(jnp.array([qm]), alpha, us, Ck, Ng, Ng, 1, 1))[:2, 0, :, 0]
    res = {}
    for name, ic2 in (("f1_c_inf", 0.0), ("f1_c_1e8", 1e-8)):
        S = R2.Setup2(Ng, Nx, Lx, [qm], [1.0], [ax], [ay], inv_c2=ic2)
        U0 = jnp.array([[u0[0], u0[1]]])
        mine = np.asarray(R2.kinetic_rhs(S, jnp.asarray(C)[None], jnp.asarray(Ex), jnp.asarray(Ey), jnp.asarray(Bz), U0))[0]
        Jx, Jy = R2.current_k(S, jnp.asarray(C)[None], U0)
        res[name] = {"rhs_rel_err": rel(mine, ref), "Jx_rel_err": rel(np.asarray(Jx), Jr[0]), "Jy_rel_err": rel(np.asarray(Jy), Jr[1])}
    res["f1_c_inf"]["pass"] = bool(max(res["f1_c_inf"].values()) <= 1e-12)
    res["f1_c_1e8"]["pass"] = bool(res["f1_c_1e8"]["rhs_rel_err"] <= 1e-6)
    out("f1", res)


def gate_f2f3():
    N, Nx, a, sig, drift = 32, 4, 0.2, 0.1, 0.3
    T, dt, ds = 10.0, 0.01, 0.1
    S = R2.Setup2(N, Nx, 1.0, [1.0], [1.0], [a], [a], inv_c2=1.0)
    gx = lambda u: np.exp(-(u - drift) ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)
    gy = lambda u: np.exp(-u ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)
    cx, cy = R.project(gx, a, 0.0, N, nq=4096, span=12.0), R.project(gy, a, 0.0, N, nq=4096, span=12.0)
    C0 = np.zeros((1, N, N, S.nk), complex)
    C0[0, :, :, 0] = np.outer(cx, cy)
    z = jnp.zeros(S.nk, complex)
    y0 = (jnp.asarray(C0), z, z, z, jnp.zeros(()))
    u0 = jnp.zeros((1, 2))
    K0 = float(R2.kinetic_energy(S, y0[0], u0))
    y, outs = R2.run(S, y0, u0, lambda t: 0.0, 1.0, T, dt, int(round(T / ds)), evolve_fields=False)
    t = ds * np.arange(1, len(outs) + 1)
    vx = np.array([float(o[0]) for o in outs]); vy = np.array([float(o[1]) for o in outs]); K = np.array([float(o[2]) for o in outs])
    zq, wq = np.polynomial.hermite_e.hermegauss(100)
    wq = wq / np.sqrt(2 * np.pi)
    UX = drift + sig * zq[:, None]; UY = sig * zq[None, :]
    W = wq[:, None] * wq[None, :]
    g = np.sqrt(1 + UX ** 2 + UY ** 2)
    ex_x = np.zeros_like(t); ex_y = np.zeros_like(t); nr_x = np.zeros_like(t); nr_y = np.zeros_like(t)
    for i, tt in enumerate(t):
        th = tt / g
        ux = UX * np.cos(th) + UY * np.sin(th); uy = -UX * np.sin(th) + UY * np.cos(th)
        gg = np.sqrt(1 + ux ** 2 + uy ** 2)
        ex_x[i] = (W * ux / gg).sum(); ex_y[i] = (W * uy / gg).sum()
        ux = UX * np.cos(tt) + UY * np.sin(tt); uy = -UX * np.sin(tt) + UY * np.cos(tt)  # nonrelativistic rigid rotation of u
        gg = np.sqrt(1 + ux ** 2 + uy ** 2)  # same velocity map, rigid-rotation momenta
        nr_x[i] = (W * ux / gg).sum(); nr_y[i] = (W * uy / gg).sum()
    err = float(max(np.max(np.abs(vx - ex_x)), np.max(np.abs(vy - ex_y))))
    disc = float(max(np.max(np.abs(nr_x - ex_x)), np.max(np.abs(nr_y - ex_y))))
    d = {"N": N, "max_abs_err_v": err, "rigid_rotation_vs_exact": disc, "max_abs_v": float(max(np.max(np.abs(ex_x)), np.max(np.abs(ex_y)))),
         "pass": bool(err <= 1e-5 and disc > 1e-3)}
    out("f2", d)
    d3 = {"K0": K0, "max_rel_K_drift": float(np.max(np.abs(K - K0)) / K0), "pass": bool(np.max(np.abs(K - K0)) / K0 <= 1e-6)}
    out("f3", d3)


def gate_f4():
    N, Nx, Lx = 24, 16, 2 * np.pi / 3
    T, dt = 30.0, 0.01
    MI = 1836.0
    ae, ai = 0.2, 0.1 * np.sqrt(2) / np.sqrt(MI)
    S = R2.Setup2(N, Nx, Lx, [-1.0, 1.0], [1.0, MI], [ae, ai], [ae, ai], inv_c2=1.0)
    C0 = np.zeros((2, N, N, S.nk), complex)
    for s, (a, sig) in enumerate(((ae, 0.1), (ai, 0.1 / np.sqrt(MI)))):
        f = lambda u, sig=sig: np.exp(-u ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)
        c = R.project(f, a, 0.0, N, nq=4096, span=12.0)
        C0[s, :, :, 0] = np.outer(c, c)
    C0[0, :, :, 1] = 1e-3 / 2 * C0[0, :, :, 0]
    k1 = 2 * np.pi / Lx
    Ex = np.zeros(S.nk, complex); Ex[1] = -1j * (-1.0 * ae * ae * C0[0, 0, 0, 1]) / k1
    z = jnp.zeros(S.nk, complex)
    y0 = (jnp.asarray(C0), jnp.asarray(Ex), z, z, jnp.zeros(()))
    u0 = jnp.zeros((2, 2))
    K0 = float(R2.kinetic_energy(S, y0[0], u0)); U0 = float(R2.field_energy(y0[1], y0[2], y0[3]))
    E0 = 1.5
    y, outs = R2.run(S, y0, u0, lambda t: E0 * jnp.cos(3.0 * t), 0.0, T, dt, 60)
    K = np.array([float(o[2]) for o in outs]); U = np.array([float(o[3]) for o in outs]); W = np.array([float(o[4]) for o in outs])
    Jy = np.array([float(o[1]) for o in outs])
    led = K - K0 + (U - U0) - W
    d = {"ledger_max": float(np.max(np.abs(led))), "max_abs_K_minus_K0": float(np.max(np.abs(K - K0))),
         "ledger_rel": float(np.max(np.abs(led)) / np.max(np.abs(K - K0))), "peak_mean_current_Jy": float(np.max(np.abs(Jy)))}
    d["pass"] = bool(d["ledger_rel"] <= 1e-5)
    out("f4", d)


if __name__ == "__main__":
    w = sys.argv[1]
    spx = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "..", "rh-spx")
    {"f1": lambda: gate_f1(spx), "f2": gate_f2f3, "f4": gate_f4}[w]()
