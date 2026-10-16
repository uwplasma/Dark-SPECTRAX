"""C09: derivatives of converged real scalar observables of the dark model, AD against finite differences.

Every objective is first converged in the forward direction (evidence recorded). Then for each parameter p:
reverse mode (``jax.grad``, Diffrax ``RecursiveCheckpointAdjoint``), forward mode (``jax.jvp``, Diffrax
``ForwardMode``) and centred finite differences (f(p+h)-f(p-h))/2h on a ladder h = |p| 10^-1 ... 10^-7, plus
an analytic derivative where one exists. AD is the exact derivative of the discrete fixed-step solve, so FD
agrees with it on a plateau bounded by truncation (large h) and round-off (small h).

(a) cold homogeneous coupled oscillator, k = 0: both eigenfrequencies omega_1,2 fitted from E(t) (variable
    projection, Gauss-Newton unrolled in JAX) vs the roots of (w^2 - n0)(w^2 - OD^2) - eta^2 n0 w^2 = 0.
    Parameters eta, Omega_D, n0 (omega_pe^2 = n0).
(b) the same oscillator, finite-time dark work W_D(T = 60). Parameters eta, Omega_D, n0, seed E0.
    Reference: a 4x4 matrix exponential of the cold linear system (J, E, E_D, A_D), W_D = -[U_D]_0^T.
(c) a short dark Landau response, k lambda_De = 0.5, v_te = 0.1 c (examples/landau.toml): P = |E_k(t=10)|^2.
    Parameters eta, Omega_D, n0, T (alpha = sqrt(2 T) in all three directions, basis = Maxwellian),
    seed amplitude. Fields start Gauss/Yukawa consistent (built here in JAX; checked against consistent_fields).

Run: python studies/c09_ad.py  -> studies/c09/run.json
"""

import json
import platform
import subprocess
import time
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm

import darkspectrax as ds

out_dir = Path(__file__).resolve().parent / "c09"
HS = [10.0 ** -e for e in range(1, 8)]


def sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parent).stdout.strip()


def ladder(f, p, ad, analytic=None):
    """Centred FD over the step ladder, relative to the reverse-mode value."""
    rows = []
    for h in HS:
        d = h * max(abs(p), 1e-300)
        fd = (float(f(p + d)) - float(f(p - d))) / (2 * d)
        rows.append({"h_rel": h, "fd": fd, "rel_err_vs_ad": abs(fd - ad) / abs(ad)})
    best = min(r["rel_err_vs_ad"] for r in rows)
    plateau = [r["h_rel"] for r in rows if r["rel_err_vs_ad"] <= max(10 * best, 1e-12)]
    out = {"ladder": rows, "best_fd_rel_err": best, "plateau_h_rel": plateau}
    if analytic is not None:
        out["analytic"] = analytic
        out["ad_rel_err_vs_analytic"] = abs(ad - analytic) / abs(analytic)
    return out


def costs(f, x, n_rep=3):
    """Wall time (best of n_rep, after compile) and XLA temp memory for f, grad f and one jvp."""
    jf, jg = jax.jit(f), jax.jit(jax.grad(f))
    res = {}
    for name, fn, args in (("forward", jf, (x,)), ("reverse", jg, (x,))):
        c = fn.lower(*args).compile()
        jax.block_until_ready(c(*args))
        ts = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            jax.block_until_ready(c(*args))
            ts.append(time.perf_counter() - t0)
        ma = c.memory_analysis()
        res[name] = {"time_s": min(ts), "temp_bytes": int(getattr(ma, "temp_size_in_bytes", -1))}
    return res


def compare(f_rev, f_fwd, x0, names, analytic=None, cost_kw=None):
    """f_rev/f_fwd: scalar functions of the parameter vector using the reverse/forward-mode adjoint."""
    x0 = jnp.asarray(x0, float)
    g = np.asarray(jax.jit(jax.grad(f_rev))(x0))
    jfwd = jax.jit(lambda x, v: jax.jvp(f_fwd, (x,), (v,))[1])
    jf = jax.jit(f_rev)
    val = float(jf(x0))
    out = {"value": val, "params": dict(zip(names, map(float, x0))), "derivatives": {}}
    for i, n in enumerate(names):
        e = jnp.zeros_like(x0).at[i].set(1.0)
        fwd = float(jfwd(x0, e))

        def f1(p, i=i):
            return jf(x0.at[i].set(p))
        d = {"reverse": float(g[i]), "forward": fwd, "fwd_rev_rel": abs(fwd - g[i]) / abs(g[i])}
        d.update(ladder(f1, float(x0[i]), float(g[i]), None if analytic is None else float(analytic[i])))
        out["derivatives"][n] = d
    v = jnp.asarray(np.random.default_rng(0).standard_normal(len(names)))
    out["random_direction"] = {"forward": float(jfwd(x0, v)), "reverse_dot": float(g @ np.asarray(v))}
    out["cost"] = costs(f_rev, x0, **(cost_kw or {}))
    t0 = time.perf_counter()
    jax.block_until_ready(jfwd(x0, v))
    out["cost"]["forward_jvp_time_s"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------- (a), (b) homogeneous oscillator
H_MODEL = ds.Model(Nx=1, Nn=3, Nm=3, Np=3, alpha_s=(0.05,) * 3, rho_background=1.0,
                   mode="self_consistent", eta=0.25, Omega_D=1.2)
T_H, N_H, DT_H = 60.0, 601, 0.01


def h_state(n0, E0):
    y = H_MODEL.zeros()
    return {**y, "Ck": y["Ck"].at[0, 0, 0, 0].set(n0 / 0.05 ** 3), "Fk": y["Fk"].at[0, 0, 0, 0].set(E0 + 0j)}


def h_solve(x, adjoint=None, fixed_dt=DT_H, rtol=1e-12, atol=1e-14):
    eta, OD, n0, E0 = x[0], x[1], x[2], x[3]
    return ds.solve(H_MODEL, h_state(n0, E0), T_H, N_H, params={"eta": eta, "Omega_D": OD},
                    fixed_dt=fixed_dt, rtol=rtol, atol=atol, max_steps=int(T_H / (fixed_dt or DT_H)) + 20, adjoint=adjoint)


def roots(eta, OD, n0):
    b, c = n0 + OD ** 2 + eta ** 2 * n0, n0 * OD ** 2
    d = jnp.sqrt(b ** 2 - 4 * c)
    return jnp.sqrt(jnp.stack([(b - d) / 2, (b + d) / 2]))


def fit_freqs(t, s, w0, iters=30):
    """Variable projection: s(t) ~ sum_j a_j cos w_j t + b_j sin w_j t; Gauss-Newton on w (unrolled)."""
    def resid(w):
        B = jnp.concatenate([jnp.cos(jnp.outer(t, w)), jnp.sin(jnp.outer(t, w))], axis=1)
        c = jnp.linalg.solve(B.T @ B, B.T @ s)
        return B @ c - s

    w = jax.lax.stop_gradient(w0)
    for _ in range(iters):
        r, Jm = resid(w), jax.jacfwd(resid)(w)
        w = w - jnp.linalg.solve(Jm.T @ Jm, Jm.T @ r)
    return w, jnp.linalg.norm(resid(w)) / jnp.linalg.norm(s)


def omega_obj(j, adjoint=None):
    def f(x3):
        x = jnp.concatenate([x3, jnp.array([0.01])])
        ys = h_solve(x, adjoint)
        t = jnp.linspace(0.0, T_H, N_H)
        w, _ = fit_freqs(t, jnp.real(ys["Fk"][:, 0, 0, 0, 0]), roots(*x3) * (1 + 1e-3))
        return w[j]
    return f


def wd_obj(adjoint=None, **kw):
    return lambda x: jnp.real(h_solve(x, adjoint, **kw)["W"][-1, 1])


def wd_exact(x):
    eta, OD, n0, E0 = x
    M = jnp.array([[0, n0, eta * n0, 0], [-1, 0, 0, 0], [-eta, 0, 0, OD ** 2], [0, 0, -1, 0.0]])
    y0 = jnp.array([0.0, E0, 0.0, 0.0])
    yT = expm(M * T_H) @ y0
    U = 0.5 * (yT[2] ** 2 + OD ** 2 * yT[3] ** 2)
    return -U  # U_D(0) = 0


def part_ab():
    x3 = jnp.array([0.25, 1.2, 1.0])
    x4 = jnp.array([0.25, 1.2, 1.0, 0.01])
    t = jnp.linspace(0.0, T_H, N_H)
    conv = {}
    for dt in (0.02, 0.01, 0.005):
        ys = h_solve(x4, fixed_dt=dt)
        w, res = fit_freqs(t, jnp.real(ys["Fk"][:, 0, 0, 0, 0]), roots(*x3) * (1 + 1e-3))
        conv[f"dt{dt}"] = {"omega": np.asarray(w).tolist(), "fit_residual": float(res),
                           "omega_rel_err_vs_root": float(jnp.max(jnp.abs(w / roots(*x3) - 1))),
                           "W_D": float(jnp.real(ys["W"][-1, 1])),
                           "W_D_rel_err_vs_expm": float(abs(jnp.real(ys["W"][-1, 1]) / wd_exact(x4) - 1))}
    ys = h_solve(x4, fixed_dt=None)
    conv["adaptive_rtol1e-12"] = {"W_D": float(jnp.real(ys["W"][-1, 1])),
                                  "W_D_rel_err_vs_expm": float(abs(jnp.real(ys["W"][-1, 1]) / wd_exact(x4) - 1))}
    A = {"forward_convergence": conv}
    J = np.asarray(jax.jacfwd(roots, argnums=(0, 1, 2))(*x3))  # (3, 2): d omega_j / d p
    for j in (0, 1):
        A[f"omega_{j + 1}"] = compare(omega_obj(j, diffrax.DirectAdjoint()), omega_obj(j, diffrax.ForwardMode()), x3,
                                      ["eta", "Omega_D", "n0"], analytic=J[:, j])
        A[f"omega_{j + 1}"]["root"] = float(roots(*x3)[j])
    A["reverse_adjoint"] = "DirectAdjoint (RecursiveCheckpointAdjoint is wrong here: see saveat_defect)"
    B = {"forward_convergence": conv, "expm_value": float(wd_exact(x4))}
    an = np.asarray(jax.grad(wd_exact)(x4))
    names = ["eta", "Omega_D", "n0", "E0"]
    B["W_D"] = compare(wd_obj(), wd_obj(diffrax.ForwardMode()), x4, names, analytic=an)
    # adaptive PID solve: AD differentiates the discrete adaptive map
    B["W_D_adaptive"] = compare(wd_obj(fixed_dt=None), wd_obj(diffrax.ForwardMode(), fixed_dt=None), x4, names,
                                analytic=an)
    return A, B


def saveat_defect():
    """Diffrax RecursiveCheckpointAdjoint with ConstantStepSize and many SaveAt(ts) on step boundaries.

    Objective: sum_i w_i E(t_i) of the dark oscillator (601 saves) and a two-variable linear oscillator.
    """
    x4 = jnp.array([0.25, 1.2, 1.0, 0.01])
    wts = jnp.asarray(np.random.default_rng(1).standard_normal(N_H))
    rows = []
    for dt in (0.01, 0.0099):
        def S(e, adj, dt=dt):
            return wts @ jnp.real(h_solve(x4.at[0].set(e), adj, fixed_dt=dt)["Fk"][:, 0, 0, 0, 0])
        fd = (float(S(0.25 + 1e-5, None)) - float(S(0.25 - 1e-5, None))) / 2e-5
        rows.append({"case": "dark oscillator, 601 saves, d/d eta", "dt": dt, "fd_h1e-5": fd,
                     "recursive_checkpoint": float(jax.grad(S)(0.25, diffrax.RecursiveCheckpointAdjoint())),
                     "direct": float(jax.grad(S)(0.25, diffrax.DirectAdjoint())),
                     "forward_mode": float(jax.jvp(lambda e: S(e, diffrax.ForwardMode()), (0.25,), (1.0,))[1])})

    def g(a, adj, dt, n, T, ms):
        ts = jnp.linspace(0, T, n)
        sol = diffrax.diffeqsolve(diffrax.ODETerm(lambda t, y, _: jnp.stack([y[1], -a * y[0]])), diffrax.Dopri8(),
                                  0.0, T, dt, jnp.array([1.0, 0.0]), saveat=diffrax.SaveAt(ts=ts),
                                  stepsize_controller=diffrax.ConstantStepSize(), adjoint=adj, max_steps=ms)
        return jnp.sum(sol.ys[:, 0] * jnp.cos(ts))
    for dt, n, T, ms in ((0.01, 601, 60.0, 6020), (0.01, 601, 60.0, 20000), (0.0099, 601, 60.0, 6080),
                         (0.01, 2, 60.0, 6020), (0.1, 11, 1.0, 1000)):
        rows.append({"case": "y'' = -a y, sum_i y(t_i) cos t_i, d/da", "dt": dt, "n_save": n, "T": T, "max_steps": ms,
                     "recursive_checkpoint": float(jax.grad(g)(0.7, diffrax.RecursiveCheckpointAdjoint(), dt, n, T, ms)),
                     "direct": float(jax.grad(g)(0.7, diffrax.DirectAdjoint(), dt, n, T, ms))})
    for r in rows:
        r["recursive_rel_err_vs_direct"] = abs(r["recursive_checkpoint"] / r["direct"] - 1)
    return rows


# ---------------------------------------------------------------- (c) dark Landau
K, VT, AMP, T_L = 5.0, 0.1, 5e-5, 10.0


def l_model(Nn):
    a = np.sqrt(2) * VT
    return ds.Model(Nx=5, Nn=Nn, Lx=2 * np.pi / K, alpha_s=(a, a, a), rho_background=1.0,
                    mode="self_consistent", eta=0.3, Omega_D=1.0)


def l_state(m, eta, OD, n0, T, amp):
    a = jnp.sqrt(2 * T)
    A = a ** 3
    y = m.zeros()
    Ck = y["Ck"].at[0, 0, 0, 0].set(n0 / A + 0j).at[0, 0, 1, 0].set(amp / A + 0j)
    nab = m.p["nabla"][:, 0, 1, 0]
    rho = -amp  # electrons, charge -1, Om0 = 1
    k2n = jnp.sum(nab ** 2)
    E = -1j * nab * rho / k2n
    phi = eta * rho / (k2n + OD ** 2)
    Fk = y["Fk"].at[:3, 0, 1, 0].set(E)
    Dk = y["Dk"].at[9, 0, 1, 0].set(phi).at[:3, 0, 1, 0].set(-1j * nab * phi)
    return {**y, "Ck": Ck, "Fk": Fk, "Dk": Dk}


def l_solve(m, x, dt, adjoint=None, fixed=True, n_save=2, checkpoints=None):
    eta, OD, n0, T, amp = x
    a = jnp.sqrt(2 * T)
    nsteps = int(round(T_L / dt))
    adj = adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint(checkpoints)
    return ds.solve(m, l_state(m, eta, OD, n0, T, amp), T_L, n_save,
                    params={"eta": eta, "Omega_D": OD, "alpha_s": jnp.stack([a, a, a])},
                    fixed_dt=dt if fixed else None, rtol=1e-12, atol=1e-18,
                    max_steps=nsteps + 10 if fixed else 20_000, adjoint=adj)


def p_obj(m, dt, adjoint=None, fixed=True, checkpoints=None):
    return lambda x: jnp.abs(l_solve(m, x, dt, adjoint, fixed, checkpoints=checkpoints)["Fk"][-1, 0, 0, 1, 0]) ** 2


def part_c():
    x = jnp.array([0.3, 1.0, 1.0, VT ** 2, AMP])
    m64 = l_model(64)
    y_ref = ds.consistent_fields(m64, ds.maxwellian(m64, [1.0], [(0, (1, 0, 0), AMP)]))
    y_me = l_state(m64, *x)
    init_err = max(float(jnp.max(jnp.abs(y_ref[k] - y_me[k]))) for k in ("Ck", "Fk", "Dk"))
    conv = {"init_vs_consistent_fields": init_err}
    for Nn in (32, 64, 128):
        for dt in (0.02, 0.01):
            conv[f"Nn{Nn}_dt{dt}"] = float(p_obj(l_model(Nn), dt)(x))
    conv["Nn64_adaptive_rtol1e-12"] = float(p_obj(m64, 0.02, fixed=False)(x))
    ref = conv["Nn128_dt0.01"]
    conv["rel_diff_vs_Nn128_dt0.01"] = {k: abs(v / ref - 1) for k, v in conv.items()
                                        if k.startswith("Nn") and isinstance(v, float)}
    names = ["eta", "Omega_D", "n0", "T", "amp"]
    dt = 0.02
    C = {"forward_convergence": conv, "Nn": 64, "dt": dt}
    C["P"] = compare(p_obj(m64, dt), p_obj(m64, dt, diffrax.ForwardMode()), x, names)
    # gradient refinement: reverse-mode gradient vs Nn and dt
    gref = {}
    for Nn, d in ((32, 0.02), (64, 0.01), (128, 0.02)):
        gref[f"Nn{Nn}_dt{d}"] = np.asarray(jax.jit(jax.grad(p_obj(l_model(Nn), d)))(x)).tolist()
    C["gradient_refinement"] = gref
    C["checkpoint_cost"] = {f"checkpoints_{c}": costs(p_obj(m64, dt, checkpoints=c), x)
                            for c in (None, 32, 8)}
    return C


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    A, B = part_ab()
    D = saveat_defect()
    C = part_c()
    rec = {"study": "C09 AD vs FD", "repository_commit": sha(), "parent_commit": ds.PARENT_COMMIT,
           "versions": {"python": platform.python_version(), "jax": jax.__version__,
                        "diffrax": diffrax.__version__, "numpy": np.__version__},
           "backend": jax.default_backend(), "dtype": "float64/complex128", "h_ladder_rel": HS,
           "a_omega": A, "b_dark_work": B, "c_landau": C, "saveat_defect": D, "wall_s": time.perf_counter() - t0}
    (out_dir / "run.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps({k: rec[k] for k in ("repository_commit", "wall_s")}))


if __name__ == "__main__":
    main()
