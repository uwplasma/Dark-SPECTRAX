"""relpic gates, declared BEFORE any production run (committed first; never amended after seeing results).

Study: independent relativistic PIC (Dark-JAX-in-Cell on the JAX-in-Cell parent pin 2d693cb, relativistic Boris)
of the HHS (arXiv:2510.13956v1) resonant run: m_i/m_e = 1836, T_e = T_i = 1e-3 m_e c^2, L = 40 c/omega_pe, 1000 cells
(dx = 0.04), uniform drive a0 = (v_q^D/c) with v_q^D/sigma_e = 0.03, quiet start, ppc = 103 per species (HHS S38:
2.06e5 markers in total for the 1000 cells), dt = 0.02/omega_pe.

Pilot done before declaring (calibration only, 100 time units, 103 ppc, 1 seed, relativistic and Newtonian): wall 22 s
per 50 time units, energy-work defect 9.5e-4 (normalised as below), Gauss and continuity ~1e-14.

Observables (history keys of the parent paper study, energies in n m_e c^2 L):
  W_th(t)  = 'spread'[:, 0]: electron 1D thermal energy (m_e/2) <(v_x - <v_x>)^2>, global mean removed (HHS B3; the
             coherent k = 0 quiver is removed, everything at k != 0 and the thermal bath is kept); initial 5e-4.
  W_E0(t)  = mean_E^2 / 2: mean-field (k = 0) electric energy;   W_Ek(t) = 'nonzero_electric': k != 0 field energy.
  <.>_200  = trailing mean over 200/omega_pe, defined at t >= 200.  dW_th = W_th - W_th(0).

Runs (all dt = 0.02 and 1000 cells unless stated):
  R   relativistic Boris, quadratic shape, ppc 103, seeds 0,1,2   to T = 2000
  N   Newtonian Boris (same loading), seeds 0,1,2                 to T = 2000
  Q   relativistic, quintic shape, seeds 0,1                      to T = 1000
  P4  relativistic, ppc 412, seed 0                               to T = 1000
  X2  relativistic, 2000 cells (dx = 0.02), dt = 0.01, seed 0     to T = 1000
  D   relativistic, dt = 0.01, seed 0                             to T = 1000
  Z   relativistic, no drive (ratio 0), seed 0                    to T = 1000
"""
import numpy as np

# ---- production gates (thresholds fixed here) ------------------------------------------------------------
GATES = dict(
    # G1 energy: parent all-step maximum |energy - initial - work| / (1e-3 n m_e c^2 L), every production run
    G1_energy_defect=0.05,
    # G2 constraints, every production run (units of the parent normalisation)
    G2_gauss=1e-10, G2_continuity=1e-10, G2_momentum=1e-12,
    # G3 early drive response: max |Ebar - homogeneous kinetic oracle| / peak, t <= 40, run R seeds
    G3_early=0.05,
    # G4/G5/G6 refinement (P4 ppc, X2 dx, D dt) vs the mean of the R seeds: pass if, at t = 400, 600, 800, 1000,
    # |dW_th(refined) - mean_R dW_th| <= max(0.15 * mean_R dW_th, 2 * std_R), trailing 200 averages
    G456_rel=0.15, G456_sigma=2.0, G456_times=(400.0, 600.0, 800.0, 1000.0),
    # G7 shape (Q vs R) same rule; reported, pass/fail recorded
    # G8 relativistic detuning: |mean_R - mean_N| of <dW_th>_200 at t = 400, 800, 1200, 1600, 2000 is 'resolved' when
    # it exceeds 3 * sqrt(var_R/3 + var_N/3) (seed standard errors); outcome recorded either way
    G8_sigma=3.0, G8_times=(400.0, 800.0, 1200.0, 1600.0, 2000.0),
    # G9 noise floor: no-drive heating dW_th(1000) must be <= 20 % of the driven mean dW_th(1000)
    G9_noise_fraction=0.20,
    # G10 Dark-SPECTRAX (Newtonian Hermite, v_q/v_te 0.03, mobile ions) vs N: <dW_th>_200 within 15 % at 400, 600, 800, 1000
    G10_rel=0.15,
)


def single_particle():
    """S1-S3: exact single-particle solutions of the parent relativistic Boris (jaxincell._core.boris_relativistic).

    Electron (q = -e) in SI units; u = gamma v is carried at half steps (leapfrog), positions advance with v(u).
    S1 uniform E along +x force, u(0) = 0.3 c: u(t) = u0 + a t exact to 1e-12 (any dt) and x(T) from the work
       identity m c^2 (gamma - gamma0) = F (x - x0): relative error <= 2e-3 at 200 steps, ratio >= 3.5 for dt/2.
    S2 uniform B: |u| conserved to 1e-13 over 400 steps; after one exact relativistic gyroperiod 2 pi gamma/Omega
       (Omega = |q| B/m) the phase error is <= 0.02 rad at 200 steps per period with ratio >= 3.5 for dt/2; the
       gyroradius |u|/Omega is reproduced to 1e-2.
    S3 crossed E = (0, 0.5 c B, 0), B = (0, 0, B), u = gamma v_d with v_d = E x B / B^2: relative drift-velocity
       error after 100 steps at Omega dt = 0.05 is <= 1e-3 and the ratio for dt/2 is >= 3.5 (relativistic Boris does
       not preserve the E x B drift exactly; Vay and Higuera-Cary do; the HHS run has B = 0, where Boris = Vay).
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxincell import elementary_charge as e, mass_electron as m, speed_of_light as c
    from jaxincell._core import boris_relativistic as push

    q, qm, z = -e, -e / m, jnp.zeros((1, 3))
    out = {}

    def velocity(u):
        return u / np.sqrt(1 + np.sum((u / c) ** 2))

    # S1
    E0, T, u0 = 2e5, 4.0e-9, 0.3 * c
    force_acc = -qm * E0          # a = q E / m with E = -E0 x_hat
    errors = []
    for steps in (200, 400):
        dt = T / steps
        u = jnp.array([[u0 - force_acc * dt / 2, 0.0, 0.0]])
        E = jnp.array([[-E0, 0.0, 0.0]])
        x = 0.0
        for n in range(steps):
            u = push(u, E, z, qm, dt)
            x += float(velocity(np.asarray(u[0]))[0]) * dt
        exact_u = u0 + force_acc * (T - dt / 2)
        g0 = np.sqrt(1 + (u0 / c) ** 2)
        g1 = np.sqrt(1 + ((u0 + force_acc * T) / c) ** 2)
        x_exact = (m * c**2 / (m * force_acc)) * (g1 - g0)
        errors.append((abs(float(u[0, 0]) - exact_u) / exact_u, abs(x - x_exact) / x_exact))
    out["S1_momentum_rel_err"] = max(errors[0][0], errors[1][0])
    out["S1_position_rel_err_200"] = errors[0][1]
    out["S1_position_ratio"] = errors[0][1] / errors[1][1]

    # S2
    B0, gamma = 1.0, 1.5
    speed = c * np.sqrt(1 - 1 / gamma**2)
    Omega = abs(q) * B0 / m
    period = 2 * np.pi * gamma / Omega
    Bv = jnp.array([[0.0, 0.0, B0]])
    phase = []
    for steps in (200, 400):
        dt = period / steps
        start = push(jnp.array([[gamma * speed, 0.0, 0.0]]), z, Bv, qm, -dt / 2)
        u, pos, track = start, np.zeros(2), []
        for n in range(steps):
            u = push(u, z, Bv, qm, dt)
            pos = pos + velocity(np.asarray(u[0]))[:2] * dt
            track.append(pos.copy())
        a1 = np.arctan2(float(u[0, 1]), float(u[0, 0]))
        a0 = np.arctan2(float(start[0, 1]), float(start[0, 0]))
        phase.append(abs(np.angle(np.exp(1j * (a1 - a0)))))
        out["S2_norm_drift_%d" % steps] = abs(float(jnp.linalg.norm(u) / jnp.linalg.norm(start)) - 1)
        track = np.array(track)
        radius = 0.5 * (track[:, 1].max() - track[:, 1].min())
        out["S2_radius_rel_err_%d" % steps] = abs(radius - gamma * speed / Omega) / (gamma * speed / Omega)
    out["S2_phase_err_200"] = phase[0]
    out["S2_phase_ratio"] = phase[0] / max(phase[1], 1e-300)

    # S3
    errs = []
    B0 = 0.5
    Ey = 0.5 * c * B0
    E, Bv = jnp.array([[0.0, Ey, 0.0]]), jnp.array([[0.0, 0.0, B0]])
    vd = np.cross(np.asarray(E[0]), np.asarray(Bv[0])) / B0**2
    ud = vd / np.sqrt(1 - (vd @ vd) / c**2)
    for factor in (1.0, 0.5):
        dt = 0.05 / (abs(qm) * B0 / 1.1547) * factor
        u = jnp.asarray(ud[None, :])
        for n in range(100):
            u = push(u, E, Bv, qm, dt)
        errs.append(float(np.linalg.norm(np.asarray(u[0]) - ud) / np.linalg.norm(ud)))
    out["S3_drift_rel_err"] = errs[0]
    out["S3_drift_ratio"] = errs[0] / max(errs[1], 1e-300)
    return out


SINGLE_PARTICLE_THRESHOLDS = dict(
    S1_momentum_rel_err=("<=", 1e-12), S1_position_rel_err_200=("<=", 2e-3), S1_position_ratio=(">=", 3.5),
    S2_norm_drift_400=("<=", 1e-13), S2_phase_err_200=("<=", 0.02), S2_phase_ratio=(">=", 3.5),
    S2_radius_rel_err_200=("<=", 1e-2),
    S3_drift_rel_err=("<=", 1e-3), S3_drift_ratio=(">=", 3.5))


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, os.environ["JIC"])
    results = single_particle()
    verdict = {}
    for key, (op, bound) in SINGLE_PARTICLE_THRESHOLDS.items():
        verdict[key] = bool(results[key] <= bound if op == "<=" else results[key] >= bound)
    print(json.dumps(dict(results=results, pass_=verdict), indent=1))
