"""Exact references for relgrid (H04): homogeneous kinetic/cold-fluid driven two-species oscillator, relativistic Langmuir dispersion.

Units c = omega_pe = n = m_e = 1; m_i = 1836; T_e = T_i = 1e-3 m_e c^2 (u_te^2 = 1e-3); drive F_ext = E0 cos(w t), E0 = v_q u_te.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import k1e

MI = 1836.0
VTE = np.sqrt(1e-3)
VTI = VTE / np.sqrt(MI)
OMEGA = np.sqrt(1 + 1 / MI)
_X, _W = np.polynomial.hermite_e.hermegauss(120)
_W = _W / np.sqrt(2 * np.pi)


def vmean(U, s, rel=True):
    """<v> for u = U + s z (z standard normal): exact quadrature of the Maxwellian-in-u load."""
    w = U + s * _X
    return (_W * (w / np.sqrt(1 + w * w) if rel else w)).sum()


def homogeneous(vq, T, w=OMEGA, rel=True, sig_e=VTE, sig_i=VTI, rtol=1e-12):
    """Exact homogeneous (k = 0) kinetic problem: every particle sees the same force, so u = u0 + dU_s(t).
    State (dU_e, dU_i, Ebar, W_ext); dU_e' = -F, dU_i' = F/m_i, Ebar' = -J, J = <v_i> - <v_e>, W' = J E_drive, F = Ebar + E_drive.
    Returns the dense solution. sig -> 0 is the cold-fluid oscillator (with gamma if rel)."""
    E0 = vq * VTE

    def rhs(t, y):
        Ue, Ui, E, _ = y
        Ed = E0 * np.cos(w * t)
        J = vmean(Ui, sig_i, rel) - vmean(Ue, sig_e, rel)
        return [-(E + Ed), (E + Ed) / MI, -J, J * Ed]

    return solve_ivp(rhs, [0, T], [0, 0, 0, 0], method="DOP853", rtol=rtol, atol=1e-15, dense_output=True)


def cold_fluid(vq, T, w=OMEGA, rel=True, rtol=1e-12):
    """Independent cold-fluid ODE written directly with gamma: u_s' = (q_s/m_s) F, v_s = u_s/sqrt(1+u_s^2)."""
    E0 = vq * VTE

    def rhs(t, y):
        ue, ui, E, _ = y
        Ed = E0 * np.cos(w * t)
        ve = ue / np.sqrt(1 + ue * ue) if rel else ue
        vi = ui / np.sqrt(1 + ui * ui) if rel else ui
        J = vi - ve
        return [-(E + Ed), (E + Ed) / MI, -J, J * Ed]

    return solve_ivp(rhs, [0, T], [0, 0, 0, 0], method="DOP853", rtol=rtol, atol=1e-15, dense_output=True)


def demod_phase(t, E, w, t0, t1):
    """Phase (rad) and amplitude of E(t) against the pump: A = <E exp(+i w t)> over [t0, t1]; E = Re(A* e^{-iwt}) * 2 for a pure tone.
    Uses a full-period-commensurate Hann weight to suppress the edge leakage. The phase is arg A."""
    m = (t >= t0) & (t <= t1)
    x = (t[m] - t0) / (t1 - t0)
    wt = np.sin(np.pi * x) ** 2
    A = 2 * (wt * E[m] * np.exp(1j * w * t[m])).sum() / wt.sum()
    return float(np.angle(A)), float(np.abs(A))


def juttner_pdf(u, theta):
    return np.exp(-(np.sqrt(1 + u * u) - 1) / theta) / (2 * k1e(1 / theta))


def juttner_dprime(u, theta):
    g = np.sqrt(1 + u * u)
    return -(u / (g * theta)) * juttner_pdf(u, theta)


def eps_langmuir(omega, k, theta_e=1e-3, mi=MI, rel=True, b_frac=0.35):
    """Relativistic electrostatic dielectric function for a Juttner electron + Juttner ion plasma (analytic continuation to Im omega < 0
    by passing the u contour below the pole: u = s - i b(s), b(s) = b0 exp(-(s/(6 sigma))^2)). eps = 1 + (1/k) sum_s (q_s^2/m_s) int f0'(u)/(omega - k v(u)) du,
    with f0 per unit u, int f0 du = 1."""
    out = 1.0 + 0j
    for theta, qq in ((theta_e, 1.0), (theta_e / mi, 1.0 / mi)):  # theta_s = T/(m_s c^2); q^2/m_s = 1, 1/m_i
        sig = np.sqrt(theta)
        s = np.linspace(-14 * sig, 14 * sig, 400001)
        b0 = b_frac * sig if theta == theta_e else 0.0
        b = b0 * np.exp(-((s / (6 * sig)) ** 2))
        u = s - 1j * b
        du = 1 - 1j * (-2 * s / (6 * sig) ** 2) * b  # du/ds
        g = np.sqrt(1 + u * u)
        f0p = -(u / (g * theta)) * np.exp(-(g - 1) / theta) / (2 * k1e(1 / theta))
        v = u / g
        integrand = f0p * du / (omega - k * v) if rel else None
        if not rel:
            f0p = -(u / theta) * np.exp(-u * u / (2 * theta)) / np.sqrt(2 * np.pi * theta)
            integrand = f0p * du / (omega - k * u)
        out += qq / k * np.trapezoid(integrand, s)
    return out


def langmuir_root(k, rel=True, guess=None, theta_e=1e-3):
    """Complex root of eps(omega, k) = 0 (secant)."""
    w0 = guess if guess is not None else complex(np.sqrt(1 + 3 * k * k * theta_e), -1e-3)
    w1 = w0 * (1 + 1e-4)
    f0, f1 = eps_langmuir(w0, k, theta_e, rel=rel), eps_langmuir(w1, k, theta_e, rel=rel)
    for _ in range(60):
        w2 = w1 - f1 * (w1 - w0) / (f1 - f0)
        if abs(w2 - w1) < 1e-13:
            return w2
        w0, f0, w1 = w1, f1, w2
        f1 = eps_langmuir(w1, k, theta_e, rel=rel)
    return w1
