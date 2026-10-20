"""R2 linear theory: driven uniform cold two-fluid plasma with the momentum friction of the Dougherty e-i operator.

Units: omega_pe = 1 (electron), n = 1, energy per unit volume in code units (divide by v_te^2 = 1e-3 for n T_e V).
Equations (the exact uniform limit of the model; studies/h08.py G0 checks the code against them):
    Ue' = -(Ebar + Ed) - nu_ei (Ue - Ui)/2,   Ui' = (Ebar + Ed)/MI - nu_ie (Ui - Ue)/2,   nu_ie = nu_ei/MI,
    Ebar' = Ue - Ui,   Ed = E0 cos(omega t),   W_ext' = J Ed,  J = Ui - Ue.
Both collision conventions (LBO-EM "dougherty", LBO-G beta = 0 "dougherty_g") give the same uniform friction (Morse rate),
so the relative velocity relaxes at  nu_eff = (nu_ei / 2)(1 + 1/MI).  With wL^2 = 1 + 1/MI the driven response obeys
    J' = wL^2 (Ebar + Ed) - nu_eff J,  Ebar' = -J,
and the steady-state absorbed power per volume is (B12 form with eps0 = 1, omega_pe^2 -> wL^2)
    P_ss = wL^2 nu_eff omega^2 E0^2 / (2 [(wL^2 - omega^2)^2 + nu_eff^2 omega^2]),   resonance: P_ss -> wL^2 E0^2 / (2 nu_eff).
The amplitude of the transient (the natural mode) decays as exp(-nu_eff t / 2).
"""
import numpy as np
from scipy.integrate import solve_ivp

MI = 1836.0
vte = np.sqrt(1e-3)
wL = np.sqrt(1 + 1 / MI)


def nu_eff(nu_ei):
    return 0.5 * nu_ei * (1 + 1 / MI)


def p_ss(E0, nu_ei, omega):
    ne = nu_eff(nu_ei)
    return wL ** 2 * ne * omega ** 2 * E0 ** 2 / (2 * ((wL ** 2 - omega ** 2) ** 2 + ne ** 2 * omega ** 2))


def exact_linear(vq, nu_ei, omega, t):
    """(Ebar, W_ext) of the uniform ODE sampled at `t` (DOP853, rtol 1e-12)."""
    E0, nie = vq * vte, nu_ei / MI

    def f(tt, y):
        Ue, Ui, Eb, W = y
        Ed = E0 * np.cos(omega * tt)
        J = Ui - Ue
        return [-(Eb + Ed) - nu_ei * (Ue - Ui) / 2, (Eb + Ed) / MI - nie * (Ui - Ue) / 2, -J, J * Ed]

    s = solve_ivp(f, (0, t[-1]), [0, 0, 0, 0], method="DOP853", t_eval=t, rtol=1e-12, atol=1e-18)
    return s.y[2], s.y[3]


def cycle_mean(t, y, tq, omega):
    """Mean of y over the window |t - tq| <= pi/omega (one drive half-period on each side clipped to the record)."""
    w = np.abs(t - tq) <= np.pi / omega
    return float(np.mean(y[w]))
