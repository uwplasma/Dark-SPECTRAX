"""Independent kinetic dispersion roots (electrostatic units: omega_pe=1 total, lambda_D/v_t explicit).
Convention exp(i k x - i w t), k>0, Z via scipy wofz (causal Landau continuation)."""
import numpy as np
from scipy.special import wofz
from scipy.optimize import fsolve

def Z(z):
    return 1j*np.sqrt(np.pi)*wofz(z)

def eps_species(w, k, frac, vt, u):
    zeta = (w - k*u)/(np.sqrt(2)*k*vt)
    return frac/(k**2*vt**2)*(1+zeta*Z(zeta))

def D(w, k, species):
    return 1 + sum(eps_species(w, k, *s) for s in species)

def root(k, species, guess):
    f = lambda x: [D(x[0]+1j*x[1], k, species).real, D(x[0]+1j*x[1], k, species).imag]
    x, info, ier, msg = fsolve(f, [guess.real, guess.imag], full_output=True, xtol=1e-14)
    w = x[0]+1j*x[1]
    assert abs(D(w, k, species)) < 1e-10, (msg, D(w, k, species))
    return w

def landau(k):
    g = np.sqrt(1+3*k**2) - 0.1j
    return root(k, [(1.0, 1.0, 0.0)], g)

def two_stream(k, u, vt):
    sp = [(0.5, vt, u), (0.5, vt, -u)]
    cold = np.sqrt(k**2*u**2 + 0.5 - np.sqrt(0.25 + 2*k**2*u**2) + 0j)  # w^2 root of cold quartic (purely imaginary branch)
    g = 1j*abs(cold.imag) if abs(cold.imag) > 1e-3 else 0.3j
    return root(k, sp, g), 1j*abs(cold.imag)

if __name__ == "__main__":
    for k in (0.3, 0.5):
        print(k, landau(k))
    print(two_stream(0.6, 1.0, 0.1))
