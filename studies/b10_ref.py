"""B10 reference: Langmuir modulational (sideband) instability of a finite-amplitude pump at k0, ordinary and dark.

Parent units: c = 1, omega_pe = 1 (n = 1, electron mass 1, eps0 = 1), v_te = sqrt(T_e/m_e), r_D = v_te,
ion Omega_cs = 1/m_i, T_i = T_e/tau. Field amplitude convention: the pump electric field is the real wave
E(x,t) = E0 cos(k0 x - w0 t) (E0 = PEAK Poisson field, Fourier |F_k0| = E0/2); w = eps0 E0^2 / (2 n T_e).
This w equals the |E|^2/(8 pi n T_e) of Zakharov 1972 (Gaussian units, psi = potential with phi = Re(psi e^{-i w t})).

Reference tiers (all derived here from Zakharov 1972 eqs 1.2, 1.9, 2.8/2.9 and checked in studies/b10.py gates):

R0  fluid Zakharov with the exact envelope structure: sideband operator (Omega - Delta_pm) with the quadratic
    Delta_pm(K) = +-v_g K + (3/2) r_D^2 K^2 (v_g = 3 k0 r_D^2 w_p), ion sound operator Omega^2 - c_s^2 K^2,
    c_s^2 = (T_e + 3 T_i)/M (1D kinetic ions; the paper's printed 3/2 T_i is recorded as a sensitivity),
        (Omega^2 - c_s^2 K^2) [ (Omega - Delta_+)(Omega + Delta_-) ] = (K^2 eps0 E0^2 / (4 M)) (w_p / 2)
                                               [ (Omega + Delta_-) - (Omega - Delta_+) ].
    At k0 = 0 and Omega -> 0 this is eq 2.8 (static, q = T_e/(T_e + T_i) with the isothermal static response) and for
    the sound-resonant limit eq 2.9. closed_form_static() returns the 2.8 rate used as an independent check.
R1  semi-kinetic: exact Langmuir dispersion roots w(k) (kinetic electrons and ions, including the dark mixing), exact
    Delta_pm = w(k0 +- K) - w(k0), ion low-frequency response by the kinetic ion function with Boltzmann electrons,
        delta n / n = -varsigma U /(T_i + varsigma T_e), varsigma = 1 + xi Z(xi), xi = Omega /(sqrt2 K v_ti),
    and the dark-modified coupling (below). Valid for Omega << K v_te, w << 1, K << k0-independent smallness.

Dark generalization (derived here; the paper has no dark term). For a longitudinal mode the electron force field is
E_f = g(w,k) E with g = 1 + eta^2 Q/(Q - Omega_D^2), Q = w^2 - k^2, (checked against D_L of studies/b04_b05.py:
D_L = (Q - Omega_D^2) eps_g, eps_g = 1 + g chi). A nonlinear current J_nl gives eps_g E_s = J_nl/(i w). With J_nl =
-e dn v_pump, v_pump driven by g0 E0: eps_g'(Omega - Delta) E_s = (g0/w0^2) (dn/n) E0, so the Langmuir coupling is
g0 / eps_g' instead of w0 / 2, the ponderomotive energy is U = g0^2 |E|^2 / (4 w0^2), and the density response uses
the low-frequency g(Omega,K) ~ 1 (K^2/Omega_D^2 ~ 1e-4). Dispersion relation for a sideband pair:
    1 = G(Omega,K) (g0/eps_g') E0^2 [ 1/(Omega - Delta_+) - 1/(Omega + Delta_-) ],
    G = -varsigma g0^2 /(4 w0^2 (T_i + varsigma T_e)).
Ordinary (eta = 0): g0 = 1, eps' = 2/w0^3, giving the Zakharov system above. Growth is Im Omega of the root with the
largest imaginary part.
"""
import numpy as np
from scipy.optimize import root
from scipy.special import wofz

eta_dark, OmegaD_dark = 0.3, 1.0


def Zp(z):
    return 1j * np.sqrt(np.pi) * wofz(z)


def chi_pop(w, k, n, q, om, vt):
    z = w / (np.sqrt(2) * k * vt)
    return n * q * q * om / (k * vt) ** 2 * (1 + z * Zp(z))


class Plasma:
    """Parent-unit plasma. eta = 0 means ordinary."""

    def __init__(self, vte=0.1, mi=1836.0, tau=1.0, eta=0.0, OmegaD=1.0):
        self.vte, self.mi, self.tau, self.eta, self.OmegaD = vte, mi, tau, eta, OmegaD
        self.Te = vte ** 2
        self.Ti = self.Te / tau
        self.vti = vte / np.sqrt(mi * tau)

    def chi(self, w, k):
        return chi_pop(w, k, 1.0, -1.0, 1.0, self.vte) + chi_pop(w, k, 1.0, 1.0, 1.0 / self.mi, self.vti)

    def g(self, w, k):
        Q = w * w - k * k
        return 1.0 + self.eta ** 2 * Q / (Q - self.OmegaD ** 2)

    def eps(self, w, k):
        return 1.0 + self.g(w, k) * self.chi(w, k)

    def omega(self, k, guess=None):
        """Complex Langmuir-branch root of eps_g = 0 nearest guess (dark: pick the branch with `guess`)."""
        w0 = guess if guess is not None else np.sqrt(1 + 3 * (k * self.vte) ** 2)
        f = lambda x: [self.eps(x[0] + 1j * x[1], k).real, self.eps(x[0] + 1j * x[1], k).imag]
        s = root(f, [np.real(w0), np.imag(w0)], tol=1e-14)
        return s.x[0] + 1j * s.x[1]

    def deps_dw(self, w, k, h=1e-6):
        return (self.eps(w + h, k) - self.eps(w - h, k)) / (2 * h)

    def branches(self, k):
        """Langmuir-like roots at k: ordinary one root; dark two roots (split near Omega_D ~ omega_pe)."""
        if self.eta == 0:
            return [self.omega(k)]
        out = []
        for g in np.linspace(0.6, 1.6, 21):
            w = self.omega(k, g + 0j)
            if abs(self.eps(w, k)) < 1e-9 and w.real > 0.3 and not any(abs(w - o) < 1e-7 for o in out):
                out.append(w)
        return sorted(out, key=lambda z: z.real)


def zakharov_params(pl, k0, w0):
    """Fluid-Zakharov frame quantities (ordinary): v_g, D coefficient."""
    return 3 * k0 * pl.vte ** 2, 1.5 * pl.vte ** 2


def closed_form_static(pl, w):
    """Eq 2.8 at k0 = 0 and the static ion response: gamma^2(K) = D(K) [ (eps0 E0^2 / (4 (T_e + T_i))) - D(K) ],
    D = (3/2) r_D^2 K^2 (w_p = 1); returns (K_max, gamma_max) from the exact maximum. E0^2 = 2 w T_e."""
    A = 2 * w * pl.Te / (4 * (pl.Te + pl.Ti))  # eps0 E0^2 / (4 (T_e+T_i)), E0^2 = 2 w T_e
    # gamma^2 = D (A - D), D = c K^2, max at D = A/2: gamma = A/2
    c = 1.5 * pl.vte ** 2
    return float(np.sqrt(A / (2 * c))), float(A / 2)


def sideband_roots(pl, k0, K, E0, branch=None, tier="R1", cs_T=3.0, cs2=None):
    """All roots Omega of the sideband dispersion relation (R0 via polynomial, R1 via Newton from R0)."""
    w0 = branch if branch is not None else pl.omega(k0)
    w0r = w0.real
    Tdenom_i = 1.0 / pl.mi
    if tier == "R0":
        vg = 3 * k0 * pl.vte ** 2
        D = 1.5 * pl.vte ** 2 * K ** 2
        Dp, Dm = vg * K + D, -vg * K + D
        cs2 = (pl.Te + cs_T * pl.Ti) * Tdenom_i if cs2 is None else cs2
        # (Om^2 - cs2 K^2)(Om - Dp)(Om + Dm) = a [(Om + Dm) - (Om - Dp)],  a = K^2 E0^2/(4M) (1/2)
        a = K ** 2 * E0 ** 2 / (4 * pl.mi) * 0.5
        p1 = np.poly1d([1, 0, -cs2 * K ** 2])
        p2 = np.poly1d([1, -Dp]) * np.poly1d([1, Dm])
        poly = p1 * p2 - np.poly1d([a * (Dm + Dp)])
        return poly.roots
    # R1
    wp, wm = pl.omega(k0 + K, w0).real, pl.omega(abs(k0 - K), w0).real
    Dp, Dm = wp - w0r, wm - w0r
    g0 = pl.g(w0r, k0).real
    ep = pl.deps_dw(w0r, k0).real
    kap = g0 / ep

    def f(Om):
        xi = Om / (np.sqrt(2) * K * pl.vti)
        vs = 1 + xi * Zp(xi)
        G = -vs * g0 ** 2 / (4 * w0r ** 2 * (pl.Ti + vs * pl.Te))
        return 1 - G * kap * E0 ** 2 * (1 / (Om - Dp) - 1 / (Om + Dm))

    # Newton from the R0 polynomial roots and from a grid of complex starts
    cs = np.sqrt((pl.Te + 3 * pl.Ti) / pl.mi)
    starts = list(sideband_roots(pl, k0, K, E0, branch, "R0"))
    span = 1.5 * max(abs(Dp), abs(Dm), cs * K)
    for re in np.linspace(-span, span, 41):
        for im in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            starts.append(re + 1j * im)
    out = []
    for s0 in starts:
        r = root(lambda x: [f(x[0] + 1j * x[1]).real, f(x[0] + 1j * x[1]).imag], [s0.real, s0.imag], tol=1e-13)
        z = r.x[0] + 1j * r.x[1]
        if r.success and abs(f(z)) < 1e-8 and not any(abs(z - o) < 1e-9 for o in out):
            out.append(z)
    return np.array(out)


def growth(pl, k0, K, E0, branch=None, tier="R1", cs_T=3.0, cs2=None):
    r = sideband_roots(pl, k0, K, E0, branch, tier, cs_T, cs2)
    if len(r) == 0:
        return 0.0, 0.0
    i = int(np.argmax(r.imag))
    return float(max(r[i].imag, 0.0)), float(r[i].real)


if __name__ == "__main__":
    import sys
    pl = Plasma(vte=0.1, mi=1836, tau=10.0)
    w = 0.01
    E0 = pl.vte * np.sqrt(2 * w)
    print("2.8 static optimum", closed_form_static(pl, w))
    for k0 in (0.0, 0.1, 0.2):
        for tier in ("R0", "R1"):
            Ks = np.linspace(0.1, 1.2, 12)
            print(k0, tier, [round(growth(pl, k0 if k0 else 1e-6, K, E0, tier=tier)[0] * 1e3, 3) for K in Ks])
