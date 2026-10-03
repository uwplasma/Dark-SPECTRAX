"""Fit E_k(t) = exp(g t) [A exp(-i w t) + B exp(+i w t)] (standing-wave pair of a real-field root)."""
import numpy as np
from scipy.optimize import least_squares

def fit_pair(t, Ek, w0, g0, window):
    m = (t >= window[0]) & (t <= window[1])
    t, y = t[m], Ek[m]
    def lin(w, g):
        M = np.stack([np.exp((g-1j*w)*t), np.exp((g+1j*w)*t)], axis=1)
        c = np.linalg.lstsq(M, y, rcond=None)[0]
        return M, c
    def res(p):
        M, c = lin(*p)
        r = (M@c - y)/np.abs(y).max()
        return np.concatenate([r.real, r.imag])
    sol = least_squares(res, [w0, g0], xtol=1e-14, ftol=1e-14)
    w, g = sol.x
    rel = np.sqrt(np.mean(res(sol.x)**2))
    return w, g, rel

def fit_window_scan(t, Ek, w0, g0, windows):
    out = [fit_pair(t, Ek, w0, g0, wdw) for wdw in windows]
    a = np.array(out)
    return a[0, 0], a[0, 1], a[:, 0].std(), a[:, 1].std(), out

def fit_growth(t, amp, window):
    m = (t >= window[0]) & (t <= window[1])
    p = np.polyfit(t[m], np.log(amp[m]), 1)
    return p[0]

def matrix_pencil(t, y, window, order=4):
    """Complex frequencies (exp(-i w t) convention: returns w = wr + i g) of y(t) on a uniform window."""
    m = (t >= window[0]) & (t <= window[1]); t, y = t[m], y[m]
    dt = t[1]-t[0]; N = len(y); Lp = N//2
    Y = np.array([y[i:i+Lp+1] for i in range(N-Lp)])
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh[:order].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1])@V[1:])
    return 1j*np.log(z)/dt
