"""Numerical check of the weakly relativistic frequency shift of a cold k=0 oscillation,
d(gamma xdot)/dt = -x (units c = omega_pe = 1): (1 - omega/omega_pe)/v_max^2 -> 3/16 for v_max << 1.
Used in note chapter 10; not a repository record."""
import numpy as np
from scipy.integrate import solve_ivp

def rhs(t, y):
    x, p = y
    return [p / np.sqrt(1.0 + p * p), -x]

for vm in (0.05, 0.1, 0.2, 0.3):
    ev = lambda t, y: y[0]
    ev.direction = -1
    s = solve_ivp(rhs, [0, 60], [vm, 0.0], events=ev, rtol=1e-11, atol=1e-13, max_step=0.01)
    T = np.diff(s.t_events[0]).mean()
    pm = np.abs(s.y[1]).max()
    v = pm / np.sqrt(1 + pm * pm)
    print(f"v_max={v:.4f}  omega={2*np.pi/T:.6f}  (1-omega)/v_max^2={(1-2*np.pi/T)/v**2:.4f}  3/16={3/16:.4f}")
