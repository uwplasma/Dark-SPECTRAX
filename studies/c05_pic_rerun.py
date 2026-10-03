"""C05 PIC side rerun on CPU with Dark-JAX-in-Cell (run in a separate environment with darkjaxincell installed).

Same construction as Dark-JAX-in-Cell examples/dark_kinetic.py (physical preset): L = 1 m, k = 2 pi/L,
omega_pe = k c/10 (sigma = 0.05 c), k lambda_D = 0.5, quiet start, exact current-neutral loading,
displacement seed 0.01/k sin(kx), eta = 0.3, Omega_D = k c, dt_over_dx_c = 0.5, t omega_pe <= 29.45.
Ladder: cells x particles chosen to refine the mesh and the particles per cell independently.

Usage: python studies/c05_pic_rerun.py CELLS PARTICLES  -> studies/c05_pic/cells{C}_particles{P}.json/.npz
"""

import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

from jaxincell import Domain, Simulation, Species, elementary_charge as e, epsilon_0, mass_electron, quiet_start
from jaxincell import speed_of_light as c
from darkjaxincell import DarkField, DarkSimulation

cells, particles = int(sys.argv[1]), int(sys.argv[2])
steps = 3000 * cells // 32  # fixed dt_over_dx_c = 0.5: same physical end time for every mesh
length, eta = 1.0, 0.3
k = 2 * np.pi / length
wp = k * c / 10
density = wp ** 2 * epsilon_0 * mass_electron / e ** 2
vth = 0.5 / k * np.sqrt(2) * wp
x, v = quiet_start(particles, length, vth=(vth, 0.0, 0.0))
v = v.at[:, 0].add(-jnp.mean(v[:, 0]))
x = x.at[:, 0].add(0.01 / k * jnp.sin(k * x[:, 0]))
electrons = Species.electrons(particles, density=density, vth=(vth, 0.0, 0.0)).replace(x=x, v=v)
plasma = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=0.5), (electrons,))
tic = time.perf_counter()
parent = plasma.run(steps, store_particles=False, verbose=False)
jax.block_until_ready(parent.E)
t1 = time.perf_counter()
result = DarkSimulation(plasma, DarkField(k * c, eta)).run(steps, store_particles=False, verbose=False)
jax.block_until_ready(result.E)
t2 = time.perf_counter()
t = np.asarray(result.ordinary.t) * wp
Ep = np.fft.rfft(np.asarray(parent.E[:, :, 0]), axis=1)[:, 1] / cells
Ed = np.fft.rfft(np.asarray(result.ordinary.E[:, :, 0]), axis=1)[:, 1] / cells


def fit(tt, A, window=(2.0, 12.0)):
    """Dark-JAX-in-Cell fixed-window rule: maxima of |E_k|, log-linear slope, mean spacing."""
    p = find_peaks(A)[0]
    p = p[(tt[p] > window[0]) & (tt[p] < window[1])]
    r = linregress(tt[p], np.log(A[p]))
    return [float(np.pi / np.mean(np.diff(tt[p]))), float(r.slope)], float(r.stderr), int(p.size)


fo, so, no = fit(t, np.abs(Ep))
fd, sd, nd = fit(t, np.abs(Ed))
U = np.asarray(result.energy()["total_with_dark"])
rec = {"cells": cells, "particles": particles, "particles_per_cell": particles / cells, "steps": steps,
       "ordinary": fo, "ordinary_slope_stderr": so, "ordinary_maxima": no,
       "dark": fd, "dark_slope_stderr": sd, "dark_maxima": nd,
       "max_closed_energy_drift": float(np.max(np.abs(U - U[0])) / U[0]),
       "wall_parent_incl_compile": t1 - tic, "wall_dark_incl_compile": t2 - t1,
       "backend": jax.default_backend(), "jax": jax.__version__, "python": platform.python_version(),
       "dark_jax_in_cell_commit": "d54757962f3b9f7ec18eb981804ef59f6db6076c"}
out = Path(__file__).resolve().parent / "c05_pic"
out.mkdir(exist_ok=True)
(out / f"cells{cells}_particles{particles}.json").write_text(json.dumps(rec, indent=1) + "\n")
s = slice(None, None, max(1, steps // 3000))
np.savez_compressed(out / f"cells{cells}_particles{particles}.npz", t=t[s], parent_Ek=Ep[s], dark_run_Ek=Ed[s])
print(json.dumps(rec))
