"""Dark-JAX-in-Cell PIC run of the README two-stream case, for the Hermite-vs-PIC phase-space comparison.

Run in an environment with jaxincell and darkjaxincell importable (Dark-JAX-in-Cell commit recorded below), on CPU:

    python studies/djic_phase_space.py [PARTICLES_PER_BEAM] [CELLS]

Same physical inputs as ``PHASE_CASES["two_stream"]`` in studies/figures.py: two electron beams with drifts
+-v0, v0 = 0.1 c, thermal standard deviation 0.3 v0, box mode k v0/omega_pe = 0.4, fixed neutralizing ions,
density seed dn_s/n_s = 1e-3 cos(k x) on both beams; dark run eta = 0.3, Omega_D = omega_pe.
PIC specifics: quiet start, omega_pe dt = 0.006 (c dt/dx = 0.49 at 128 cells), displacement seed.
Writes artifacts/djic_phase/two_stream.npz (sampled markers per frame; not committed) and
studies/djic_phase/run.json + data.npz (E_k1 histories); per-frame marker histograms go to the cache.
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxincell import Domain, Simulation, Species, elementary_charge as e, epsilon_0, mass_electron, quiet_start
from jaxincell import speed_of_light as c
from darkjaxincell import DarkField, DarkSimulation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies"))
from figures import PHASE_CASES, PHASE_FRAMES  # noqa: E402

per_beam = int(sys.argv[1]) if len(sys.argv) > 1 else 131072
cells = int(sys.argv[2]) if len(sys.argv) > 2 else 128
case = PHASE_CASES["two_stream"]
(fr, vt_es, u_es), _ = case["pops"]
beta, k_es, seed, T = 0.1, case["k"], case["seed"], case["T"]
length, eta = 1.0, 0.3
k = 2 * np.pi / length
v0 = u_es * beta * c
wp = k * v0 / k_es
density = wp ** 2 * epsilon_0 * mass_electron / e ** 2
vth = np.sqrt(2) * vt_es * beta * c  # quiet_start convention: sqrt(2) x standard deviation (studies/c05_pic_rerun.py)
stride = 125  # steps per movie frame: omega_pe dt = 0.006, c dt/dx = 0.49 at 128 cells (explicit Proca margin)
dt = T / (PHASE_FRAMES - 1) / stride / wp
frames = PHASE_FRAMES - 1
species = []
for sign, name in ((1, "plus"), (-1, "minus")):
    x, v = quiet_start(per_beam, length, vth=(vth, 0.0, 0.0), drift=(sign * v0, 0.0, 0.0))
    x = x.at[:, 0].add(-seed / k * jnp.sin(k * x[:, 0]))  # n_s(x) = n_s (1 + seed cos k x) to first order
    species.append(Species.electrons(per_beam, density=fr * density, vth=(vth, 0.0, 0.0), name=name).replace(x=x, v=v))
plasma = Simulation(Domain(length=length, cells=cells, time_step=dt), tuple(species))
dark = DarkSimulation(plasma, DarkField(wp, eta))
sample = np.concatenate([np.linspace(0, per_beam - 1, 20000, dtype=int),
                         per_beam + np.linspace(0, per_beam - 1, 20000, dtype=int)])
v_edges = np.linspace(*case["v"], 81)
x_edges = np.linspace(0, 1, 65)

out, timing = {}, {}
for label, sim in (("ordinary", plasma), ("dark", dark)):
    tic = time.perf_counter()
    state, ts, E1, xs, vs, H = None, [], [], [], [], []
    for first in range(0, frames, 10):  # saves at t = stride*dt, ..., T (the run does not store t = 0)
        n = min(10, frames - first)
        r = sim.run(n * stride, store_every=stride, state=state, verbose=False)
        state = r.state
        o = r if label == "ordinary" else r.ordinary
        jax.block_until_ready(o.E)
        ts.append(np.asarray(o.t) * wp)
        E1.append(np.fft.rfft(np.asarray(o.E[:, :, 0]), axis=1)[:, 1] / cells)
        xp = (np.asarray(o.x[:, :, 0]) / length + 0.5) % 1.0  # PIC box is [-L/2, L/2): shift to [0, L)
        vp = np.asarray(o.v[:, :, 0]) / (beta * c)  # electrostatic units of the Hermite case (v0 = u_es)
        xs.append(xp[:, sample])
        vs.append(vp[:, sample])
        H.append(np.stack([np.histogram2d(a, b, bins=(x_edges, v_edges))[0] for a, b in zip(xp, vp)]))
    timing[label] = time.perf_counter() - tic
    out[label] = {"t": np.concatenate(ts), "E1": np.concatenate(E1), "x": np.concatenate(xs),
                  "v": np.concatenate(vs), "H": np.concatenate(H)}
    print(label, out[label]["t"].shape, f"{timing[label]:.1f}s", flush=True)

cache = ROOT / "artifacts" / "djic_phase"
cache.mkdir(parents=True, exist_ok=True)
np.savez_compressed(cache / "two_stream.npz", **{f"{m}_{q}": out[m][q] for m in out for q in out[m]})
rec_dir = ROOT / "studies" / "djic_phase"
rec_dir.mkdir(parents=True, exist_ok=True)
djic = Path(sys.modules["darkjaxincell"].__file__).resolve().parents[1]
sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=djic, capture_output=True, text=True).stdout.strip()
rec = {"case": "two_stream", "command": "python studies/djic_phase_space.py " + " ".join(sys.argv[1:]),
       "dark_jax_in_cell_commit": sha, "cells": cells, "particles_per_beam": per_beam, "dt_omega_pe": dt * wp, "c_dt_over_dx": c * dt * cells / length,
       "frame_stride_steps": stride, "frames": int(out["ordinary"]["t"].size), "eta": eta, "Omega_D_over_wp": 1.0,
       "v0_over_c": v0 / c, "seed": seed, "wall_s": timing, "backend": jax.default_backend(), "jax": jax.__version__,
       "python": platform.python_version(),
       "histogram": {"x_edges_over_L": [0, 1, 64], "v_edges_over_v0": [*case["v"], 80]}}
(rec_dir / "run.json").write_text(json.dumps(rec, indent=2) + "\n")
np.savez_compressed(rec_dir / "data.npz", **{f"{m}_{q}": out[m][q] for m in out for q in ("t", "E1")})
