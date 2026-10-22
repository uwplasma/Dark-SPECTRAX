"""relpic: one PIC run of the HHS (arXiv:2510.13956v1, Fig. 2, App. B) set-up with Dark-JAX-in-Cell.

    python studies/relpic/relpic_run.py --tag NAME --cells 1000 --ppc 103 --dt 0.02 --T 1000 \
        [--ratio 0.03] [--shape 2|5] [--nonrel] [--seed 0] [--chunk 200]

Loading, drive and diagnostics are the Dark-JAX-in-Cell `examples/dark_reservoir.py` paper study (`paper_case`):
quiet start (equal spacing, co-located e/i), m_i/m_e = 1836, T_e = T_i = 1e-3 m_e c^2, L = 40 c/omega_pe, uniform
prescribed E = (m_e c omega_pe/e) a0 cos(omega_pe t), a0 = ratio * sqrt(1e-3).  `--nonrel` swaps the parent's
relativistic Boris pusher for the Newtonian one (everything else identical).  The run is cut into chunks that
restart from the exact archived state (`paper_continue`), so no run is longer than one process budget.
Environment: DJIC (Dark-JAX-in-Cell checkout), JIC (JAX-in-Cell checkout at the parent pin).
"""
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
OUT = Path(__file__).resolve().parent / "runs"


def load_module(nonrel):
    sys.path.insert(0, os.environ["DJIC"])
    sys.path.insert(0, os.environ["JIC"])
    m = importlib.import_module("examples.dark_reservoir")
    if nonrel:
        from jaxincell import Simulation, Solver
        original = m.paper_plasma

        def paper_plasma(*a, **k):
            plasma, wp = original(*a, **k)
            order = plasma.solver.shape_order if hasattr(plasma.solver, "shape_order") else 2
            solver = Solver(relativistic=False) if order == 2 else Solver(relativistic=False, shape_order=order)
            return Simulation(plasma.domain, plasma.species, solver), wp

        m.paper_plasma = paper_plasma
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cells", type=int, default=1000)
    ap.add_argument("--ppc", type=int, default=103)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--ratio", type=float, default=0.03)
    ap.add_argument("--shape", type=int, default=2)
    ap.add_argument("--nonrel", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=float, default=200.0)
    a = ap.parse_args(argv)
    m = load_module(a.nonrel)
    folder = OUT / a.tag
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    first = min(a.chunk, a.T)
    history, settings, results = m.paper_case(
        folder / "c0", a.cells, a.ppc * a.cells, a.dt, first, a.seed, a.ratio, block_horizon=min(100.0, first),
        shape_order=a.shape)
    done, index = first, 0
    while done < a.T - 1e-9:  # each chunk restarts from the exact archived state of the previous one
        index += 1
        done = min(done + a.chunk, a.T)
        history, settings, results = m.paper_continue(
            folder / f"c{index}", folder / f"c{index - 1}" / "final_state.npz", done)
        print(f"chunk {index} to t = {done}", flush=True)
    wall = time.perf_counter() - t0
    keep = ("t", "spread", "kinetic", "electric", "nonzero_electric", "mean_E", "rms", "mean", "work",
            "balance", "momentum", "max_speed")
    arrays = {k: np.asarray(history[k]) for k in keep if k in history}
    np.savez_compressed(folder / "history.npz", **arrays)
    settings = {k: v for k, v in settings.items() if k not in ("initial_fingerprints",)}
    record = dict(args=vars(a), nonrelativistic=a.nonrel, wall_s=wall, settings=settings, results=results,
                  devices=[str(d) for d in jax.devices()])
    (folder / "record.json").write_text(json.dumps(record, indent=1, default=float))
    print("done", a.tag, f"{wall:.0f} s")


if __name__ == "__main__":
    main()
