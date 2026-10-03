# Independent ordinary-plasma references (imported)

Produced in the 2026-10-03 review by a separate agent workspace and imported here unchanged, except that
machine-specific paths in `code/` were replaced by relative paths or environment variables
(`GKEYLL_EXE`, `GKEYLL_LANDAU_INPUT`, `GKEYLL_WORKDIR`).

- `code/slv.py`: NumPy semi-Lagrangian 1D1V Vlasov-Poisson solver (Strang splitting, spectral shifts); no SPECTRAX code.
- `code/disp.py`: wofz plasma-dispersion roots. `code/fit.py`: fit rules used by every comparison.
- `code/b0*.py`, `code/common.py`: drivers that wrote the records. `code/sx.py` is their SPECTRAX adapter for the
  unmerged integration commit 6781d80 (its `initial_projection_residual` key does not exist at ab87385).
- `code/gk.py`: driver for an unmodified local Gkeyll build; no Gkeyll input or source is included here.
- `B00 B01 B02 B06 B07/`: `run.json` (configuration, versions, windows, numbers, timings) and `README.md`;
  `data.npz` is kept for B00 and B02, which `studies/compare_refs.py` uses.

The SPECTRAX numbers inside these records come from 6781d80. The Hermite side of B00 and B02 is rerun on the
pinned parent by `python studies/compare_refs.py` (results in `studies/refs_rerun/`, summarized in
docs/results.md). Units: omega_pe = 1, v_te = 1, electrons only, fixed uniform ions.
