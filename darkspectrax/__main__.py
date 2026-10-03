"""Thin TOML command line: ``python -m darkspectrax input.toml --out DIR``."""

from __future__ import annotations

import argparse
import tomllib

import numpy as np

from ._model import Model
from ._simulation import consistent_fields, maxwellian, proca_mode, run, save_record


def build(cfg):
    """Model and initial state from a parsed TOML dictionary."""
    mcfg = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg["model"].items()}
    model = Model(**mcfg)
    ini = cfg.get("init", {})
    pert = [(s, tuple(idx), complex(re, im)) for s, idx, re, im in ini.get("perturbations", [])]
    y = maxwellian(model, ini.get("densities", [1.0] * model.Ns), pert)
    y = consistent_fields(model, y, ini.get("E_mean", (0.0, 0.0, 0.0)))
    for mode in ini.get("proca", []):
        A = np.asarray(mode["A_re"]) + 1j * np.asarray(mode.get("A_im", [0.0, 0.0, 0.0]))
        y = proca_mode(model, y, tuple(mode["index"]), A)
    return model, y


def main(argv=None):
    ap = argparse.ArgumentParser(prog="darkspectrax", description=__doc__)
    ap.add_argument("input", help="TOML file with [model], [init], [run] tables")
    ap.add_argument("--out", required=True, help="output directory for run.json/run.npz")
    ap.add_argument("--t-max", type=float, help="override [run] t_max")
    ap.add_argument("--resume", help="run.npz of a previous run of the same input: continue "
                    "from its final kinetic and field state")
    args = ap.parse_args(argv)
    with open(args.input, "rb") as fh:
        cfg = tomllib.load(fh)
    model, y = build(cfg)
    if args.resume:
        old = np.load(args.resume)
        y = {**y, "Ck": old["Ck_final"], "Fk": old["Fk"][-1], "Dk": old["Dk"][-1]}
    r = dict(cfg.get("run", {}))
    if args.t_max is not None:
        r["t_max"] = args.t_max
    out = run(model, y, r.pop("t_max", 10.0), **r)
    rec = save_record(args.out, model, out, {"input": cfg, "resumed_from": args.resume})
    print(f"{rec['status']}: {out['num_steps']} steps, max ledger defect "
          f"{rec['max_abs_ledger_defect']:.2e}, Gauss {rec['max_gauss_residual']}")
    return 0 if rec["status"] == "success" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
