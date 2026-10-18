"""H08 G1 follow-up (new study, declared before running; the H08 gate G1 itself is not amended).

Question: why does the H08 nu = 0 rerun (v_q 0.03, Nx 32, Nn 64, parent dd80b0f, restart chunk 200) differ from the
nxconv record vq0.03_Nx32_Nn64_c4_nf1e-14 (parent 16400f8, restart chunk 240) by 6e-6 in dK_e?
Hypothesis (from the parent diff 16400f8..dd80b0f: collision_model=None is documented bitwise unchanged; ode_rtol/ode_atol
default to ode_tolerance; FlooredPIDController is unchanged): the only setting that differs between the two scripts'
runs is the restart chunk length (ds.run_adaptive is restarted every chunk, which resets the PID controller and step size).
Test: rerun h08.py nu = 0 at dd80b0f with --chunk 240 (nxconv's), 200 (H08's) and 120, same everything else, and compare each
with the record as in G1 (max |x - x_rec| / max |x_rec| over common saves, dK_e and W_ext).
Attribution criteria, declared now: chunk 240 matches the record to <= 1e-12 (then the parent change is not a cause) and the
chunk-200 and chunk-120 deviations are >= 1e-8 (then restart placement alone produces a deviation of the observed size).

Run: python studies/h08_g1_cause.py   (about 3 min; writes studies/h08_g1_cause/summary.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h08  # noqa: E402
import darkspectrax as ds  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402

D = Path(__file__).resolve().parent / "h08_g1_cause"
REC = Path(__file__).resolve().parent / "nxconv" / "runs" / "vq0.03_Nx32_Nn64_c4_nf1e-14"


def cat(files):
    t, K, W = [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        s = slice(0, None) if i == 0 else slice(1, None)
        t.append(d["t"][s]), K.append(d["K_species"][s]), W.append(d["W"][s])
    return np.concatenate(t), np.concatenate(K), np.concatenate(W)


def parts(stem):
    return sorted(stem.parent.glob(stem.name + ".part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))


def series(files):
    t, K, W = cat(files)
    return t, K[:, 0] - K[0, 0], (W[:, 2] - W[0, 2]).real


def main():
    out = {"parent_commit": ds.PARENT_COMMIT, "repository_commit": _sim._git_sha(), "dev_vs_record": {}}
    tr, dk_r, w_r = series(parts(REC))
    for chunk in (240, 200, 120):
        h08.RUNS = D / f"runs_chunk{chunk}"
        for _ in range(12):
            if h08.run(argparse.Namespace(vq=0.03, nu=0.0, Nx=32, Nn=64, T=1000.0, chunk=float(chunk), control=False)):
                break
            if parts(h08.RUNS / "vq0.03_nu0_Nx32_Nn64") and json.loads(
                    Path(str(parts(h08.RUNS / "vq0.03_nu0_Nx32_Nn64")[-1])[:-4] + ".json").read_text())["t_end"] >= 1000 - 1e-9:
                break
        t, dk, w = series(parts(h08.RUNS / "vq0.03_nu0_Nx32_Nn64"))
        n = min(t.size, tr.size)
        assert np.allclose(t[:n], tr[:n])
        out["dev_vs_record"][str(chunk)] = {
            "parts": len(parts(h08.RUNS / "vq0.03_nu0_Nx32_Nn64")), "t_end": float(t[-1]),
            "dKe": float(np.max(np.abs(dk[:n] - dk_r[:n])) / np.max(np.abs(dk_r[:n]))),
            "Wext": float(np.max(np.abs(w[:n] - w_r[:n])) / np.max(np.abs(w_r[:n])))}
    d = out["dev_vs_record"]
    out["criteria"] = {"chunk240_matches_1e-12": d["240"]["dKe"] <= 1e-12 and d["240"]["Wext"] <= 1e-12,
                       "chunk200_and_120_ge_1e-8": min(d["200"]["dKe"], d["120"]["dKe"]) >= 1e-8}
    (D / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
