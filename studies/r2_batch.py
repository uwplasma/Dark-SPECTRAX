"""R2: run the case list of r2_gates.cases() with a few worker processes, each case chunked and resumable.
Usage: python studies/r2_batch.py WORKERS [filter-substring ...]   (single-threaded XLA per worker; nice 10)"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_gates as g  # noqa: E402
import r2_run  # noqa: E402


def one(c):
    conv, vq, nu, om, mem, ctl = c
    Nx, Nn = (8, 64) if ctl else g.MEMBERS[mem]
    case = r2_run.case_name(conv, vq, nu, om, Nx, Nn, ctl)
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               OMP_NUM_THREADS="1")
    for _ in range(8):
        cmd = ["nice", "-n", "10", sys.executable, str(HERE / "r2_run.py"), "--conv", conv, "--vq", str(vq), "--nu", str(nu),
               "--omega", om, "--Nx", str(Nx), "--Nn", str(Nn), "--c", str(g.C_CLOSURE), "--T", str(g.T_END),
               "--chunk", str(g.CHUNK)] + (["--control"] if ctl else [])
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        with open(HERE / "r2" / "log.txt", "a") as fh:
            fh.write(f"{case} rc={r.returncode} {r.stdout.strip().splitlines()[-1][:200] if r.stdout.strip() else r.stderr[-300:]}\n")
        parts = sorted((r2_run.OUT).glob(case + ".part*.json"), key=lambda p: int(p.suffixes[-2][5:]))
        if r.returncode or not parts:
            return case, "error"
        info = json.loads(parts[-1].read_text())
        if info["t_end"] >= g.T_END - 1e-9 or info["status"] not in ("success", "short"):
            return case, info["status"]
    return case, "unfinished"


if __name__ == "__main__":
    (HERE / "r2").mkdir(exist_ok=True)
    nw = int(sys.argv[1])
    flt = sys.argv[2:]
    todo = [c for c in g.cases() if not flt or any(f in r2_run.case_name(c[0], c[1], c[2], c[3], *((8, 64) if c[5] else g.MEMBERS[c[4]]), c[5]) for f in flt)]
    with ThreadPoolExecutor(nw) as ex:
        for case, st in ex.map(one, todo):
            print(case, st, flush=True)
