"""Run the local (unmodified) Gkeyll build on a copied/edited VP Landau input; parse field-energy dynvector."""
import numpy as np, os, re, subprocess, time
GK = os.environ.get("GKEYLL_EXE", "gkeyll")
SRC = os.environ.get("GKEYLL_LANDAU_INPUT", "rt_vp_landau_damping_1x1v_p2.lua")
WD = os.environ.get("GKEYLL_WORKDIR", "gk_work")

def read_dynvec(fn):
    b = open(fn, "rb").read(); off = 0; ts = []; ds = []
    while off < len(b):
        assert b[off:off+5] == b"gkyl0"; off += 5
        ver, ftype, meta = np.frombuffer(b, np.uint64, 3, off); off += 24 + int(meta)
        rtype, esz, n = np.frombuffer(b, np.uint64, 3, off); off += 24
        n = int(n); esz = int(esz)
        ts.append(np.frombuffer(b, np.float64, n, off)); off += 8*n
        ds.append(np.frombuffer(b, np.float64, n*esz//8, off).reshape(n, esz//8)); off += n*esz
    return np.concatenate(ts), np.concatenate(ds)

def run(k, tend, Nx=32, Nv=32, vmax=6.0, p=2, alpha=1e-4, tag=None):
    tag = tag or f"gk_k{k}_Nx{Nx}_Nv{Nv}_p{p}"
    s = open(SRC).read()
    s = s.replace("k0 = 0.5 / lambda_D", f"k0 = {k} / lambda_D")
    s = re.sub(r"Nx = 32", f"Nx = {Nx}", s); s = re.sub(r"Nvx = 32", f"Nvx = {Nv}", s)
    s = s.replace("vx_max = 6.0 * vte", f"vx_max = {vmax} * vte")
    s = s.replace("t_end = 100.0 / omega_pe", f"t_end = {tend} / omega_pe")
    s = s.replace("alpha = 1.0e-4", f"alpha = {alpha}")
    s = s.replace("poly_order = 2", f"poly_order = {p}")
    os.makedirs(WD, exist_ok=True)
    fn = os.path.join(WD, tag + ".lua"); open(fn, "w").write(s)
    t0 = time.time()
    subprocess.run([GK, fn], cwd=WD, check=True, capture_output=True)
    wall = time.time() - t0
    t, W = read_dynvec(os.path.join(WD, tag + "-field-energy.gkyl"))
    return t, W[:, 0], wall, s
