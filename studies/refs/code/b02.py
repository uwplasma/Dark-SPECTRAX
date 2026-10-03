"""B02 symmetric two-stream: two electron beams (density 1/2 each, drift +-u, thermal sd vt), fixed ions, omega_pe=1 total.
Growth of modes k=0.2..0.8 (u=1) vs cold root and warm (wofz) root."""
import numpy as np, time
from disp import two_stream
from fit import fit_growth, matrix_pencil
from slv import VP, maxwellian
import sx
from common import save
u = 1.0; k0 = 0.2; L = 2*np.pi/k0; ms = (1, 2, 3, 4); eps = 1e-12; tmax = 60.0
res = {}; arrays = {}
for vt in (0.1, 0.3):
    R = dict(u=u, vt=vt)
    roots = {}
    for m in ms:
        k = m*k0; w, cold = two_stream(k, u, vt)
        roots[f"k{k:.1f}"] = dict(cold_growth=cold.imag, warm_root=w)
    R["roots"] = roots
    seeds = {m: eps/2 for m in ms}
    win = (1.0, 20.0)
    def fits(t, E):
        out = {}
        for j, m in enumerate(ms):
            k = m*k0
            A = np.abs(E[:, j]); big = np.nonzero(A > 1e-5)[0]
            t2 = min(tmax, t[big[0]] if len(big) else tmax); t1 = 25.0
            wins = [(t1, t2), (t1+5, t2), (t1, t2-5), (t1-5, t2)]
            gs = [fit_growth(t, A, w) for w in wins]
            gmp = matrix_pencil(t, E[:, j], (1.0, 20.0), order=4).imag.max()
            out[f"k{k:.1f}"] = dict(g=gs[0], g_win_std=float(np.std(gs)), window=wins[0], g_matrix_pencil_early=gmp,
                                    max_amp_in_window=float(A[t <= t2].max()))
        return out
    G = {}
    for Nv, dt, vmax in [(1024, 0.025, 3.0), (2048, 0.0125, 3.5)]:
        s = VP(L, 32, vmax, Nv)
        f0 = 0.5*maxwellian(s.v, vt, u) + 0.5*maxwellian(s.v, vt, -u)
        f = (1+eps*sum(np.cos(m*k0*s.x) for m in ms))[:, None]*f0[None, :]
        t0 = time.time(); f, t, E, d = s.run(f, dt, tmax, modes=ms, save_every=int(round(0.05/dt))); wall = time.time()-t0
        G[f"Nv{Nv}_dt{dt}"] = dict(fits=fits(t, E), wall=wall)
        arrays[f"vt{vt}_grid_Nv{Nv}_t"] = t; arrays[f"vt{vt}_grid_Nv{Nv}_E"] = E
        print(vt, "grid", Nv, {kk: round(v['g'], 5) for kk, v in G[f'Nv{Nv}_dt{dt}']['fits'].items()}, wall)
    R["grid"] = G
    S = {}
    for Nn in (8, 16, 32, 64):
        r = sx.run([(0.5, vt, u), (0.5, vt, -u)], L, 16, Nn, tmax, int(tmax/0.05)+1, seeds, beta=0.1)
        t, E = r['t'], r['Ek'][:, list(ms)]
        S[f"N{Nn}"] = dict(fits=fits(t, E), wall_compile_run=r['wall'], stats=r['stats'])
        arrays[f"vt{vt}_sx_N{Nn}_t"] = t; arrays[f"vt{vt}_sx_N{Nn}_E"] = E
        print(vt, "sx", Nn, {kk: round(v['g'], 5) for kk, v in S[f'N{Nn}']['fits'].items()}, r['wall'])
    R["spectrax"] = S
    print(vt, {kk: (round(v['cold_growth'], 5), round(v['warm_root'].imag, 5)) for kk, v in roots.items()})
    res[f"vt{vt}"] = R
save("B02", dict(model="two electron beams (+-u, sd vt, density 1/2 each), fixed ions, omega_pe=1 total; units: u=1",
                 seed="n = 1 + 1e-12 sum_{m=1..4} cos(m k0 x), k0=0.2, Gauss-consistent E",
                 observable="|E_k|; growth = slope of log|E_k| on [25, t(|E_k|=1e-5)] (variants shift ends by 5); early [1,20] 4-term matrix-pencil kept as a diagnostic (biased at k=0.2)",
                 spectrax_setup="Ns=2 electron species each with its own drifting basis (u=+-beta u, a=sqrt2 beta vt), Nx=16, nu=0, Dopri8 1e-10, beta=0.1",
                 results=res), arrays)
