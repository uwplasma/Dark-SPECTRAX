"""B01 nonlinear Landau damping, k=0.5, eps=0.01,0.02,0.05: envelope extrema of |E_k1| vs O'Neil bounce time
tau_B = 2 pi / sqrt(k E0), E0 = eps/k (initial field amplitude) => tau_B = 2 pi/sqrt(eps)."""
import numpy as np, time
from scipy.ndimage import maximum_filter1d
from slv import VP, maxwellian
import sx
from common import save
from scipy.signal import find_peaks
def envelope_extrema(t, E):
    dt = t[1]-t[0]; w = int(round(2*np.pi/1.1/dt))
    env = maximum_filter1d(np.abs(E), size=w, mode="nearest")
    le = np.log(env)
    sel = t > 5
    imax, _ = find_peaks(le, prominence=0.3, distance=w)
    imin, _ = find_peaks(-le, prominence=0.3, distance=w)
    return env, [float(t[i]) for i in imin if t[i] > 5], [float(t[i]) for i in imax if t[i] > 5]
res = {}; arrays = {}
from disp import landau
for k, eps, tmax in [(0.5, 0.05, 100.0), (0.3, 0.01, 200.0), (0.3, 0.02, 200.0), (0.3, 0.05, 200.0)]:
    L = 2*np.pi/k; tag = f"k{k}_eps{eps}"
    gL = -landau(k).imag
    R = dict(k=k, eps=eps, tmax=tmax, tauB_oneil=2*np.pi/np.sqrt(eps), gammaL=gL, gammaL_tauB=gL*2*np.pi/np.sqrt(eps))
    G = {}
    for Nx, Nv, dt in [(32, 1024, 0.025), (64, 2048, 0.0125)]:
        s = VP(L, Nx, 8.0, Nv)
        f = (1+eps*np.cos(k*s.x))[:, None]*maxwellian(s.v)[None, :]
        t0 = time.time(); f, t, E, d = s.run(f, dt, tmax, modes=(1, 2), save_every=int(round(0.05/dt))); wall = time.time()-t0
        env, mins, maxs = envelope_extrema(t, E[:, 0])
        G[f"Nx{Nx}_Nv{Nv}_dt{dt}"] = dict(env_min_times=mins, env_max_times=maxs, wall=wall,
                                          energy_err_rel_total=float(abs(d[:, 1]+d[:, 2]-d[0, 1]-d[0, 2]).max()/(d[0, 1]+d[0, 2])),
                                          mass_err=float(abs(d[:, 0]/d[0, 0]-1).max()), fmin=float(d[:, 4].min()))
        arrays[f"{tag}_grid_Nx{Nx}_Nv{Nv}_t"] = t; arrays[f"{tag}_grid_Nx{Nx}_Nv{Nv}_E"] = E
        print(tag, "grid", Nx, Nv, mins[:3], maxs[:3], wall)
    R["grid"] = G
    rt, rE = arrays[f"{tag}_grid_Nx64_Nv2048_t"], arrays[f"{tag}_grid_Nx64_Nv2048_E"][:, 0]
    S = {}
    for Nn, nu in [(256, 0.0), (512, 0.0), (1024, 0.0), (256, 1.0), (512, 1.0)]:
        r = sx.run([(1, 1, 0)], L, 16, Nn, tmax, int(tmax/0.05)+1, {1: eps/2}, beta=0.1, nu=nu)
        t, E = r['t'], r['Ek'][:, 1]
        env, mins, maxs = envelope_extrema(t, E)
        Eg = np.interp(t, rt, rE.real)+1j*np.interp(t, rt, rE.imag)
        dt_ = t[1]-t[0]; w_ = int(round(2*np.pi/1.1/dt_))
        envg = maximum_filter1d(np.abs(Eg), size=w_, mode="nearest")
        dev = np.abs(np.log(env/envg))
        i = np.argmax(dev > 0.1); tdev = float(t[i]) if dev.max() > 0.1 else None
        S[f"N{Nn}_nu{nu}"] = dict(env_min_times=mins, env_max_times=maxs, wall_compile_run=r['wall'], stats=r['stats'],
                                  first_t_logenv_dev_gt_0p1=tdev, max_logenv_dev=float(dev.max()))
        arrays[f"{tag}_sx_N{Nn}_nu{nu}_t"] = t; arrays[f"{tag}_sx_N{Nn}_nu{nu}_E"] = E
        print(tag, "sx", Nn, nu, mins[:3], maxs[:3], tdev, r['wall'])
    R["spectrax"] = S
    res[tag] = R
save("B01", dict(model="ordinary VP/Ampere electrons, fixed ions", seed="n=1+eps cos(kx), Gauss-consistent E",
                 observable="|E_k1(t)|, envelope = running max over ~one plasma period; extrema = find_peaks(log env, prominence 0.3); SPECTRAX-vs-grid agreement = |log(env_sx/env_grid)|, first time > 0.1",
                 spectrax_setup="Ns=1, Nx=16 (keeps |m|<=5), Dopri8 rtol=atol=1e-10, beta=0.1; nu = parent hypercollision coefficient (numerical closure, col=n(n-1)(n-2)/((N-1)(N-2)(N-3)))",
                 results=res), arrays)
