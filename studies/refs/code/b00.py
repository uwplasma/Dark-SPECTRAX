"""B00 linear Landau damping, k lambda_D = 0.3, 0.5; eps = 1e-4 density seed.
Independent root (wofz) vs grid-SL fit vs Gkeyll DG fit vs SPECTRAX Hermite N=32..256 fits."""
import numpy as np, time
from scipy.optimize import least_squares
from disp import landau
from slv import VP, maxwellian
from fit import fit_pair
import sx, gk
from common import save

def fit_energy(t, W, w0, g0, window):
    m = (t >= window[0]) & (t <= window[1]); t, y = t[m], W[m]
    def res(p):
        w, g = p
        M = np.stack([np.exp(2*g*t), np.exp(2*g*t)*np.cos(2*w*t), np.exp(2*g*t)*np.sin(2*w*t)], 1)
        c = np.linalg.lstsq(M, y, rcond=None)[0]
        return (M@c - y)/y.max()
    s = least_squares(res, [w0, g0], xtol=1e-14, ftol=1e-14)
    return s.x[0], s.x[1], np.sqrt(np.mean(res(s.x)**2))

def windows(base):
    a, b = base
    return [(a, b), (a+2, b), (a, b-4), (a+2, b-4), (a-2, b-2)]

def scan(t, E, w0, g0, base):
    fits = [fit_pair(t, E, w0, g0, wd) for wd in windows(base)]
    a = np.array(fits)
    return dict(w=a[0, 0], g=a[0, 1], w_win_std=a[:, 0].std(), g_win_std=a[:, 1].std(), resid=a[0, 2],
                windows=windows(base), fits=a)

eps = 1e-4
res = {}; arrays = {}
for k, tmax, gwin in [(0.5, 45.0, (10.0, 40.0)), (0.3, 160.0, (16.0, 150.0))]:
    root = landau(k); w0, g0 = root.real, root.imag
    R = dict(k=k, root_wofz=root)
    # --- grid SL: dt and v refinement
    grid = {}
    for dt, Nv, vmax in [(0.1, 256, 8.0), (0.05, 256, 8.0), (0.025, 256, 8.0), (0.0125, 256, 8.0), (0.025, 512, 10.0)]:
        s = VP(2*np.pi/k, 16, vmax, Nv)
        f = (1+eps*np.cos(k*s.x))[:, None]*maxwellian(s.v)[None, :]
        t0 = time.time(); f, t, E, d = s.run(f, dt, tmax, save_every=max(1, int(round(0.1/dt)))); wall = time.time()-t0
        sc = scan(t, E[:, 0], w0, g0, gwin)
        Wf0 = d[0, 2]
        sc.update(dt=dt, Nv=Nv, vmax=vmax, Nx=16, wall=wall, mass_err=abs(d[:, 0]/d[0, 0]-1).max(),
                  energy_err_over_W0=abs(d[:, 1]+d[:, 2]-d[0, 1]-d[0, 2]).max()/Wf0, recurrence_time=2*np.pi/(k*s.dv))
        grid[f"dt{dt}_Nv{Nv}_v{vmax}"] = sc
        arrays[f"k{k}_grid_dt{dt}_Nv{Nv}_t"] = t; arrays[f"k{k}_grid_dt{dt}_Nv{Nv}_E"] = E[:, 0]
        print(k, "grid", dt, Nv, sc['w'], sc['g'], wall)
    # Richardson (2nd-order Strang) from dt=0.05, 0.025
    a1, a2 = grid["dt0.025_Nv256_v8.0"], grid["dt0.0125_Nv256_v8.0"]
    wR = a2['w'] + (a2['w']-a1['w'])/3; gR = a2['g'] + (a2['g']-a1['g'])/3
    R['grid'] = grid
    R['grid_best'] = dict(w=wR, g=gR, unc_w=max(abs(wR-a2['w']), a2['w_win_std'], abs(a2['w']-grid["dt0.025_Nv512_v10.0"]['w'])),
                          unc_g=max(abs(gR-a2['g']), a2['g_win_std'], abs(a2['g']-grid["dt0.025_Nv512_v10.0"]['g'])))
    # --- Gkeyll DG (p=2, serendipity, RK3) on field energy
    gks = {}
    for Nx, Nv in ([(32, 32), (32, 64)] if k == 0.5 else [(32, 32), (32, 64), (32, 128)]):
        t, W, wall, _ = gk.run(k, tmax, Nx=Nx, Nv=Nv)
        fe = [fit_energy(t, W, w0, g0, wd) for wd in windows(gwin)]
        fe = np.array(fe)
        gks[f"Nx{Nx}_Nv{Nv}"] = dict(w=fe[0, 0], g=fe[0, 1], w_win_std=fe[:, 0].std(), g_win_std=fe[:, 1].std(), resid=fe[0, 2], wall=wall)
        arrays[f"k{k}_gk_Nv{Nv}_t"] = t; arrays[f"k{k}_gk_Nv{Nv}_W"] = W
        print(k, "gkeyll", Nv, fe[0], wall)
    R['gkeyll'] = gks
    # --- SPECTRAX
    sxr = {}
    ref_t = arrays[f"k{k}_grid_dt0.0125_Nv256_t"]; ref_E = arrays[f"k{k}_grid_dt0.0125_Nv256_E"]
    for Nn in (32, 64, 128, 256):
        r = sx.run([(1, 1, 0)], 2*np.pi/k, 4, Nn, tmax, int(round(tmax/0.1))+1, {1: eps/2}, beta=0.1)
        r2 = sx.run([(1, 1, 0)], 2*np.pi/k, 4, Nn, tmax, int(round(tmax/0.1))+1, {1: eps/2}, beta=0.1)  # warm (post-compile) timing
        t, E = r['t'], r['Ek'][:, 1]
        Eg = np.interp(t, ref_t, ref_E.real) + 1j*np.interp(t, ref_t, ref_E.imag)
        dev = np.abs(E-Eg)/abs(Eg[0])
        # contamination time: first t where deviation from grid reference exceeds 1e-3 * |E(0)|
        idx = np.argmax(dev > 1e-3); tc = t[idx] if dev.max() > 1e-3 else np.inf
        tb = min(gwin[1], 0.9*tc) if np.isfinite(tc) else gwin[1]
        win = (gwin[0], tb)
        ok = tb - gwin[0] > 8
        sc = scan(t, E, w0, g0, win) if ok else dict(w=np.nan, g=np.nan, w_win_std=np.nan, g_win_std=np.nan)
        sc.update(Nn=Nn, contamination_time=tc, fit_window=win, compile_plus_run=r['wall'], run_only=r2['wall'], stats=r['stats'],
                  hermite_front_estimate=np.sqrt(Nn)/k, beta=0.1, max_dev_over_E0=dev.max())
        sxr[f"N{Nn}"] = sc
        arrays[f"k{k}_sx_N{Nn}_t"] = t; arrays[f"k{k}_sx_N{Nn}_E"] = E
        print(k, "sx", Nn, sc['w'], sc['g'], tc, win, r['wall'], r2['wall'])
    # beta-independence check at N=128
    r = sx.run([(1, 1, 0)], 2*np.pi/k, 4, 128, tmax, int(round(tmax/0.1))+1, {1: eps/2}, beta=1.0)
    R['beta_check_max_rel_diff_N128'] = float(np.abs(r['Ek'][:, 1]-arrays[f"k{k}_sx_N128_E"]).max()/eps)
    R['spectrax'] = sxr
    res[f"k{k}"] = R
save("B00", dict(model="ordinary 1D1V Vlasov-Poisson/Ampere, electrons only, fixed uniform ions, omega_pe=1, v_te=1, lambda_D=1",
                 seed="n_e = 1 + 1e-4 cos(k x), Gauss-consistent E", observable="E_k(t), k=mode 1 (complex Fourier coefficient)",
                 fit_model="E_k = exp(g t)[A exp(-i w t) + B exp(i w t)], LSQ on complex samples, windows varied (5 variants)",
                 spectrax_setup="Ns=1, Nx=4 (strict 2/3 keeps |m|<=1), Nm=Np=1, nu=0, D=0, Dopri8, PID rtol=atol=1e-10, beta=v_t/c=0.1",
                 results=res), arrays)
