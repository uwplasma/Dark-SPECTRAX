"""B06 two-pulse echo. Pulse 1 (t=0): density seed eps1 cos(k1 x), Gauss-consistent E.
Pulse 2 (t=tau): instantaneous external velocity kick f(x,v) -> f(x, v - d2 cos(k2 x)).
Self-consistent Vlasov-Poisson/Ampere otherwise. Echo expected at k3=k2-k1, t = k2 tau/(k2-k1)."""
import numpy as np, time, sys
from scipy.linalg import expm
from slv import VP, maxwellian
import sx
from common import save
k0 = 0.25; m1, m2 = 4, 6; m3 = m2 - m1
k1, k2, k3 = m1*k0, m2*k0, m3*k0
tau = 10.0; tE = k2*tau/(k2-k1); tmax = 45.0
eps1 = 0.01; d2 = 0.05
L = 2*np.pi/k0
res = {}; arrays = {}
# ---------------- grid reference
def grid(Nv, dt, d2=d2, Nx=64, vmax=8.0):
    s = VP(L, Nx, vmax, Nv)
    f = (1+eps1*np.cos(k1*s.x))[:, None]*maxwellian(s.v)[None, :]
    t0 = time.time()
    f, t, E, d = s.run(f, dt, tmax, kicks=[(tau, lambda x: d2*np.cos(k2*x))], modes=(m1, m2, m3), save_every=int(round(0.05/dt)))
    return t, E, time.time()-t0, 2*np.pi/(k2*s.dv)
G = {}
for Nv, dt in [(512, 0.025), (512, 0.0125), (1024, 0.0125)]:
    t, E, wall, TR = grid(Nv, dt)
    w = (t > tau+5)
    i = np.argmax(np.abs(E[:, 2])*w); G[f"Nv{Nv}_dt{dt}"] = dict(t_echo=float(t[i]), amp=float(abs(E[i, 2])), wall=wall, grid_recurrence_k2=TR)
    arrays[f"grid_Nv{Nv}_dt{dt}_t"] = t; arrays[f"grid_Nv{Nv}_dt{dt}_E"] = E
    print("grid", Nv, dt, G[f"Nv{Nv}_dt{dt}"])
t, E, wall, _ = grid(512, 0.0125, d2=d2/2)
i = np.argmax(np.abs(E[:, 2])*(t > tau+5))
G["half_kick_amp_ratio"] = float(abs(E[i, 2])/G["Nv512_dt0.0125"]["amp"])
ref_t, ref_E = arrays["grid_Nv1024_dt0.0125_t"], arrays["grid_Nv1024_dt0.0125_E"]
res["grid"] = G
# ---------------- SPECTRAX: run to tau, kick in real space on a padded grid, restart
beta = 0.1
def hermite_kick(Ck, Nn, Nx, a):
    """Apply f(x,v)->f(x,v-d(x)) exactly in the truncated basis: C(x) <- expm(s(x) R) C(x), (RC)_n = sqrt(2n) C_{n-1}, s=d/a."""
    Nf = 128
    Cf = np.zeros((Nn, Nf//2+1), complex); Cf[:, :Nx//2+1] = Ck[:, 0, :, 0]
    Cx = np.fft.irfft(Cf, n=Nf, axis=1, norm="forward")
    xf = np.arange(Nf)*(L*beta)/Nf
    R = np.diag(np.sqrt(2*np.arange(1, Nn)), -1)
    s = beta*d2*np.cos(k2/beta*xf)/a
    for j in range(Nf):
        Cx[:, j] = expm(s[j]*R)@Cx[:, j]
    Cn = np.fft.rfft(Cx, axis=1, norm="forward")[:, :Nx//2+1]
    out = Ck.copy(); out[:, 0, :, 0] = Cn
    return out
S = {}
Nx = 33
for Nn in (64, 128, 256, 512):
    r1 = sx.run([(1, 1, 0)], L, Nx, Nn, tau, int(tau/0.05)+1, {m1: eps1/2}, beta=beta)
    Ck = hermite_kick(r1['Ck_last'], Nn, Nx, r1['a'][0])
    t0 = time.time()
    r2 = sx.run([(1, 1, 0)], L, Nx, Nn, tmax-tau, int((tmax-tau)/0.05)+1, {}, beta=beta, Ck0=Ck, Fk0=r1['Fk_last'])
    t = np.concatenate([r1['t'], tau + r2['t'][1:]]); E = np.concatenate([r1['Ek'], r2['Ek'][1:]])[:, [m1, m2, m3]]
    i = np.argmax(np.abs(E[:, 2])*(t > tau+5))
    Eg = np.interp(t, ref_t, np.abs(ref_E[:, 2]))
    dev = np.abs(np.abs(E[:, 2])-Eg)
    S[f"N{Nn}"] = dict(Nn=Nn, t_echo=float(t[i]), amp=float(abs(E[i, 2])), wall=r1['wall']+r2['wall'],
                       max_abs_dev_k3_over_grid_echo=float(dev.max()/G["Nv1024_dt0.0125"]["amp"]),
                       phase_mixing_index_k1_at_tau=(k1*tau)**2, stats=[r1['stats'], r2['stats']])
    arrays[f"sx_N{Nn}_t"] = t; arrays[f"sx_N{Nn}_E"] = E
    print("sx", Nn, S[f"N{Nn}"])
res["spectrax"] = S
save("B06", dict(model="ordinary VP/Ampere electrons, fixed ions; external impulsive kick at tau",
                 params=dict(k0=k0, k1=k1, k2=k2, k3=k3, tau=tau, eps1=eps1, kick_dv=d2, tmax=tmax, t_echo_theory=tE),
                 observable="|E_k3(t)| (k3 = k2-k1), echo peak = max for t>tau+5",
                 spectrax_setup="Ns=1, Nx=33 (keeps |m|<=10), nu=0, Dopri8 rtol=atol=1e-10, restart at tau with kick applied by exact truncated-basis shift expm(sR) on a 128-point padded x grid, beta=0.1",
                 results=res), arrays)
