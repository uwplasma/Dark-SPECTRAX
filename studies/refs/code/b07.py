"""B07 free streaming of one Fourier-Maxwellian perturbation, force off (q=0) in SPECTRAX.
Exact: C_n(t) = C_0(0) (-i b)^n exp(-b^2/4)/sqrt(2^n n!), b = k_code a t = sqrt(2) k v_t t
=> |g_n|^2 ∝ (k v_t t)^{2n}/n! exp(-(k v_t t)^2)."""
import numpy as np
from scipy.special import gammaln
import sx
from common import save
k = 0.5; eps = 1e-4
res = {}; arrays = {}
for Nn in (16, 32, 64, 128, 256, 512):
    tmax = float(np.ceil(3.0*np.sqrt(Nn)/k))
    nt = int(tmax/0.05)+1
    r = sx.run([(1, 1, 0)], 2*np.pi/k, 4, Nn, tmax, nt, {1: eps/2}, beta=0.1, qforce=False, tol=1e-12)
    t = r['t']; C = r['Ck'][:, :, 1]; a = r['a'][0]
    C00 = eps/2/a**3
    b = np.sqrt(2)*k*t
    n = np.arange(Nn)
    logmag = n[None, :]*np.log(np.maximum(b[:, None], 1e-300)) - b[:, None]**2/4 - 0.5*(n*np.log(2) + gammaln(n+1))[None, :]
    exact = C00*np.exp(logmag)*((-1j)**n)[None, :]
    err_low = np.abs(C[:, :4]-exact[:, :4]).max(1)/C00
    err_all = np.sqrt((np.abs(C-exact)**2).sum(1))/C00
    tc = {}
    for thr in (1e-8, 1e-6, 1e-3):
        i = np.argmax(err_low > thr); tc[f"{thr:g}"] = float(t[i]) if err_low.max() > thr else None
    # where is the front: mean index of |C_n|^2 distribution vs (k v t)^2
    P = np.abs(C)**2; nmean = (P*n).sum(1)/P.sum(1)
    res[f"N{Nn}"] = dict(Nn=Nn, tmax=tmax, contamination_time_lowmodes=tc, sqrtN_over_kvt=np.sqrt(Nn)/k,
                         tc1e6_times_k_over_sqrtN=(tc["1e-06"]*k/np.sqrt(Nn) if tc["1e-06"] else None),
                         max_err_all_before_half_front=float(err_all[t < 0.5*np.sqrt(Nn)/k].max()),
                         wall_compile_run=r['wall'], stats=r['stats'])
    arrays[f"N{Nn}_t"] = t; arrays[f"N{Nn}_err_low"] = err_low; arrays[f"N{Nn}_nmean"] = nmean
    arrays[f"N{Nn}_absC_over_C00"] = np.abs(C[::20])/C00; arrays[f"N{Nn}_absexact_over_C00"] = np.abs(exact[::20])/C00
    print(Nn, tc, res[f"N{Nn}"]["tc1e6_times_k_over_sqrtN"], res[f"N{Nn}"]["max_err_all_before_half_front"], r['wall'])
save("B07", dict(model="free streaming, force off (qs=0), Ns=1, Nx=4, Nm=Np=1, nu=0, Dopri8 rtol=atol=1e-12, beta=0.1",
                 k_lambdaD=k, seed="n = 1 + 1e-4 cos(kx); f1 = (eps/2) f_M e^{ikx}",
                 exact="C_n(t)=C_0(0)(-i b)^n e^{-b^2/4}/sqrt(2^n n!), b=sqrt(2) k v_t t",
                 contamination_definition="first t with max_{n<=3}|C_n-exact|/C_0(0) > threshold",
                 results=res), arrays)
