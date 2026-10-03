"""B02 supplement: k=0.2 seeded alone (1e-12) to remove the nonlinear (k0.6-k0.4) beat; grid + SPECTRAX N=32."""
import numpy as np, json
from slv import VP, maxwellian
from fit import fit_growth
from disp import two_stream
import sx
from common import OUT
out = {}
for vt in (0.1, 0.3):
    s = VP(2*np.pi/0.2, 16, 3.0, 1024); f0 = 0.5*maxwellian(s.v, vt, 1)+0.5*maxwellian(s.v, vt, -1)
    f = (1+1e-12*np.cos(0.2*s.x))[:, None]*f0[None, :]
    f, t, E, d = s.run(f, 0.025, 80.0, modes=(1,), save_every=2)
    r = sx.run([(0.5, vt, 1.0), (0.5, vt, -1.0)], 2*np.pi/0.2, 4, 32, 80.0, 1601, {1: 0.5e-12}, beta=0.1)
    w = two_stream(0.2, 1.0, vt)[0]
    out[f"vt{vt}"] = dict(root=w.imag, grid=[fit_growth(t, np.abs(E[:, 0]), wd) for wd in [(40, 70), (50, 80)]],
                          spectrax_N32=[fit_growth(r['t'], np.abs(r['Ek'][:, 1]), wd) for wd in [(40, 70), (50, 80)]])
    print(vt, out[f"vt{vt}"])
json.dump(out, open(f"{OUT}/B02/k0.2_single_mode.json", "w"), indent=1)
