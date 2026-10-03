"""Write docs/results.md and the README result paragraph from the generated B00 record."""

import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rec = json.loads((root / "docs/_static/b00/run.json").read_text())
inst = json.loads((root / "docs/_static/b02_b03/run.json").read_text())
refs = json.loads((root / "studies/refs_rerun/run.json").read_text())
c05 = json.loads((root / "studies/c05/run.json").read_text())


def _times(key):
    return [r[key] for r in runs] + [r[key] for r in inst["results"]]


def tmin(key):
    return min(_times(key))


def tmax(key):
    return max(_times(key))


def grid_note(refs):
    out = []
    for key, v in refs["B00"].items():
        for lab, d, u, dr in (("frequency", v["dw_grid"], v["grid_unc"][0], v["dw_root"]),
                              ("damping", v["dg_grid"], v["grid_unc"][1], v["dg_root"])):
            if abs(d) > u:
                out.append(f"B00 {key} {lab}: Hermite - grid = {d:.1e} exceeds the grid uncertainty {u:.1e}; "
                           f"Hermite - root = {dr:.1e}")
    g = max(abs(v["dg_grid"]) for v in refs["B02"].values())
    head = "All other B00 Hermite-grid differences are inside the grid's stated uncertainty. " if out else \
        "All B00 Hermite-grid differences are inside the grid's stated uncertainty. "
    return head + "; ".join(out) + (". " if out else "") + f"B02 growth differences are at most {g:.1e}."


def extra_sections():
    L = ["## B02/B03: two-stream and bump-on-tail growth, ordinary and dark", "",
         f"Input: `{inst['command']}`; repository `{inst['repository_commit']}`. Electrostatic inputs mapped with "
         f"v_t/c = {inst['settings']['beta']}; eta = {inst['settings']['eta']}, Omega_D = {inst['settings']['Omega_D']} "
         "omega_pe; seed 1e-13; growth = slope of log|E_k| over a 25-unit window ending where |E_k| reaches 1e-5 "
         "(electrostatic units). Reference: D_L roots with drifting-Maxwellian chi (wofz).", "",
         "| case | model | Nn | fitted growth | root growth | difference | window spread |", "|---|---|---|---|---|---|---|"]
    for r in inst["results"]:
        L.append(f"| {r['case']} | {r['model']} | {r['Nn']} | {r['growth_fit']:.7f} | {r['root'][1]:.7f} | "
                 f"{r['growth_error']:.1e} | {r['growth_window_spread']:.1e} |")
    L += ["", "The bump-on-tail difference falls from about 7e-7 to 2e-8 when Nn doubles (64 to 128); two-stream "
          "results are unchanged between Nn = 32 and 64. Mixing raises each growth rate here; this is one "
          "deliberately large coupling and one wavenumber per case, not a growth-band map.", "",
          "## Non-Hermite continuum reference (ordinary plasma)", "",
          "Independent references (NumPy semi-Lagrangian Vlasov-Poisson solver, wofz roots, Gkeyll p=2 DG records) "
          "are imported with provenance in [studies/refs](../studies/refs/README.md). The Hermite side was rerun "
          f"here on the pinned parent `{refs['parent_commit'][:7]}` through `darkspectrax` (mode ordinary), with "
          "the reference fit rules (`python studies/compare_refs.py`).", "",
          "| case | Hermite fit | grid SL (Richardson) | wofz root | Hermite - grid |", "|---|---|---|---|---|"]
    for key, v in refs["B00"].items():
        L.append(f"| B00 {key} | {v['w']:.7f}{v['g']:+.7f}i | {v['grid'][0]:.7f}{v['grid'][1]:+.7f}i "
                 f"(+-{v['grid_unc'][0]:.0e}, {v['grid_unc'][1]:.0e}) | {v['root'][0]:.7f}{v['root'][1]:+.7f}i | "
                 f"{v['dw_grid']:.1e}, {v['dg_grid']:.1e} |")
    for key, v in refs["B02"].items():
        L.append(f"| B02 {key} growth | {v['g']:.7f} | {v['grid']:.7f} | {v['warm_root']:.7f} | {v['dg_grid']:.1e} |")
    dmax = max(max(abs(v['w'] - v['spectrax_6781d80'][0]), abs(v['g'] - v['spectrax_6781d80'][1]))
               for v in refs["B00"].values())
    L += ["", f"The rerun on `{refs['parent_commit'][:7]}` matches the reference SPECTRAX fits made on the unmerged "
          f"integration commit 6781d80 to {dmax:.0e} (B00). " + grid_note(refs), "",
          "## C05: matched Landau case against Dark-JAX-in-Cell PIC records", "",
          "Same physical inputs as Dark-JAX-in-Cell `physical_kinetic_*` (sigma = 0.05c, k lambda_De = 0.5, "
          "eta = 0.3, Omega_D = kc, density seed 0.01, Yukawa-consistent start); the PIC side is its published "
          f"record at commit `{c05['dark_jax_in_cell_commit'][:7]}` (not rerun here). Same fit rule: neighbouring maxima "
          "of |E_k| on 2 <= t <= 12. Shared: field conventions, D_L, work definition. Not shared: kinetic "
          "discretization, field solver, integrator, loading.", "",
          f"Independent D_L roots agree with the Dark-JAX-in-Cell roots to {c05['root_cross_check']['dark_abs_diff']:.0e}.", "",
          "| run | ordinary (maxima rule) | dark (maxima rule) | RMS log-amplitude diff vs Hermite (ord / dark) |",
          "|---|---|---|---|"]
    for n in ("Nn128", "Nn256"):
        o, dk = c05["spectrax"][f"ordinary_{n}"]["maxima_fit"], c05["spectrax"][f"self_consistent_{n}"]["maxima_fit"]
        L.append(f"| Hermite {n} | {o[0]:.5f}{o[1]:+.5f}i | {dk[0]:.5f}{dk[1]:+.5f}i | - |")
    for c, v in c05["pic"].items():
        L.append(f"| PIC {c}, {v['particles']} particles | {v['ordinary'][0]:.5f}{v['ordinary'][1]:+.5f}i | "
                 f"{v['dark'][0]:.5f}{v['dark'][1]:+.5f}i (stderr {v['damping_slope_stderr']:.4f}) | "
                 f"{v['rms_log_amplitude_diff_parent_Ek_2_12']:.3f} / {v['rms_log_amplitude_diff_ordinary_Ek_2_12']:.3f} |")
    r = c05["roots"]
    L += [f"| roots | {r['ordinary'][0]:.5f}{r['ordinary'][1]:+.5f}i | {r['dark'][0]:.5f}{r['dark'][1]:+.5f}i | |", "",
          "The PIC values move toward the Hermite values under joint cell/particle refinement and the amplitude "
          "difference roughly halves from 32 to 128 cells, but the 128-cell PIC fits still differ from the Hermite "
          "fits by about 1-2% in damping. The maxima rule with a 1% seed is itself biased by about 1% relative to the "
          "linear root (Hermite rows); compare like with like. Not a claim of PIC convergence."]
    return "\n".join(L)

runs = [r for r in rec["results"] if "roots" not in r]
roots = {r["k_lambda_De"]: r["roots"] for r in rec["results"] if "roots" in r}
s = rec["settings"]


def c(z):
    return f"{z[0]:.6f}{z[1]:+.6f}i"


rows = []
for kl in sorted(roots):
    for name in ("ordinary", "dark"):
        sel = [r for r in runs if r["k_lambda_De"] == kl and r["model"] == name]
        best = max(sel, key=lambda r: (r["Nn"], r["Nx"]))
        spread = max(abs(complex(*r["omega_fit"]) - complex(*sel[0]["omega_fit"])) for r in sel)
        rows.append(f"| {kl} | {name} | {c(best['omega_fit'])} | {c(roots[kl][name])} | "
                    f"{best['rel_error']:.1e} | {abs(best['omega_fit'][1] - roots[kl][name][1]):.1e} | {spread:.1e} | {best['fit_window'][0]:g}-{best['fit_window'][1]:g} | "
                    f"{max(r['max_ledger_defect'] for r in sel):.1e} |")
table = ("| k lambda_De | model | fitted omega (Nn=128, Nx=8) | independent root | rel. error | abs. gamma error | "
         "spread over Nn, Nx | fit window | max ledger defect |\n|---|---|---|---|---|---|---|---|---|\n"
         + "\n".join(rows))
scan_lines = []
for r in runs:
    if r["Nn"] == 128 and r["Nx"] == 8:
        sc = ", ".join(f"t>={float(k):g}: {v[2]:.1e}" for k, v in r["window_start_scan"].items())
        scan_lines.append(f"- k lambda_De = {r['k_lambda_De']}, {r['model']}: {sc}")

shift = {kl: (roots[kl]["dark"][0] / roots[kl]["ordinary"][0] - 1,
              roots[kl]["dark"][1] / roots[kl]["ordinary"][1] - 1) for kl in roots}
worst = max(r["rel_error"] for r in runs)
para = (
    f"Electrons with $v_{{te}}=0.1c$ on fixed ions damp a $10^{{-4}}$ density seed at "
    f"$k\\lambda_{{De}}=0.2$-$0.7$, with and without a mixed field ($\\eta={s['eta']}$, "
    f"$\\Omega_D={s['Omega_D']}\\,\\omega_{{pe}}$, Yukawa-consistent start). An independent SciPy root of "
    "$D_L=(Q-\\Omega_D^2)(1+\\chi)+\\eta^2Q\\chi$ predicts that mixing raises the frequency by "
    f"{100 * shift[0.3][0]:.1f}% / {100 * shift[0.5][0]:.1f}% and lowers the damping rate by "
    f"{-100 * shift[0.3][1]:.1f}% / {-100 * shift[0.5][1]:.1f}% at $k\\lambda_{{De}}=0.3/0.5$. Fitted complex frequencies agree with "
    f"the roots to a relative {worst:.1e} or better for all {len(runs)} runs (Hermite orders 64/128, grids 5/8), "
    "and the work ledger closes to roundoff. At $k\\lambda_{De}=0.2$ the damping rate itself (about $5\\times10^{-5}$) "
    "is not resolved by the Hermite-limited window. This is a linear, deliberately large-coupling verification; it does not "
    "address nonlinear or late-time behavior.\n\n"
    "<img src=\"docs/_static/b00/figure.png\" width=\"860\" alt=\"Ordinary and dark Landau damping against kinetic roots\">\n\n"
    "[Script](examples/plasma.py) · [record](docs/_static/b00/run.json) · [table](docs/results.md)")

(root / "docs/results.md").write_text(f"""# Results

Generated by `python docs/make_results.py` from records; do not edit numbers by hand.

## B00: linear Landau damping, ordinary and dark

- Question: does the companion reproduce the independent kinetic Maxwell-Proca root, and how does mixing
  change the Landau mode?
- Input: `{rec['command']}`; repository `{rec['repository_commit']}`, parent `{rec['parent_commit']}`;
  versions {json.dumps(rec['versions'])}.
- Method: electrons, $v_{{te}}={s['vte']}c$, fixed neutralizing background, density seed {s['seed']},
  ordinary Gauss field plus static Yukawa dark near field, Dopri8 rtol {s['rtol']}, atol {s['atol']},
  no closure (nu = 0). Fit $E_{{x,k}}(t)=\\sum_j c_je^{{-i\\omega_jt}}$ with the $\\pm$ Landau pair
  (plus the $\\pm$ massive longitudinal branch for dark runs) by nonlinear least squares.
  Reference: SciPy `wofz` plasma dispersion function, $Q=\\omega^2-c^2k^2$, $c=1$.

{table}

Fit-window sensitivity (relative error versus the window start; early windows include the ballistic transient):

{chr(10).join(scan_lines)}

- Convergence: results are identical to about 1e-9 across Nn = 64, 128 and Nx = 5, 8 (single linear mode
  inside the fit window, Hermite front below order 64). This shows the window is Hermite-resolved; it is not a
  test of nonlinear spatial convergence.
- At k lambda_De = 0.2 the root damping rate (~5e-5) changes the amplitude by about 0.1% over the 26-unit window
  allowed before the Hermite front reaches order 64; the fitted rate is off by about 7e-5 absolute (factor ~2).
  The frequency is accurate; the damping there is unresolved, not validated.
- Limitation: large coupling ($\\eta=0.3$) chosen for a visible shift; linear regime only; the massive
  longitudinal branch ($\\omega\\approx\\sqrt{{k^2c^2+\\Omega_D^2}}$) is weakly excited and only included in the fit.

{extra_sections()}

## Test suite (local, CPU, float64)

A00 moments and Lorentz operator by independent quadrature (agreement 1e-12 or better), A01 zero-mixing RHS equal
to the parent to 1e-14 and a trajectory match, A02 vacuum Proca dispersion/polarization/constraints, A03 coupled
homogeneous oscillator against a matrix exponential (1e-11) with both roots, A04 finite-k cold transverse and
longitudinal branches from full determinants (relative 2e-5 / 1e-4, the size of the thermal correction at
v_t = 0.005c), A05 static Yukawa, A06 eta-sign symmetry, A07 exact mean pump (1e-9), A08 eighth-order time
convergence of fixed-step Dopri8 and Hermite-order convergence of exact free streaming, A09 one to three populations
on odd/even grids with ledger closure, A10 failure policy (nonfinite state, step budget, short stiff transient
without a step floor), rfft Parseval weights for even and odd grids. Deliberate sign/source/mass-potential/weight
mutations of the companion were each caught by at least one test. A08 uses exact solutions; a manufactured-source
hook is not implemented.

Timing (CPU, from the records): compile {tmin('compile_time'):.2f}-{tmax('compile_time'):.2f} s per configuration,
integration {tmin('run_time'):.2f}-{tmax('run_time'):.2f} s for the B00/B02/B03 runs above.
""")

readme = (root / "README.md").read_text()
readme = re.sub(r"(## Result: Landau damping with a dark field\n\n).*?(\n\n## Ordinary and dark plasma tests)",
                lambda m: m.group(1) + para + m.group(2), readme, flags=re.S)
(root / "README.md").write_text(readme)


def readme_table():
    refmax = max([abs(v["dw_grid"]) for v in refs["B00"].values()] + [abs(v["dg_grid"]) for v in refs["B00"].values()]
                 + [abs(v["dg_grid"]) for v in refs["B02"].values()])
    L = ["| Case | Ordinary: measured / reference | Dark ($\\eta=0.3$, $\\Omega_D=\\omega_{pe}$): measured / root |", "|---|---|---|"]
    for kl in sorted(roots):
        o = [r for r in runs if r["k_lambda_De"] == kl and r["model"] == "ordinary" and r["Nn"] == 128 and r["Nx"] == 8][0]
        d = [r for r in runs if r["k_lambda_De"] == kl and r["model"] == "dark" and r["Nn"] == 128 and r["Nx"] == 8][0]
        L.append(f"| Landau $k\\lambda_{{De}}={kl}$, $\\omega$ | {c(o['omega_fit'])} / {c(o['omega_root'])} | "
                 f"{c(d['omega_fit'])} / {c(d['omega_root'])} |")
    for name in dict.fromkeys(r["case"] for r in inst["results"]):
        rr = [r for r in inst["results"] if r["case"] == name]
        top = max(r["Nn"] for r in rr)
        o = [r for r in rr if r["model"] == "ordinary" and r["Nn"] == top][0]
        d = [r for r in rr if r["model"] == "dark" and r["Nn"] == top][0]
        L.append(f"| {name.split('_', 1)[1].replace('_', ' ')}, growth | {o['growth_fit']:.6f} / {o['root'][1]:.6f} | "
                 f"{d['growth_fit']:.6f} / {d['root'][1]:.6f} |")
    p = c05["pic"]["cells128"]
    h = c05["spectrax"]
    L.append(f"| C05 Landau vs Dark-JAX-in-Cell PIC (128 cells), same fit rule | Hermite {c(h['ordinary_Nn256']['maxima_fit'])} / "
             f"PIC {c(p['ordinary'])} | Hermite {c(h['self_consistent_Nn256']['maxima_fit'])} / PIC {c(p['dark'])} |")
    return "\n".join(L) + ("\n\nLandau and growth references are independent kinetic roots; the ordinary Hermite runs also "
                           f"agree with an independent semi-Lagrangian solver to within {refmax:.1e} (B00, B02). "
                           "Details, windows and limitations: [docs/results.md](docs/results.md).\n\n"
                           "<img src=\"docs/_static/b02_b03/figure.png\" width=\"860\" "
                           "alt=\"Two-stream and bump-on-tail growth, ordinary and dark, against kinetic roots\">\n\n"
                           "[Script](examples/instabilities.py) · [record](docs/_static/b02_b03/run.json)")


readme = (root / "README.md").read_text()
readme = re.sub(r"(## Ordinary and dark plasma tests\n\n).*?(\n\n## Install)",
                lambda m: m.group(1) + readme_table() + m.group(2), readme, flags=re.S)
(root / "README.md").write_text(readme)
