"""Write docs/results.md and the README result paragraph from the generated B00 record."""

import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rec = json.loads((root / "docs/_static/b00/run.json").read_text())
inst = json.loads((root / "docs/_static/b02_b03/run.json").read_text())
refs = json.loads((root / "studies/refs_rerun/run.json").read_text())
c05 = json.loads((root / "studies/c05/run.json").read_text())
b16 = json.loads((root / "studies/b01_b06/run.json").read_text())
hhs = json.loads((root / "studies/hhs/run.json").read_text())


def nonlinear_and_hhs():
    B1, B6 = b16["B01"], b16["B06"]
    L = ["## B01/B06: nonlinear Landau and echo, ordinary and dark (`python studies/b01_b06.py`)", "",
         "k = 0.3, eps = 0.05 (B01) and k1 = 1, k2 = 1.5 kick at tau = 10, echo at k3 = 0.5 (B06), electrostatic units, "
         "beta = 0.1. Dark: eta = 0.3, Omega_D = omega_pe. Ordinary runs are compared with the imported grid solver; "
         "there is no independent dark reference. nu is the parent numerical hypercollision coefficient.", "",
         f"B01 grid: first envelope minimum t = {B1['grid']['env_min'][0]:.2f}, maximum t = {B1['grid']['env_max'][0]:.2f} "
         "(not reached in t <= 100 here).", "",
         "| B01 run | envelope minima (t <= 100) | first t with abs(log env/env_grid) > 0.1 | dark-ordinary RMS log-env diff (t <= 60) | run time (s) |",
         "|---|---|---|---|---|"]
    for Nn, nu in ((512, 0.0), (1024, 0.0), (512, 1.0)):
        for mdl in ("ordinary", "dark"):
            r = B1[f"{mdl}_N{Nn}_nu{nu}"]
            dev = r.get("first_t_logenv_dev_gt_0p1", "-")
            dev = "none" if dev is None else dev
            diff = B1[f"dark_vs_ordinary_N{Nn}_nu{nu}_rms_log_env_diff_t_le_60"] if mdl == "dark" else "-"
            L.append(f"| {mdl} N={Nn} nu={nu} | {[round(x, 2) for x in r['env_min']]} | {dev} | {diff if isinstance(diff, str) else f'{diff:.3f}'} | {r['run_time']:.1f} |")
    L += ["", "With nu = 0, N = 1024 reproduces the grid's first envelope minimum (31.45) and stays within 10% of the grid "
          "envelope through t = 100; N = 512 departs at t = 85 and its envelope minimum is not detected. The dark run "
          "reaches its first minimum earlier (30.5). The hypercollision closure nu = 1 removes the trapping minimum "
          "while staying within 10% of the grid envelope to t = 100: a closure can match an envelope and still erase "
          "the physical trapping signature.", "",
          f"B06 grid echo: t = {B6['grid']['t_echo']:.2f}, abs(E_k3) = {B6['grid']['amp']:.5e} (ballistic estimate {B6['t_echo_ballistic']:.0f}).", "",
          "| B06 run | echo t | echo amplitude | vs grid | max dev of abs(E_k3) / grid echo | k1 after t = 20 | segment ledger defect |",
          "|---|---|---|---|---|---|---|"]
    for Nn, nu in ((256, 0.0), (512, 0.0), (512, 1.0)):
        for mdl in ("ordinary", "dark"):
            r = B6[f"{mdl}_N{Nn}_nu{nu}"]
            vs = f"{100 * r['amp_rel_to_grid']:+.3f}%" if "amp_rel_to_grid" in r else "-"
            dv = f"{r['max_abs_dev_k3_over_grid_echo']:.1e}" if mdl == "ordinary" else "-"
            k1 = f"{r['max_k1_after_t20']:.1e}" if mdl == "ordinary" else "-"
            L.append(f"| {mdl} N={Nn} nu={nu} | {r['t_echo']:.2f} | {r['amp']:.5e} | {vs} | {dv} | {k1} | {r['ledger_defect_after_kick']:.0e} |")
    o, d = B6["ordinary_N512_nu0.0"], B6["dark_N512_nu0.0"]
    L += ["", f"Mixing lowers the echo amplitude by {100 * (1 - d['amp'] / o['amp']):.1f}% at N = 512, nu = 0. The kick is an "
          "external velocity shift applied at a restart (exact in the truncated basis); its kinetic energy "
          f"({o['kinetic_energy_from_kick']:.2e}) is an external term outside the ledger. N = 256 matches the echo peak "
          "but its k1 Hermite recurrence (about 27 in the imported t_c table) contaminates the curve (8% deviation); "
          "nu = 1 suppresses the recurrence and also lowers the physical echo by about 3%.", "",
          "## HHS-v1-inspired nonrelativistic ladder (`python studies/hhs_ladder.py`)", "",
          "Not a reproduction of HHS: nonrelativistic 1D3V Hermite, deterministic declared seeds, no quiet-start PIC "
          "noise. Inputs from arXiv:2510.13956v1 App. B: v_te = sqrt(1e-3)c, m_i/m_e = 1836, T_i = T_e, L = 40 c/omega_pe, "
          "uniform drive E0 cos(omega t), E0 = v_q omega_pe with v_q/v_te = 0.03 (strong) or 1e-3 (weak). In parent units "
          "the force is q_s Omega_cs[s] E, so with Omega_cs[0] = 1 the code field equals the electron acceleration "
          "amplitude. omega = 1 (electron omega_pe) and 1.000272 (with ions) are both run; v1 does not state which.", "",
          "| H00 homogeneous, fixed ions, t <= 1000 | max abs(Ebar - exact) / max abs(Ebar) | W_ext rel. error | max quiver / v_te |",
          "|---|---|---|---|"]
    for k, v in hhs["H00"].items():
        L.append(f"| {k} | {v['max_abs_Ebar_err_over_max']:.1e} | {v['W_ext_rel_err']:.1e} | {v['max_quiver_over_vte']:.2f} |")
    L += ["", "The homogeneous mean field is exact in the Hermite system (only orders 0-1 enter), so H00 is meaningful even "
          "at quiver speeds of 15 v_te; finite-k strong-drive runs were limited to t = 100.", "",
          "| H01/H02 finite-k (Nx = 8, Nn = 32) | T | identity residual / (omega_L^2 E0) | Ebar error | max abs(Q_i) / E0 | Q_i phase vs pump (rad) | run + diag time (s) |",
          "|---|---|---|---|---|---|---|"]
    for grp in ("H01", "H02"):
        for k, v in hhs[grp].items():
            if "T" not in v:
                continue
            L.append(f"| {grp} {k} | {v['T']:g} | {v['max_identity_residual_over_wL2E0']:.1e} | "
                     f"{v.get('max_abs_Ebar_err_over_max', float('nan')):.1e} | {v.get('Qi_over_n0E0_max', 0):.3f} | "
                     f"{v.get('Qi_phase_vs_pump_rad', float('nan')):.2f} | {v['run_time'] + v['diagnostic_time']:.1f} |")
    L += ["", "Seeds: 1e-3 density at k1 = 2 pi/40 (both species when ions move) and 1e-4 electron-only at k2. The mean-pump "
          "identity, evaluated from the code's own RHS, holds to about 1e-13 with mobile ions; the gate (T = 100 residual "
          "< 1e-6) passed, so T = 1000 was run. The ion-density/field correlation reaches about 6% of the drive term by "
          "t = 1000 and is roughly in antiphase with the pump, but a single-frequency fit leaves 34% of Q_i unexplained, "
          "so it is not a pure pump-frequency response. This is a seed-dependent pilot, not a detuning measurement.", "",
          "| H07 homogeneous, eta E_D(0) = E0 (weak) | max Ebar error vs 4x4 exponential | max fraction of U_D(0) transferred | max abs(Ebar) reservoir / prescribed |",
          "|---|---|---|---|"]
    for k, v in hhs["H07"].items():
        L.append(f"| {k} | {v['Ebar_vs_matrix_exponential_max_err_over_max']:.1e} | {v['max_fraction_of_reservoir_transferred']:.3f} | "
                 f"{v['max_abs_Ebar_reservoir'] / v['max_abs_Ebar_prescribed']:.3f} |")
    L += ["", "At matched initial effective force a finite reservoir follows the prescribed drive only while eta t is small: "
          "with eta = 0.03 it transfers essentially all of U_D(0) and the plasma field peaks at 7% of the prescribed "
          "secular value over t <= 1000; with eta = 1e-3 the beat period (about 2 pi/eta = 6300) exceeds the run."]
    return "\n".join(L)


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

{nonlinear_and_hhs()}

## Test suite (local, CPU, float64)

A00 moments and Lorentz operator by independent quadrature (agreement 1e-12 or better), A01 zero-mixing RHS equal
to the parent to 1e-14 and a trajectory match, A02 vacuum Proca dispersion/polarization/constraints, A03 coupled
homogeneous oscillator against a matrix exponential (1e-11) with both roots, A04 finite-k cold transverse and
longitudinal branches from full determinants (relative 2e-5 / 1e-4, the size of the thermal correction at
v_t = 0.005c), A05 static Yukawa, A06 eta-sign symmetry, A07 exact mean pump (1e-9), A08 eighth-order time
convergence of fixed-step Dopri8 and Hermite-order convergence of exact free streaming, A09 one to three populations
on odd/even grids with ledger closure, A10 failure policy (nonfinite state, step budget, short stiff transient
without a step floor), C00 random states satisfying both Gauss laws (exact energy-work theorems for particles,
Maxwell and Proca, continuity, both constraints and B_D = curl A_D preserved by the RHS, zero magnetic work), C07
chunked restart equal to a single run with a continuous ledger, rfft Parseval weights for even and odd grids. Deliberate sign/source/mass-potential/weight
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
    o, d = b16["B06"]["ordinary_N512_nu0.0"], b16["B06"]["dark_N512_nu0.0"]
    L.append(f"| B06 echo amplitude (N=512), grid {b16['B06']['grid']['amp']:.5e} | {o['amp']:.5e} | {d['amp']:.5e} (no reference) |")
    h = hhs["H00"]["weak_omega_e"]
    L.append(f"| H00 resonant mean field, t <= 1000 (HHS-v1-inspired) | error {h['max_abs_Ebar_err_over_max']:.0e}, W_ext {h['W_ext_rel_err']:.0e} | - |")
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
