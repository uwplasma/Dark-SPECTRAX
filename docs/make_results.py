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
dg = json.loads((root / "studies/dark_grid/run.json").read_text())
pic = sorted((json.loads(p.read_text()) for p in (root / "studies/c05_pic").glob("*.json")),
             key=lambda r: (r["cells"], r["particles"]))
hs = json.loads((root / "studies/hhs_scan/run.json").read_text())


def third_round():
    L = ["## Independent dark reference: grid Vlasov-Ampere-Proca (`python studies/dark_grid_reference.py`)", "",
         "`studies/refs/code/slv_proca.py` extends the imported semi-Lagrangian solver with Ampere's law and the "
         "longitudinal Proca fields (E_D, A_D, phi_D at every k including k = 0, exact per-k propagator, mid-step "
         "current from the exact velocity shift). It shares no code with SPECTRAX or Dark-SPECTRAX; it shares the "
         "field convention and the D_L equation. c = 10 (beta = 0.1), eta = 0.3, Omega_D = omega_pe.", "",
         "| Verification alone | dt = 0.05 | dt = 0.025 | dt = 0.0125 | Richardson - root |", "|---|---|---|---|---|"]
    for k, v in dg["linear_verification"].items():
        f = v["fits"]
        L.append(f"| Landau {k}: omega | " + " | ".join(f"{f[d]['w']:.6f}{f[d]['g']:+.6f}i" for d in ("0.05", "0.025", "0.0125"))
                 + f" | {v['richardson_minus_root'][0]:.1e}, {v['richardson_minus_root'][1]:.1e} |")
        L.append(f"| Landau {k}: dark ledger / U_E(0) | " + " | ".join(f"{f[d]['dark_ledger_defect_over_U_E0']:.1e}"
                                                                     for d in ("0.05", "0.025", "0.0125")) + " | |")
    b1, b6 = dg["B01_dark"], dg["B06_dark"]
    L += ["", "Errors fall as dt^2 (Strang); Gauss residuals of both laws stay below 5e-9.", "",
          "| Dark comparison | grid (dt = 0.0125) | Hermite | agreement |", "|---|---|---|---|",
          f"| B01 first envelope minimum (prominence 0.2) | {b1['grid_dt0.0125']['env_min_prominence_0p2']} | "
          f"N=1024: {b1['hermite_dark_N1024_nu0.0']['env_min_prominence_0p2']} | max log-envelope difference to t = 80: "
          f"{b1['hermite_dark_N1024_nu0.0']['max_logenv_dev_t_le_80']:.3f} (N=1024), "
          f"{b1['hermite_dark_N512_nu1.0']['max_logenv_dev_t_le_80']:.3f} (N=512, nu=1) |",
          f"| B06 dark echo | t = {b6['grid_dt0.0125']['t_echo']:.2f}, {b6['grid_dt0.0125']['amp']:.6e} | "
          f"N=512: {b6['hermite_dark_N512_nu0.0']['amp']:.6e} | {100 * b6['hermite_dark_N512_nu0.0']['amp_rel_to_grid']:+.4f}%, "
          f"curve {b6['hermite_dark_N512_nu0.0']['max_abs_dev_over_grid_echo']:.1e}; N=512 nu=1: "
          f"{100 * b6['hermite_dark_N512_nu1.0']['amp_rel_to_grid']:+.2f}% |", "",
          "At prominence 0.3 (the B01 rule) the grid dark minimum (prominence 0.25) is not flagged while the Hermite one "
          "(0.30) is; at 0.2 both give 30.5-30.55. The dark echo is reproduced by an independent solver to 2e-5.", "",
          "## C05 rerun: Dark-JAX-in-Cell PIC on CPU (`python studies/c05_pic_rerun.py CELLS PARTICLES`)", "",
          "Dark-JAX-in-Cell commit d547579, run here on CPU in its own environment with the construction of its "
          "`dark_kinetic.py` physical preset (quiet start, current-neutral, displacement seed 0.01/k). Same fit rule as "
          "the Hermite rows (maxima on 2 <= t <= 12). The 32/40000, 64/80000 and 128/160000 rows reproduce the published "
          "GPU records exactly.", "",
          "| cells | particles | per cell | ordinary | dark | dark slope stderr | energy drift | wall (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in pic:
        L.append(f"| {r['cells']} | {r['particles']} | {r['particles_per_cell']:.0f} | {r['ordinary'][0]:.5f}{r['ordinary'][1]:+.5f}i | "
                 f"{r['dark'][0]:.5f}{r['dark'][1]:+.5f}i | {r['dark_slope_stderr']:.4f} | {r['max_closed_energy_drift']:.1e} | "
                 f"{r['wall_parent_incl_compile'] + r['wall_dark_incl_compile']:.0f} |")
    h = c05["spectrax"]
    L += [f"| Hermite (Nn = 256) | | | {h['ordinary_Nn256']['maxima_fit'][0]:.5f}{h['ordinary_Nn256']['maxima_fit'][1]:+.5f}i | "
          f"{h['self_consistent_Nn256']['maxima_fit'][0]:.5f}{h['self_consistent_Nn256']['maxima_fit'][1]:+.5f}i | | | |", "",
          "From 64 cells / 80000 particles, doubling the particles at fixed mesh and doubling the mesh at fixed particles "
          "per cell move the dark damping fit by similar amounts (about 5e-3, comparable to the slope standard error), so "
          "neither refinement direction is converged yet. The finest PIC run (128 cells, 320000 particles) is within "
          "0.1% in frequency and 1% in damping of the Hermite values for both ordinary and dark runs.", "",
          "## H05/H06 pilots: drive-amplitude scan and swept drive (`python studies/hhs_scan.py`)", "",
          f"Record from parent `{hs['parent_commit'][:7]}` (not rerun at the current pin: the scan takes over 30 minutes). "
          "HHS-v1-inspired, nonrelativistic, mobile ions (1836), Nx = 8, declared seeds, omega = 1.000272 (total). Each run "
          "had a 250000-step budget. Several runs exhausted it; before that, the Hermite state became inadmissible "
          "(a species' moment kinetic energy went negative) while the work ledger still closed. Agreement time = first "
          "time the electron kinetic-energy change differs by > 10% between Nn = 32 and 64 (nu = 0), or between nu = 0 "
          "and 1 (Nn = 64). Values are reported only up to that time.", "",
          "| v_q/v_te | resolved until | W_ext / (n T_e) | dK_e / (n T_e) | dK_i / (n T_e) | runs that failed |",
          "|---|---|---|---|---|---|"]
    for ratio in ("0.001", "0.003", "0.01", "0.03", "0.1"):
        a = hs["H05"][f"vq{ratio}_agreement"]
        v = a["Nn64_nu0_at_resolved"]
        fails = [k.split("_", 1)[1] for k, r in hs["H05"].items()
                 if k.startswith(f"vq{ratio}_Nn") and isinstance(r, dict) and r.get("status") == "failure"]
        fails = ", ".join(f"{f} (t = {hs['H05'][f'vq{ratio}_' + f]['t_reached']:.0f})" for f in fails) or "none"
        L.append(f"| {ratio} | {a['resolved_until']:.0f} | {v['W_ext'] / 1e-3:.3f} | {v['dK_electron'] / 1e-3:.3f} | "
                 f"{v['dK_ion'] / 1e-3:.1e} | {fails} |")
    L += ["", "Within the resolved windows W_ext follows the linear resonant estimate E0^2 t^2/8 and the ions receive "
          "less than 0.1% of the kinetic-energy change: no saturation is resolved. Beyond v_q/v_te = 1e-3 the fixed "
          "Hermite basis (width matched to the initial Maxwellian) cannot follow the growing quiver motion past these "
          "times; this needs a moving/rescaled basis or a different closure, not a longer run.", "",
          "| H06 swept drive, v_q/v_te = 0.1 | instantaneous frequency | homogeneous check vs solve_ivp | Nn32/nu0 vs Nn64/nu1 agree until | t reached |",
          "|---|---|---|---|---|"]
    for d in ("up", "down"):
        r = hs["H06"][f"{d}_Nn64_nu1"]
        L.append(f"| {d}: theta = ({r['omega0']} + ({r['a']:g}) t) t | {r['instantaneous_frequency']} | "
                 f"{r['homogeneous_max_err_over_max']:.1e} | {hs['H06'][f'{d}_agreement']['Nn32nu0_vs_Nn64nu1'][0]:.0f} | "
                 f"{r['t_reached']:.0f} (budget) |")
    L += ["", "The homogeneous swept-drive mean field matches an independent ODE solution to 1e-9 over 0 <= t <= 1e4. "
          "The kinetic swept runs lose resolution (t ~ 2400-2850) before the resonance crossing at t = 5000: no swept-drive "
          "kinetic result is claimed."]
    return "\n".join(L)


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

{third_round()}

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
    pf = pic[-1]
    L.append(f"| C05 Landau, same fit rule: Hermite / PIC rerun ({pf['cells']} cells, {pf['particles']} particles) | "
             f"{c(c05['spectrax']['ordinary_Nn256']['maxima_fit'])} / {c(pf['ordinary'])} | "
             f"{c(c05['spectrax']['self_consistent_Nn256']['maxima_fit'])} / {c(pf['dark'])} |")
    p = c05["pic"]["cells128"]
    h = c05["spectrax"]
    0 and L.append(f"| C05 Landau vs Dark-JAX-in-Cell PIC (128 cells), same fit rule | Hermite {c(h['ordinary_Nn256']['maxima_fit'])} / "
             f"PIC {c(p['ordinary'])} | Hermite {c(h['self_consistent_Nn256']['maxima_fit'])} / PIC {c(p['dark'])} |")
    o, d = b16["B06"]["ordinary_N512_nu0.0"], b16["B06"]["dark_N512_nu0.0"]
    g6 = dg["B06_dark"]["grid_dt0.0125"]["amp"]
    L.append(f"| B06 echo amplitude (N=512) vs grid | {o['amp']:.5e} / {b16['B06']['grid']['amp']:.5e} | "
             f"{d['amp']:.5e} / {g6:.5e} (grid Vlasov-Ampere-Proca) |")
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
