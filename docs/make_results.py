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
gate = json.loads((root / "studies/h05_pump/run.json").read_text())
lc = json.loads((root / "studies/lane_c/summary.json").read_text())
lb = json.loads((root / "studies/lane_b/summary.json").read_text())
fid = {f"{c}_N{n}": json.loads((root / f"studies/lane_b/fidelity_{c}_N{n}.json").read_text())
       for c in ("B01", "B06") for n in (128, 512)}
reb = json.loads((root / "studies/consolidate/b01_rebound.json").read_text())
nfs = {}
for p in sorted((root / "studies/noisefloor").glob("scan_*.jsonl")):
    for line in p.read_text().splitlines():  # last record per (part, rtol, floor, case) wins
        d = json.loads(line)
        nfs[(p.stem, d["floor"], d.get("vq", d.get("case")))] = d
ba = {p.stem[len("before_after_"):]: json.loads(p.read_text()) for p in (root / "studies/consolidate").glob("before_after_*.json")}


def _tr(x, end=1000.0):
    return "-" if x is None else (f">= {end:g}" if x >= end else f"{x:g}")


def lanes():
    """H05 strong-drive lane comparison, the declared stabilizer, the B01 correction and the run_adaptive fixes."""
    sc, fb = lc["scan"], gate["fixed_basis_resolved_until"]
    L = ["## H05 strong resonant drive: lane comparison (pump frame + remap, parent c0910a1)", "",
         "HHS-v1-inspired nonrelativistic pilot: mobile ions (m_i/m_e = 1836), v_te = sqrt(1e-3), L = 40, Nx = 8, gate "
         "seeds at 5e-4, drive at omega = sqrt(1 + 1/1836), rtol 1e-10. t_res = first t > 50 at which electron dK differs "
         "by more than 10% of its running max between refinements (W_ext never limits). Lane C: nu = 0, Nn 64/128/256 "
         "(`studies/lane_c_run.py`). Lane B: order-2 hypercollision, Nn64/Nn128 rows, pairs Nn and nu/2-2nu "
         "(`studies/lane_b_closure.py`). Lane A (width floor + exponential filter) is deferred: branch `lane-a` and "
         "SPECTRAX #61 stay open, its records are not on main and it is not compared here.", "",
         "| v_q/v_te | fixed basis | pump gate (Nn32/64, nu 0/1) | Lane C nu = 0 (certified) | Lane C first negative K or T_x (Nn128) | Lane B nu = 1 | Lane B nu = 2 |",
         "|---|---|---|---|---|---|---|"]
    real = lc["realizations"]
    for vq in ("0.01", "0.03", "0.1"):
        g = gate["H05"].get(f"vq{vq}_agreement", {}).get("resolved_until")
        cert = sc[vq]["certified"]
        tp = real[vq]["t_pos_Nn128"]["0"]
        b = [" / ".join(_tr((lb.get(f"vq{vq}_o2_Nn{n}_nu{nu}", {}).get("t_res") or {}).get("dK")) for n in (64, 128))
             for nu in (1, 2)]
        L.append(f"| {vq} | {_tr(fb.get(vq))} | {_tr(g)} | {_tr(cert['t_cert'])} ({cert['pair']}) | "
                 f"{'none' if tp is None else tp} | {b[0]} | {b[1]} |")
    bf = lc["boundary_fit"]
    hr = bf["heating_at_t_res_range"]
    L += ["", f"Lane C boundary for 0.02 <= v_q/v_te <= 0.1: t_res = {bf['t_res ~ C vq^p']['C']:.0f} "
          f"(v_q/v_te)^{bf['t_res ~ C vq^p']['p']:.2f} (4 points, residuals <= 1%); W_ext/(n T_e) at t_res ranges "
          f"{hr[0]:.0f}-{hr[1]:.0f}, so it is not a heating threshold. Loss of resolution coincides with loss of "
          "positivity. Empirical, for these seeds and Nx = 8 only.", "",
          "Inside the certified windows (Lane C, cycle averages at t_cert; seed spread over three phase realizations "
          "in `studies/lane_c/summary.json`):", "",
          "| v_q/v_te | t_cert | W_ext / exact linear resonant W | dK_e / W_ext | dU_E / W_ext | dK_i / W_ext | electron k=0 random / W_ext |",
          "|---|---|---|---|---|---|---|"]
    for vq in ("0.001", "0.01", "0.02", "0.03", "0.05", "0.1"):
        r = sc[vq]
        pa = r["partition_at_t_cert"]
        L.append(f"| {vq} | {r['certified']['t_cert']:g} | {r['W_ext_over_W_lin']['cycle_avg_at_t_cert']:.4f} | "
                 f"{pa['dK_e/W_ext']:.3f} | {pa['dU_E/W_ext']:.3f} | {pa['dK_i/W_ext']:.1e} | {pa['dThermal_e(k=0)/W_ext']:.4f} |")
    un = lc["unseeded"]
    L += ["", "The unseeded control follows the exact uniform two-fluid oscillator to "
          f"{max(v['max_rel_dev_from_W_lin'] for v in un.values()):.0e} over 20 < t <= 1000 (ds.CONTROLS; rtol-limited, see the "
          "control-run noise floor section); the seeded deficit (about 1%) behaves like a fixed "
          "detuning and is unexplained. The ion correlation force <dn_i dE_x> is about 1e-4 of the mean force. An "
          "independent mobile-ion grid code (`studies/lane_c_grid_mobile.py`) agrees on dK_e up to "
          + ", ".join(f"{v['dK_e']:g} (v_q/v_te = {k.split('_')[1][2:]})" for k, v in lc["grid_mobile"].items()
                      if "Nx16_Nv4096" in k) + ".", "",
          "### Declared stabilizer: order-2 hypercollision, nu = 1-2 (Lane B), for v_q/v_te <= 0.03 only", "",
          "Before blow-up the H05 observables do not depend on nu (spread <= 1.4e-6 of the mean at t = 600). At 0.1 every "
          "nu (0.25-64) and both orders blow up before t = 1000; the closure postpones loss of positivity, it does not "
          "regularize. Fidelity cost on ordinary benchmarks (`studies/lane_b_fidelity.py`, against the grid references):", "",
          "| nu (order 2) | B01 rebound retained N=128 / N=512 (> 100%: recurrence overshoot) | B06 echo amplitude error N=128 / N=512 | B06 echo t N=128 (grid 29.05) | H05 0.03 steps Nn64 / wall (s) |",
          "|---|---|---|---|---|"]
    for nu in (0, 1, 2):
        k = f"nu{nu}_o2"
        b1 = " / ".join(f"{100 * (1 - fid[f'B01_N{n}'][k]['erased_fraction']):.0f}%" for n in (128, 512))
        b6 = " / ".join(f"{100 * fid[f'B06_N{n}'][k]['amp_rel_to_grid']:+.1f}%" if "amp_rel_to_grid" in fid[f"B06_N{n}"][k]
                        else "-" for n in (128, 512))
        h = lb.get(f"vq0.03_o2_Nn64_nu{nu}")
        cost = f"{h['steps']} / {h['wall']:.0f}" if h else "- (nu = 0: step budget at t = 900)"
        L.append(f"| {nu} | {b1} | {b6} | {fid['B06_N128'][k]['t_echo']:.2f} | {cost} |")
    L += ["", "Use it only as a declared numerical closure at Nn >= 64 and v_q/v_te <= 0.03; at N <= 128 it removes most "
          "of the B06 echo, so it is not a collisionless proxy there.", "",
          "### What can and cannot be claimed", "",
          "- Can: v_q/v_te <= 0.01 collisionless (nu = 0) to omega_pe t = 1000, Hermite-converged (Nn 64/128/256), positive, "
          "W_ext on the exact linear resonant law to about 1%, energy split equally between electron kinetic and mean field, "
          "ions at m_e/(2 m_i). For 0.02-0.1 the same holds up to the tabulated t_res.",
          "- Can: v_q/v_te = 0.03 to t = 1000 with the declared order-2 closure, nu = 1-2, Nn 64/128 agreeing in W_ext "
          "and (nu = 2) in dK_e.",
          "- Cannot: any physics beyond t_res (saturation, late heating, partition); any v_q/v_te = 0.1 result at t = 1000 "
          "with any lane; x-convergence (Hermite Nx = 16 fails earlier through spurious high-k growth); portability of "
          "the t_res law to other seeds, Nx, mass ratio or relativistic drive; local temperatures (k = 0 random energy "
          "includes non-uniform flow); a pass of the common gate (0.03 and 0.1 to t = 1000 collisionless).", "",
          "## Correction: B01 trapping rebound under the hypercollision closure "
          "(`python studies/consolidate/b01_rebound.py`)", "",
          "Retracted: the c-refs record (`studies/refs/B01`) stated that nu = 1 at N = 256/512 removes the trapping "
          "rebound. That statement rested on a prominence-0.3 extremum detector, which also finds no extremum in the "
          "nu = 0, N = 512 run. Detector-free re-measurement, envelope ratio R = log(env(65.6)/env(31.45)) at the grid "
          "extremum times and R_window = log(max env on [50, 80] / min env on [20, 45]), relative to the grid "
          f"(R_grid = {reb['grid']['R_fixed']:.3f}):", "",
          "| run | R / R_grid | R_window / R_grid_window |", "|---|---|---|"]
    for key in ("fresh_ordinary_N256_nu0", "fresh_ordinary_N256_nu1", "fresh_ordinary_N512_nu0", "fresh_ordinary_N512_nu1",
                "record_ordinary_N1024_nu0.0", "record_dark_N512_nu0.0", "record_dark_N512_nu1.0"):
        r = reb[key]
        L.append(f"| {key.split('_', 1)[1].replace('_', ' ')} | {r['retained_fixed']:.2f} | {r['retained_window']:.2f} |")
    L += ["", "nu = 1 keeps the rebound (93% at N = 512, 82% at N = 256); at N = 256, nu = 0 recurrence corrupts it instead. "
          "The fresh N = 512 runs reproduce the committed `studies/b01_b06` arrays exactly.", "",
          "## run_adaptive fixes (`python studies/consolidate/before_after.py VQ NN REAL [FLOOR]`)", "",
          "- One compilation per run: `t0` is a traced argument and the compiled solve is reused across segments.",
          "- Positivity stop: a segment with a saved K_s <= 0 or k = 0 T_x <= 0 ends the run with "
          "`failure_reason = 'segment k: positivity lost at t = ...'` instead of spending the step budget.",
          "- Unseeded pump-frame stall: the k = 0, n >= 1 ion coefficients are round-off of C_000,i ~ 1/a_i^3 ~ 1e9, so "
          "plain PID at atol 1e-14 rejects steps on noise. `noise_floor` adds noise_floor*|C_000,s| to atol on species "
          "s (default off, so recorded runs are unchanged).", "",
          "| case | floor | before: status, t, steps, compile s, wall s | after: status, t, steps, compile s, wall s | max diff dK_e / W_ext (rel.) |",
          "|---|---|---|---|---|"]
    for key in sorted(ba):
        r = ba[key]
        f = lambda d: (f"{d['status']}, {d['t_reached']:g}, {d['steps']}, {d['compile_time']:.1f}, "  # noqa: E731
                       f"{d['wall_time']:.0f}")
        L.append(f"| {r['case']} | {r['noise_floor']} | {f(r['before'])} | {f(r['after'])} | "
                 f"{r['max_rel_diff_dK_e']:.1e} / {r['max_rel_diff_W_ext']:.1e} |")
    L += ["", noise_floor_section()]
    return "\n".join(L)


def noise_floor_section():
    fl = ("1e-16", "1e-15", "1e-14", "1e-13", "1e-12", "1e-10")
    g = lambda part, f, key: nfs.get((part, None if f is None else float(f), key))  # noqa: E731
    L = ["## Control-run noise floor: `ds.CONTROLS = {\"noise_floor\": 1e-14}` (`python studies/noisefloor/scan.py`)", "",
         "Control runs (H00, H01, unseeded pump-frame controls) pass `**ds.CONTROLS` explicitly; `run` and "
         "`run_adaptive` keep `noise_floor=None`, so seeded records are unchanged. Scan, rtol 1e-10 / atol 1e-14 unless noted:", "",
         "| noise_floor | (a) unseeded pump 0.01: steps, max abs(W/W_lin - 1) | (a) unseeded pump 0.1: steps, dev | "
         "(b) H00 weak omega_e: steps, Ebar err | (b) H01 weak: steps, Ebar err | (b) H01 strong: steps, Ebar err | "
         "(c) seeded 0.03 Nn64 r0: t reached, steps, diff dK_e / W_ext |", "|---|---|---|---|---|---|---|"]
    for f in (None,) + fl:
        a = [g("scan_a", f, v) for v in (0.01, 0.1)]
        aa = [f"{x['steps']}, {x['max_rel_dev_W_lin']:.1e}" if x else "fails (committed plain-PID record)" for x in a]
        b = [g("scan_b", f, k) for k in ("H00_weak_omega_e", "H01_weak_omega_e", "H01_strong_omega_e")]
        c = g("scan_c", f, "vq0.03_Nn64_r0")
        L.append(f"| {f or 'None (plain PID)'} | {aa[0]} | {aa[1]} | "
                 + " | ".join(f"{x['steps']}, {x['Ebar_err']:.1e}" for x in b)
                 + f" | {c['t_reached']:g}, {c['steps']}, {c['max_rel_diff_dK_e']:.0e} / {c['max_rel_diff_W_ext']:.0e} |")
    r = {k[0][len("scan_a_rtol"):] + f" floor {k[1]:g}, v_q/v_te {k[2]}": d for k, d in nfs.items() if k[0].startswith("scan_a_rtol")}
    L += ["", "Unseeded pump-frame controls at tighter rtol (same floor scan): "
          + "; ".join(f"rtol {k}: {d['steps']} steps, {d['max_rel_dev_W_lin']:.1e}" for k, d in sorted(r.items())) + ".", "",
          "Reading: the unseeded deviation from the exact uniform two-fluid law is set by rtol, not by the floor "
          "(flat over 1e-16..1e-10; the maximum sits at t = 20-30, the start of the window); plain PID agreed to 3e-10 only "
          "because it crawled on noise and stalled (t = 158 and 31.5). H00 is floor-insensitive up to 1e-13; H01's Ebar "
          "error grows roughly linearly with the floor (3x at 1e-14, 40x at 1e-12, 800x at 1e-10, where it exceeds 2e-9). "
          "The seeded case moves by <= 4e-8 of dK_e at any floor and stops at the same t, far below the realization spread "
          "(committed t reached 869.5-937.5 over r0-r2; 860 here is the positivity stop for every floor). Larger floors save few steps (pump: 3238 at 1e-14 vs 3194 at 1e-10), so 1e-14 is "
          "kept as the control preset; controls keep the seeded runs' rtol 1e-10 so they test the same numerics, and "
          "rtol 1e-11 would bring the unseeded agreement below 2e-9 for about 15-30% more steps."]
    return "\n".join(L)


def phase_space():
    ph = json.loads((root / "docs/_static/phase_space/run.json").read_text())["cases"]
    rows = []
    for k, e in ph.items():
        i, r = e["inputs"], e["runs"]
        reach = [(v["valid"] - 1) * i["T"] / 400 for v in r.values()]
        st = ", ".join(sorted({v["status"] for v in r.values()}))
        rows.append(f"| {k} | {i['Nn']} / {2 * i['Nn']}, nu = {i['nu']:g} | {st} ({min(reach):g}-{max(reach):g}) | "
                    f"{e['t_res_ordinary']:g} / {e['t_res_dark']:g} | {e['t_res_f_ordinary']:g} / {e['t_res_f_dark']:g} | "
                    f"{e['movie']['min_f_over_max_ordinary']:.3f} / {e['movie']['min_f_over_max_dark']:.3f} |")
    return """## Phase-space movies: resolution of the reconstructed f (`python studies/figures.py phase`)

Question: how long do fixed-basis Hermite runs resolve f(x, v) itself (not only the field) once trapping starts?
Input: `PHASE_CASES` in studies/figures.py (electrostatic units mapped with v_t/c = 0.1, Nx = 16, seeds at the box
mode, ordinary and dark eta = 0.3, Omega_D = omega_pe), each at Nn and 2Nn, Dopri8 rtol 1e-8, 200000-step budget.
Measurement: t_res by field energy (10% of the running max, as for H05) and by f (first frame with
max|f_Nn - f_2Nn| > 0.1 max f on a 96 x 160 grid); frames are drawn only before the earlier one.

| case | Nn / 2Nn, closure | solver status (t reached) | t_res field energy (ord / dark) | t_res f (ord / dark) | min f / max f drawn (ord / dark) |
|---|---|---|---|---|---|
""" + "\n".join(rows) + """

The field-energy rule alone is too lenient for phase-space pictures: in the two-stream and bump-on-tail runs f
differs by more than 10% between Nn and 2Nn 13-18 time units before the field energies do, and every two-stream
and bump-on-tail run later exhausts its step budget. Negative f is not removed by doubling Nn (two-stream: about
-31% / -37% of max f at the last drawn frame while Nn and 2Nn agree to 0.8%); Nx was not varied, so x truncation and
the nu = 1 closure are the untested candidates. The nonlinear Landau runs (no closure) stay above -0.7% of max f to
the drawn end. The movies illustrate the onset of trapping only."""


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
          "reaches its first minimum earlier (30.5). The prominence-based extremum detector used in this table misses "
          "minima in several runs (N = 512, nu = 0 and nu = 1); the earlier statement that nu = 1 removes the trapping "
          "minimum is retracted, see the B01 correction section below.", "",
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
    f"Electrons with $v_{{te}}=0.1c$ on fixed ions, with and without a mixed field ($\\eta={s['eta']}$, "
    f"$\\Omega_D={s['Omega_D']:g}\\,\\omega_{{pe}}$, Yukawa-consistent start). Mixing splits the longitudinal response into a "
    "Langmuir-like and a Proca-like branch (left panel). Fitted complex frequencies of "
    f"{len(runs)} runs (Hermite orders 64/128, grids 5/8) agree with independent SciPy roots of "
    "$D_L=(Q-\\Omega_D^2)(1+\\chi)+\\eta^2Q\\chi$ to a relative "
    f"{worst:.1e} or better, and the work ledger closes to roundoff. Mixing raises the frequency by "
    f"{100 * shift[0.3][0]:.1f}% / {100 * shift[0.5][0]:.1f}% and lowers the damping rate by "
    f"{-100 * shift[0.3][1]:.1f}% / {-100 * shift[0.5][1]:.1f}% at $k\\lambda_{{De}}=0.3/0.5$. At $k\\lambda_{{De}}=0.2$ the "
    "damping rate (about $5\\times10^{-5}$) is not resolved inside the Hermite-limited window and is not plotted. "
    "This is a linear, deliberately large-coupling verification.")

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

{lanes()}

{phase_space()}

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
    return "\n".join(L)




def _between(text, tag, body):
    """Replace the README block between <!-- tag --> and <!-- /tag --> (generated from the records)."""
    return re.sub(rf"(<!-- {tag} -->\n).*?(\n<!-- /{tag} -->)", lambda m: m.group(1) + body + m.group(2), text,
                  flags=re.S)


readme = (root / "README.md").read_text()
readme = _between(readme, "tests-paragraph", para)
readme = _between(readme, "tests-table", readme_table())
(root / "README.md").write_text(readme)
