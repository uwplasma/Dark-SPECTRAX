"""Write docs/results.md and the README result paragraph from the generated B00 record."""

import json
import re
from pathlib import Path

import numpy as np

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
hk = json.loads((root / "studies/highk/summary.json").read_text())
hk_lin = json.loads((root / "studies/highk/linear_vq0.1.json").read_text())
hk_eig = json.loads((root / "studies/highk/eig.json").read_text())
hk_flq = json.loads((root / "studies/highk/floquet.json").read_text())
otsi = json.loads((root / "studies/instab/otsi.json").read_text())
ba = {p.stem[len("before_after_"):]: json.loads(p.read_text()) for p in (root / "studies/consolidate").glob("before_after_*.json")}


def _tr(x, end=1000.0):
    return "-" if x is None else (f">= {end:g}" if x >= end else f"{x:g}")


def lanes():
    """H05 strong-drive lane comparison, the declared stabilizer, the B01 correction and the run_adaptive fixes."""
    sc, fb = lc["scan"], gate["fixed_basis_resolved_until"]
    L = ["## H05 strong resonant drive: lane comparison (pump frame + remap, Nx = 8, records made at parent c0910a1)", "",
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
          "with any lane; x-convergence without the field-scaled closure (see the high-k section below); portability of "
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


def highk_section():
    """x-refinement of H05: numerical AW truncation instability, field-scaled closure (SPECTRAX #66), physical high-k limit."""
    L = ["## H05 x-refinement: AW truncation instability, field-scaled closure, physical high-k limit (`studies/highk_*.py`)", "",
         "Supersedes the lane comparison above for x-convergence and the certified windows at v_q/v_te = 0.03 and 0.1. "
         "Records were made at parent c0910a1 with the closure implemented in the study; the study now calls the parent "
         "helper `spectrax.field_scaled_closure_rate` (SPECTRAX #66, in the pinned integration branch), which equals the "
         "study implementation to round-off (`python studies/highk_parent_equiv.py`, record `studies/highk/parent_equiv.txt`).", "",
         "### Numerical: asymmetric-Hermite truncation instability", "",
         "Without a closure, Nx = 16 fails earlier than Nx = 8 (t = 476 against >= 700 at v_q/v_te = 0.1), and raising Nn "
         "does not help. In a non-uniform field the truncated AW Galerkin operator -v d_x + (q/m)E d_v is not "
         "anti-self-adjoint: its growing modes sit in the top third of the Hermite ladder at the largest kept |k|, with a "
         "rate that rises with the Fourier cut-off K and with Nn. The symmetric (SW) basis is neutral to round-off. "
         "Largest real part of the frozen-field spectrum (`studies/highk_eig.py`, u = 0) and Floquet exponent in a "
         "pump-modulated field (`studies/highk_floquet.py`):", "",
         "| basis | K | Nn | max Re, E1 = 1e-3 | max Re, E1 = 1e-2 | Floquet, E1 = 8e-3 | Floquet, E1 = 2e-2 |", "|---|---|---|---|---|---|---|"]
    eig = {(r["basis"], r["K"], r["Nn"], r["E1"]): r["max_re"] for r in hk_eig if r["u"] == 0.0}
    flq = {(r["basis"], r["K"], r["Nn"], r["E1"]): r["floquet"] for r in hk_flq}
    for b in ("AW", "SW"):
        for K in (2, 5, 10):
            for nn in (32, 64, 128):
                f = [flq.get((b, K, nn, e)) for e in (0.008, 0.02)]
                L.append(f"| {b} | {K} | {nn} | {eig[(b, K, nn, 0.001)]:.1e} | {eig[(b, K, nn, 0.01)]:.1e} | "
                         + " | ".join("-" if x is None else f"{x:.1e}" for x in f) + " |")
    L += ["", "Ruled out: aliasing (padded convolution equals the 2/3-masked RHS to 1e-16, `studies/highk/alias_check.txt`), "
          "step size (fixed dt = 0.01 fails at the same t) and segment remaps (with remaps off it fails at 472).", "",
          "Closure: nu_s = c |q/m|_s sqrt(2N) max|E - <E>| / a_s on n(n-1)(n-2)/((N-1)(N-2)(N-3)) (density, momentum "
          "and energy rows untouched), c = 1. c = 0.5 and 2 and Nn = 128 agree with c = 1 under the 10% rule; before "
          "t of about 550 the run is identical to the unregularized one to 5 digits.", "",
          "| v_q/v_te | Nx | status, t reached | first non-positive K or T | steps (rejected) | compile + run s | c = 0.5 / c = 2 / Nn 128 t_res(dK_e) | grid t_res(dK_e), refinement pair |",
          "|---|---|---|---|---|---|---|---|"]
    for vq in ("0.1", "0.03"):
        for nx in (8, 16, 32):
            r = hk[f"vq{vq}_Nx{nx}"]
            var = " / ".join(str(r[k]["t_res_dKe"]) for k in ("c0.5", "c2", "Nn128")) if "c2" in r else "-"
            L.append(f"| {vq} | {nx} | {r['status']}, {r['t_reached']:g} | {r['t_pos'] if r['t_pos'] is not None else 'none'} | "
                     f"{r['steps']} ({r['rejected']}) | {r['compile_s']:g} + {r['run_s']:g} | {var} | {r['grid']['t_res_dKe']} |")
    L += ["", "Hermite x-refinement agreement (t_res dK_e / W_ext): "
          + "; ".join(f"{k.replace('_', ' ')} {hk[k]['t_res_dKe']} / {hk[k]['t_res_W']}"
                      for k in ("vq0.1_Nx8_vs_Nx16", "vq0.1_Nx16_vs_Nx32", "vq0.03_Nx8_vs_Nx16", "vq0.03_Nx16_vs_Nx32")) + ". "
          "Nx = 32 runs use `noise_floor` 1e-14 (without it atol 1e-14 sits below FFT round-off and the runs hit the cap); "
          "at Nx = 16 the floor changes nothing.", "",
          "### Physical: broadband finite-k instability of the quivering plasma", "",
          "The exact Volterra solution and the linear Hermite model (`studies/highk_linear.py`, v_q/v_te = 0.1, T = 700) "
          "agree that the driven plasma amplifies finite-k fields, more strongly at larger k up to k lambda_D of about 0.1. "
          "Hermite Nn = 32/64/128 agree with each other; they follow the exact solution until the tabulated t (first 10% "
          "difference) and end below it by a factor 1.3-4, so the Hermite model understates, not invents, the growth:", "",
          "| mode | exact max E_k^2 / initial | exact end / initial | Hermite Nn 32 / 64 / 128 end / initial | Hermite vs exact 10% at t |",
          "|---|---|---|---|---|"]
    for k, r in hk_lin["res"].items():
        L.append(f"| {k} | {r['exact_E2_max_over_E2_0']:.2e} | {r['exact_E2_end_over_0']:.2e} | "
                 + " / ".join(f"{r[f'herm{n}_E2_end_over_0']:.3e}" for n in (32, 64, 128)) + f" | {r['herm64_t_10pct']:g} |")
    L += ["", "This instability, not the closure, ends x-converged agreement: at Nx = 32 the Hermite run loses positivity "
          f"at t = {hk['vq0.1_Nx32']['t_pos']:g} (0.1) and {hk['vq0.03_Nx32']['t_pos']:g} (0.03), and the grid at Nx = 32 "
          "blows up at the same time. Correction (2026-10-19): this section originally read the v_q/v_te = 0.1 limit "
          "(~600-750) as the resolution of the OTSI high-k cascade, not the basis. The GPU resolution study (gpures section) "
          "refutes that: with c = 4 more Nx or Nn ends the run sooner (0.1: t = 595, 383, 453, 328), so the limit is the "
          "asymmetric-Hermite truncation instability, whose frozen-field rate rises with the Fourier cut-off and Nn, outgrowing "
          "the closure. The physical OTSI amplification is real but has not been resolved to t = 1000 with this basis; the "
          "grid-reference caveat (certified only to ~750) stands.", "",
          "### Certified collisionless windows (Hermite, all criteria)", "",
          f"- v_q/v_te = 0.1: t of about {hk['vq0.1_Nx32']['t_pos']:g} (Nx 32 positivity; grid agreement to "
          f"{hk['grid_vq0.1_Nx16_vs_Nx32']['t_res_dKe']}-{hk['vq0.1_Nx16']['grid']['t_res_dKe']}). Previously 606.5 (lane C).",
          f"- v_q/v_te = 0.03: t of about {hk['vq0.03_Nx32']['t_pos']:g}-{hk['vq0.03_Nx16_vs_Nx32']['t_res_dKe']} "
          "(Nx 32 positivity and Nx 16 vs 32). Previously 879 (lane C).",
          "- v_q/v_te = 0.01: unchanged, resolved to 1000 (lane C).",
          "- No drive reaches t = 1000 x-converged at 0.03 or 0.1.", "",
          "Superseded: the certified-time panel of `docs/_static/conversion/figure.png` and the Lane C t_res values for "
          "0.03 and 0.1 above are Nx = 8 results; they are kept as records but the windows here replace them. "
          "The t_res law in the lane section is an Nx = 8 law. Grid caveat: the grid reference goes negative "
          "(f < -1e-3) before the Hermite runs do, so it is not a better reference late in the run."]
    return "\n".join(L)


def nxconv_section():
    d = json.loads((root / "studies/nxconv/summary.json").read_text())
    L = ["## H05 Nx convergence to t = 1000: Hermite (Nx 32/64/128) vs masked spectral grid (`studies/nxconv_*.py`)", "",
         "Records made with parent integration/dark-baseline 16400f8 (pump frame, remap, #66 field-scaled closure "
         "strength c, noise_floor 1e-14). Each entry is the last time two runs agree within 10% (`>=` = agree to the "
         "end of the shorter run). Field spectra count only modes with |E_k|^2 >= 1e-6 of the running max of |E_1|^2 "
         "in the reference run.", "",
         "| Hermite pair | dK_e | W_ext | field spectrum |", "|---|---|---|---|"]
    for k, v in d["pairs"].items():
        L.append(f"| {k} | {v['dK_e']} | {v['W_ext']} | {next(x for kk, x in v.items() if kk.startswith('Ek2'))} |")
    L += ["", "| Hermite c = 4 vs grid | dK_e | W_ext | field spectrum |", "|---|---|---|---|"]
    for k, v in d["hermite_vs_grid"].items():
        L.append(f"| {k} | {v['dK_e']} | {v['W_ext']} | {next(x for kk, x in v.items() if kk.startswith('Ek2'))} |")
    L += ["", "| grid run | first f < -1e-3 max f | min f / max f | max edge fraction | max energy defect / W |",
          "|---|---|---|---|---|"]
    for k, v in d["grid"].items():
        L.append(f"| {k} | {v.get('t_fneg_1e-3')} | {v['min_f_rel']:.2g} | {v['max_edge']:.2g} | "
                 f"{v['max_energy_defect_over_W']:.1e} |")
    L += ["", "**Certified windows (supersede the Nx = 8 lane windows and the conversion-figure panel):** "
          "v_q/v_te = 0.03: dK_e and W_ext to t ~ 960-1000 (c = 4, Nx >= 32), field spectra to ~720-770. "
          "v_q/v_te = 0.1: dK_e and W_ext to t <~ 600, limited by the asymmetric-Hermite truncation instability "
          "(more Nx or Nn fails sooner); field spectra to ~555. The grid reference at v_q/v_te = 0.1 is valid only to "
          "t ~ 645 for positivity and t ~ 700-755 for the box edge; widening the box does not extend it to t = 1000 "
          "(next paragraph)."]
    return "\n".join(L + gridref_lines(d))


def rclosure_section():
    m = json.loads((root / "studies/rclosure/measure.json").read_text())
    f = json.loads((root / "studies/rclosure/stagef.json").read_text())
    tr = [x for x in m if x["peak"] in (1e-3, 4e-3)]
    n1 = [x for x in tr if x["peak"] == 1e-3]
    n4 = [x for x in tr if x["peak"] == 4e-3]
    null = lambda xs: sum(x["c_req"] is None for x in xs)  # noqa: E731
    L = ["## Can a resolution-consistent closure strength remove the AW truncation instability? (`studies/rclosure_gates.py`)", "",
         "Gates, revisions and consequences are committed in the docstring of `studies/rclosure_gates.py` before each run "
         "(v1 dense eigenvalues; v2 time-domain growth because the AW operator is so non-normal that eigenvalues are round-off "
         "sensitive: E = 1e-2, K = 5, Nn = 64 gave 0.756 dense, 0.846 ARPACK, 0.702 time-domain; v3 ladder of closure strengths).",
         "", "**Stage M (frozen field).** The truncated AW operator -v d/dx + (q/m) E(x) d/dv, kept |k| <= K k1, Hermite order Nn, "
         "a = sqrt(2e-3), with the #66 hypercollision shape nu s(n), nu = c |q/m| sqrt(2 Nn) max|E - <E>| / a; gamma is the "
         "asymptotic growth rate of exp(L t) applied to a fixed random vector. Ladder c = 0, 4, 16, 64, 256, 1024; field shapes "
         "single cosine, flat and 1/j multi-mode spectra; (K, Nn) in {2, 5, 10, 21} x {32, 64, 128} plus (5, 256), (10, 256); peak field 1e-3 and 4e-3.", "",
         f"- No ladder c reaches gamma <= 1e-3 at {null(n4)}/{len(n4)} points with peak 4e-3 (the k1 field at t ~ 200-450 of the v_q 0.1 runs) "
         f"and at {null(n1)}/{len(n1)} points with peak 1e-3. Raising c leaves a floor. Declared gate Q1 (a c(K, Nn) keeping gamma_AW <= 0 at every "
         "refinement) therefore FAILS: the premise is false for this closure family, so the decisive stage D (scaled closure at Nx 32/64/128 x "
         "Nn 64/128/256 to t = 1000, B10 rerun) was not run and no SPECTRAX option was added.",
         "- Fits (log-linear, rms residual 0.4-0.5 in ln gamma, factor ~1.6): gamma(c = 0) ~ 32 Nn^0.57 peak^1.2 with no dependence on K "
         "(exponent 0.00; the K dependence of the single-mode eigenvalue table above comes from a different operator, not from a rising "
         "frozen-field rate); gamma(c = 4) ~ 2.4 Nn^0.78 peak^1.1 (more Nn makes c = 4 worse); floor gamma(c = 1024) ~ 11 K^0.15 Nn^0.30 peak^1.26. "
         "At peak 4e-3 the c = 4 growth is 0.04-0.46 and the c = 1024 floor 0.02-0.14, i.e. e^20 to e^100 over t = 1000 from a 1e-14 seed.", "",
         "**Stage F (strong fixed closure on the real run).** v_q/v_te = 0.1, Nx 64, Nn 128, GPU, c in {16, 64, 256} against the c = 4 record "
         "(ends at 595):", "",
         "| c | t reached | first K_s or T_x <= 0 | ledger / max W_ext | max dK_e diff to c = 4, t <= 500 | max W_ext diff to c = 4, t <= 500 |",
         "|---|---|---|---|---|---|"]
    for k, v in f["runs"].items():
        L.append(f"| {k[1:]} | {v['t_reached']:g} | {v['t_pos']:g} | {v['ledger_rel']:.1e} | {v['fidelity_t500']['dK_e']:.1e} | {v['fidelity_t500']['W_ext']:.1e} |")
    L += ["", "F2 (fidelity 1e-3 up to t = 500) holds for all three and the ledger is below 1e-9, but the window saturates near 710-760 "
          "(c = 16: 731.5, c = 64: 757, c = 256: 713) and F1 (c = 256 reaching 744) fails; every run ends by loss of positivity. "
          "A stronger closure therefore buys about 120-160 time units at no cost in resolved physics, then stops. Because 750 is also the "
          "limit of the grid reference, the saturation cannot be assigned to the AW floor alone: the physical OTSI stage and the AW "
          "floor both act there. Nx/Nn refinement pairs with a scaled closure were not run (no derivation to scale); the route to t = 1000 at "
          "v_q/v_te = 0.1 still needs a stable basis (design option B)."]
    return "\n".join(L)


def stiffdiag_section():
    sm = json.loads((root / "studies/stiffdiag/summary.json").read_text())
    cf = json.loads((root / "studies/stiffdiag/counterfactual.json").read_text())
    L = ["## What limits the step at the H05 v_q/v_te = 0.1 failure: stiffness or instability? (`studies/stiffdiag_gates.py`)", "",
         "Hypotheses, gates and four dated revisions/addenda (stability threshold, frozen-closure split, fine step history, "
         "similarity-scaled eigensolve and time-domain tangent growth, closure-removal counterfactual) are committed in the docstring "
         "of `studies/stiffdiag_gates.py` before the measurements they govern. Nx 64, Nn 128, v_q/v_te 0.1, pump frame, noise floor 1e-14, "
         "rtol 1e-10, atol 1e-14, GPU, closure c = 4 (run ends t = 578 here, 595 in the gpures record: the blow-up is segment-sensitive) "
         "and c = 64 (ends 760 by positivity). Per checkpoint: the exact Jacobian J of the RHS (jax.linearize, 11272 real unknowns: dealiased "
         "Ck, Fk and the pump-frame velocities), its dense spectrum at t = 300 / the checkpoint before the end / the last one and ARPACK "
         "(LM, LR) everywhere, J_L = the same RHS linearized about the homogeneous part of the state with the closure rate frozen "
         "(streaming + closure + Maxwell, exactly block diagonal in k: leakage 2e-16), the Dopri8 stability bound "
         "h_stab = largest h with |R(h lambda)| <= 1 + 1e-6 over the spectrum (real axis 5.17, imaginary axis 3.70), and a tolerance test "
         "(1 time unit at rtol 1e-10 ... 1e-4). A step set by accuracy scales as rtol^(-1/8) (steps 1e-10 / 1e-4 = 5.6); a step set by the "
         "explicit stability bound does not respond to the tolerance (1).", ""]
    for key, name in (("c4", "c = 4"), ("c64", "c = 64")):
        d = sm[key]
        L += [f"**{name}** (t_f = {d['t_f']:g}; last measured {d['last_measured_t']:g}): HA (stiffness-limited at every near-failure checkpoint) "
              f"{'holds' if d['HA'] else 'fails'}, HC (accuracy-limited) {'holds' if d['HC'] else 'fails'}, HB (growth realized as predicted) "
              f"{'holds' if d['HB'] else 'fails'}; declared verdict **{d['verdict']}**.", "",
              "| t | dt_acc | nu_e (closure) | rho(J) | max Re lambda(J) | h_stab | dt/h_stab | steps 1e-10/1e-4 | top-3rd rate: run / tangent | steps/unit live / closure removed |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in d["rows"]:
            tg = "-" if r["top3_rate_tangent"] is None else f"{r['top3_rate_tangent']:.2f}"
            c_ = "-" if r["steps_live_1unit"] is None else f"{r['steps_live_1unit']} / {r['steps_no_closure_1unit']}"
            L.append(f"| {r['t']:g} | {r['dt_acc']:.2g} | {r['nu_e']:.3g} | {r['rho_J']:.3g} | {r['abscissa_J']:.2f} | {r['h_stab_min']:.2g} | "
                     f"{r['dt_over_hstab']:.2f} | {r['steps_ratio_1e-10_over_1e-4']:.2f} | {r['top3_rate_measured']:.2f} / {tg} | {c_} |")
        L += ["", f"Step collapse t = 300 to {d['last_measured_t']:g}: C = {d['collapse_C']:.3g}; f_stab = {d['f_stab']:.2f} "
              f"(dt/h_stab {d['dt_over_hstab_ref_last'][0]:.2f} -> {d['dt_over_hstab_ref_last'][1]:.2f}); f_tol at the last checkpoint = {d['f_tol']:.2f}; "
              f"share of the last unit's steps removable by removing the closure damping = {d['curable_step_share_last']:.2f}.", ""]
    L += ["**Reading.** (1) The stiff eigenvalue is the closure itself: rho(J) equals the closure rate nu_e = c sqrt(2 Nn) max|E|/a to 0.2-7% "
          "for t >= 500 at c = 4 (20% at t <= 300, where the light wave and mode coupling still matter) and to 0.0% at c = 64, real and negative, and rho(J_L) = rho(J) at c = 64 (835.628 and 2318.37), so the explicit step is bounded by "
          "5.17/nu_e and the stiff part lives entirely in the linear block (diagonal, so an integrating factor or exponential treatment is exact). "
          "(2) c = 64 is stiffness-limited throughout (dt/h_stab 0.7-1.9, steps independent of the tolerance 1.02-1.12 at five of six "
          "near-failure checkpoints; at t = 720 the tolerance ratio is 2.3, so the strict HA rule fails there and the declared verdict is MIXED); "
          "removing the closure from the RHS cuts the steps per unit by 6-9 times (548 to 88 at t = 760). The run ends by positivity loss at 760 "
          "without a step collapse (dt 3.7e-3). (3) c = 4 is mixed: dt/h_stab 0.16-0.49 and tolerance ratios 2.5-5.3 (partly accuracy-limited) "
          "until t = 576, where the closure rate has risen to 2500 with |E|max = 4 and the closure-removed counterfactual needs 7 times fewer "
          "steps; the final collapse (dt 1e-3 at 576, 3.6e-5 at 578, max|Ck| 2e70, |E| 320, K_e < 0) is the blow-up itself and its Jacobian "
          "was not measured (the dense and ARPACK measurement of that state did not finish and was stopped). (4) A true instability is present in "
          "both runs: the tangent growth rate of the top-third Hermite amplitude, 2.2 at c = 4 (t = 560) and 1.8-2.7 at c = 64 (t = 740, 760), "
          "equals max Re lambda(J) (2.1, 1.6, 2.7), and it comes from the field-nonlinear part (max Re lambda(J_L) is 0.16-0.43 only). "
          "The nonlinear run does not realize it as a steady exponential at c = 64 (top-third rate -0.2 to +0.2) so HB, which needs the measured "
          "and tangent rates within a factor 2, fails at both; at c = 4 the measured rate climbs 0.42, 1.15, 4.28 at t = 566, 572, 576 against "
          "max Re lambda 2.4, 3.6, 6.0. (5) Consequence: the part of the step collapse that is stiffness (f_stab = 1.0 for both, since the "
          "stability bound shrinks faster than dt; 84-86% of the steps of the last unit are removable by exponential/implicit treatment of the "
          "closure) is curable; the window is not: the AW instability with max Re lambda 2-3 ends both runs (positivity lost at 760 and about 578-581), "
          "an integrator that took 6-9 times fewer steps would reach the same loss of positivity. Caveats: eigenvalues of this operator are "
          "round-off sensitive (dense and ARPACK abscissa differ by 1.8% at c = 64, t = 740 and by 0.036 at t = 300, which fail the S0 agreement gate; "
          "rho agrees to 1e-14), so the time-domain tangent growth and the tolerance test are the primary evidence; the S0 finite-difference "
          "check as declared failed (0.005-1.0, perturbation scaled by |v| dominated by the 1e4-1e20 Hermite coefficients), the component-wise "
          "follow-up S0b passes (<= 3e-7 on finite states; the blown-up c = 4 state is non-finite).", ""]
    return "\n".join(L)


def r1nl_section():
    """R1 second step: certified windows with the strong closure (Lawson) and nonlinear-detuning diagnostics at the strongest drive."""
    s1 = json.loads((root / "studies/r1nl/stage1_summary.json").read_text())
    s2 = json.loads((root / "studies/r1nl/stage2_summary.json").read_text())
    v = s2["vstar"]
    f = lambda x, fmt=".3g": "-" if x is None else format(x, fmt)
    L = ["## R1 nonlinear: detuning and saturation of the energy transfer at the strongest certified drive (`studies/r1nl_gates.py`)", "",
         "Gates, the case lists and the decision rules were committed in `studies/r1nl_gates.py` (69a56e9) before any production run; "
         "the analysis (`studies/r1nl_analysis.py`) implements them without change. H05 set-up (mobile ions, pump frame + remap, seeds r0, "
         "omega = omega_L, noise floor 1e-14, rtol 1e-10, atol 1e-14), strong field-scaled closure c = 64 integrated exactly by the Lawson wrapper "
         "(`run(..., integrating_factor=)`), saves every 0.5, GPU (RTX A4000, float64). v_q/v_te is the effective initial force. Stage-1 runs go to T = 1000 (or to the first positivity loss / step budget); stage-2 runs were integrated to T = 800, which is above the window of the strongest drive (the Nx 64 member loses positivity at 791.5), so nothing inside the window is cut. One stage-2 coherence realization (tau = 300, seed 1) loses positivity at 792, just beyond the window.", "",
         "### Stage 1: certified windows (members B = Nx 32/Nn 64, X = Nx 64/Nn 64, N = Nx 32/Nn 128, XN = Nx 64/Nn 128, C = Nx 32/Nn 64 at c = 32)", "",
         "| v_q/v_te | window | limiter | pair B-X | B-N | X-XN | N-XN | B-C (c 32/64) | first K_s or T_x <= 0 (B / X / N / XN / C) | t reached (B / X / N / XN / C) | max ledger | grid Nx 32 / 64 agree with B until |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in s1["vqs"].items():
        pn = r["pairs"]
        fp = " / ".join(f(r["first_nonpositive"][m], "g") for m in ("B", "X", "N", "XN", "C"))
        tr = " / ".join(f(r["t_reached"][m], "g") for m in ("B", "X", "N", "XN", "C"))
        g = r["grid"]
        gg = " / ".join(f(g.get(f"grid_Nx{n}_vs_B", {}).get("agree_until"), "g") for n in (32, 64))
        L.append(f"| {k} | {r['window']:g} | {r['limiter']} | {pn['B-X']:g} | {pn['B-N']:g} | {pn['X-XN']:g} | {pn['N-XN']:g} | {pn['B-C']:g} | "
                 f"{fp} | {tr} | {max(r['ledger'].values()):.1e} | {gg} |")
    L += ["", f"Window = min(1000, all pair times (10% rule on dK_e and W_ext, t > 50), all positivity times, all t reached). All three drives are certified "
          f"(ledger gate 1e-6). The pairs agree to the end of the shorter run except where noted in the table; the windows are set by positivity "
          f"loss of the Nx 64 members, i.e. by the AW truncation instability reaching the resolved range, not by disagreement of the refinement pairs. "
          f"Strongest certified drive (declared rule): v_q/v_te = {v:g}, window {s2['window']:g}. The grid comparison is a reported cross-check (not in the window).", "",
          f"### Stage 2 at v_q/v_te = {v:g}, window {s2['window']:g}", ""]
    n1, n2, n3 = s2["N1"], s2["N2"], s2["N3"]
    L += ["**N1, W_ext against the linear resonant law (the homogeneous mean-field ODE).** "
          f"D = W_ext/W_lin - 1: max |D| over (50, window] B {n1['max_abs_D_by_member']['B']:.2e}, X {n1['max_abs_D_by_member']['X']:.2e}, "
          f"N {n1['max_abs_D_by_member']['N']:.2e}, XN {n1['max_abs_D_by_member']['XN']:.2e}; at the window end D = {n1['D_at_window_end']['B']:+.2e} (B), "
          f"{n1['D_at_window_end']['XN']:+.2e} (XN); W_ext(window)/(n T_e V) = {n1['Wext_over_nTe_at_window']['B']:.1f} kinetic, {n1['Wext_lin_over_nTe_at_window']:.1f} linear. "
          f"Departure rule (|D_B| > 0.10 and all refinements > 0.05): **{'MET' if n1['departure'] else 'not met'}**; upper bound on the nonlinear correction to W_ext "
          f"(including refinement spread): {n1['bound_max_abs_D']:.2e}.", "",
          "**N2, effective frequency shift of the mean field.** "
          f"Slope of the lock-in phase difference kinetic - linear over t >= {n2['lo']:.0f}: delta_omega (omega_pe units) = "
          + ", ".join(f"{m} {n2['delta_omega_by_member'][m]:.2e}" for m in ("B", "X", "N", "XN"))
          + f" (delta_omega/omega_L = {n2['delta_omega_over_wL_B']:.1e} for B); refinement spread {n2['refinement_spread']:.1e}; resolved (> 1e-3 omega_L and > 3 spread): **{'yes' if n2['resolved'] else 'no'}**; "
          f"bound |delta_omega| <= {n2['bound']:.1e}. Peak |E-bar| kinetic/linear in the last 31 units: "
          + ", ".join(f"{m} {n2['Ebar_amp_ratio_kin_over_lin_at_window'][m]:.4f}" for m in ("B", "X", "N", "XN")) + ".", "",
          "**N3, energy partition at the window end (units n T_e V), Q_i and OTSI onset (reported, ungated).**", "",
          "| member | dK_e | dK_i | W_ext | dU_field | dTh_e (k=0 thermal) | dTh_i | ion share dK_i/W_ext | max abs Q_i | rms Q_i (last 20%) | corr(Q_i, Ebar) last 20% | max sum_k>=1 abs E_k^2 / E0^2 | t_OTSI (4 <= k < Nx/3) | energy growth rate in +-50 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("B", "X", "N", "XN"):
        r = n3[m]
        L.append(f"| {m} | {r['dKe']:.2f} | {r['dKi']:.3f} | {r['Wext']:.2f} | {r['dU_gamma']:.2f} | {r['dTh_e']:.3f} | {r['dTh_i']:.4f} | {r['ion_share']:.2e} | "
                 f"{r['Qi_max_abs']:.1e} | {r['Qi_rms_last20pct']:.1e} | {r['Qi_corr_with_Ebar_last20pct']:.2f} | {r['Ek_fluct_max_ratio_to_E0sq']:.0f} | "
                 f"{f(r['t_otsi'], 'g')} | {f(r['growth_rate_energy_of_k>=4_sum_within_50'], '.3f')} |")
    L += ["", "The upper-k band of the OTSI onset definition has 9 modes at Nx 32 and 20 at Nx 64 (floor(Nx/3) - 1), so t_OTSI and its growth rate are compared within equal Nx only.", ""]
    L += ["**N4, finite reservoir (self-consistent dark field, Omega_D = omega_L).**", "",
          "| u = U_D0/(n T_e V) | eta | window | max abs(U_D - U_D^lin)/U_D0 (B / XN) | min U_D/U_D0 kin / lin | W_D/U_D0 at window kin / lin | max abs(W_D/W_D^lin - 1) B / XN | departure | t_split kin / lin | ledger B / XN |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in s2["N4"].items():
        if r.get("missing"):
            L.append(f"| {k} | missing |"); continue
        dp = r["WD_departure"]
        L.append(f"| {r['U_D0_over_nTe']:.3g} | {r['eta']:.3e} | {r['window_sc']:g} | {r['max_dev_UD_over_UD0']['B']:.2e} / {r['max_dev_UD_over_UD0']['XN']:.2e} | "
                 f"{r['min_UD_over_UD0_kin']:.3f} / {r['min_UD_over_UD0_lin']:.3f} | {r['WD_over_UD0_at_window_kin_lin'][0]:.3f} / {r['WD_over_UD0_at_window_kin_lin'][1]:.3f} | "
                 f"{dp['max_abs_D_by_member']['B']:.2e} / {dp['max_abs_D_by_member']['XN']:.2e} | {'MET' if dp['departure'] else 'no'} | "
                 f"{f(r['t_split_kin'], 'g')} / {f(r['t_split_lin'], 'g')} | {r['ledger'][0]:.1e} / {r['ledger'][1]:.1e} |")
    L += ["", "**N5, finite coherence (v2 definition, tau = field coherence time; realizations seeds 0-5 in B, seed 0 in XN).**", "",
          "| tau omega_pe | window | R = <W_kin>/<W_lin> late (se), per-realization | XN r0 D at window | departure | bound if none | max abs D by realization (relative; large where W_lin is small) | max abs(W_kin - W_lin) / peak W_lin by realization | late slope W_ext /(n T_e V) | ledger max |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in s2["N5"].items():
        if r.get("missing"):
            L.append(f"| {k} | missing |"); continue
        L.append(f"| {k[3:]} | {r['window_coh']:g} | {r['R']:.4f} ({r['se']:.1e}); " + ", ".join(f"{x:.4f}" for x in r["R_by_realization"])
                 + f" | {r['XN_r0_D_at_window']:+.2e} | {'MET' if r['departure'] else 'no'} | {r['bound_if_none']:.2e} | "
                 + ", ".join(f"{x:.1e}" for x in r["max_abs_D_by_realization"]) + " | " + ", ".join(f"{x:.1e}" for x in r["max_abs_diff_over_peak_Wlin_by_realization"]) + f" | {r['late_slope_over_nTe']:.3g} | {r['ledger_max']:.1e} |")
    n6 = s2["N6"]
    L += ["", f"N6 ledger over all {n6['n_runs']} stage-1 and stage-2 runs: max {n6['ledger_max']:.1e} ({'pass' if n6['pass'] else 'FAIL'}, gate 1e-6).", ""]
    fu = json.loads((root / "studies/r1nl/followup_summary.json").read_text())
    L += ["### Follow-up: what sets the small W_ext deficit? (`studies/r1nl_followup.py`, declared after stage 2 and before its runs)", "",
          "D(t) of the coherent drive is the same at v_q/v_te = 0.03, 0.05 and 0.07 to three digits (D = -1.0e-8 t^2: -1.0e-4 at 100, -4.1e-4 at 200, -1.66e-3 at 400) until the OTSI stage, "
          "so it is not a drive nonlinearity. Seed-scale test at v_q/v_te = 0.07, Nx 32/Nn 64, T = 400 (seeds multiplied by s):", "",
          "| seed scale s | D(100) | D(200) | D(300) | D(400) | ledger |", "|---|---|---|---|---|---|"]
    for k, lab in (("s0", "0 (unseeded)"), ("s0.1", "0.1"), ("s1", "1 (stage-1 B run)")):
        d = fu["D"][k]
        L.append(f"| {lab} | {d['100']:.2e} | {d['200']:.2e} | {d['300']:.2e} | {d['400']:.2e} | {fu['ledger'][k]:.1e} |")
    L += ["", f"F1 (deficit scales as the seed amplitude squared: D(s=0.1)/D(s=1) in [0.005, 0.02] and the unseeded control below 10% of D(s=0.1)): "
          f"ratio {fu['F1']['ratio_s0.1_over_s1']:.4f}, unseeded/D(s=0.1) {fu['F1']['unseeded_over_s0.1']:.1e}: **{'pass' if fu['F1']['pass'] else 'FAIL'}**. "
          f"F2 (the deficit is a property of the seedless homogeneous Hermite system): unseeded/D(s=1) = {fu['F2']['unseeded_over_s1']:.1e}: **{'pass' if fu['F2']['pass'] else 'FAIL'}** "
          f"(so that hypothesis is rejected). F3 ledger: {'pass' if fu['F3']['pass'] else 'FAIL'}.", "",
          "**Reading.** (1) At the strongest certified drive (v_q/v_te = 0.07, window 791.5, 10% pairs over four resolutions plus a closure-strength pair and a grid "
          "cross-check) the energy transfer follows the linear resonant law: W_ext is within 0.68% of it over the whole window, the four refinement members agree on that deficit "
          "to 2e-4, and the frequency shift of the mean field is below 7e-6 omega_pe (the declared resolution threshold was 1e-3 omega_L). No saturation or detuning is resolved: "
          "the upper bound on the nonlinear correction to W_ext is 0.7%. (2) The 0.7% is not a drive effect. It is identical at v_q/v_te = 0.03-0.07, grows as t^2 and is "
          "proportional to the square of the seed amplitude (unseeded control 2e-9 at t = 400), i.e. it is the back-reaction of the quiver-advected seed modulation on the mean "
          "field (the <dn dE> term of the mean-field identity), set by the fluctuation level that was imposed, not by the pump. A physical fluctuation level other than 5e-4 would rescale it. "
          "(3) The reason the transfer stays linear is energetic: 96-97% of W_ext sits in coherent k = 0 electron quiver energy (dK_e = 370 of 381 n T_e V); the thermal gain of the "
          "electrons is 0.14-0.24 n T_e V (about 0.04-0.06% of W_ext) and the ions receive 0.05%, and the fluctuating-field amplitude (sum of |E_k|^2 over k >= 1) reaches ~2000 E_0^2 only when the OTSI "
          "turns on at t of about 780-860 (onset definition in the gates), after which the window ends by positivity loss. Within the window the instability has not yet extracted a "
          "resolvable fraction of the work. (4) Finite reservoir at the same drive (u = 1, 10): U_D follows the linear coupled-oscillator prediction to 0.23% / 0.81% of U_D0, "
          "W_D follows the matched prescribed drive until t_split = 50.5 / 72, equal to the linear t_split; the reservoir gives up 47% of its energy (u = 1, W_D/U_D0 at the window end) and for u = 10 exchanges "
          "energy through beats (first minimum of U_D at 4e-4 U_D0, linear 4e-6; W_D/U_D0 = 1.6% at the window end against 0.7% linear), with W_D within 1.0% / 2.7% of the linear prediction while it is above 20% of its running maximum. (5) Finite coherence (tau omega_pe = 100, 300; six realizations "
          "each): R = <W_kin>/<W_lin> = 1.0050 +- 0.0058 and 1.0000 +- 0.0038; the bound on the ensemble nonlinear correction is 1.7% and 0.8%, and the single refined "
          "realization agrees (+1.8%, -0.03%). Individual realizations deviate more (up to 4.7% of the peak linear W_ext at tau = 100, 1.5% at tau = 300), with ratios on both sides of 1. (6) Scope: one box (L = 40), "
          "one ion mass, T_i = T_e, drives up to 0.07, windows set by the AW truncation instability of the Hermite basis (not by the physical OTSI), fluctuation seeds of fixed "
          "amplitude, Lorentzian line only, reservoir and coherence not combined in one run. Beyond the window (> 790) the result says nothing; at v_q/v_te = 0.1 no window "
          "beyond ~760 exists (rclosure).", ""]
    return "\n".join(L)


def r3_section():
    """R3 first step: dark-photon conversion at a smooth density ramp, with and without B0; cold reference vs Dark-SPECTRAX."""
    S = json.loads((root / "studies/r3/summary.json").read_text())
    F = json.loads((root / "studies/r3/followup_summary.json").read_text())
    f3 = lambda x: f"{x:.3f}"
    L = ["## R3 inhomogeneous and magnetized conversion: a density ramp and B0 (`studies/r3_gates.py`, `studies/r3_followup_gates.py`)", "",
         "**Question.** How do a smooth density variation n(x) and a magnetic field B0 change dark-photon to plasma conversion and escape? "
         "**Set-up** (declared in `studies/r3_gates.py` before any run; two pre-run corrections of the reference integrator are recorded there): "
         "periodic box 640, immobile ions, electron density n(x) = omega_pe^2(x) rising from 0.02 to 2.98 through a tanh ramp of width w "
         "(the resonance omega_pe = Omega_D = 1 sits at n = 1, L_c = n_c/n' = 0.77 w), a plateau of 260 and a down ramp the packet does not reach; "
         "incident right-moving free Proca packet (k0 = 1, omega0 = sqrt 2, group velocity 0.71, sigma_k = 0.04, peak |E_D| = 1e-4), polarization x "
         "(longitudinal), y or z, eta = 0.15 and 0.3, T = 330, |Omega_c| = 0.3 with B0 along (cos th, 0, sin th), th = 0, 45, 90 deg. "
         "**Reference** (`studies/r3_ref.py`, NumPy/SciPy only): cold electron fluid + Maxwell + canonical Proca in 1D, pseudo-spectral in x (4096 points), "
         "DOP853 (rtol 1e-12), same discrete spectrum as the kinetic runs. **Kinetic**: Dark-SPECTRAX, v_te = 0.01, Nn = Nm = Np = 4, Nx 512 (and Nx 768, Nn 6 refinement), "
         "starting from an exact kinetic equilibrium: Maxwellian n(x) in the static field E0 = -v_te^2 n'/n, which is solved on the discrete C_1 row (residual "
         f"{S['K1']['balance_residual_max']:.0e} against a base {1e1:.0f}) with the dark near field of the residual charge; the equilibrium stays put to "
         f"{S['K1']['max_dn_over_nhi']:.0e} in n/n_hi (K1). Channels are energy fractions of the packet at t = T: reflected/transmitted dark and photon "
         "(cold eigen-branch projection of the fields in uniform windows, energy inner product), energy stored at the ramp, and (kinetic) absorbed thermal energy; "
         "fluxes S = E x B and S_D = E_D x B_D + Omega_D^2 phi_D A_D at planes xu -/+ 7w. "
         "Deviation from the declared set-up: the reflected photon (speed 0.9) reaches x = 0 and wraps around before T, so 1.0-1.1% (w = 4) of the energy "
         "sits in the declared 'beyond' window; it is reported as the 'wrap-corrected' reflected photon (left-moving branches in x > xd + 6w) and the declared "
         "left-window value is also tabulated.", ""]
    L += ["![transverse conversion](_static/r3/transverse_conversion.png)", ""]
    r1_, r2_, r3_ = S["R1"], S["R2"], S["R3"]
    L += ["### Declared gates", "",
          "| Gate | Rule | Result |", "|---|---|---|",
          f"| R1 | cold energy drift <= 1e-8 | {'PASS' if r1_['pass'] else 'FAIL'}: max {r1_['max_energy_drift']:.1e} over {r1_['n_cases']} cases |",
          f"| R2 | N 4096 -> 6144, rtol 1e-12 -> 1e-13: channels change <= 1e-4 (3 cases) | {'PASS' if r2_['pass'] else 'FAIL'}: max {max(x['max_channel_change'] for x in r2_['cases']):.1e} |",
          f"| R3 | flux budget with the phi_D A_D term <= 1e-4 of E_in; without it > 1e-2 for B0 = 0 longitudinal | "
          f"with term: {'PASS' if r3_['pass_with'] else 'FAIL'} (max {r3_['max_residual_with_phiA']:.1e}; every transverse-decoupled case <= 3.5e-6, the 14 failures are the cases with energy at a cold resonance layer, where the layer wavenumber exceeds the grid); "
          f"omission: {'PASS' if r3_['pass_omission'] else 'FAIL'} (smallest omission residual {min(r3_['omission_residuals_B0_x'].values()):.2f} of E_in) |"]
    r4 = S["R4"]
    bad = [x for x in r4["rows"] if x["valid"] and not x["pass"]]
    bad_txt = ", ".join(f"{x['case']} {x['max_abs_dev']:.3f}" for x in bad)
    L += [f"| R4 | Landau-Zener law within 0.03 in V (k Lc >= 3, eta f n_c <= 0.25 kbar^2) | FAIL: {len(bad)} of {r4['n_valid']} cases in V ({bad_txt}); see below |"]
    r5 = S["R5"]
    L += [f"| R5 | longitudinal: stored = WKB tunnelling, refl_dark = 1 - T (0.10) | FAIL ({sum(1 for x in r5['rows'] if x['gated'] and x['pass'])} of {sum(1 for x in r5['rows'] if x['gated'])} gated cases): the WKB number is the TRANSMITTED dark fraction, not the absorbed one (declared prediction wrong) |"]
    k2, k3 = S["K2"], S["K3"]
    L += [f"| K1 | background-only run | PASS: |dn|/n_hi {S['K1']['max_dn_over_nhi']:.0e}, fields {S['K1']['max_field_final']:.0e} |",
          f"| K2 | success and ledger <= 1e-6 E_in | {'PASS' if k2['pass'] else 'FAIL'}: {k2['n_runs']} runs, ledger <= {k2['max_ledger_rel_E_in']:.0e} E_in |",
          f"| K3 | closure <= 2e-3, flux budget <= 5e-3 | closure max {k3['max_closure']:.1e} PASS; flux residual: transverse-decoupled cases up to 1.5e-2 at Nx 512 (eta 0.3, w 2: FAIL; 3.9e-3 at w 4 and 3e-4 at w 8: pass) and <= 3.4e-4 at Nx 768 / Nn 6 (pass); resonance cases 0.05-0.2: FAIL |"]
    k4 = [r for r in S["kin_rows"] if "K4_pass" in r]
    k4_512 = [r for r in k4 if r["Nx"] == 512]
    dec = [r for r in k4 if (r["pol"] == "y" and r["theta"] in (None, 0)) or (r["pol"] == "z" and r["theta"] == 90)]
    L += [f"| K4 | kinetic vs cold twin (band-limited profile of the same Nx): 0.02 / 0.03 / longitudinal 0.05 | Nx 512: {sum(1 for r in k4_512 if r['K4_pass'])} of {len(k4_512)} pass. All transverse-decoupled cases (B0 = 0 y, th 0 y, th 90 z: {len(dec)} runs) agree to {max(r['K4_max'] for r in dec):.0e}; "
          "longitudinal and oblique-B0 cases FAIL at Nx 512/768 (resolution, next paragraph) |",
          f"| K5 | refinement pair (Nx 768, Nn 6) changes <= 0.005 | {sum(1 for x in S['K5'] if x['pass'])} of {len(S['K5'])} pass; not converged: " +
          ", ".join(x['case'] for x in S['K5'] if not x['pass']) + " |",
          f"| K6 | amplitude x 3 | {'PASS' if S['K6']['pass'] else 'FAIL'}: change {S['K6']['max_channel_change']:.0e} (linear) |", ""]
    L += ["### Transverse conversion and the Landau-Zener law", "",
          "Cold reference (kinetic Nx 512 agrees to <= 4e-4 at v_te = 0.01), B0 = 0, polarization y; T = transmitted dark, Rg and Rd = reflected photon (wrap-corrected) and "
          "dark. LZ: P_d = exp(-pi g^2/(kbar |dDelta/dx|)), g = eta n_c, n_c = Omega_D^2/(1 - eta^2), kbar^2 = omega^2 - n_c, T = P_d, Rg = P_d(1 - P_d), Rd = (1 - P_d)^2, averaged over the packet spectrum. "
          "Refined law (follow-up G, declared before it was computed): the Dykhne complex-turning-point exponent of the exact local eigenwavenumbers.", "",
          "| eta | w | L_c k | s = eta n_c/kbar^2 | T ref | Rg ref | Rd ref | T LZ | Rg LZ | Rd LZ | max dev LZ | max dev refined |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(F["FU4"]["rows"], key=lambda r: (r["eta"], r["w"])):
        if r["theta"] is not None:
            continue
        L.append(f"| {r['eta']:g} | {r['w']:g} | {r['kbarLc']:.1f} | {r['eta_fnc_over_kbar2']:.2f} | {f3(r['ref']['trans_dark'])} | {f3(r['ref']['refl_photon'])} | {f3(r['ref']['refl_dark'])} | "
                 f"{f3(r['simple']['trans_dark'])} | {f3(r['simple']['refl_photon'])} | {f3(r['simple']['refl_dark'])} | {r['max_dev_simple']:.3f} | {r['max_dev_refined']:.3f} |")
    L += ["", f"**Domain of validity.** The simple law is accurate where s = eta n_c/kbar^2 <~ 0.1 (deviation <= 0.01, eta <= 0.1 at kbar ~ 1); it systematically "
          "overestimates the exponent, ln T_ref/ln P_LZ ~ 1 - 4.5 s^2 (0.99, 0.95, 0.89, 0.80, 0.61 at eta = 0.05, 0.1, 0.15, 0.2, 0.3; "
          "![exponent ratio](_static/r3/lz_exponent_ratio.png)) independently of L_c for L_c kbar >= 3. The declared predicate V (s <= 0.25) was too permissive: "
          f"R4 fails for eta 0.15, w 8 and 12. The refined exponent (FU4: {'PASS' if F['FU4']['pass'] else 'FAIL'}) reproduces all three channels to "
          f"{F['FU4']['max_dev_refined_gated']:.3f} (gated: k Lc >= 3) and {F['FU4']['max_dev_refined_all']:.3f} (all cases, worst at w = 2 where the ramp is no longer slow), "
          "including eta = 0.3, where the simple law is off by 0.25. The relative exponent error scales with eta^2/kbar^2, so for non-relativistic dark photons (kbar = v Omega_D) the simple formula needs "
          "eta << 0.3 v; nothing here tests v << 1 (needs a longer box), and the 4% bandwidth is included in the average. "
          "The reflected photon returns through the resonance once (turning point at n = omega^2 above the resonance), so the three channels follow T, T(1 - T) and (1 - T)^2 with no interference term.", ""]
    L += ["### Longitudinal polarization: tunnelling to a resonance layer", "",
          "![longitudinal](_static/r3/longitudinal_conversion.png)", "",
          "The cold longitudinal dispersion k^2 = omega^2 - Omega_D^2 (omega^2 - n)/(omega^2 - (1 + eta^2) n) (checked against the 19x19 cold matrix to 2e-15) has an evanescent window between "
          "n_t = omega^2 k^2/(omega^2 (1 + eta^2) - Omega_D^2) and the cold resonance n_p = omega^2/(1 + eta^2). The reference shows that the WKB tunnelling factor exp(-2 int kappa dx) is the "
          "**transmitted dark fraction** (eta 0.3, w 4: T_ref 0.274 against 0.274; eta 0.15, w 8: 0.451 against 0.450; |difference| <= 1.4e-3 for w >= 3, and 0.014 / 0.09 at w = 2 where k Lc < 3); the remainder splits into reflected dark and energy left in the layer (up to 28%). "
          "That layer energy phase-mixes with wavenumber k(t) = |d omega_pe/dx| t, about 40 at T = 330 for w = 4, which the first-step grids (k <= 1.67 at Nx 512) cannot hold. "
          "Follow-up F (declared before it was run) refines Nx to 3072 (k <= 10): "]
    for label in ("longitudinal_B0", "th45_y"):
        lad = F["ladder"][label]
        k, c = lad["rows"][-1]["kin"], lad["rows"][-1]["cold"]
        L.append(f"{label.replace('_', ' ')}: FU1 {'PASS' if lad['FU1']['converged'] else 'FAIL'} (change {lad['FU1']['max']:.1e} between Nx 2048 and 3072), FU2 {'PASS' if lad['FU2']['pass'] else 'FAIL'} "
                 f"(refl_dark {k['refl_dark']:.4f} / {c['refl_dark']:.4f}, trans_dark {k['trans_dark']:.4f} / {c['trans_dark']:.4f}, layer remnant {k['remnant']:.3f} / {c['remnant']:.3f}, kinetic / cold); ")
    th3 = F["ladder"]["longitudinal_B0"]["FU3"]
    L += [f"thermal dependence at Nx 3072 (v_te 0.005 / 0.01 / 0.02): longitudinal remnant {th3['0.005']['remnant']:.3f} / {th3['0.01']['remnant']:.3f} / {th3['0.02']['remnant']:.3f}, "
             f"transmitted {th3['0.005']['trans_dark']:.4f} / {th3['0.01']['trans_dark']:.4f} / {th3['0.02']['trans_dark']:.4f}: no resolved kinetic correction to the propagating channels at v_te <= 0.02 over T = 330; "
             "thermal absorption of the layer energy is below the 1% noise of the thermal-energy diagnostic (the layer wavenumber 40 times v_te/omega = 0.3 would start Landau damping only near the end). "
             "Accounting at Nx >= 1536 is limited by the fixed 4096-point analysis grid (closure to 4e-3, absorbed noise +-1%).", ""]
    L += ["### Effect of B0 (cold reference, eta 0.3, w 4; kinetic agrees where converged)", "", "![B0](_static/r3/b0_effect.png)", "",
          "| B0 | pol | T dark | Rd | Rg (wrap-corr.) | stored | change of T vs B0 = 0 |", "|---|---|---|---|---|---|---|"]
    for r in sorted(S["R6"], key=lambda r: (r["theta"], r["pol"])):
        if r["w"] != 4.0:
            continue
        c = r["channels"]
        L.append(f"| th = {r['theta']} | {r['pol']} | {f3(c['trans_dark'])} | {f3(r['wrap']['refl_dark_wrapcorr'])} | {f3(r['wrap']['refl_photon_wrapcorr'])} | {f3(c['stored'])} | {r['delta_vs_B0']['trans_dark']:+.3f} |")
    L += ["", "- B0 along the gradient (th = 0, R and L parts with n -> n omega/(omega -/+ Omega_c)): no change within 0.002 (the shifts of the two circular parts cancel in the packet average) and "
          "the refined circular law (n -> f n in the two-level system) reproduces it (deviation 0.003); B0 perpendicular with the field along B0 (th = 90, pol z, ordinary mode): identical to B0 = 0.",
          "- The transmitted dark fraction is insensitive to B0 at w >= 4 (<= 0.004), because it is set by the tunnelling/LZ exponent, which B0 does not change at |Omega_c| = 0.3; at w = 2 it moves by 0.02-0.06.",
          "- What B0 changes is the fate of the rest: with oblique B0 the longitudinal resonance energy that was trapped at the layer (0.20 of the packet at B0 = 0) escapes as reflected ordinary photons "
          "(th 45, 90 x: stored 0.04-0.05, reflected photon 0.26-0.27), and a transversely polarized packet leaves 0.05-0.10 at the layer (hybrid-type resonance) while the reflected photon fraction drops "
          "(0.25 -> 0.13 at th 45 y, 0.05 at th 90 y; X mode).", "",
          "### Scope and limits", "",
          "First step: linear amplitude (K6), one packet (k0 = 1, 4% bandwidth), immobile ions, v_te = 0.01 T <= 330, eta 0.15-0.3 (large against the physical eta), |Omega_c| = 0.3, Hermite order 4 (no collisions, no closure). "
          "Not done: non-relativistic packets (k0 << Omega_D) where the photon turning point approaches the resonance, a frequency-domain boundary-value solution of the Beadle-Caputo-Ellis kind "
          "(arXiv:2405.13882, abstract read only), thermal absorption at the longitudinal layer (needs Nx >~ 12000 and Nn >> 4), mobile ions, 2D geometry, nonlinear amplitudes. "
          "The cold reference is exact in the linear cold limit, so for the converged cases the kinetic runs add no new physics at v_te = 0.01; they certify the reference and the equilibrium construction. "
          "Kinetic records have the plane-flux time series stripped (`--keep-series` regenerates it).", ""]
    return "\n".join(L)


def expint_section():
    sm = json.loads((root / "studies/expint/summary.json").read_text())
    cost = json.loads((root / "studies/expint/cost.json").read_text())
    L = ["## Exact integration of the diagonal closure damping (Lawson wrapper) (`studies/expint_gates.py`)", "",
         "SPECTRAX `LawsonSolver` (integration/dark-baseline, PR #81) wraps Dopri8: with `w = exp(r (t - t0)) y` and the rate "
         "`r = nu_s s(n)` frozen at the step start, the diagonal closure is integrated exactly and everything else (streaming, force, "
         "fields, the variation of nu over the step) stays explicit; the split is an identity for any `r`; the order of Dopri8 is retained for h -> 0 at fixed rate (Lawson 1967, SIAM J. Numer. Anal. 4, 372-380; the step-start rate is Lawson's re-evaluated-A form), with no stiff-order claim when r h >> 1, because the stiff order of Lawson methods drops (to 1 for Lawson-Runge; Hochbruck and Ostermann 2010, Acta Numerica 19, 209-286, Thm 2.17, Ex. 2.32, Table 2.2); the empirical support is measured agreement with plain Dopri8 records (5e-10 in H05 c=64) (derivation in `spectrax/_lawson.py`). Opt-in: `darkspectrax.run(..., integrating_factor=True)` (constant `nu col`) or "
         "`integrating_factor=rate_fn` with `rate_fn` from `darkspectrax.field_closure(model, c)`; default off is bitwise the earlier path. Fields, dark "
         "fields, work ledger and pump-frame basis have rate zero, rows n <= 2 are untouched. Streaming is not exponentiated. Steps are clipped onto the "
         "saves (the Lawson dense output is accurate only for r h <~ 1). Gates, the platform note and the CPU controls are in the docstring of "
         "`studies/expint_gates.py`, committed before the Lawson runs. The Lawson runs and the plain CPU controls ran on the office CPU (GPU 0 was "
         "busy); the baseline is the committed GPU stiffdiag record.", "",
         "| case | T_cmp | steps plain / Lawson | step ratio (t <= 500) | same-CPU wall ratio (t <= 500) | max rel diff dK_e / W_ext (t <= 500) | ledger | first non-positive t plain / Lawson | window end plain / Lawson | D2 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in ("4", "64"):
        r = sm["cases"][c]
        L.append(f"| c = {c} | {r['T_cmp']:g} | {r['steps_base']:.0f} / {r['steps_lawson']:.0f} | {r['steps_ratio_500']:.2f} | "
                 f"{r['wall_ratio_same_cpu_500']:.2f} | {r['A']['dKe']:.1e} / {r['A']['Wx']:.1e} | {r['L']:.1e} | "
                 f"{r['t_pos_base']:g} / {r['t_pos_lawson']:g} | {r['t_end_base']:g} / {r['t_end_lawson']:g} | {'pass' if r['D2'] else 'FAIL'} |")
    b = sm["b10"]
    L += ["", f"Gates (declared): D2 steps ratio >= 3 at T_cmp per case (c = 4: {sm['cases']['4']['steps_ratio']:.2f} at 560 **FAIL**; c = 64: "
          f"{sm['cases']['64']['steps_ratio']:.2f} at 740 pass), so D2 as a whole (both cases) **fails**; accuracy A <= 1e-6 on dK_e and W_ext to t = 500 passes "
          f"for both (the CPU-plain control differs from the GPU record by {sm['cases']['64']['A_ungated_cpu_plain_vs_gpu_record']['dKe']:.1e}, "
          f"so the platform alone explains most of it); ledger <= 1e-9 passes; positivity not earlier passes (c = 4: 571 vs 568, c = 64: 760 both). "
          f"Per-step cost (c = 64 from t = 300, 4 units, CPU): plain {cost['plain']['steps']} steps, Lawson {cost['lawson']['steps']} steps, "
          f"cost per step ratio {cost['cost_per_step_ratio']:.2f}.", "",
          f"**Ordinary B10** (`M_mi100_ord_j10`, constant nu = 100, Nn 32): {b['steps_record']} -> {b['steps_lawson']} steps ({b['steps_ratio']:.2f}x, B1 pass by steps); "
          f"wall {b['wall_rerun']:.0f} s -> {b['wall_lawson']:.0f} s on the same machine ({b['wall_ratio']:.2f}x: the per-step cost is about 3x higher on this tiny state); "
          f"gamma_fit {b['gamma_fit_lawson']:.9g} vs {b['gamma_fit_record']:.9g} (relative {b['gamma_rel_diff']:.1e}, window identical; B2 pass); "
          f"**B3 FAIL**: ledger / U_field0 = {b['ledger_over_U0']:.1e} against the declared 1e-9 (plain 8.4e-12). The plain value is a by-product of its "
          "stability-limited steps being far smaller than accuracy needs; the Lawson steps are accuracy-limited, so the quadratic field-energy "
          "ledger closes only at the solver tolerance (absolute 1.7e-12 with atol 1e-12). The gate was too strict for an accuracy-limited step and is not amended.", "",
          "**Reading.** (1) Where the closure is the stiff mode (c = 64, t >= 100) the exact treatment cuts the steps 5.0x to t = 500 and 4.3x to t = 740 "
          "(the counterfactual of stiffdiag predicted 6-9x for the last unit; the realized factor is lower because streaming and the force now set "
          "the step) at the same per-step cost, so a same-CPU wall ratio of 5.3x including compile, with dK_e and W_ext equal to the plain record to 5e-10. "
          "(2) Where it is not (c = 4 to t = 540; nu_e <= 100, dt/h_stab <= 0.5) there is no gain (0.96x to 500) and the clipping onto the saves costs a few steps. "
          "(3) The window does not move: c = 64 still ends at 760 by positivity loss and c = 4 at 579.5 (plain 578) by the step budget inside the AW instability, as the design "
          "note predicted; the instability is not an integrator problem. (4) Streaming Lawson (Gauss-Hermite nodal transform) was not built.", ""]
    return "\n".join(L)


def stream_section():
    sm = json.loads((root / "studies/stream/summary.json").read_text())
    cost = json.loads((root / "studies/stream/cost.json").read_text())
    b = sm["b10"]
    L = ["## Exact Hermite streaming in the integrating factor (`studies/streaming_gates.py`)", "",
         "SPECTRAX PR #83 (stacked on the Lawson wrapper #82) integrates the free-streaming operator exactly per Fourier mode: one real eigh of the Hermite "
         "tridiagonal (Gauss-Hermite nodal transform, shared by all k, species, widths and drifts; widths and drifts from the state basis frozen at the "
         "step start) and the Strang product `A(h/2) B(h) A(h/2)` with the field-scaled closure rate (`darkspectrax.run(..., integrating_factor=darkspectrax.streaming(rate_fn))`). "
         "The composition is an exact identity (the split `w' = Phi^-1 (f(Phi w) - Phi' w)` has no splitting error); what is left explicit is of the order "
         "`(h |G|)^2 |r|`, which the gates were designed to test. Gates were committed in `studies/streaming_gates.py` before any run; runs and the same-session "
         "closure-only control ran on a laptop CPU (2 threads), not on the office machine of the expint records (so wall times are comparable within this section only).", "",
         "| case | steps closure-only (`_li2`) / streaming (`_ls`) to 500 | wall s closure-only / streaming | dK_e / W_ext vs plain CPU | ledger (ls) | status ls | S1 | S2 | S3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in ("4", "64"):
        r = sm["cases"][c]
        st = f"{r['steps_ls']:.0f}" if r['steps_ls'] else f"not reached (ends t = {r['t_end_ls']:g}, step budget)"
        wl = f"{r['wall_ls']:.0f}" if r['wall_ls'] else "-"
        L.append(f"| c = {c} | {r['steps_li2']:.0f} / {st} | {r['wall_li2']:.0f} / {wl} | {r['A_vs_plain_cpu']['dKe']:.1e} / {r['A_vs_plain_cpu']['Wx']:.1e} "
                 f"(common saves to {r['A_vs_plain_cpu']['t_common_max']:g}) | {r['ledger_ls']:.1e} | {r['status_ls']} | "
                 f"{'pass' if r['S1'] else 'FAIL'} | {'pass' if r['S2'] else 'FAIL'} | {'pass' if r['S3'] else 'FAIL'} |")
    L += ["", f"**Verdict S4: streaming does NOT earn its place in this regime** (S1 fails both cases, S3 fails c = 4). c = 64: the same {sm['cases']['64']['steps_li2']:.0f} vs "
          f"{sm['cases']['64']['steps_ls']:.0f} steps (ratio {sm['cases']['64']['steps_ratio_li2_over_ls']:.3f}), so streaming was not what limited the step (as stiffdiag found: the stiff mode is the "
          f"closure), and the {cost['cost_per_step_ratio_streaming_over_lawson']:.2f}x higher cost per step (c = 64 from t = 300, 4 units: closure-only {cost['lawson']['steps']} steps at "
          f"{1e3*cost['lawson']['time_per_step']:.2f} ms, streaming {cost['streaming']['steps']} steps at {1e3*cost['streaming']['time_per_step']:.2f} ms, plain {cost['plain']['steps']} steps at "
          f"{1e3*cost['plain']['time_per_step']:.2f} ms) makes it {1/sm['cases']['64']['wall_ratio_li2_over_ls']:.1f}x slower in wall time. c = 4: identical steps to t = 400, then the run "
          f"ends at t = {sm['cases']['4']['t_end_ls']:g} on the step budget (201927 steps in a 100-unit chunk against {5444} for the closure-only control): with a large closure rate and "
          "a large `h k a sqrt(2 Nn)` the Strang leftover `O((h|G|)^2 |r|)` is stiff in the transformed variable and the step collapses. "
          f"Ordinary B10 (`M_mi100_ord_j10`, nu = 100): closure-only {b['steps_li2']} steps; streaming reached the 400000-step budget ({b['wall_ls']:.0f} s against {b['wall_li2']:.0f} s), "
          f"B1s, B2s and B3s fail (the closure-only ledger/U0 is {b['ledger_over_U0_li2']:.1e}, streaming {b['ledger_over_U0_ls']:.1e}). Accuracy where it finished is fine "
          "(dK_e, W_ext within 3.4e-10 and 2.7e-11 of the plain CPU record at c = 64). The streaming factor itself is exact (unit tests: analytic Hermite front, generator equals the "
          "streaming part of the RHS in fixed and pump bases, unitarity, order, gradient); it helps where the streaming stability bound is the limit and nothing stiff is present "
          "(SPECTRAX default two-stream, Nx 128 Nn 128, tolerance 1e-8: 1049 -> 29 steps, 12x wall), which is not the H05 / B10 regime.", ""]
    return "\n".join(L)


def gpures_section():
    g = json.loads((root / "studies/gpures/summary.json").read_text())
    c8 = json.loads((root / "studies/gpures/c08.json").read_text())
    L = ["## H05 resolution extension on GPU: Nn {128, 256} x Nx {64, 128} at v_q/v_te 0.1 and 0.05 (`studies/gpures_*.py`)", "",
         "RTX A4000, float64, jax 0.11.2 CUDA, parent 3de6b2b; same drivers as `studies/nxconv_run.py` (pump frame, remap, "
         "#66 closure c = 4, noise_floor 1e-14, rtol 1e-10, atol 1e-14, T = 1000, 100-unit checkpointed chunks). Gates were "
         "committed in `studies/gpures_gates.py` before any run (G1 pairs within 10% on dK_e and W_ext to T = 1000; G2 K_s "
         "and k=0 T_x positive; G3 ledger defect <= 1e-6 of max |W_ext|; G4 solver success to T).", "",
         f"**C08 (GPU vs CPU, same venv, v_q 0.1, Nx 32, Nn 64, c = 4, T = 100):** max relative difference "
         f"{c8['max_rel']:.1e} (dK_e {c8['dK_e']:.1e}, W_ext {c8['W_ext']:.1e}, T_x {c8['Tx']:.1e}, |E_k|^2 k <= 8 "
         f"{c8['Ek2_k<=8']:.1e}); declared gate 1e-6 {'passed' if c8['gate_1e-6'] else 'FAILED'}. Adaptive step counts "
         f"differ ({c8['steps'][0]} CPU, {c8['steps'][1]} GPU), so this is agreement of two adaptive trajectories, not "
         f"bitwise identity. Integration time {c8['run_s'][0]:.1f} s CPU (4 threads) vs {c8['run_s'][1]:.1f} s GPU.", "",
         "| run | t reached | status | first K_s or T_x <= 0 | G3 ledger defect / max W_ext | steps (rejected) | GPU integration s | wall s incl. diagnostics |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in g["runs"].items():
        L.append(f"| {k.replace('_c4_nf1e-14_gpu', '')} | {v['t_reached']:g} | {'ended by the step budget' if v['status'] == 'failure' else v['status']} | "
                 f"{v['t_pos'] if v['t_pos'] is not None else 'never'} | {v['ledger_rel']:.1e} | {v['steps']} ({v['rejected']}) | "
                 f"{v['run_s']:.0f} | {v['wall_s']:.0f} |")
    L += ["", "| pair | dK_e | W_ext | G1 (to 1000) |", "|---|---|---|---|"]
    for k, v in g["pairs"].items():
        L.append(f"| {k} | {v['dK_e']} | {v['W_ext']} | {'pass' if v['G1'] else 'FAIL'} |")
    L += ["", "**Verdict: both v_q values FAIL G1 and G4; no new certified window.** "
          + "; ".join(f"v_q {k[2:]}: every pair agrees within 10% to the end of its shorter run, certified window {v['certified_window']:g}"
                      for k, v in g["verdict"].items())
          + ". Every run ends at the adaptive step budget (200000 steps per 100-unit chunk) inside the AW truncation "
          "instability, and raising Nn or Nx makes it end sooner (v_q 0.1: t = 595, 383, 453, 328 for Nx/Nn 64/128, 64/256, "
          "128/128, 128/256; v_q 0.05: 752, 555, 671, 440). The 10% pairs never separate before the shorter run stops, so "
          "this is a failure of the resolution route, not a disagreement: the existing windows (0.03 to ~960-1000; 0.1 to "
          "~600) stand, and v_q 0.05 is certified only to ~440-555 (Nn 256 pairs) or ~667 (Nx 64 vs 128 at Nn 128, "
          "which separate there). G2 fails for three runs (first non-positive K_s or "
          "k=0 T_x at 581 for v_q 0.1 Nx 64 Nn 128, 741 and 662.5 for v_q 0.05 Nx 64 and 128 at Nn 128); G3 holds in every run (<= 1.7e-7)."
          " Cost: one GPU run is 7-15 min of integration (up to 20 min wall including diagnostics), the 8 runs 1.3 h of GPU integration "
          "(one chunk was interrupted by a SIGTERM and resumed from its checkpoint), mostly spent in the unstable tail; the GPU is ~4x faster than 4 CPU threads on the C08 case. Going "
          "further needs a different basis (the stable-Galerkin route of `docs/design/l2-stable-basis.md`, option B), not "
          "more Hermite orders or Fourier modes."]
    return "\n".join(L)


def first_above(x, t, th):
    i = np.nonzero(np.abs(x) > th)[0]
    return f"{t[i[0]]:g}" if i.size else "none"


def gridref_lines(d):
    """Batch 11: wide-box (vmax 48/64 v_te) grid reference for H05 v_q/v_te = 0.1."""
    G = root / "studies/nxconv/grid"
    L = ["", "**Wide-box grid reference at v_q/v_te = 0.1 (batch 11, `studies/nxconv/batch11.sh`, records at "
         "`7a4582a`).** Reference = Nx 64, Nv 16384, vmax 64 v_te, dt 0.02, masked spectral, to t = 1000. Pairs, "
         "same 10% rule:", "",
         "| grid pair (reference vmax 64, Nv 16384, Nx 64, dt 0.02) | dK_e | W_ext | field spectrum |",
         "|---|---|---|---|"]
    for k, v in d["grid_pairs"].items():
        if k.startswith("gridref"):
            L.append(f"| {k[len('gridref vq0.1 '):]} | {v['dK_e']} | {v['W_ext']} | "
                     f"{next(x for kk, x in v.items() if kk.startswith('Ek2'))} |")
    L += ["", "| wide-box grid run | first f < -1e-3 max f | first edge > 1e-6 / 1e-3 | "
          "first energy defect / W > 1e-6 / 1e-3 | Hermite c = 4 vs grid: dK_e, W_ext, spectrum |",
          "|---|---|---|---|---|"]
    for nx, nv, vm in ((64, 8192, 32), (64, 12288, 48), (64, 8192, 64), (64, 16384, 64), (32, 16384, 64)):
        tag = f"g_vq0.1_Nx{nx}_Nv{nv}_dt0.02_osc_v{vm}_spectral_xmask23"
        z = np.load(G / f"{tag}.npz")
        h = d["hermite_vs_grid"][tag]
        L.append(f"| Nx {nx}, Nv {nv}, vmax {vm} | {d['grid'][tag]['t_fneg_1e-3']} | "
                 f"{first_above(z['edge'], z['t'], 1e-6)} / {first_above(z['edge'], z['t'], 1e-3)} | "
                 f"{first_above(z['energy_defect'], z['t'], 1e-6)} / {first_above(z['energy_defect'], z['t'], 1e-3)} | "
                 f"{h['dK_e']}, {h['W_ext']}, {next(x for kk, x in h.items() if kk.startswith('Ek2'))} |")
    L += ["", "Findings. (1) Widening the box from 32 to 64 v_te moves the first edge contact only from t ~ 700 to "
          "~755: the tail is accelerated past 64 v_te within ~50 time units, so no affordable box reaches t = 1000. "
          "(2) The first negativity (t = 645.5 at Nx 64, 498 at Nx 32) does not depend on vmax, Nv or (to t = 200) dt; "
          "it is set by the x resolution. The earlier attribution of the t ~ 650 limit to the box edge was wrong: at "
          "vmax 32 the edge is still below 1e-6 at t = 700. (3) The wide-box grids agree with each other in dK_e to "
          "t ~ 790-800 and in W_ext to 1000; Hermite c = 4 agrees with every grid in dK_e and W_ext to the end of the "
          "Hermite run (760) and in the field spectrum to 555 (k4), as at vmax 32. **Certified windows at v_q/v_te = "
          "0.1:** grid reference dK_e and W_ext to t ~ 750 (edge contact and energy-defect onset; the later "
          "agreement is not certified), field spectrum to ~640 (Nx 32 vs 64 at k4); Hermite c = 4 matches it in dK_e "
          "and W_ext to t >= 760 and in the field spectrum to 555. Past ~750 this grid cannot certify v_q 0.1; the "
          "Hermite-pair limit near 600 is a Hermite truncation effect, since the grid is converged there."]
    return L


def otsi_section():
    """The physical finite-k instability of H05 is the oscillating two-stream instability (studies/instab_otsi.py)."""
    sc = otsi["scan"]
    un = [r for r in sc if r["gamma_floquet"] > 1e-8]
    dmax = max(abs(r["gamma_floquet"] - r["gamma_nishikawa"]) / r["gamma_floquet"] for r in un)
    fixed = max(abs(r["gamma_fixed_ions"]) for r in sc)
    det = ", ".join(f"{d['w0']:g}: {d['floquet'][0] if d['floquet'][0] > 1e-8 else 0:.2g}" for d in otsi["detune"])
    L = ["## The physical high-k instability is the oscillating two-stream instability (`studies/instab_otsi.py`)", "",
         "Linear pump-frame Hermite model under a constant dipole pump E0 cos(w0 t); Floquet rate from the one-period "
         "monodromy, compared with the exact kinetic Silin matrix dispersion relation (Bessel orders |l| <= 4).", "",
         f"- Floquet and the Silin relation agree to {100 * dmax:.1f}% at all {len(un)} unstable (k, pump) points "
         "(v_os/v_te 0.3-10, k = 2-30 k1); the modes are purely growing in the ion frame (OTSI, not decay/PDI).",
         f"- With fixed ions every rate is below {fixed:.0e}: there is no electron-only mechanism.",
         f"- Detuning (k12, v_os = 3 v_te), w0/w_pe: rate = {det}. Unstable only on the OTSI side; "
         "the decay side is stable at T_i = T_e. Insensitive to T_i/T_e; rate scales as m_i^-0.29.",
         "- Integrating 2 int gamma(k, v_rel(t)) dt with the secular H05 excursion predicts log10 |E_k|^2 gain at t = 700 of "
         + ", ".join(f"k{k} {otsi['adiabatic_h05'][k]['log10_amp_700']:.2f}" for k in ("2", "5", "8"))
         + " against the Hermite H05 linear run "
         + ", ".join(f"{__import__('math').log10(hk_lin['res']['k' + k]['herm64_E2_end_over_0']):.2f}" for k in ("2", "5", "8"))
         + " (uniformly 0.6-0.8 decades high): the H05 high-k growth is the quasi-static OTSI.",
         "- This is known physics (Silin/Nishikawa) in a new regime (resonant pump, secular excursion, T_i = T_e).", "",
         "Two-stream movie negativity (`studies/instab_twostream.py`): at Nx = 16 it is x-truncation, insensitive to Nn "
         "and the closure; raising Nx without the closure triggers the AW truncation instability; Nx = 64 with the "
         "#66 closure (c = 1) brings min f from -0.37 to -0.02. The L2-stable basis options are compared in "
         "`docs/design/l2-stable-basis.md`."]
    return "\n".join(L)


def sw_section():
    sw = root / "studies/sw"
    g = json.loads((sw / "summary.json").read_text())["gates"]
    runs = [json.loads(f.read_text()) for f in sorted((sw / "runs").glob("*.json"))]
    ends = ", ".join(f"{r['case'].split('_')[2]} {r['case'].split('_')[3]}: t = {r['T']:g}"
                     + ("" if r["finite"] else " (non-finite)") for r in runs)
    g4 = ", ".join(f"{k.split('_')[2]} {k.split('_')[3]} {v:.1e}" for k, v in sorted(g["G4_max"].items()))
    L = ["## Symmetric-weight Hermite (option B) on H05 v_q 0.1, nu = 0 (`studies/sw/`)", "",
         "Gates declared before the runs (`studies/sw/sw_analysis.py` docstring): G3 = dK_e and W_ext within 10% of the "
         "masked grid reference to t >= 700 with Nn and Nx refinement pairs agreeing to t >= 700; G4 = energy-ledger "
         "drift <= 1e-3 W_ext for 50 < t <= 600; reject if G3 fails before t = 650 or G4 fails. RK4, dt 0.02, T 1000. "
         "Tables below are generated by `python studies/sw/sw_analysis.py` (`studies/sw/summary.md`); drift columns "
         "over the whole run read nan where a run later goes non-finite.", "",
         "\n".join(ln for ln in (sw / "summary.md").read_text().splitlines()
                   if ln.startswith("|") or not ln.strip()).strip(), "",
         f"Run end (last saved time): {ends}. Positivity of the reconstructed f was not measured.", "",
         f"Verdict: {'REJECTED' if (not g['G3_pass'] or not g['G4_pass']) else 'PASSED'} as declared. G3 pairs agree "
         f"only to t = {g['G3_pairs_min']:g}{' (< 650: rejection rule)' if g['G3_reject_before_650'] else ''}; "
         f"G4 max ledger/W for 50 < t <= 600: {g4}. Option A (#66 closure) stays in production."]
    return "\n".join(L)


def c09_records():
    return json.loads((root / "studies/c09/run.json").read_text())


def c09_rows(r):
    """(objective, parameter, derivative dict) for every compared derivative."""
    objs = [("omega_1", r["a_omega"]["omega_1"]), ("omega_2", r["a_omega"]["omega_2"]),
            ("W_D(60), fixed dt", r["b_dark_work"]["W_D"]), ("W_D(60), adaptive", r["b_dark_work"]["W_D_adaptive"]),
            ("abs(E_k(10))^2", r["c_landau"]["P"])]
    return [(n, p, d) for n, o in objs for p, d in o["derivatives"].items()]


def c09_section():
    r = c09_records()
    a, b, c = r["a_omega"], r["b_dark_work"], r["c_landau"]
    fc = c["forward_convergence"]
    L = ["## C09: derivatives of converged dark observables, AD vs finite differences (`studies/c09_ad.py`)", "",
         f"- Record: `studies/c09/run.json`, repository `{r['repository_commit'][:7]}`, parent `{r['parent_commit'][:7]}`, "
         f"JAX {r['versions']['jax']}, Diffrax {r['versions']['diffrax']}, CPU float64; wall {r['wall_s']:.0f} s.",
         "- Method: `ds.solve` (new, returns JAX arrays) with traced `eta`, `Omega_D`, `alpha_s` via `ds.with_params`. "
         "Reverse mode `jax.grad` (Diffrax `RecursiveCheckpointAdjoint`, or `DirectAdjoint` for (a)), forward mode `jax.jvp` "
         "(Diffrax `ForwardMode`), centred FD over h = |p| 1e-1 ... 1e-7, analytic derivative where one exists. "
         "Fixed-step Dopri8 unless stated, so AD is the exact derivative of the discrete solve.",
         "- (a) cold homogeneous oscillator (eta 0.25, Omega_D 1.2, n0 1, k = 0, T = 60, dt 0.01): both frequencies "
         "fitted from E(t) by variable projection (Gauss-Newton unrolled in JAX); analytic: roots of "
         "(w^2 - n0)(w^2 - Omega_D^2) - eta^2 n0 w^2 = 0.",
         f"  Forward: fitted roots equal the analytic roots to {a['forward_convergence']['dt0.01']['omega_rel_err_vs_root']:.0e} "
         f"(fit residual {a['forward_convergence']['dt0.01']['fit_residual']:.0e}) at dt 0.02, 0.01, 0.005.",
         f"- (b) same oscillator, W_D(T = 60), parameters eta, Omega_D, n0, seed E0 = 0.01; reference: 4x4 matrix exponential, "
         f"W_D = -U_D(T). Forward: W_D = {b['expm_value']:.10e}; solve error "
         + ", ".join(f"{k} {v['W_D_rel_err_vs_expm']:.0e}" for k, v in b["forward_convergence"].items()) + ".",
         f"- (c) dark Landau, k lambda_De = 0.5, v_te = 0.1c, eta 0.3, Omega_D 1, seed 5e-5, Nn {c['Nn']}, dt {c['dt']}: "
         "P = abs(E_k(t = 10))^2; parameters eta, Omega_D, n0, T (alpha = sqrt(2T) isotropic), seed amplitude. "
         f"Forward: P = {c['P']['value']:.10e}; relative change vs Nn 128, dt 0.01: "
         + ", ".join(f"{k} {v:.0e}" for k, v in fc["rel_diff_vs_Nn128_dt0.01"].items())
         + f"; start equals `consistent_fields` to {fc['init_vs_consistent_fields']:.0e}. "
         "The reverse gradient at Nn 32/dt 0.02, Nn 64/dt 0.01, Nn 128/dt 0.02 agrees to "
         f"{max(abs(x / y - 1) for k in c['gradient_refinement'] for x, y in zip(c['gradient_refinement'][k], [c['P']['derivatives'][n]['reverse'] for n in c['P']['derivatives']])):.0e} relative.",
         f"- Seed checks: the cold oscillator is linear, so W_D is exactly quadratic in E0 and FD of it is exact at every h "
         f"(whole ladder on the plateau). P is quadratic in the seed only to the weak nonlinearity: 2P/amp = "
         f"{2 * c['P']['value'] / c['P']['params']['amp']:.10e} vs AD {c['P']['derivatives']['amp']['reverse']:.10e} "
         f"({abs(2 * c['P']['value'] / c['P']['params']['amp'] / c['P']['derivatives']['amp']['reverse'] - 1):.0e} relative).",
         "", "| Objective | Parameter | Reverse AD | forward vs reverse | best FD (rel.) | plateau h/abs(p) | AD vs analytic |",
         "|---|---|---|---|---|---|---|"]
    for n, p, d in c09_rows(r):
        an = f"{d['ad_rel_err_vs_analytic']:.0e}" if "analytic" in d else "-"
        pl = f"{min(d['plateau_h_rel']):.0e}-{max(d['plateau_h_rel']):.0e}"
        L.append(f"| {n} | {p} | {d['reverse']:.10e} | {d['fwd_rev_rel']:.0e} | {d['best_fd_rel_err']:.0e} | {pl} | {an} |")
    L += ["", "FD error vs AD over the h ladder (h/abs(p) = 1e-1 ... 1e-7) for (c), showing truncation (h^2) then round-off:", ""]
    L += [f"- {p}: " + ", ".join(f"{x['rel_err_vs_ad']:.0e}" for x in d["ladder"]) for n, p, d in c09_rows(r)
          if n.startswith("abs")]
    cst = [("(a) omega_1", a["omega_1"]["cost"]), ("(b) W_D fixed dt", b["W_D"]["cost"]),
           ("(b) W_D adaptive", b["W_D_adaptive"]["cost"]), ("(c) P", c["P"]["cost"])]
    L += ["", "Cost (best of 3 after compile; memory = XLA temp buffer of the compiled call; laptop shared, load about 20):", "",
          "| Objective | forward s | reverse s (ratio) | one jvp s | forward temp | reverse temp |", "|---|---|---|---|---|---|"]
    for n, k in cst:
        f, g = k["forward"], k["reverse"]
        L.append(f"| {n} | {f['time_s']:.3f} | {g['time_s']:.3f} ({g['time_s'] / f['time_s']:.1f}x) | "
                 f"{k['forward_jvp_time_s']:.3f} | {f['temp_bytes'] / 1e3:.0f} kB | {g['temp_bytes'] / 1e3:.0f} kB |")
    L += ["", "(c) checkpoint count (500 steps): " + ", ".join(
        f"{k.split('_')[1]}: reverse {v['reverse']['time_s']:.2f} s, {v['reverse']['temp_bytes'] / 1e3:.0f} kB"
        for k, v in c["checkpoint_cost"].items()) + ".", "",
          "Diffrax defect found: with `ConstantStepSize` and many `SaveAt(ts)` points, `RecursiveCheckpointAdjoint` "
          "returned wrong gradients in some long runs; `DirectAdjoint`, `ForwardMode` and FD agree with each other:", "",
          "| Case | dt | saves | max_steps | recursive | direct | rel. error |", "|---|---|---|---|---|---|---|"]
    for d in r["saveat_defect"]:
        L.append(f"| {d['case']} | {d['dt']} | {d.get('n_save', 601)} | {d.get('max_steps', int(60 / d['dt']) + 20)} | "
                 f"{d['recursive_checkpoint']:.10g} | {d['direct']:.10g} | {d['recursive_rel_err_vs_direct']:.0e} |")
    L += ["", "The wrong value depends on `max_steps` (same solve, 6020 vs 20000), which points at the checkpoint "
          "bookkeeping; the cause was not isolated. Reported upstream as [diffrax#788](https://github.com/patrick-kidger/diffrax/issues/788). Final-state objectives (2 saves) and a short 11-save case were exact. "
          "So (a), which fits 601 saves, uses `DirectAdjoint`; (b) and (c) use only the final state. This affects the parent's "
          "checkpointed reverse mode (#44) whenever a loss uses many saved times with a constant step.",
          "", "Not covered: gradients through the pump-frame remap and `run_adaptive`, through a fitted damping rate of a "
          "Landau mode, nonlinear-regime objectives, and GPU."]
    return "\n".join(L)


def dai_section():
    """Dai (arXiv 2608.09827) stable AW Galerkin, linear prototype (studies/dai/)."""
    fr = json.loads((root / "studies/dai/spectra_frozen.json").read_text())["results"]
    fq = json.loads((root / "studies/dai/spectra_floquet.json").read_text())["results"]
    ot = json.loads((root / "studies/dai/otsi.json").read_text())["rows"]
    nm = json.loads((root / "studies/dai/spectra_norm.json").read_text())["results"]
    cc = json.loads((root / "studies/dai/conserve_cost.json").read_text())
    def mx(rs, var, key, nn):
        v = [r[key] for r in rs if r["variant"] == var and r["Nn"] == nn]
        return f"{max(v):.1e}" if v else "-"
    L = ["## Dai stable AW Galerkin: linear prototype (`studies/dai/`)", "",
         "Gate 1: max Re lambda (frozen, K 2/5/10, E1 1e-3..1e-2) and Floquet exponent (E1 8e-3/2e-2, w = 1, Magnus), "
         "maximum over the points at each Nn. DAI_AW = the paper's eq. 27 form in float64 (AW tridiagonal + one dense "
         "column); DAI_SW = the same Galerkin method in orthonormal Hermite functions of width a/sqrt2.", "",
         "| Nn | AW_c0 frozen | DAI_AW frozen | DAI_SW frozen | AW_c0 Floquet | DAI_AW Floquet | DAI_SW Floquet |",
         "|---|---|---|---|---|---|---|"]
    for nn in (32, 64, 128):
        L.append(f"| {nn} | {mx(fr, 'AW_c0', 'max_re', nn)} | {mx(fr, 'DAI_AW', 'max_re', nn)} | {mx(fr, 'DAI_SW', 'max_re', nn)} | "
                 f"{mx(fq, 'AW_c0', 'floquet', nn)} | {mx(fq, 'DAI_AW', 'floquet', nn)} | {mx(fq, 'DAI_SW', 'floquet', nn)} |")
    rel = lambda v: max(abs(r[v]["rel"]) for r in ot)
    big = [r for r in nm if r["variant"] == "DAI_AW" and r["Nn"] == 64]
    cst = cc["cost"]
    L += ["", f"Gate 2 (OTSI Floquet rate vs Silin, {len(ot)} points): max relative error AW {100 * rel('AW'):.2f}%, "
          f"DAI_SW {100 * rel('DAI_SW'):.2f}%, DAI_AW {rel('DAI_AW'):.0f}x (wrong beyond Nn = 32).",
          f"L2(dv) norm of DAI_AW under exact stepping, Nn 64, t = 1000: grows by {big[0]['l2_norm_ratio']['1000']:.1e} "
          "(DAI_SW: 1 to 1e-14). Gate 3 (structural): Dai's projected top mode breaks the moments of the parity of N+1: "
          "even Nn keeps momentum exactly, odd Nn keeps density and energy exactly, never all three (AW keeps all three). "
          f"Gate 4: block nnz {cst['DAI_AW']['nnz_ratio_vs_AW']:.2f}x AW, matvec "
          f"{cst['DAI_AW']['matvec_s'] / cst['AW']['matvec_s']:.1f}x, largest entry {cst['DAI_AW']['max_abs_entry']:.0e}.", "",
          "Verdict: gate 1 FAILS for Dai's AW-coordinate form at Nn >= 64 (round-off: the AW-to-orthonormal change of "
          "basis has condition ~2^N), so the study stops as declared; the large Floquet cases (floquet_big) were not run. "
          "The method itself is stable and Silin-accurate, but only when written as symmetric Hermite functions of width "
          "a/sqrt2, i.e. option B of `docs/design/l2-stable-basis.md`, not a fix that keeps the AW basis."]
    return "\n".join(L)


def antisym_section():
    """Square-root symmetric-weight Hermite (Issan et al.), frozen/Floquet/OTSI prototype and H05 pilot (studies/antisym/)."""
    A = root / "studies/antisym"
    sm = json.loads((A / "summary.json").read_text())
    hs = json.loads((A / "h05_summary.json").read_text())
    S = sm["summary"]
    cs = sm["cons"]
    cd = [r for r in sm["cond"]]
    g = lambda v, k: f"{S[v][k]:.1e}"  # noqa: E731
    L = ["## Anti-symmetric (square-root symmetric-weight) Hermite candidate (`studies/antisym/`)", "",
         "Gates are in the docstring of `studies/antisym/antisym_gates.py` (committed before any run; amendments A1-A2 there). "
         "Literature: Issan et al. arXiv:2312.05439 (anti-symmetric SW Galerkin, centred or Fourier-Galerkin x, truncation closure: "
         "sum |C|^2 = int f^2 conserved for any E(t), u(t); square-root form g = sqrt f with the same linear equation: "
         "sum C^2 = particle number, density positive; momentum and energy not conserved), 2407.09626 (no closure restores "
         "number, momentum and energy in SW), 2412.07073 (AW with hypercollisions; no stability theorem for AW). "
         "SQ = that square-root form, s = sqrt2 sigma. Its frozen-field linear operator is the SW operator already rejected on H05, "
         "so stage 1 cannot discriminate SQ from SW; the new content is the quadratic moment map, tested in stage 2.", "",
         "**Stage 1 (linear, 18 frozen points: E1 1e-3/4e-3, K 2/5/10, Nn 32/64/128; 12 Floquet points).**", "",
         "| variant | frozen: points with max Re lambda > 1e-10 | frozen max Re lambda | Floquet points > 1e-10 | Floquet max |",
         "|---|---|---|---|---|"]
    for v in ("AW_c0", "AW_c1", "SQ"):
        L.append(f"| {v} | {S[v]['n_frozen_gt_1e10']} / {S[v]['n_frozen']} | {g(v, 'frozen_max')} | "
                 f"{S[v]['n_floquet_gt_1e10']} / {S[v]['n_floquet']} | {g(v, 'floquet_max')} |")
    c0 = next(r for r in cd if r["var"] == "AW_c0" and r["K"] == 5 and r["E1"] == 4e-3)
    q0 = next(r for r in cd if r["var"] == "SQ" and r["K"] == 5 and r["E1"] == 4e-3)
    L += ["", f"G1 {'PASS' if sm['G1'] else 'FAIL'} (SQ <= 4e-15 everywhere). G2 {'PASS' if sm['G2'] else 'FAIL'}: "
          f"{sm['G2_points']} OTSI points (k5/k12/k20, Nn 32-128), SQ vs Silin max error {100 * sm['G2_max_rel']:.2f}% "
          f"(AW 1.36% on the full dai scan); SQ equals the AW Floquet rate to {sm['G2_max_sq_vs_aw']:.1e} relative "
          "(the linearisation of f = g^2 about the Maxwellian is the AW system in the free-energy norm int df^2/f_M). "
          f"G5 {'PASS' if sm['G5'] else 'FAIL'}: SQ normality defect {q0['normality_defect']:.0e}, dense vs time-domain growth "
          f"{sm['SQ_dense_vs_td_max']:.0e}; AW_c0 normality defect {c0['normality_defect']:.2f}, eigenvector condition "
          f"{c0['cond_eigvec']:.0e}, dense max Re {c0['dense_max_re']:.3f} against time-domain {c0['td_rate']:.3f} "
          f"(largest dense/time-domain gap over the 18 points {sm['AW_dense_vs_td_max']:.2f}).", "",
          "G3 (finite-N conservation of the SQ moments, instantaneous relative defect of d/dt of N, P, KE on random states, K = 5, "
          "frozen random field; AW rows 0-2 are exact by construction, not re-run):", "",
          "| Nn | spectrum | top-2 fraction | dN | dP | dKE |", "|---|---|---|---|---|---|"]
    for r in cs:
        L.append(f"| {r['Nn']} | n^-{r['p']} | {r['top2_frac']:.1e} | {r['dN_rel']:.0e} | {r['dP_rel']:.1e} | {r['dK_rel']:.1e} |")
    ct = sm["G4_ratio"]
    L += ["", f"G4 (banded kernels, pseudo-spectral RHS, Nx 32): SQ/AW wall ratio {ct[0]:.2f} (Nn 64), {ct[1]:.2f} (Nn 128); frozen "
          "operator nnz ratio 1.5 (SW couples n+-1 in the force, AW only n-1). The first, dense-kernel record (ratios 3.2 and 5.8) "
          "is kept as `cost_dense_firstrun.json` (amendment A2).", "",
          "**Stage 2 (H05 v_q/v_te 0.1, nu = 0, no closure, RK4 dt 0.02, standalone pump-frame 1D1V two-species Vlasov-Ampere; "
          "gates identical to the SW study).** Declared after stage 1 and before the runs; the SQ driver reproduces the SW driver's "
          "dK_e to 1e-9 relative for t <= 150 (ungated check).", "",
          "| SQ run | t_res(dK_e) vs grid | t_res(W_ext) | last finite t | ledger/W > 1e-3 first at | SW same case: t_res(dK_e) |",
          "|---|---|---|---|---|---|"]
    swr = {"Nx16_Nn64": 634.5, "Nx16_Nn128": 703.5, "Nx32_Nn64": 681.0, "Nx32_Nn128": 737.5, "Nx32_Nn256": 685.0}
    gname = "g_vq0.1_Nx64_Nv16384_dt0.02_osc_v64_spectral_xmask23"
    for led in hs["ledger"]:
        tag = led["case"].split("sq_vq0.1_")[1].split("_dt")[0]
        row = next(r for r in hs["vs_grid"] if r["run"] == led["case"] and r["grid"] == gname)
        L.append(f"| {tag} | {row['dK_e']} | {row['W_ext']} | {led['T']:g} | {led['first_energy_defect_over_W_gt_1e-3']} | {swr[tag]} |")
    gt = hs["gates"]
    d = {}
    for tag in ("sq_vq0.1_Nx16_Nn64_dt0.005_T420", "sq_vq0.1_Nx64_Nn64_dt0.02_T1000"):
        p = A / "runs" / f"{tag}.json"
        if p.exists():
            d[tag] = json.loads(p.read_text())
    L += ["", f"G3 {'PASS' if gt['G3_pass'] else 'FAIL'} (refinement pairs agree only to t = {gt['G3_pairs_min']:g}; reject-before-650 "
          f"{'triggered' if gt['G3_reject_before_650'] else 'not triggered'}); G4 {'PASS' if gt['G4_pass'] else 'FAIL'}. "
          "Ungated diagnostics (not in the declared batch): "
          + "; ".join(f"{k.split('sq_vq0.1_')[1]}: last finite t {v['T']:g}, ledger/W > 1e-3 first at {v['first_energy_defect_over_W_gt_1e-3']}"
                      for k, v in d.items()) + ".", "",
          "**Verdict.** SQ is linearly neutral to round-off, Silin-accurate, normal (no round-off pathology) and costs 1.5x AW, "
          "but it fails the nonlinear H05 test earlier than the SW basis it contains linearly: it departs from the grid at t = 391 "
          "(Nx 16) and 498 (Nx 32), with Nn 64-256 identical and dt 0.02 -> 0.005 unchanged, i.e. the loss is not Hermite truncation or "
          "time step; the Nx 64 diagnostic fails at 538. The bounded quantity (particle number, sum C^2) stays at 1e-9 until the "
          "top Hermite band fills within about 10 time units (top fraction 2e-11 -> 5e-2), after which RK4 diverges and the energy "
          "ledger leaves 1e-3 W_ext. SQ is rejected; A (#66 closure) stays in production. See `docs/design/l2-stable-basis.md`, "
          "Verdict on E."]
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
        rows.append(f"| {k} | {i['Nn']} / {2 * i['Nn']}, Nx {i['Nx']} / {2 * i['Nx']}, nu = {i['nu']:g}, c = {i.get('field_nu', 0):g} | "
                    f"{st} ({min(reach):g}-{max(reach):g}) | "
                    f"{e['t_res_ordinary']:g} / {e['t_res_dark']:g} | {e['t_res_x_ordinary']:g} / {e['t_res_x_dark']:g} | "
                    f"{e['t_res_f_ordinary']:g} / {e['t_res_f_dark']:g} | {e['t_res_fx_ordinary']:g} / {e['t_res_fx_dark']:g} | "
                    f"{e['t_drawn']:g} | {e['movie']['min_f_over_max_ordinary']:.4f} / {e['movie']['min_f_over_max_dark']:.4f} |")
    return """## Phase-space movies: resolution of the reconstructed f (`python studies/figures.py phase --Nx 64 --field-nu 1`)

Question: how long do fixed-basis Hermite runs resolve f(x, v) itself (not only the field) once trapping starts?
Input: `PHASE_CASES` in studies/figures.py (electrostatic units mapped with v_t/c = 0.1, seeds at the box mode,
ordinary and dark eta = 0.3, Omega_D = omega_pe), Nx = 64 with the field-scaled AW closure of SPECTRAX #66
(c = 1), each at (Nn, Nx), (2Nn, Nx) and (Nn, 2Nx); Dopri8 rtol 1e-8, 200000-step budget, step floor 1e-6.
Measurement: t_res by field energy (10% of the running max, as for H05) and by f (first frame with
max|f_a - f_b| > 0.1 max f on a 256 x 160 grid), each for Nn vs 2Nn and for Nx vs 2Nx; frames are drawn only
before the earliest of the eight times (t drawn); a time equal to T means no disagreement up to T.

| case | Nn, Nx, closures | solver status (t reached) | t_res U_E Nn (ord / dark) | t_res U_E Nx | t_res f Nn | t_res f Nx | t drawn | min f / max f drawn (ord / dark) |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + """

At Nx = 64 with the closure the drawn time is set by f: Nn vs 2Nn for bump-on-tail (the x rule fails later), and
both rules at the same frame for two-stream, so x and Hermite truncation end the two-stream window together. Negative f is much smaller than in the earlier Nx = 16 movies (two-stream -31% / -37%, bump-on-tail
-7% / -9%, nonlinear Landau -0.6% at t = 71.25): the Nx = 16 negativity was x truncation
(`studies/instab_twostream.py`). The two-stream runs stop at the 1e-6 step floor after the drawn window; the
other runs reach T. The nonlinear Landau runs are resolved by all rules to T = 100. Two-stream and bump-on-tail
remain illustrations of trapping onset only."""


def r2_section():
    """R2: sustained dark-to-plasma absorption with Dougherty collisions vs the linear drag law."""
    s = json.loads((root / "studies/r2/summary.json").read_text())
    fu = json.loads((root / "studies/r2/followup_summary.json").read_text())
    C, gt = s["cases"], s["gates"]
    om = {"res": "resonant", "0.9": "0.9", "1.1": "1.1"}
    L = ["## R2: sustained absorption with collisions vs the linear drag law (`studies/r2_gates.py`)", "",
         "Gates, case list and decision rules were committed in `studies/r2_gates.py` (92e0775) before any production run; the analysis "
         "(`studies/r2_analysis.py`) implements them without rule changes, records are in `studies/r2/` (`summary.json`, slim records, `followup_summary.json`), "
         "figure ![R2](_static/r2/r2_absorption.png). H05 set-up (mobile ions 1836, pump frame + remap, seeds r0, c = 64 field-scaled closure with Lawson, noise floor 1e-14, "
         "T = 1000), Dougherty e-e/e-i/i-i with nu_ee = nu_ei, nu_ii = nu_ei sqrt(m_e/m_i), both conventions (EM = `dougherty`, G = `dougherty_g` beta = 0). "
         "Grid: v_q/v_te 0.01/0.03/0.05 x nu_ei 1e-3/3e-3/1e-2/3e-2 x omega resonant (omega_L), 0.9, 1.1 (units of omega_pe); members Nx 32 Nn 64 (B), Nx 64 (X), Nn 128 (N) for EM, B for G; "
         "unseeded controls (Nx 8). 192 runs, 1.3e6 steps.", "",
         "**Linear law** (`studies/r2_theory.py`). The uniform relative velocity relaxes at nu_eff = (nu_ei/2)(1 + m_e/m_i) (Morse rate, identical in EM and G beta = 0), so the "
         "uniform driven cold response obeys J' = omega_L^2 (Ebar + Ed) - nu_eff J and the steady absorbed power per volume is "
         "P_ss = omega_L^2 nu_eff omega^2 E0^2 / (2 [(omega_L^2 - omega^2)^2 + nu_eff^2 omega^2]) (resonant: omega_L^2 E0^2 / (2 nu_eff)); the transient (natural mode) decays as exp(-nu_eff t/2) "
         "(1/e time 4/nu_ei: 4000, 1333, 400, 133). Literature form (read for this study, paraphrased): Dubovsky and Hernandez-Chifflet (arXiv:1509.00039) write the same cold two-fluid friction "
         "-m_e nu (u_e - u_i) with nu = (4 sqrt(2 pi) alpha^2 n_e / (3 m_e^(1/2) T_e^(3/2))) log Lambda, obtain the electron response u_e = -e E / (m_e (nu - i omega)), and find that "
         "plasma shielding suppresses the hidden-photon coupling by m^2/omega_p^2 for nu << m << omega_p and by i nu m/omega_p^2 for m << nu (a Drude response with complex plasma frequency omega_p^2/(1 + i nu/omega)); McDermott and Witte (arXiv:1911.05086) use the "
"off-shell (non-resonant) absorption rate proportional to nu m^2/omega_p^2 with the same Spitzer-like nu (their Eq. 5-6), plus resonant conversion at m = omega_p. The B12 form above is the "
         "same Drude law; the nu of those papers corresponds to nu_eff here (friction on the electron-ion relative velocity), not to nu_ei. In the model nu is **constant**: the Spitzer "
         "dependence nu ~ T_e^(-3/2) (heating reduces the absorption in the cosmological estimates) is not present.", "",
         f"Gates: T0 (theory self-check as declared) **FAILED by declaration defect** (80-period averaging window longer than the 400-unit record, result NaN); T0b (declared as a follow-up, theory only: "
         f"1000 units, 40 periods) pass, deviations from P_ss {max(abs(v['rel']) for v in s['T0b'].values()):.1e}; C0 controls pass (Ebar {max(v['Ebar_err'] for v in s['controls'].values()):.1e}, "
         f"W_ext {max(v['Wext_err'] for v in s['controls'].values()):.1e}, gate 1e-6; 48 runs); L1 ledger {'pass' if gt['L1'] else 'FAIL'} (max {max(v['max_ledger_over_Wext'] for v in s['runs'].values()):.1e}); "
         f"L2 no net collisional work {'pass' if gt['L2'] else 'FAIL'} (max {max(v['max_Psum_over_Pe'] for v in s['runs'].values()):.1e}); L3 identity {'pass' if gt['L3'] else 'FAIL'} "
         f"(max {max(v['max_R_over_wL2E0'] for v in s['runs'].values()):.1e}); CV convention check {s['counts']['CV'][0]}/{s['counts']['CV'][1]} pass; K3 steady-state law {s['counts']['K3'][0]}/{s['counts']['K3'][1]} pass; "
         f"K2 kinetic = linear within 5% (late rate and cumulative work, plus refinement spread <= 0.02) **{s['counts']['K2'][0]}/{s['counts']['K2'][1]}** (the six failures are the nu_ei = 1e-3 detuned cases, see below).", "",
         "Windows: every EM case reaches Wn = 1000 (pairs B/X, B/N agree to 1000 in dK_e and W_ext; refinement spread of the late rate <= 2.2e-4). The nonrelativistic window Wp (T_e/T_e(0) < 90) is shorter only for "
         "v_q/v_te = 0.05 resonant: " + ", ".join(f"nu {k.split('_')[1][2:]}: {r['Wp']:g}" for k, r in C.items() if r["Wp"] < 1000) + ". Everything below is evaluated at tb = Wn - pi/omega = 996.5 (rate interval Delta = 250 resonant, 188-189 detuned).", "",
         "### Late-time rate, kinetic vs exact linear (B member; R = P_kin/P_lin on the same interval; f_ss = P_lin/P_ss; T_e/T_e(0), ion heating and exchange at tb)", "",
         "| omega | v_q/v_te | nu_ei | R_late | R_cum | f_ss | P_kin/P_ss | K2 | T_e/T_e(0) | T_i/T_i(0) | dK_i/W_ext | Q_i/W_ext |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in C.items():
        v, n, w = k.split("_")
        b, p = r["B"], r["report"]
        L.append(f"| {om[w[1:]]} | {v[2:]} | {n[2:]} | {b['R_late']:.4f} | {b['R_cum']:.4f} | {b['f_ss']:.2f} | {b['P_kin_over_pss']:.3f} | {'pass' if r['K2'] else 'FAIL'} | "
                 f"{p['Te_over_Te0']:.3g} | {p['Ti_over_Ti0']:.3f} | {p['dKi_over_Wext']:.1e} | {p['Qi_over_Wext']:.1e} |")
    L += ["", "### Deposition law (K4, reported)", "", "Fit ln P = a + p ln(v_q/v_te) + q ln(nu_ei) over the 12 cases per omega (each interval as above; standard errors from the residual scatter, so they include the "
          "transient-state curvature, not only noise). The exact linear fit differs from kinetic only by the seed effect.", "",
          "| omega | p kinetic | p linear | q kinetic | q linear | kappa = geometric mean of P_kin/P_lin | range of P_kin/P_lin |", "|---|---|---|---|---|---|---|"]
    for w, f in s["fits"].items():
        L.append(f"| {om[w]} | {f['kin']['p']:.3f} +- {f['kin']['se_p']:.3f} | {f['lin']['p']:.3f} +- {f['lin']['se_p']:.3f} | {f['kin']['q']:.3f} +- {f['kin']['se_q']:.3f} | "
                 f"{f['lin']['q']:.3f} +- {f['lin']['se_q']:.3f} | {f['kappa']:.3f} +- {f['kappa_se']:.3f} | {f['R_late_range'][0]:.3f}-{f['R_late_range'][1]:.3f} |")
    L += ["", "### Follow-up F: is the nu_ei = 1e-3 detuned departure a seed effect? (`studies/r2_followup.py`, declared after the main analysis and before its runs)", "",
          "Seeds multiplied by 0.1 at v_q/v_te = 0.03, nu_ei = 1e-3 (EM, B member):", "", "| case | D = R_cum - 1 at s = 1 | at s = 0.1 | ratio | E = R_late - 1 at s = 1 | at s = 0.1 | ratio | ledger |", "|---|---|---|---|---|---|---|---|"]
    for k, r in fu.items():
        if k != "gates":
            L.append(f"| {k} | {r['D_s1']:+.4f} | {r['D_s0.1']:+.5f} | {r['D_ratio']:.4f} | {r['E_s1']:+.4f} | {r['E_s0.1']:+.5f} | {r['E_ratio']:.4f} | {r['ledger']:.1e} |")
    L += ["", f"F1 (ratios in [0.005, 0.02], i.e. s^2 = 0.01 within a factor 2) **{'pass' if fu['gates']['F1'] else 'FAIL'}**, F2 ledger **{'pass' if fu['gates']['F2'] else 'FAIL'}**.", "",
          "**Reading.** (1) Where collisions act and the linear transient has decayed, the kinetic absorption equals the linear drag law: the late rate is within 0.2% of P_ss at nu_ei = 3e-2 for all three frequencies "
          "and all drives (f_ss = 1.00, P_kin/P_ss 0.996-1.003), and K3 passes in 12/12 of the cases where P_lin is within 10% of P_ss. For the other cases the kinetic rate equals the exact transient-inclusive linear law within 5% "
          "except the six cases below, but the steady state is not reached inside t = 1000: for nu_ei = 1e-3, 3e-3, 1e-2 the resonant rate is only 0.20, 0.48, 0.89 of P_ss (the amplitude transient has 1/e time 4/nu_ei). "
          "The sustained law is therefore tested directly for nu_ei = 3e-2 (and 1e-2 detuned), and through the linear ODE for the rest. "
          "(2) The v_q scaling is quadratic: p = 1.99 +- 0.10 (kinetic) vs 2 (exact), 2.00 +- 0.03, 2.00 +- 0.02 for the detuned frequencies. The nu_ei scaling follows the transient-inclusive linear law, not a pure power: "
          "q (kinetic, resonant) = -0.51 vs -0.52 (linear) and 0.75 / 0.88 vs 0.79 / 0.90 for 0.9 / 1.1 (detuned rate grows with nu: off-resonant absorption proportional to nu, the McDermott-Witte non-resonant limit, until the beat transient decays). "
          "(3) Kinetic corrections are small and are seed back-reaction: the resonant late rate is 1.5-4% low at nu_ei = 1e-3 and <= 0.2% low at 3e-2, the detuned late rate is 9-14% high at nu_ei = 1e-3 "
          "(2.6% at 3e-3, < 0.1% at >= 1e-2). The detuned excess is independent of v_q (1.131/1.141/1.143 at 0.01/0.03/0.05 for omega = 0.9), the unseeded controls agree with the linear law to 3e-8, and the "
          "seed-scale test scales it as s^2 (ratios 0.0105 and 0.0099; F1 pass): it is the quiver-advected seed modulation acting back on the mean field, as in R1, large only where W_ext is itself tiny. "
          "The resonant deficit at v_q/v_te = 0.05, nu_ei = 1e-3 (3.7% in the late rate, T_e/T_e(0) = 129) is larger than at 0.01 and 0.03 (1.5%, 1.7%); drive amplitude and heating are not separable here "
          "(T_e rises with v_q^2), so the data allow a few-percent heating/drive correction but not its attribution. (4) Heating does not change the rate through nu in the model because nu is constant: R_late along the "
          "run stays within 1% of 1 while T_e/T_e(0) rises from 7 to 70 (v_q 0.03, nu 1e-2, resonant; `path_t_R_Te` in the summary). "
          "(5) Energy partition at tb: dK_e is 57-95% of W_ext in the resonant runs and 76-96% in the detuned ones (coherent quiver plus heat; the rest is field energy), electron temperature rises by factors 5-231 (v_te beyond 0.3 c for T_e/T_e(0) > 90, "
          "marked by Wp), ion heating is small (T_i/T_i(0) <= 2.8, dK_i/W_ext <= 1.3e-2) and the e-to-i exchange is Q_i/W_ext = 4e-5 (nu 1e-3) to 1.6e-2 (nu 3e-2, detuned), growing in proportion to nu_ei. "
          "(6) EM and G agree in W_ext to 2.5e-6, in dK_e to 8.3e-3 and in T_e to 6.6e-3 (CV pass in all 36 cases); the late-rate ratio differs by <= 7e-6; T_i differs by up to 31% and the e-i exchange Q_i by 0.4999 of its EM value (halved), as in H08. "
          "(7) Not done / not claimed: a cosmological deposition law needs the velocity-dependent Coulomb nu(T_e) (heating then lowers the rate), radiative and Compton cooling, expansion, ionization, "
          "relativistic electrons (the model is nonrelativistic: T_e/T_e(0) > 90 is beyond v_te = 0.3 c), the k != 0 coupling of a real dark-photon field, and the nu_ee, nu_ii choices (Z = 1 order-of-magnitude values; they do not enter the uniform drag). "
          "Domain of the law: v_q/v_te <= 0.05, nu_ei in 1e-3..3e-2 omega_pe, omega in {0.9, omega_L, 1.1} omega_pe, t <= 1000 omega_pe^-1, one box (L = 40) and mass ratio, one seed level (5e-4).", ""]
    return "\n".join(L)


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

def b04_b05_section():
    b4 = json.loads((root / "studies/b04_b05/b04.json").read_text())
    b5 = json.loads((root / "studies/b04_b05/b05.json").read_text())
    s4, r4 = b4["settings"], b4["results"]
    ok4 = [r for r in r4 if r["status"] == "success"]
    bad4 = [r for r in r4 if r["status"] != "success"]
    L = ["## B04: ion-acoustic damping, ordinary and dark (`python studies/b04_b05.py b04`)", "",
         f"Electrons (v_te = {s4['v_te']}c) and mobile ions (m_i/m_e = {s4['mass_ratio']:.0f}), T_e/T_i = 1, 5, 20, "
         f"k lambda_De = 0.1, 0.2, 0.3, density seed {s4['seed']}, hypercollision nu = {s4['nu']}, Dopri8 rtol "
         f"{s4['rtol']}. Dark: eta = {s4['eta']}, Omega_D = omega_pe. The complex frequency is fitted on E_k with the "
         "ion-acoustic and Langmuir pairs as guesses and compared with the kinetic Maxwell-Proca root (both species "
         f"in the susceptibility). Record: [b04.json](../studies/b04_b05/b04.json) (made at "
         f"{b4['repository_commit'][:7]}). Rows below are Nn = 128, Nx = 5.", "",
         "| T_e/T_i | k lambda_De | model | kinetic root | fit | freq. rel. error | damping rel. error | gamma/omega |",
         "|---|---|---|---|---|---|---|---|"]
    for r in ok4:
        if r["Nn"] == 128 and r["Nx"] == 5:
            L.append(f"| {r['Te_over_Ti']:g} | {r['k_lambda_De']:g} | {r['model']} | {c(r['omega_root'])} | "
                     f"{c(r['omega_fit'])} | {r['rel_error']:.1e} | {r['damping_rel_error']:+.1e} | "
                     f"{r['gamma_over_omega_root']:.3f} |")
    def worst(key, te, Nn):
        return max(abs(r[key]) for r in ok4 if r["Te_over_Ti"] == te and r["Nn"] == Nn and r["Nx"] == 5)
    nx = [(a, b) for a in ok4 if a["Nx"] == 8 for b in ok4
          if b["Nx"] == 5 and b["Nn"] == a["Nn"] and b["case"] == a["case"] and b["model"] == a["model"]]
    dnx = max(abs(complex(*a["omega_fit"]) - complex(*b["omega_fit"])) / abs(complex(*b["omega_fit"])) for a, b in nx)
    sh = []
    for te in (1.0, 5.0, 20.0):
        for kl in (0.1, 0.3):
            o, d = [[r for r in ok4 if r["Te_over_Ti"] == te and r["k_lambda_De"] == kl and r["model"] == m
                     and r["Nn"] == 128 and r["Nx"] == 5][0]["omega_root"] for m in ("ordinary", "dark")]
            sh.append((d[0] / o[0] - 1, d[1] / o[1] - 1))
    L += ["", f"Refinement (Nn = 32/64/128, Nx = 5): worst frequency error {worst('rel_error', 1.0, 128):.1e} / "
          f"{worst('rel_error', 5.0, 128):.1e} / {worst('rel_error', 20.0, 128):.1e} at T_e/T_i = 1/5/20 (Nn = 128); "
          f"worst damping error {worst('damping_rel_error', 1.0, 128):.1e} / {worst('damping_rel_error', 5.0, 128):.1e} / "
          f"{worst('damping_rel_error', 20.0, 128):.1e}. Nn = 32 -> 64 removes the electron-recurrence error at T_e/T_i = 5 "
          f"(damping error up to {worst('damping_rel_error', 5.0, 32):.0e} at Nn = 32); 64 -> 128 changes little. "
          f"At T_e/T_i = 20 the fitted damping stays about 8% weaker than the root at Nn = 64 and 128 (gamma/omega about "
          "0.013, about 0.3 e-folds over the fit window): not resolved here and not claimed; the frequency agrees to 1e-3. "
          f"Nx = 8 at Nn = 32 reproduces Nx = 5 to {dnx:.0e} where it ran (T_e/T_i = 1); the {len(bad4)} Nx = 8 runs at "
          f"T_e/T_i = 5 and 20 hit the {max(r['steps'] for r in bad4)}-step budget without a nonfinite field (recorded as "
          "failures). Ledger defect <= "
          f"{max(r['max_ledger_defect'] for r in ok4):.0e} in all successful runs. Mixing shifts the roots by at most "
          f"{100 * max(abs(a) for a, _ in sh):.2f}% in frequency and {100 * max(abs(b) for _, b in sh):.1f}% in damping "
          "at k lambda_De = 0.1/0.3; the fitted dark-ordinary shifts follow the roots within the errors above.", ""]

    s5, r5 = b5["settings"], b5["results"]
    base = [r for r in r5 if r["Nn"] == 16 and r["Nx"] == 5]
    ref = max(abs(r["growth_fit"] / b["growth_fit"] - 1) for r in r5 if (r["Nn"], r["Nx"]) != (16, 5)
              for b in base if b["case"] == r["case"] and b["model"] == r["model"])
    rc = s5["return_current_carrier"]
    L += [f"## B05: Buneman growth, ordinary and dark (`python studies/b04_b05.py b05`)", "",
          f"Electrons drift at u0 = {s5['u0']}c through ions at rest, v_te = {s5['v_te'] / s5['u0']:.1f} u0, T_i = T_e, "
          "mobile ions with m_i/m_e = 100 and 1836. The k = 0 current is cancelled by an inert return-current carrier "
          f"(charge +{rc['q']:g}, mass {rc['mass']:.0e} m_e, drifting with the electrons, uniform background "
          f"{rc['rho_background']:g}); its susceptibility is kept in the roots, and max |E_0| stays below "
          f"{max(r['max_abs_E0'] for r in r5):.0e} (seed {s5['seed']}). Dark: eta = {s5['eta']}, Omega_D = omega_pe. "
          f"nu = {s5['nu']}, Dopri8 rtol {s5['rtol']}. Growth is the log-slope of |E_k| over the explicit window "
          f"[t_end - 3/gamma, t_end], t_end the first time |E_k| > {s5['threshold']}; the spread is over three "
          f"sub-windows. Record: [b05.json](../studies/b04_b05/b05.json) (made at {b5['repository_commit'][:7]}).", "",
          "| m_i/m_e | k u0 | model | kinetic root | fitted growth | rel. error | window | root / fluid max |",
          "|---|---|---|---|---|---|---|---|"]
    for r in base:
        L.append(f"| {r['mass_ratio']:.0f} | {r['k_u0']:g} | {r['model']} | {c(r['root'])} | {r['growth_fit']:.6f} | "
                 f"{r['growth_rel_error']:+.1e} | [{r['window'][0]:.1f}, {r['window'][1]:.1f}] | {r['root_over_fluid']:.3f} |")
    gs = [r for r in base]
    def mix(mi, ku, i):
        o, d = [[r for r in base if r["mass_ratio"] == mi and r["k_u0"] == ku and r["model"] == m][0]["root"][i]
                for m in ("ordinary", "dark")]
        return 100 * (d / o - 1)
    fr = [mix(mi, ku, 0) for mi in (100.0, 1836.0) for ku in (0.9, 1.0)]
    fl = {r["mass_ratio"]: r["fluid_gamma_max"] for r in base}
    L += ["", f"Rows are Nn = 16, Nx = 5; Nn = 32 and Nx = 8 change the fitted rate by at most {ref:.1e} relative "
          f"({sum(r['status'] == 'success' for r in r5)} of {len(r5)} runs succeed, ledger defect <= "
          f"{max(r['max_ledger_defect'] for r in r5):.0e}), so the residual error is the fit window (sub-window spread "
          f"{min(r['growth_window_spread'] for r in r5):.0e} to {max(r['growth_window_spread'] for r in r5):.1e}), not "
          f"truncation. All errors are within {max(abs(r['growth_rel_error']) for r in r5):.1e} of the kinetic roots, "
          "ordinary and dark. Fluid check: the cold limit gamma_max = (sqrt3/2)(m_e/2m_i)^(1/3) is "
          f"{fl[100.0]:.3f} (m_i = 100) and {fl[1836.0]:.4f} (1836); the kinetic roots at the two sampled k are "
          f"{100 * min(r['root_over_fluid'] for r in gs):.0f}-{100 * max(r['root_over_fluid'] for r in gs):.0f}% of it "
          "(warm electrons and the off-optimum k lower it), consistent with, not a test of, the fluid scaling. "
          f"Mixing changes the growth by {mix(100.0, 0.9, 1):+.1f}/{mix(100.0, 1.0, 1):+.1f}% (m_i = 100) and "
          f"{mix(1836.0, 0.9, 1):+.1f}/{mix(1836.0, 1.0, 1):+.1f}% (m_i = 1836) at k u0 = 0.9/1.0, and lowers the real "
          f"frequency by {-max(fr):.0f}-{-min(fr):.0f}%."]
    return "\n".join(L)


def b09_section():
    d = root / "studies/b09"
    lin, lad = (json.loads((d / f"{n}.json").read_text()) for n in ("linear", "ladder"))
    L = ["## B09: Weibel and current filamentation in 1D3V, ordinary and dark (`python studies/b09.py linear|ladder`)", "",
         f"Records: [linear.json](../studies/b09/linear.json), [ladder.json](../studies/b09/ladder.json) (made at "
         f"{lin['repository_commit'][:7]} / {lad['repository_commit'][:7]}). Gates and the independent kinetic transverse "
         "dispersion relation (with the Proca coupling, eta = 0.3) are in the script docstring, declared before running. "
         f"Part 1 overall: {'passed' if lin['passed'] else 'FAILED'}; Part 1b ladder: {'passed' if lad['passed'] else 'FAILED'}.", "",
         "Relative growth error |gamma_fit - gamma_root|/gamma_root. Levels (Nx, Nn, Nm): base (5,32,4), v-refined (5,64,8), "
         "x-refined (9,32,4); ladder (5, Nn, 4).", "",
         "| case | model | k | gamma_root | base | v-refined | x-refined | Nn=128 | Nn=256 | Nn=512 | ladder step 256->512 | gate |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r0 in [r for r in lin["results"] if (r["Nx"], r["Nn"], r["Nm"]) == (5, 32, 4)]:
        key = lambda r: r["case"] == r0["case"] and r["model"] == r0["model"] and r["k"] == r0["k"]
        lv = {(r["Nx"], r["Nn"]): r for r in lin["results"] if key(r)}
        ld = {r["Nn"]: r for r in lad["results"] if key(r)}
        e = lambda r: f"{r['rel_error']:.1e}" if r else ""
        if ld:
            ok, step = ld[512]["passed"], f"{ld[512]['refinement_change']:.1e}"
        else:
            ok, step = all(r["passed"] for r in lv.values()), ""
        L.append(f"| {r0['case']} | {r0['model']} | {r0['k']} | {r0['gamma_root']:.6f} | {e(lv[(5, 32)])} | {e(lv[(5, 64)])} | "
                 f"{e(lv[(9, 32)])} | {e(ld.get(128))} | {e(ld.get(256))} | {e(ld.get(512))} | {step} | {'pass' if ok else 'FAIL'} |")
    led = max(max(r["ledger_rel"]) for r in lin["results"] + lad["results"])
    L += ["", f"- Ledger: max relative defect over all {len(lin['results']) + len(lad['results'])} linear runs {led:.1e} (gate 1e-6); "
          "Gauss residuals at or below 1e-15 of max |F_k| everywhere (gate 1e-10).",
          "- k <= 1.2: all levels pass. k >= 1.6: Nn = 32/64 are unconverged (small xi = gamma/(k a_x), resonant width gamma/k << a_x); "
          "the ladder reaches the 1e-3 accuracy at Nn = 512 for every case, and the 1e-4 Nn/2 -> Nn step everywhere except Weibel at "
          "k = 2.4 (all three models), which fails the step gate.",
          "- One Part-1 failure is a fit-selection artifact: Weibel dark Omega_D = 0.3, k = 0.4, x-refined level, where the largest-Im fitted "
          "mode was spurious (Re w / gamma = 8.5); the ladder uses the mode nearest i gamma_root instead.",
          "- Nonlinear part of `studies/b09.py` (no record): the full run ended in status failure at the base and v-doubled Weibel levels; "
          "it is replaced by the chunked closure study below."]
    L += ["", b09_closure_lines()]
    return "\n".join(L)


def _b10_trace_rate(name, lo=1.5e-3, hi_cap=2e-2):
    """Post-hoc growth estimate from the stored S(t) trace (every 250 time units): slope of ln S where lo A_p <= S <= min(hi_cap,
    0.9 S_max) A_p. Not a declared gate."""
    rec = json.loads((root / f"studies/b10/{name}.json").read_text())
    x = rec["results"][0]
    t, S = np.array(x["trace_t"]), np.exp(np.array(x["trace_lnS"])) / rec["A_p"]
    hi = min(hi_cap, 0.9 * S.max())
    m = (S >= lo) & (S <= hi)
    return float(np.polyfit(t[m], np.log(S[m]), 1)[0]) if m.sum() >= 4 else None


def b10_notes(rows, fl, th):
    g = lambda n: rows[n]["gamma_fit"]
    rel = lambda n: 100 * rows[n]["rel_R1"]
    tr = {n: _b10_trace_rate(n) for n in ("S2_nu300_dk_up_j7", "S2_nu300_dk_up_j8")}
    sm = lambda n: rows[n]["S_max_over_Ap"]
    L = ["", "Findings (all from the records above; nothing past a window end is claimed):", "",
         f"- **Ordinary, R1 and R0.** At T_e/T_i = 10, w = 0.04 the fitted sideband growth is {rel('M_mi1836_ord_j5'):+.1f}%, "
         f"{rel('M_mi1836_ord_j8'):+.1f}%, {rel('M_mi1836_ord_j16'):+.1f}% from R1 at K r_D = 0.0125, 0.02, 0.04 (m_i/m_e = 1836) and "
         f"{rel('M_mi100_ord_j7'):+.1f}%, {rel('M_mi100_ord_j10'):+.1f}%, {rel('M_mi100_ord_j16'):+.1f}% (m_i/m_e = 100): inside the declared 15% "
         "(G1) and 30% against R0 (G1b, 7/7). R0 and R1 differ by less than 1% (ion kinetic corrections to the fluid sound response are "
         "negligible at T_e/T_i = 10 here), so the fit is a test of the Zakharov structure itself. Growth is 0.7-1.8e-3 omega_pe at "
         "1836 (an e-folding of 500-1400 / omega_pe), the window is 2.3 e-folds, the pump changed by at most 0.25% at every window end "
         "(G4 19/21; the two non-passes are the S2 points that never reached a window), ledger <= 1.1e-9 and Gauss <= 1e-13 of the pump scales.",
         f"- **Closure sensitivity.** The Hermite closure is the weak point. nu = 1, 3, 10, 40 all blew up in shakedown runs (not records) "
         "and nu = 100 is the smallest value found stable; a run with nu = 300 changes the ordinary growth by "
         f"{100 * (g('R_nu300_ord_j8') / g('M_mi1836_ord_j8') - 1):+.1f}% (m_i/m_e = 1836, j = 8) and "
         f"{100 * (g('S3_nu300_mi100_ord_j10') / g('M_mi100_ord_j10') - 1):+.1f}% (m_i/m_e = 100, j = 10): the declared G2 gate (3%) fails, so the "
         "ordinary agreement above carries a closure uncertainty of 5-9%, and the sign (nu = 300 is further below R1) suggests the nu -> 0 value "
         "is closer to R1. Nx + 12 reproduces the base to 12 digits. The Nn = 64 refinements (nu = 100, and ordinary also at nu = 300) ended "
         "in a closure blow-up (G3 failed); the Nn refinement gate is therefore NOT met and no Nn-converged claim is made.",
         f"- **Dark upper mode.** The derived dark reference R1 agrees with the fit to {rel('M_mi1836_dk_up_j5'):+.1f}% ({rel('S3_nu300_dk_up_j5'):+.1f}% "
         f"at nu = 300) at the R1 maximum for m_i/m_e = 1836 (j = 5) and {rel('M_mi100_dk_up_j5'):+.1f}% (m_i/m_e = 100, j = 5), and over-predicts at "
         f"the other K: {rel('M_mi100_dk_up_j3'):+.1f}% (j = 3), {rel('M_mi100_dk_up_j7'):+.1f}% (j = 7), {rel('M_mi100_dk_up_j10'):+.1f}% (j = 10); "
         f"the nu = 300 repeat gives {rel('S3_nu300_mi100_dk_up_j3'):+.1f}%, {rel('S3_nu300_mi100_dk_up_j5'):+.1f}%, {rel('S3_nu300_mi100_dk_up_j7'):+.1f}%, "
         f"{rel('S3_nu300_mi100_dk_up_j10'):+.1f}%, i.e. the dark departure is not a closure effect. G1 fails at 4 of 5 m_i/m_e = 100 dark "
         "points and passes only at j = 5. The derived dark envelope (g0, eps_g', Doppler) is therefore the right structure at the peak and "
         "incomplete away from it; which neglected term (low-frequency dark force, second-order electron coupling of the mixed mode, "
         "branch mixing at the sidebands) is responsible was not isolated.",
         "- **Dark upper, m_i/m_e = 1836, j = 7, 8 at nu = 100: failed (G1, G3).** The runs show growth 3-4 times R1 followed by an explosive "
         "stage and ended in a step-budget failure. Re-running them at nu = 300 (S2) removes the explosive stage (G3 ok) and shows slow growth "
         f"(post-hoc trace estimate, S in [1.5e-3, 2e-2] A_p: j = 7 {1e3 * tr['S2_nu300_dk_up_j7']:.2f}e-3 against R1 {1e3 * rows['S2_nu300_dk_up_j7']['gamma_R1']:.2f}e-3; "
         f"j = 8: S_max {sm('S2_nu300_dk_up_j8'):.3f} A_p at t_max, no window), so the nu = 100 anomaly is a closure artifact, and no gated "
         "agreement exists for these two points.",
         f"- **Dark lower mode and stability.** R1 predicts weak growth (0.25e-3 for m_i/m_e = 1836, j = 5; 0.77e-3 for 100, j = 3), which the declared "
         f"windows cannot resolve; the records reached S_max = {sm('M_mi1836_dk_lo_j5'):.4f} A_p and {sm('M_mi100_dk_lo_j3'):.4f} A_p (no window, no gate).",
         "- **Dark change.** Measured gamma_dark/gamma_ord at the same K and w: see the table; at the common peak points the dark upper mode grows "
         "faster than the ordinary one (m_i/m_e = 1836, j = 5: x1.10 against R1 x1.19; m_i/m_e = 100, j = 7: x1.31 against x1.44) and slower "
         "at the larger K (m_i/m_e = 100, j = 10: x0.58 against x0.90). The qualitative R1 statements that the dark modes move the unstable band to "
         "smaller K (stable below K = k0) and reduce the rate at large K are supported by the measured ordering; the K < k0 stability was not run.",
         "- **Outside the Zakharov regime** (no gate). T_e/T_i = 1 (strong ion Landau damping): ordinary "
         f"{rel('O1_tau1_ord_j8'):+.1f}%, dark upper {rel('O1_tau1_dk_up_j5'):+.1f}% from R1 (R1 contains the kinetic ion function; R0 differs from R1 by "
         f"{100 * (rows['O1_tau1_ord_j8']['gamma_R0'] / rows['O1_tau1_ord_j8']['gamma_R1'] - 1):+.1f}% there). k0 r_D = 0.04: ordinary {rel('O3_k04_ord_j5'):+.1f}% from R1; the dark "
         "upper run (R1: stable) failed in the closure. w = 0.15: both runs failed in the closure (step budget, ledger blow-up); no growth "
         "is claimed. The kinetic correction to the Zakharov (R0) growth that these runs can resolve is therefore the 5-14% closure-limited "
         "shift above, not a separate effect.",
         "- **Floquet cross-check (follow-up F1, diagnostic).** The exact tangent flow of the same truncated model over one pump period "
         "(jvp through fixed-step Dopri8, dt = 0.02, Arnoldi) gives gamma_F = "
         f"{1e3 * fl['M_mi100_ord_j10']['gamma_floquet']:.2f}e-3 for m_i/m_e = 100 ordinary j = 10 (fit {1e3 * g('M_mi100_ord_j10'):.2f}e-3, R1 "
         f"{1e3 * rows['M_mi100_ord_j10']['gamma_R1']:.2f}e-3): the fit and the exact flow agree to {100 * (g('M_mi100_ord_j10') / fl['M_mi100_ord_j10']['gamma_floquet'] - 1):+.1f}%. For the dark upper j = 7 (m_i/m_e = 100) it gives "
         f"{1e3 * fl['M_mi100_dk_up_j7']['gamma_floquet']:.2f}e-3 against fit {1e3 * g('M_mi100_dk_up_j7'):.2f}e-3 and R1 {1e3 * rows['M_mi100_dk_up_j7']['gamma_R1']:.2f}e-3 "
         "(not consistent with either; unresolved). The m_i/m_e = 1836 Floquet runs returned 7e-3 and 1.6e-2, which contradict the stable "
         "simulations at the same points; they are a failure of the fixed-step tangent at that mass ratio and are NOT used.",
         "- **Not done:** an Nn-converged result (closure blow-up at Nn = 64), a kinetic Floquet/Silin finite-k pump reference beyond the diagnostic above, "
         "K < k0 runs, the dark lower branch with a resolved window, pump amplitude scans, and any state after a window end (saturation, "
         "soliton formation).", ""]
    return "\n".join(L)


def b10_section():
    d = root / "studies/b10"
    th = json.loads((d / "theory.json").read_text())
    S = json.loads((d / "summary.json").read_text())
    fl = {p.stem[len("floquet_"):]: json.loads(p.read_text()) for p in d.glob("floquet_*.json")}
    rows = S["cases"]
    cm = sorted({json.loads(p.read_text())["repository_commit"][:7] for p in d.glob("[MROS]*.json")})
    commit = ", ".join(cm)
    f = lambda x, n=2: "-" if x is None else f"{x * 1e3:.{n}f}"
    pc = lambda x: "-" if x is None else f"{100 * x:+.1f}%"
    gs = S["gate_summary"]
    L = ["## B10: Langmuir sideband (modulational) instability of a finite-amplitude pump, ordinary and dark "
         "(`python studies/b10.py`, reference `studies/b10_ref.py`)", "",
         f"Records: [studies/b10/](../studies/b10/) (runs made at {commit}); gates, the declared revisions (v2, v3, "
         "follow-ups F1, S2 and S3) and the setup are in the `studies/b10.py` docstring and were committed before the runs they "
         "govern. Units: parent (c = omega_pe = 1), v_te = 0.1 c, T_e/T_i = 10, w = E0^2/(2 n T_e) = 0.04 for the pump "
         "E0 cos(k0 x - w0 t) (E0 the peak Poisson field; this w equals Zakharov's |E|^2/(8 pi n T_e)), k0 = 0.05 (k0 r_D = "
         "0.005), pump at Fourier index n = 2, hypercollision nu = 100, Nn = 32, Nx per run (= 2 ceil(1.5 (max kept index + 1))). "
         "The pump is one eigenvector of the linearized model itself (a pure right-moving wave; for the dark model one of the "
         "two Langmuir-Proca mixed normal modes, 1.17 (upper) or 0.87 (lower) omega_pe at Omega_D = omega_pe, eta = 0.3); "
         "the sidebands are seeded (3e-4 A_p, random phase) with the eigenvectors of the same branch. One sideband pair per run: "
         "the nonlinear terms keep only the Fourier indices {0, j, n, n + j, |n - j|} (pump, sidebands, ion density, mean), "
         "which is the truncation the references assume.", "",
         "**References** (`studies/b10_ref.py`): R0 = Zakharov 1972 system (eqs 1.2, 1.9) with the exact quadratic "
         "Delta_pm = +-v_g K + (3/2) r_D^2 K^2, v_g = 3 k0 r_D^2, and c_s^2 = (T_e + 3 T_i)/M (the paper's printed 3/2 T_i is not "
         "used); its K-dependent limits are eq 2.8 (static: maximum gamma = (w/4) T_e/(T_e + T_i), NOT the printed w/16) and eq 2.9 "
         "(hydrodynamic: gamma -> (mu w/3)^(1/2), mu = m_e/m_i). R1 = semi-kinetic: exact Langmuir roots of the kinetic "
         "dispersion function, Delta_pm = omega(k0 +- K) - omega(k0), kinetic ion response varsigma = 1 + xi Z(xi) with Boltzmann "
         "electrons, and for the dark model the derived coupling: electrons feel the force field E_f = g E with g = 1 + eta^2 Q/(Q - "
         "Omega_D^2), Q = omega^2 - k^2, so the sideband operator is eps_g'(Omega - Delta), the quiver and the ponderomotive "
         "potential scale with g0 and g0^2, and 1 = G (g0/eps_g') E0^2 [1/(Omega - Delta_+) - 1/(Omega + Delta_-)], "
         "G = -varsigma g0^2/(4 w0^2 (T_i + varsigma T_e)). Dark change at this order: the Langmuir wave becomes a mixed mode with "
         f"group velocity of order the Proca one (v_g = {th['cases']['M_mi1836_dk_up_j5']['v_g']:.3f} c for the upper mode against "
         f"{th['cases']['M_mi1836_ord_j5']['v_g']:.4f} c ordinary at k0 = 0.05), a dispersion ~1/(2 omega) instead of 3 r_D^2/2, and "
         "g0 = 1.37 (upper) / 0.76 (lower); the model is stable for K below k0 and the instability moves to K of order 0.1-0.2. "
         "Declared theory checks: "
         f"G0a (R1 static maximum vs eq 2.8: {100 * th['G0a']['rel']:+.2f}%, gate 5%) {'pass' if th['G0a']['pass'] else 'FAIL'}; "
         f"G0b (R0 hydrodynamic limit vs eq 2.9: {th['G0b']['rel']:+.1e}) {'pass' if th['G0b']['pass'] else 'FAIL'}; "
         f"G0c (dark R1 at eta -> 1e-8 equals ordinary R1: {th['G0c']['max_rel']:.1e}) {'pass' if th['G0c']['pass'] else 'FAIL'}.", ""]
    L += [f"**Declared gates (passes / points):** " + ", ".join(f"{g}: {v['pass']}/{v['of']}" for g, v in gs.items()) +
          ". G1 = growth within 15% of R1 at the gated points (window: first S >= 6e-3 A_p to first S >= 6e-2 A_p, "
          "S = rss of the two sideband amplitudes; nothing after the window end is claimed), G1b = within 30% of R0 (ordinary), "
          "G2 = refinement change <= 3%, G3 = ledger <= 1e-6 and Gauss <= 1e-8, G4 = pump amplitude change <= 5% at the window end.", ""]
    L += ["| case | m_i/m_e | model | K r_D | gamma R0 (1e-3) | gamma R1 (1e-3) | gamma sim (1e-3) | sim/R1 - 1 | Floquet (1e-3) | "
          "pump change | gates |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = lambda r: (r["tag"], -r["mi"], r["model"], r["j"])
    for r in sorted((r for r in rows.values() if r["tag"].startswith("M_")), key=order):
        gt = " ".join(f"{g} {'ok' if r[g] else 'FAIL'}" for g in ("G1", "G1b", "G3", "G4") if g in r)
        if not r["gated"]:
            gt = "ungated; " + gt
        fq = fl.get(r["name"]) if r["mi"] < 500 else None  # mi 1836 Floquet runs are rejected (see findings)
        bad = r["status"] != "success"
        L.append(f"| {r['name']} | {r['mi']:g} | {r['model']} | {r['K_rD']:.4f} | {f(r.get('gamma_R0'))} | {f(r['gamma_R1'])} | "
                 f"{'run failed' if bad else f(r['gamma_fit'])} | {'-' if bad else pc(r['rel_R1'])} | {f(fq['gamma_floquet']) if fq else '-'} | "
                 f"{pc(r['pump_drift'])} | {gt}{'' if r['status'] == 'success' else ' [run ended: ' + r['status'] + ']'} |")
    L += ["", "Refinement and follow-ups (change of the fitted growth against the stated base):", "",
          "| case | gamma sim (1e-3) | against | change | G2 | G3 | note |", "|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows.values() if r["tag"].startswith(("R_", "S2"))), key=order):
        base = f"M_mi1836_{r['model']}_j{r['j']}" if r["tag"] != "S2_Nn64nu300" else f"R_nu300_{r['model']}_j{r['j']}"
        bad = r["status"] != "success"
        L.append(f"| {r['name']} | {'run failed' if bad else f(r['gamma_fit'])} | {base} | {pc(r.get('G2_rel_to_base')) if not bad else '-'} | "
                 f"{'ok' if r.get('G2') else ('FAIL' if 'G2' in r else '-')} | {'ok' if r['G3'] else 'FAIL'} | "
                 f"{'ledger/Gauss ' + format(r['ledger_over_U0'], '.0e') + ' / ' + format(r['gauss_over_Ap'], '.0e')} |")
    L += ["", "Outside the Zakharov regime (no pass/fail gate; kinetic corrections are the departures from R0 and R1; "
          "O1: T_e/T_i = 1, O2: w = 0.15, O3: k0 = 0.4 (k0 r_D = 0.04)):", "",
          "| case | model | K r_D | gamma R1 (1e-3) | gamma sim (1e-3) | sim/R1 - 1 | sim/R0 - 1 | G3 |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows.values() if r["tag"].startswith("O")), key=order):
        bad = r["status"] != "success"
        L.append(f"| {r['name']} | {r['model']} | {r['K_rD']:.4f} | {f(r['gamma_R1'])} | {'run failed' if bad else f(r['gamma_fit'])} | "
                 f"{'-' if bad else pc(r['rel_R1'])} | {'-' if bad else pc(r.get('rel_R0'))} | {'ok' if r['G3'] else 'FAIL'} |")
    # dark change at common K
    L += ["", "Dark-coupling change at common K (upper mixed mode against ordinary, same K, same w): measured and R1 ratio of "
          "the growth rates:", "", "| m_i/m_e | j | K r_D | gamma_dark/gamma_ord measured | R1 |", "|---|---|---|---|---|"]
    for mi in (1836.0, 100.0):
        for j in sorted({r["j"] for r in rows.values() if r["tag"].startswith("M_") and r["mi"] == mi and r["model"] == "dk_up"}):
            o = rows.get(f"M_mi{int(mi)}_ord_j{j}")
            dk = rows.get(f"M_mi{int(mi)}_dk_up_j{j}")
            if o and dk and o["gamma_fit"] and dk["gamma_fit"] and o["status"] == "success" and dk["status"] == "success":
                L.append(f"| {mi:g} | {j} | {o['K_rD']:.4f} | {dk['gamma_fit'] / o['gamma_fit']:.2f} | {dk['gamma_R1'] / o['gamma_R1']:.2f} |")
    L += [b10_notes(rows, fl, th)]
    return "\n".join(L)


def b09_closure_lines():
    d = root / "studies/b09"
    pr = {n: json.loads((d / f"closure_probe_{n}_amend1.json").read_text()) for n in ("none_c0", "E_c1", "EB_c1")}
    sp = {T: json.loads((d / f"closure_spectrum_t{T}.json").read_text()) for T in (100, 108)}
    sat = json.loads((d / "closure_saturation.json").read_text())
    L = ["### B09 nonlinear: step collapse = AW truncation instability driven by the magnetic force "
         "(`python studies/b09_closure.py probe|spectrum|sat|satan`)", "",
         "Weibel, k0 = 1.2, seed B_z = 1e-6, chunks of 2 time units with checkpoints; declared criteria and three amendments "
         "(each committed before its runs) in the script docstring. Closures (order-2 hypercollision, rows of degree <= 2 "
         "untouched): E = the SPECTRAX #66 rate c |q/m| sqrt(2N) max|E - <E>|/a; EB = the same with the per-axis Lorentz "
         "force scale |E_i - <E_i>| + sum_j V_j |B_l - <B_l>|, V_j = |u_j| + a_j sqrt(2 N_j).", "",
         "| closure (base level, ordinary) | chunk that fails (20000-step budget) | top-third Hermite energy > 0.1 at |",
         "|---|---|---|"]
    for n, r in pr.items():
        L.append(f"| {n} | t = {r['t_budget_failure']} | t = {r['t_top_third_gt_0.1']} |")
    L += ["", "Frozen-field kinetic spectrum (fields frozen at the c = 0 state, ARPACK on the JVP): largest Re lambda "
          "(the Weibel growth rate is 0.108) and the fraction of that mode's energy in the top third of the Hermite ladder.", "",
          "| t | U_B | c = 0 | E, c = 1 | EB, c = 1 |", "|---|---|---|---|---|"]
    for T, r in sp.items():
        c = {x["closure"]: x for x in r["results"]}
        f = lambda x: f"{x['max_re']:.3g} ({x['top_third_fraction_of_LR_mode']:.2g})"
        L.append(f"| {T} | {r['UB_at_t']:.2e} | {f(c['none'])} | {f(c['E'])} | {f(c['EB'])} |")
    L += ["", "Saturation with EB (to t = 140). B_peak = 2|B_force,z(k0)| at the first maximum of U_B; B_sat = gamma^2/(k0 v_perp). "
          "G3a over all four comparisons at c = 1 fails for every model, because the velocity-doubled run (16,64,32) at c = 1 "
          "fails before the peak; amendment 3 pairs base and velocity-doubled at c = 2.", "",
          "| model | gamma | B_sat | t_peak | B_peak (base c=1) | B_peak/B_sat | x-doubled, c=0.5, c=2: max peak change | "
          "v-doubled c=2 vs base c=2: peak change, T_res | dark/ordinary B_peak (estimate) |", "|---|---|---|---|---|---|---|---|---|"]
    for g in sat["groups"]:
        ch = max(v for k, v in g["peak_agreement"].items() if v is not None)
        p2 = g["pair_c2"]
        L.append(f"| {g['model']} | {g['gamma_root']:.4f} | {g['B_sat_estimate']:.3e} | {g['t_peak']} | {g['B_peak']:.4f} | "
                 f"{g['ratio_to_estimate']:.2f} | {ch:.1e} | {p2['peak_agreement']:.1e}, {p2['T_res']} | "
                 f"{g.get('dark_over_ordinary_B_peak', 1):.3f} ({g.get('dark_over_ordinary_estimate', 1):.3f}) |")
    led = max(v["ledger_rel"] for g in sat["groups"] for v in g["runs"].values())
    L += ["", f"- Ledger <= {led:.1e} of the max field perturbation energy, Gauss <= 1e-14 in every saturation run.",
          "- Mechanism: at c = 0 the frozen-field operator has a growing mode with Re lambda 12-25x the Weibel rate, all in the top "
          "third of the Hermite ladder (the AW truncation instability of the H05 section, here driven by v x B); the electric-only #66 "
          "rate barely changes it (E_x is small in Weibel), the EB rate removes it at t = 100 and reduces it 10x at t = 108.",
          "- EB at c = 1 only postpones the collapse (t = 110 -> 146 at the base level), c = 0.5 becomes truncation-dominated at t = "
          "126-132, and the velocity-doubled level fails at t = 112-118 at c = 1 and 126-138 at c = 2 (there with top-third energy "
          "~1e-6, so that late failure is not the Hermite top-mode instability). Nothing is claimed after the first U_B maximum.",
          "- Claim (within the resolved window, t <= T_res): the first magnetic-energy maximum is at t = 115.5-120.5 with "
          "B_peak = 1.93-1.97 B_sat for ordinary and both dark models, unchanged to <= 0.5% by x-doubling, closure strength 0.5/2 "
          "and (at c = 2) velocity doubling. The dark/ordinary ratio of B_peak follows the bounce estimate to 2.3% (OD 0.3) "
          "and 1.3% (OD 1.2). Filamentation was not run."]
    return "\n".join(L)


def b11_section():
    r = json.loads((root / "studies/b11/run.json").read_text())
    res = {x["case"]: x for x in r["results"]}
    A, B, D = res["A two-beam"], res["B anisotropy 3V"], res["D drift relaxation m_r/m_s=4"]
    C4, C100 = res["C T-equilibration m_r/m_s=4"], res["C T-equilibration m_r/m_s=100"]
    ok = lambda x: "pass" if x["passed"] else "FAIL"
    L = ["## B11: collisional relaxation with the conservative Dougherty operator (`python studies/b11.py`)", "",
         f"Record: [run.json](../studies/b11/run.json) (made at {r['repository_commit'][:7]}, parent "
         f"{r['parent_commit'][:7]} = SPECTRAX integration with #71). Operator, analytic references and gates G1-G5 are "
         "in the script docstring, declared before the recorded run. Homogeneous and uncharged (qs = 0), so the "
         f"collision operator is the whole dynamics; Dopri8, rtol = atol = 1e-12. Overall: {'passed' if r['passed'] else 'FAILED'}.", "",
         "| case | reference | error | conservation | gate |", "|---|---|---|---|---|",
         f"| A two beams (d = 0.6, v_t = 0.5), 1V, Nn = 96, nu = 1 | C_n = C_n(0) exp(-n nu t), shifted-centred basis | "
         f"abs {max(A['mode_abs_errors'].values()):.1e} of C_0; rel {max(A['mode_rel_errors'].values()):.1e} (n = 4, 6, 8) | "
         f"rows 0-2 {A['rows_0_2_change']:.0e} | {ok(A)} |",
         f"| B anisotropy T = (0.25, 0.16, 0.09), 3V, N = 12^3, nu = 0.5 | T_i - theta ~ exp(-2 nu t) | {B['rel_error']:.1e} of theta | "
         f"theta {B['theta_change']:.0e} | {ok(B)} |",
         f"| C T_s = 1, T_r = 0.2, m_r/m_s = 4 | dT ~ exp(-2 (nu_sr m_s + nu_rs m_r) t / (m_s + m_r)) | {C4['rel_error']:.1e} | "
         f"energy {C4['energy_change']:.0e} | {ok(C4)} |",
         f"| C same, m_r/m_s = 100 | same | {C100['rel_error']:.1e} | energy {C100['energy_change']:.0e} | {ok(C100)} |",
         f"| D drift U = (0.3, -0.075), m_r/m_s = 4 | dU ~ exp(-(nu_sr + nu_rs) t / 2) | {D['rel_error']:.1e} | "
         f"momentum {D['momentum_change']:.0e}, energy {D['energy_change']:.0e} | {ok(D)} |",
         "",
         f"- A: the non-Maxwellian part (modes n >= 3) falls from {A['nonmaxwellian_norm'][0]:.3f} to "
         f"{A['nonmaxwellian_norm'][1]:.1e} by t = 5. The relative mode error grows as C_n decays toward the 1e-12 absolute "
         "tolerance; G1 was restated from a whole-window relative error to absolute plus windowed relative after an "
         "uncommitted pre-run (stated in the script).",
         "- B and C use truncated series of Maxwellians whose width differs from the basis. Only rows n <= 2 set the "
         "moments and the operator is lower-triangular in n, so the temperature rates are exact despite the truncation.",
         f"- D: the drift energy becomes heat; final T = ({D['T_final'][0]:.4f}, {D['T_final'][1]:.4f}) from 0.5 each.",
         "- Cross-species coefficients: the parent's #71 form is the LBO-EM of Francisquez et al. (JPP 88, 905880303, "
         "arXiv:2109.10381, eqs 2.18-2.20; checked 2026-10-18), not Greene's beta = 0 form. With their nu_sr = alpha_E "
         "(m_s + m_r)/(m_s n_s) it matches the Morse momentum rate and gives twice the Morse temperature-difference rate "
         "(any mass ratio); their LBO-G with beta = 0 (parent #73, collision_model = 'dougherty_g') matches both. The "
         "conservation and the stated model rates are what B11 tests; no B11 gate compares to the Morse rate.",
         "- Not covered: inhomogeneous collisional runs (collisional Landau damping vs a kinetic root with the Dougherty "
         "operator), dark-coupled collisional runs, and charged two-species relaxation (fields respond to the current)."]
    return "\n".join(L)


def b11_landau_section():
    r = json.loads((root / "studies/b11/landau.json").read_text())
    f = json.loads((root / "studies/b11/landau_followup.json").read_text())
    g, gf = r["gates"], f["gates"]
    g0p, g3p = gf["G0'"], gf["G3b'"]
    ok = lambda x: "pass" if x else "FAIL"
    L = ["### B11 extension: collisional Landau damping, ordinary and dark (`python studies/b11_landau.py`)", "",
         f"Record: [landau.json](../studies/b11/landau.json) (made at {r['repository_commit'][:7]}, parent {r['parent_commit'][:7]}); "
         f"follow-up [landau_followup.json](../studies/b11/landau_followup.json) ({f['repository_commit'][:7]}). B00 setup (v_te = 0.1 c, "
         "eta = 0.3, Omega_D = 1, Nx = 8, B00 fit windows) plus Dougherty self collisions nu (omega_pe units). Two independent "
         "references for the linear Dougherty susceptibility, derived in the script docstring: R1 the Hermite tridiagonal with "
         "collision diagonal nu n (n >= 3; the restoring terms cancel rows 1 and 2) solved at N = 4000 and 8000, and R2 a "
         "continuum form (Fourier transform in v, Lenard-Bernstein kernel along s in [0, k/nu], momentum and temperature "
         "restoring terms closed by a 3x3 system). Both are inserted in the B00 dispersion D_L = (Q - Omega_D^2)(1 + chi) + "
         "eta^2 Q chi.", "",
         "| k lambda_De | model | nu | R2 root | fitted omega (Nn = 256) | rel. error | Nn 128 vs 256 | R1 vs R2 root | max ledger defect |",
         "|---|---|---|---|---|---|---|---|---|"]
    refs = {(x["k_lambda_De"], x["model"], x["nu"]): x for x in r["references"]}
    for x in r["results"]:
        if x["Nn"] != 256:
            continue
        rf = refs[(x["k_lambda_De"], x["model"], x["nu"])]
        L.append(f"| {x['k_lambda_De']} | {x['model']} | {x['nu']:g} | {c(x['omega_ref'])} | {c(x['omega_fit'])} | "
                 f"{x['rel_error']:.1e} | {x['Nn_refinement']:.0e} | {rf['R1_vs_R2']:.0e} | {x['max_ledger_defect']:.0e} |")
    worst = max(x["rel_error"] for x in r["results"] if x["Nn"] == 256)
    rows = max(x["rows_0_2_over_global_max"] for x in f["G3b_prime"])
    L += ["", f"Declared gates: G1 fit within 1e-3 {ok(g['G1'])} (worst {worst:.1e}); G2 Nn refinement {ok(g['G2'])}; "
          f"G4 status {ok(g['G4'])}; G0 {ok(g['G0'])}; G3 {ok(g['G3'])}. **G0 and G3 failed as declared, both mis-specified:** "
          "G0 compared R2 at nu = 1e-6 with the collisionless chi to 1e-6, but the physical change of chi is O(nu) (5.7e-7, "
          "1.9e-6); G3's per-state row ratio is 1 by construction at t = 0 (the Dougherty rate of the seeded Maxwellian is "
          "round-off in every row). The ledger part of G3 passed (<= 1e-17). Follow-up gates declared in a new committed "
          f"script before running: G0' (d(nu) = abs(chi_R2 - chi_wofz) linear in nu) {ok(g0p)}, ratios "
          + ", ".join(f"{q:.4f}" for x in f["G0_prime"] for q in x["ratios"]) +
          f"; G3b' (rows 0-2 <= 1e-12 of the run's max Dougherty rate) {ok(g3p)}: worst {rows:.1e}, i.e. "
          "round-off of the O(350) C_0 row, not a conservation defect; the ledger closes to <= 1e-17.",
          "- Effect of collisions on the root: at k lambda_De = 0.5 the damping falls with nu (ordinary 0.1534 at nu = 0 to "
          "0.0823 at nu = 0.2; dark 0.1391 to 0.0762): the restoring terms keep rows n <= 2 collisionless and the damping of "
          "the n >= 3 tail weakens the Landau phase mixing. At 0.3 the change is non-monotonic and within 7% (0.01262 to "
          "0.01287, 0.01350, 0.01275). R1 and R2 agree to <= 4e-12 in the root; the fits are Nn-converged to <= 6e-7.",
          "- The largest fit errors (5-6e-4, ordinary nu = 0.2) are not from the reference (R1 = R2 to 1e-12); the short B00 "
          "fit windows are the likely cause, not tested here."]
    return "\n".join(L)


def r1_section():
    """R1: finite reservoir (mobile ions) and finite temporal coherence, from studies/r1/*_summary.json."""
    pa, pb = root / "studies/r1/reservoir_summary.json", root / "studies/r1/coherence_summary.json"
    if not (pa.exists() and pb.exists()):
        return ""
    A, B1 = json.loads(pa.read_text()), json.loads(pb.read_text())
    p2 = root / "studies/r1/coherence2_summary.json"
    B2 = json.loads(p2.read_text()) if p2.exists() else None
    ga = A["gates"]
    f = lambda x, fmt="{:g}": "-" if x is None else fmt.format(x)  # noqa: E731
    L = ["## R1: finite dark reservoir and finite temporal coherence (`python studies/r1_reservoir.py`, `python studies/r1_coherence.py`)", "",
         "Gates were declared in `studies/r1_common.py` (commit 4f3e1f0) before any run; records in `studies/r1`. H05 seeded setup "
         "(Nx = 8, pump frame + remap, mobile ions, parent 3de6b2b), omega = omega_L, v_q/v_te is the effective initial force. "
         "Claim windows are min(refinement pairs Nn 32/64 and, where run, Nx 8/16, certified window 1000 / 940, positivity, t reached).", "",
         "### (a) Self-consistent finite reservoir vs matched prescribed drive", "",
         "Free k = 0 Proca mode, Omega_D = omega_L, eta E_D(0) = E0; U_D0/(n T_e V) = u. Linear coupled-oscillator prediction "
         "(Ebar, J, E_D, A_D) with the 4x4 matrix exponential (checked against solve_ivp: "
         f"{ga['G6_max']:.1e}, gate 1e-8).", "",
         "| v_q/v_te | u = U_D0/(n T_e V) | eta | claim window | min U_D/U_D0 kin (t) | min U_D/U_D0 lin (t) | pi/eta | max abs(U_D kin - lin)/U_D0 | "
         "first > 0.1 | W_D/U_D0 kin / lin at window | W_D/(n T_e) | W_ext prescribed/(n T_e) (at t) | t_split kin / lin |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in A["cases"].items():
        if not k.startswith("sc_"):
            continue
        L.append(f"| {v['vq']:g} | {v['u']:g} | {v['eta']:.2e} | {v['claim_window']:.0f} | {v['U_D_min_over_U_D0_kin_in_window']:.3f} ({v['t_min_kin']:.0f}) | "
                 f"{v['U_D_min_over_U_D0_lin_in_window']:.3f} ({v['t_min_lin']:.0f}) | {v['t_first_min_U_D_pred_pi_over_eta']:.0f} | "
                 f"{v['max_dev_U_D_over_U_D0_in_window']:.3f} | {f(v['t_dev_10pct'], '{:.0f}')} | "
                 f"{v['W_D_final_over_U_D0']:.3f} / {v['W_D_lin_at_window_over_U_D0']:.3f} | {v['W_D_over_nTe_at_window_vs_prescribed']:.3g} | "
                 f"{v['W_ext_prescribed_over_nTe_at_window']:.3g} (t = {v['window_vs_prescribed']:.0f}) | {f(v['t_split_kin_10pct'], '{:.0f}')} / {f(v['t_split_lin_10pct'], '{:.0f}')} |")
    L += ["", f"Gates: G1 ledger max {ga['G1_ledger_max']:.1e} ({'pass' if ga['G1_pass'] else 'FAIL'}, tolerance 1e-4); "
          f"G4 mean-field identity max {ga['G4_identity_max']:.1e} ({'pass' if ga['G4_pass'] else 'FAIL'}, 1e-6); "
          f"G5 linear reservoir prediction within 0.1 U_D0 in the window: pass {', '.join(ga['G5_pass_cases']) or 'none'}; "
          f"FAIL {', '.join(ga['G5_fail_cases']) or 'none'}; G7 t_split vs linear within a factor 1.5: FAIL {', '.join(ga['G7_fail_cases']) or 'none'}; "
          f"G2 first non-positive times {ga['G2_first_nonpositive_per_run'] or 'none'}: the two prescribed v_q/v_te = 0.03 runs lose positivity at 886 / 854 (so the comparison with the prescribed drive is cut at 854), the self-consistent u = 10 runs at 968 / 958 (beyond the 940 window).", "",
          "### (b) Finite temporal coherence of a prescribed drive (v_q/v_te = 0.01)", "",
          "Drive E0 cos(omega_L t + Phi(t)); Phi' = piecewise-linear interpolation (knot spacing 5) of iid Gaussian frequency samples, Phi(0) = 0; "
          "the drive is C^1 (no phase resets) and the injected work is the ledger W_ext. 6 kinetic realizations per finite tau_c (seeds 0-5); "
          "400 linear mean-field realizations; derived expectation in `studies/r1_common.py::expected_work`. Late slope = least-squares dW/dt "
          "over the last 20% of the claim window, in code units (energy per volume, n = 1, omega_pe = 1; divide by v_te^2 = 1e-3 for n T_e V per omega_pe^-1).", "",
          "**Declaration defect (v1).** The v1 text declared a phase variance of 2 dt/tau_c; the realized variance is dt/tau_c, so the v1 field "
          "coherence function is exp(-dt/(2 tau_c)): the v1 coherence times are 2 tau_c = 60, 200, 600 omega_pe^-1. The derived expectation "
          "and Monte Carlo use the exact realized covariance, so v1 is internally consistent; only its label was wrong. v1 records are kept and "
          "tabulated with the realized coherence time; v2 (declared in `studies/r1_common.py::V2_NOTE` before it was run) uses variance "
          "2/(tau_c h) so that tau_c is the field coherence time, <exp(i dPhi)> = exp(-abs(dt)/tau_c) (Lorentzian, half width 1/tau_c).", ""]
    for tag, B, mult in (("v1 (realized coherence time = 2 x nominal)", B1, 2), ("v2 (coherence time = nominal)", B2, 1)):
        if B is None:
            continue
        gb = B["gates"]
        L += [f"#### {tag}", "",
              "| coherence time tau omega_pe | claim window | kinetic realizations | late slope kin | late slope lin (same realizations) | "
              "late slope derived expectation | late slope MC (400) | asymptote E0^2 tau wL^2/4 | <W_kin>/<W_lin> late (se) | "
              "first 10% departure | G9 max z | G12 z |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for nm, v in B["tau"].items():
            tau = nm.replace("tau", "")
            lab = tau if tau == "inf" else f"{float(tau) * mult:g}"
            ast = None if v["late_slope_asymptote_E0sq_tau_wL2_over_4"] is None else v["late_slope_asymptote_E0sq_tau_wL2_over_4"] * mult
            L.append(f"| {lab} | {v['claim_window']:.0f} | {v['n_realizations_kinetic']} | {v['late_slope_kin']:.3e} | "
                     f"{v['late_slope_lin_same_real']:.3e} | {v['late_slope_expected']:.3e} | {f(v['late_slope_MC400'], '{:.3e}')} | "
                     f"{f(ast, '{:.3e}')} | {v['ratio_late_mean']:.4f} ({f(v['ratio_late_se'], '{:.1e}')}) | "
                     f"{', '.join(f(p['t_dev_10pct'], '{:.0f}') for p in v['per_realization'])} | {f(v.get('G9_max_z'), '{:.2f}')} | "
                     f"{f(v.get('G12_z_kin_mean_vs_expected'), '{:.2f}')} |")
        L += ["", f"Gates: G1 ledger max {gb['G1_ledger_max']:.1e} ({'pass' if gb['G1_pass'] else 'FAIL'}); G4 identity max {gb['G4_identity_max']:.1e} "
              f"({'pass' if gb['G4_pass'] else 'FAIL'}); G8 quadrature check {gb['G8_quadrature_rel_err']:.1e} ({'pass' if gb['G8_pass'] else 'FAIL'}); "
              f"G9 Monte Carlo vs derived expectation ({'pass' if gb['G9_pass'] else 'FAIL'}); G10 no 10% kinetic-vs-linear departure: "
              + ", ".join(f"{k.replace('tau', '')} {'pass' if v['G10_pass'] else 'FAIL'}" for k, v in B["tau"].items())
              + (f"; G11 realized phase variance rate x tau: {gb['G11_rate_x_tau_c']} ({'pass' if gb['G11_pass'] else 'FAIL'})" if "G11_pass" in gb else "") + ".", ""]
    return "\n".join(L)


def h08_section():
    p = root / "studies/h08/summary.json"
    if not p.exists():
        return ""
    s = json.loads(p.read_text())
    g = s["gates"]
    ok = lambda x: "pass" if x else "FAIL"
    L = ["## H08 pilot: resonant drive with Dougherty e-e and e-i collisions (`python studies/h08.py`)", "",
         f"Summary: [summary.json](../studies/h08/summary.json) (analysis at {s['repository_commit'][:7]}, parent "
         f"{s['parent_commit'][:7]}); runs in `studies/h08/runs`. H05 nxconv setup (mobile ions 1836, pump frame + remap, "
         "#66/#72 field-scaled closure c = 4, noise floor 1e-14, Nn 64) plus nu_dougherty = [[nu_ei, nu_ei], [0, nu_ei "
         "sqrt(m_e/m_i)]]; the parent sets nu_ie = nu_ei m_e/m_i. Convention and limitations (velocity-independent nu, so "
         "absorption is a model value; nu_ee = nu_ei and nu_ii as Z = 1 order-of-magnitude choices; cross-species moments "
         "are Francisquez's LBO-EM (momentum friction at the Morse rate, e-i temperature exchange 2x Morse, negligible at "
         "m_e/m_i) are in the script docstring. HHS-v1-inspired, not a "
         "reproduction.", "",
         f"Gates: G0 homogeneous controls vs the exact driven two-fluid oscillator with drag {ok(g.get('G0'))}; "
         f"G1 nu = 0 reproduces the nxconv record {ok(g.get('G1'))}; G2 collisions do no net work and the ledger closes "
         f"{ok(g.get('G2'))}; G3 mean-field identity with the collisional current term {ok(g.get('G3'))}.", "",
         "| homogeneous control | Ebar error | W_ext error | exact W_ext(1000) |", "|---|---|---|---|"]
    for k, v in s["controls"].items():
        L.append(f"| {k} | {v['Ebar_err']:.1e} | {v['Wext_err']:.1e} | {v['Wext_exact_end']:.3e} |")
    if "G1_reproduction" in s:
        d = s["G1_reproduction"]["max_rel_dev"]
        L += ["", f"G1: nu = 0, v_q 0.03, Nx 32 vs the nxconv record: max deviation dK_e {d['dKe']:.0e}, W_ext {d['Wx']:.0e}."]
    L += ["", "| run | t reached | status | identity R / (w_L^2 E0) | without collisional term | max abs(P_e + P_i)/max abs(P_e) | ledger / W_ext |",
          "|---|---|---|---|---|---|---|"]
    for k, v in s["runs"].items():
        L.append(f"| {k} | {v['t_end']:g} | {v['status']} | {v['max_R_over_wL2E0']:.0e} | {v['max_R_nocoll_over_wL2E0']:.0e} | "
                 f"{v.get('max_Psum_over_Pe', float('nan')):.0e} | {v['max_ledger_over_Wext']:.0e} |")
    L += ["", "Refinement pairs (nxconv rule: last t with 10% agreement; window = min over observables and the collisionless "
          "certified window):", "", "| pair | dK_e | W_ext | T_e/T_i - 1 | Q_e | claim window |", "|---|---|---|---|---|---|"]
    for k, v in s["pairs"].items():
        L.append(f"| {k} | {v['dKe']:g} | {v['Wx']:g} | {v['TeTi']:g} | {v.get('Qe', '-')} | {v['window']:g} |")
    L += ["", "Cycle averages (Nx 32, Nn 64) at the times inside the claim windows; energies per unit box, n = 1:", "",
          "| run | t | W_ext | dK_e / W_ext | dK_i / W_ext | T_e/T_e(0) | T_i/T_i(0) | Q_e (collisional, e) | Q_i |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in s["runs"].items():
        if "_Nx32_Nn64" not in k:
            continue
        vq, nu = k.split("_")[0][2:], k.split("_")[1][2:]
        win = min([pv["window"] for pk, pv in s["pairs"].items() if pk.startswith(f"vq{vq}_nu{nu}_")] or [0.0])
        for t, a in v["cycle_avg"].items():
            if float(t) > win:
                continue
            Te0, Ti0 = v["cycle_avg"][t]["Te"], v["cycle_avg"][t]["Ti"]
            L.append(f"| {k} | {t} | {a['W_ext']:.3e} | {a['dK_e'] / a['W_ext']:.3f} | {a['dK_i'] / a['W_ext']:.1e} | "
                     f"{Te0 / 1e-3:.3f} | {Ti0 / 1e-3:.4f} | {a['Q_e']:.2e} | {a['Q_i']:.2e} |")
    L += ["", "Collisional / collisionless (same Nx 32, Nn 64), cycle averages:", "", "| case | t | W_ext ratio | dK_e ratio |",
          "|---|---|---|---|"]
    for k, v in s["compare_collisionless"].items():
        for t, a in v.items():
            L.append(f"| {k} | {t} | {a['W_ext_ratio']:.3f} | {a['dK_e_ratio']:.3f} |")
    g1c = root / "studies/h08_g1_cause/summary.json"
    if g1c.exists():
        c = json.loads(g1c.read_text())
        d = c["dev_vs_record"]
        L += ["", "G1 cause (follow-up study `studies/h08_g1_cause.py`, declared after the G1 failure; G1 itself is not amended): "
              f"the same nu = 0 case rerun at parent {c['parent_commit'][:7]} with only the restart chunk length changed, "
              "compared with the nxconv record as in G1:", "", "| restart chunk | dK_e deviation | W_ext deviation |", "|---|---|---|"]
        L += [f"| {k} | {v['dKe']:.1e} | {v['Wext']:.1e} |" for k, v in d.items()]
        L += ["", f"Declared criteria: chunk 240 matches the record to 1e-12 {ok(c['criteria']['chunk240_matches_1e-12'])}; chunks 200 and 120 "
              f"deviate by >= 1e-8 {ok(c['criteria']['chunk200_and_120_ge_1e-8'])}. nxconv restarted the adaptive integrator every 240 time "
              "units and H08 every 200 (the restart resets the PID controller and first step); with the chunk matched the new parent "
              "reproduces the record exactly, so the parent change 16400f8..dd80b0f is not the cause. The deviation is the "
              "adaptive-step trajectory difference from a different restart placement; it is a few 1e-6 although each step is "
              "controlled at rtol 1e-10, because the pump-frame resonant dynamics amplify step-level differences. It is far below "
              "the 10% resolution rule. The G1 gate (1e-6) was mis-specified relative to this restart sensitivity: it presumed a "
              "bitwise or tolerance-level match without matching the chunking; the failure is recorded as is."]
    R = s["runs"]
    hot = R["vq0.03_nu0.01_Nx32_Nn64"]["cycle_avg"]["950"]
    ctl = s["controls"]["vq0.03_nu0.01"]["Wext_cycle_avg_950"]
    rc = [v["max_R_nocoll_over_wL2E0"] for k, v in R.items() if "_nu0_" not in k]
    L += ["", "Findings (inside the claim windows only):",
          "- G1 FAILED as declared: the nu = 0 rerun deviates from the nxconv record by "
          f"{s['G1_reproduction']['max_rel_dev']['dKe']:.0e} of the running max in dK_e (gate 1e-6; W_ext "
          f"{s['G1_reproduction']['max_rel_dev']['Wx']:.0e}). The record was made at parent 16400f8, this run at dd80b0f; the "
          "cause is attributed below (restart chunk length, not the parent). It is far below the 10% resolution rule, so the collisionless baseline used below is "
          "this rerun, not the old record.",
          "- Every Nx 32/64 and Nn 64/128 pair agrees to t = 1000 in dK_e, W_ext, T_e/T_i and Q_e; claims stop at 950 for "
          "v_q 0.03 (the collisionless certified window).",
          "- The W_ext ratio collisional/collisionless is the same at v_q 0.01 and 0.03 to 3 digits, and the seeded W_ext "
          f"follows the homogeneous control (v_q 0.03, nu 1e-2, cycle averages at t = 950: {hot['W_ext']:.4f} vs {ctl:.4f}, "
          f"{hot['W_ext'] / ctl - 1:+.1%}): the absorption is set by the uniform driven oscillator with drag; the seeded finite-k dynamics adds little "
          "at these amplitudes.",
          "- With nu_ei = 1e-2 the electron share dK_e/W_ext rises from 0.48 to 0.85 (the rest stays in the field), and "
          "collisions turn quiver energy into heat: T_e/T_e(0) reaches "
          f"{hot['Te'] / 1e-3:.0f} at v_q 0.03 (v_t ~ {np.sqrt(hot['Te']):.2f} c), outside the regime where the "
          "nonrelativistic model is physical, although numerically resolved. Ion heating stays small (T_i/T_i(0) "
          f"{hot['Ti'] / 1e-3:.2f}); the e-to-i collisional exchange Q_i is {hot['Q_i'] / hot['W_ext']:.1e} of W_ext.",
          f"- The collisional current-moment term is O(1) of the drive term in the mean-field identity (up to {max(rc):.1f} "
          "omega_L^2 E0 without it); with it the identity closes to <= 3e-13. Collisions do no net work (|P_e + P_i| <= "
          "2e-11 of |P_e|).",
          "- Not checked: the cross-species beta convention, a velocity-dependent (Coulomb) collision frequency, relativistic "
          "electrons at the hottest point; the seeds are the deterministic H05 gate seeds (one realization)."]
    return "\n".join(L)


def h08_lbog_section():
    p = root / "studies/h08_lbog/summary.json"
    if not p.exists():
        return ""
    s = json.loads(p.read_text())
    g = s["gates"]
    ok = lambda x: "pass" if x else "FAIL"
    L = ["### H08 with the Francisquez LBO-G cross-species form (`python studies/h08_lbog.py`)", "",
         f"Summary: [summary.json](../studies/h08_lbog/summary.json) (analysis at {s['repository_commit'][:7]}, parent "
         f"{s['parent_commit'][:7]}). Same setup as H08 (collision_model = \"dougherty_g\", beta = 0, parent #73, which matches both "
         "Morse rates; the H08 records above use LBO-EM, with 2x Morse electron-ion temperature exchange), Nx 32, Nn 64, "
         "restart chunk 240, T = 1000, v_q/v_te 0.01 and 0.03, nu_ei 1e-3 and 1e-2. Gates declared in the script before "
         f"running: G0 homogeneous controls {ok(g['G0'])}; G2 no net collisional work and ledger {ok(g['G2'])}; G3 mean-field "
         f"identity with the collisional current term {ok(g['G3'])}; G4 (declared small effect: |relative change| <= 2e-2 in "
         f"W_ext, dK_e, T_e, T_i vs the LBO-EM records at the cycle-averaged times) {ok(g['G4'])}.", "",
         "| run | identity R / (w_L^2 E0) | max abs(P_e + P_i)/max abs(P_e) | ledger / W_ext |", "|---|---|---|---|"]
    for k, v in s["runs"].items():
        L.append(f"| {k} | {v['max_R_over_wL2E0']:.0e} | {v['max_Psum_over_Pe']:.0e} | {v['max_ledger_over_Wext']:.0e} |")
    L += ["", "Relative change LBO-G (beta = 0) vs LBO-EM, cycle averages (the claim window of H08 is 1000 at v_q 0.01 and 950 at 0.03):",
          "", "| case | t | W_ext | dK_e | dK_e/W_ext | T_e | T_i | Q_e = -Q_i (exchange) | dK_i |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in s["vs_em"].items():
        for t, a in v.items():
            r = a["rel_change"]
            L.append(f"| {k.replace('_Nx32_Nn64', '')} | {t} | {r['Wx']:+.1e} | {r['dKe']:+.1e} | {r['dKe_over_Wext']:+.1e} | "
                     f"{r['Te']:+.1e} | {r['Ti']:+.1e} | {r['Qe']:+.4f} | {r['dKi']:+.3f} |")
    bad = [(k.replace("_Nx32_Nn64", ""), t, a["rel_change"]["Ti"]) for k, v in s["vs_em"].items() for t, a in v.items()
           if abs(a["rel_change"]["Ti"]) > 2e-2]
    L += ["", "Findings:",
          "- The electron-ion collisional exchange is halved (Q_e = -Q_i changes by -0.4999 to -0.4987 of the EM value in every "
          "case), as expected for a 2x-Morse temperature exchange replaced by the Morse one.",
          "- W_ext is unchanged to <= 9e-8 at all times (the absorption is set by the momentum friction, identical in the two forms); "
          "dK_e and T_e change by <= 1.8e-3 over the window, the electron share dK_e/W_ext by the same.",
          "- G4 FAILED as declared, only through T_i: " + ", ".join(f"{c} at t = {t} ({x:+.1%})" for c, t, x in bad) + ". "
          "The gate treated T_i like the electron observables, but T_i is fed by the e-i exchange itself (ion heating, dK_i changes by "
          "-42% at nu_ei = 1e-2), so the halved exchange shows up there at order one in the relative sense, where T_i barely "
          "moves from its initial value. The gate was wrong about T_i and is not amended; the other three observables pass at all "
          "times, and the T_i change is the measured effect of the prediction (halved exchange).",
          "- Not checked: other beta values, no refinement pairs for LBO-G (the H08 EM pairs apply to the same grid), "
          "the restart chunk differs from the EM records (240 vs 200), which alone gives about 1e-5 (G1 cause study)."]
    return "\n".join(L)


def b08_section():
    d = root / "studies/b08"
    rot, wav, flx = (json.loads((d / f"{n}.json").read_text()) for n in ("rotation", "waves", "flux"))
    rr, gr = rot["results"], rot["gates"]
    dense = [r for r in rr if r["n"] == 1.0]
    tp = [r for r in rr if r["n"] < 1e-3]
    L = ["## B08: uniform-B0 rotation and cold/warm electromagnetic waves, ordinary and dark "
         "(`python studies/b08.py rotation|waves|flux`)", "",
         f"Records: [rotation.json](../studies/b08/rotation.json), [waves.json](../studies/b08/waves.json), "
         f"[flux.json](../studies/b08/flux.json) (made at {rot['repository_commit'][:7]} / {wav['repository_commit'][:7]} / "
         f"{flx['repository_commit'][:7]}). Gates were declared in the script docstring before the runs.", "",
         "### Rotation in a uniform B0 (Nx = 1, Hermite 4x4x4)", "",
         f"Drifting anisotropic Maxwellian, U0 = {rot['settings']['U0']}, v_t = {rot['settings']['v_t']}, |B0| = "
         f"{rot['settings']['B0_mag']}, B0 along z, x, (1,1,1) and (0.3,-0.5,0.8), q = -1 and +1, density 1 (the "
         "perpendicular mean current drives the mean field E_0) and 1e-4. Reference: matrix exponential of the exact "
         "6x6 mean-flow/mean-field system and P(t) = R P0 R^T.", "",
         "| n | worst U error / U0 | worst E_0 error | worst covariance error / P0 | worst gyration-frequency error "
         "| worst ledger / K0 | K_perp change |", "|---|---|---|---|---|---|---|"]
    for lab, g in (("1", dense), ("1e-4", tp)):
        L.append(f"| {lab} | {max(r['U_rel_error'] for r in g):.1e} | {max(r['E0_rel_error'] for r in g):.1e} | "
                 f"{max(r['P_rel_error'] for r in g):.1e} | {max(r['omega_rel_error'] for r in g):.1e} | "
                 f"{max(r['ledger_rel'] for r in g):.1e} | {max(r['Kperp_rel_change'] for r in g):.1e} |")
    fails = []
    for key, rk in (("U", "U_rel_error"), ("P", "P_rel_error"), ("omega", "omega_rel_error"), ("ledger", "ledger_rel")):
        bad = [r for r in rr if r[rk] > gr[key]]
        if bad:
            fails.append(f"{key} (gate {gr[key]:g}: {len(bad)} of {len(rr)} runs, worst {max(r[rk] for r in bad):.1e}, "
                         f"all with n = {sorted(set(r['n'] for r in bad))})")
    kp = [r for r in tp if r["Kperp_rel_change"] > gr["Kperp_test_particle"]]
    if kp:
        fails.append(f"K_perp test-particle conservation (gate {gr['Kperp_test_particle']:g}: {len(kp)} of {len(tp)} runs, "
                     f"worst {max(r['Kperp_rel_change'] for r in kp):.1e})")
    L += ["", f"All {len(rr)} runs succeed. Mean flow, mean field and the full covariance tensor follow the exact rotation "
          f"to {max(r['U_rel_error'] for r in rr):.0e} / {max(r['P_rel_error'] for r in rr):.0e} for every direction and both "
          "charge signs (including the off-diagonal covariance that an oblique B0 generates from the diagonal start, up to "
          f"{max(r['P_offdiag_max'] for r in rr):.1e} absolute against a largest P0 entry of "
          f"{max(rot['settings']['v_t']) ** 2:.0e}). The kinetic energy changes by up to "
          f"{100 * max(r['K_rel_change'] for r in dense):.0f}% at n = 1, entirely through the electric work: K - K0 - W_em "
          "closes to round-off, so the magnetic work is zero. "
          + ("Gates failed: " + "; ".join(fails) + ". Causes: at n = 1e-4 the coupled mean-field mode has "
             "omega ~ omega_p = 0.01, less than one period in the window, so its fitted frequency is poorly determined "
             "(the trajectories match to 1e-13); K0 is 5e-8 there, so the ledger ratio is the absolute round-off "
             "(about 1e-17) over a small K0; and K_perp is exchanged with E_0 at order omega_p^2/Omega_c^2 = 4e-4, so the "
             "test-particle gate was mis-specified (the exact 6x6 reference, which includes E_0, is matched)."
             if fails else "All declared gates pass.")]

    wr, gw, sw = wav["results"], wav["gates"], wav["settings"]
    L += ["", "### Cold magnetized electromagnetic waves (k along x, Nx = 5)", "",
          f"Electrons, omega_pe = 1, |Omega_ce| = {sw['B0']}; k parallel (B0 along x) and perpendicular (B0 along z) at "
          f"k = 0.5 and 1.1; dark eta = {sw['eta']}, Omega_D = {sw['Omega_D']} omega_pe. Reference: eigenvalues of the full cold "
          "field-current operator (no determinant). Every eigenfrequency also annihilates the conductivity-based field "
          f"system M(omega) (smallest singular value / largest <= "
          f"{max(m['sigma_residual'] for e in wav['eigen'] for m in e['modes']):.0e}). Each positive-frequency eigenvector "
          f"seeds a Hermite run (amplitude {sw['amplitude']}), fitted with one mode on [0, min(6 periods, 200)]. "
          "dark fraction = |E_D|^2/(|E|^2+|E_D|^2) of the eigenvector.", "",
          "| geometry | model | k | branch | dark frac. | cold omega | fit (v_t = 0.01, Nn = 4) | rel. error | "
          "error at v_t = 0.005 | Nn 4 -> 8 change |", "|---|---|---|---|---|---|---|---|---|---|"]
    def get(r, vt, Nn):
        m = [x for x in wr if x["geometry"] == r["geometry"] and x["model"] == r["model"] and x["k"] == r["k"]
             and x["omega_cold"] == r["omega_cold"] and x["v_t"] == vt and x["Nn"] == Nn]
        return m[0] if m else None
    base = [r for r in wr if r["v_t"] == 0.01 and r["Nn"] == 4]
    dn, ratios = [], []
    for r in base:
        h, f = get(r, 0.005, 4), get(r, 0.01, 8)
        ch = abs(complex(*f["omega_fit"]) - complex(*r["omega_fit"])) / r["omega_cold"] if f else float("nan")
        dn.append(ch)
        if h:
            ratios.append(r["rel_error"] / max(h["rel_error"], 1e-300))
        L.append(f"| {r['geometry']} | {r['model']} | {r['k']:g} | {r['kind']} | {r['dark_fraction']:.2f} | "
                 f"{r['omega_cold']:.6f} | {c(r['omega_fit'])} | {r['rel_error']:.1e} | "
                 f"{h['rel_error'] if h else float('nan'):.1e} | {ch:.0e} |")
    ok = [r for r in wr if r["status"] == "success"]
    led_k = max(r["ledger_rel"][0] for r in ok)
    led_f = max(max(r["ledger_rel"][1:]) for r in ok)
    gmax = max(max(r["gauss_rel"]) for r in ok)
    nbad = sum(r["rel_error"] > gw["rel_error_vt0.01"] for r in base)
    mixed = [r["dark_fraction"] for r in base if r["k"] == 1.1 and r["model"] == "dark" and 0.2 < r["dark_fraction"] < 0.8]
    lx = min((r for r in base if r["geometry"] == "perpendicular" and r["model"] == "ordinary" and r["k"] == 1.1),
             key=lambda r: r["omega_cold"])
    lxh = get(lx, 0.005, 4)
    nref = sum(x > gw["Nn_refinement"] for x in dn)
    L += ["", f"{len(ok)} of {len(wr)} runs succeed. Gates: v_t = 0.01 error <= {gw['rel_error_vt0.01']:g}: "
          f"{len(base) - nbad} of {len(base)} pass (worst {max(r['rel_error'] for r in base):.1e}); Nn refinement <= "
          f"{gw['Nn_refinement']:g}: {len(dn) - nref} of {len(dn)} pass (worst {max(dn):.0e}); kinetic ledger "
          f"{led_k:.0e} U_wave (gate {gw['kinetic_ledger_rel']:g}), field ledgers {led_f:.0e} U_wave (gate "
          f"{gw['field_ledger_rel']:g}), Gauss residuals {gmax:.0e} of |E_k| (gate {gw['gauss_rel']:g}). The remaining error "
          "is the warm correction: halving v_t divides it by "
          f"{np.median(ratios):.1f} (median; range {min(ratios):.1f}-{max(ratios):.1f}; v_t^2 scaling predicts 4), while "
          "Nn = 4 -> 8 changes nothing at that level. Mixing (eta = 0.3) moves each plasma branch and opens avoided "
          "crossings with the transverse and longitudinal Proca branches; near k = 1.1 the R and upper-X branches cross "
          f"sqrt(k^2 + Omega_D^2) and those eigenvectors are strongly mixed (dark fraction {min(mixed):.2f}-{max(mixed):.2f}). "
          f"The lower X branch at k = 1.1 (omega = {lx['omega_cold']:.3f}, next to 2|Omega_ce| = {2 * sw['B0']:g}) has the "
          f"largest warm damping, Im omega = {lx['omega_fit'][1]:.1e} at v_t = 0.01 and {lxh['omega_fit'][1]:.1e} at 0.005 "
          f"(dark: {max(abs(r['omega_fit'][1]) for r in base if r['model'] == 'dark' and r['geometry'] == 'perpendicular'):.1e} "
          "at v_t = 0.01); the v_t^2 scaling is consistent with second-harmonic cyclotron (Bernstein) coupling, which cold "
          "theory omits (not checked against a warm dispersion relation here)."]

    fr = flx["results"]
    L += ["", "### Proca energy flux in a packet", "",
          f"Packets of five Proca modes (transverse, longitudinal, mixed polarization) with cold plasma, B0, eta = "
          f"{flx['settings']['eta']}, evolved to t = 5 and checked pointwise on a {flx['settings']['fine_grid']}-point grid: "
          "d u_D/dt + d S_D/dx + P_D = 0 with S_D = Om0^2[(E_D x B_D)_x + Omega_D^2 phi_D A_D,x]. Residual / max|du_D/dt|: "
          + ", ".join(f"{r['packet']} {r['residual_rel']:.0e} (without phi_D A_D: {r['residual_rel_without_phiA']:.2f})"
                      for r in fr) + f". Gate {flx['gates']['residual']:g}: "
          + ("pass." if all(r["residual_rel"] <= flx["gates"]["residual"] for r in fr) else "FAIL.")
          + " The phi_D A_D term is required whenever the packet has a longitudinal part."]
    return "\n".join(L)


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

{b04_b05_section()}

{b08_section()}

{b09_section()}

{b10_section()}

{b11_section()}

{b11_landau_section()}

{h08_section()}

{h08_lbog_section()}

{r1_section()}

{r1nl_section()}

{r2_section()}

{r3_section()}

{third_round()}

{lanes()}

{highk_section()}

{nxconv_section()}

{gpures_section()}

{rclosure_section()}

{stiffdiag_section()}

{expint_section()}

{stream_section()}

{otsi_section()}

{dai_section()}

{sw_section()}

{antisym_section()}

{c09_section()}

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
Maxwell and Proca, continuity, both constraints and B_D = curl A_D preserved by the RHS, zero magnetic work), B08 oblique-B0 rotation of the mean flow, mean field and full covariance against the exact
matrix exponential for both charge signs (nonzero perpendicular current), C07
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
    b4 = json.loads((root / "studies/b04_b05/b04.json").read_text())["results"]
    b5 = json.loads((root / "studies/b04_b05/b05.json").read_text())["results"]
    for te in (5.0, 20.0):
        o, d = [[r for r in b4 if r["Te_over_Ti"] == te and r["k_lambda_De"] == 0.3 and r["model"] == m
                 and r["Nn"] == 128 and r["Nx"] == 5][0] for m in ("ordinary", "dark")]
        L.append(f"| B04 ion-acoustic $T_e/T_i={te:g}$, $k\\lambda_{{De}}=0.3$, $\\omega$ | {c(o['omega_fit'])} / {c(o['omega_root'])} | "
                 f"{c(d['omega_fit'])} / {c(d['omega_root'])} |")
    o, d = [[r for r in b5 if r["mass_ratio"] == 1836.0 and r["k_u0"] == 1.0 and r["model"] == m
             and r["Nn"] == 16 and r["Nx"] == 5][0] for m in ("ordinary", "dark")]
    L.append(f"| B05 Buneman $m_i/m_e=1836$, $ku_0=1$, growth | {o['growth_fit']:.6f} / {o['root'][1]:.6f} | "
             f"{d['growth_fit']:.6f} / {d['root'][1]:.6f} |")
    w8 = json.loads((root / "studies/b08/waves.json").read_text())["results"]
    for geom, kind in (("parallel", "R"), ("perpendicular", "X")):
        o, d = [max((r for r in w8 if r["geometry"] == geom and r["model"] == m and r["k"] == 1.1 and r["kind"] == kind
                     and r["v_t"] == 0.01 and r["Nn"] == 4), key=lambda r: r["omega_cold"]) for m in ("ordinary", "dark")]
        L.append(f"| B08 cold {geom} {kind} (upper), $k=1.1$, $\\omega$ ($v_t=0.01$; dark column $\\Omega_D=1.2$) | {c(o['omega_fit'])} / {o['omega_cold']:.6f} | "
                 f"{c(d['omega_fit'])} / {d['omega_cold']:.6f} |")
    b9 = json.loads((root / "studies/b09/linear.json").read_text())["results"]
    for case in ("weibel", "filament"):
        o, d = [[r for r in b9 if r["case"] == case and r["model"] == m and r["k"] == 1.2 and (r["Nx"], r["Nn"]) == (5, 64)][0]
                for m in ("ordinary", "dark_OD0.3")]
        L.append(f"| B09 {case} $k=1.2$, growth (dark column $\\Omega_D=0.3$) | {o['omega_fit'][1]:.6f} / {o['gamma_root']:.6f} | "
                 f"{d['omega_fit'][1]:.6f} / {d['gamma_root']:.6f} |")
    sm = json.loads((root / "studies/b10/summary.json").read_text())["cases"]
    o, d = sm["M_mi1836_ord_j8"], sm["M_mi1836_dk_up_j5"]
    L.append(f"| B10 Langmuir sideband growth ($m_i/m_e=1836$, $w=0.04$, ordinary $Kr_D=0.02$; dark upper mode $Kr_D=0.0125$) vs semi-kinetic Zakharov R1 "
             f"| {o['gamma_fit']:.6f} / {o['gamma_R1']:.6f} | {d['gamma_fit']:.6f} / {d['gamma_R1']:.6f} |")
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
    w = max(d["best_fd_rel_err"] for _, _, d in c09_rows(c09_records()))
    wa = max(d["ad_rel_err_vs_analytic"] for _, _, d in c09_rows(c09_records()) if "analytic" in d)
    L.append(f"| C09 gradients (eigenfrequencies, W_D, Landau abs(E_k)^2): AD vs FD plateau / analytic | - | "
             f"worst {w:.0e} / {wa:.0e} over {len(c09_rows(c09_records()))} derivatives |")
    return "\n".join(L)




def _between(text, tag, body):
    """Replace the README block between <!-- tag --> and <!-- /tag --> (generated from the records)."""
    return re.sub(rf"(<!-- {tag} -->\n).*?(\n<!-- /{tag} -->)", lambda m: m.group(1) + body + m.group(2), text,
                  flags=re.S)


readme = (root / "README.md").read_text()
readme = _between(readme, "tests-paragraph", para)
readme = _between(readme, "tests-table", readme_table())
(root / "README.md").write_text(readme)
