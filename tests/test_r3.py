"""R3 study helpers: the refined Landau-Zener exponent reduces to the simple one, and the cold reference conserves energy."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
import r3_ref as r3  # noqa: E402


def test_refined_law_reduces_to_landau_zener_for_small_eta():
    case = r3.Case(0.02, 8.0, None, "y")
    lz, rf = r3.lz_prediction(case), r3.refined_prediction(case)
    assert abs(lz["trans_dark"] - rf["trans_dark"]) < 2e-3
    assert abs(lz["refl_photon"] - rf["refl_photon"]) < 2e-3


def test_cold_reference_conserves_energy_and_flux_budget():
    r = r3.run_cold(r3.Case(0.3, 4.0, None, "y"), N=1024, rtol=1e-10)
    assert r["energy_drift"] < 1e-8
    assert abs(r["budget"]["residual"]) < 1e-6
    assert abs(sum(r["channels"][k] for k in ("refl_dark", "refl_photon", "trans_dark", "beyond")) - 1) < 2e-3


def test_longitudinal_local_dispersion_matches_cold_matrix():
    assert r3.local_dispersion_check() < 1e-12
    assert np.isfinite(r3.wkb_longitudinal(r3.Case(0.3, 4.0, None, "x"))["T_wkb"])
