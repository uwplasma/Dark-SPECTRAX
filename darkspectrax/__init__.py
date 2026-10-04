"""Dark-SPECTRAX: kinetic Maxwell-Proca companion to SPECTRAX."""

from ._model import FRAMES, MODES, PARENT_COMMIT, Model, basis_of, hermite_index, inner, rhs
from ._diagnostics import charge_density, energies, fit_modes, gauss_residuals, moments
from ._simulation import (adapt_basis, consistent_fields, maxwellian, proca_mode, run, run_adaptive,
                          save_record)

__version__ = "0.1.0"
__all__ = ["FRAMES", "MODES", "PARENT_COMMIT", "basis_of", "adapt_basis", "run_adaptive",
           "Model", "hermite_index", "inner", "rhs", "charge_density",
           "energies", "fit_modes", "gauss_residuals", "moments", "consistent_fields",
           "maxwellian", "proca_mode", "run", "save_record", "__version__"]
