"""Dark-SPECTRAX: kinetic Maxwell-Proca companion to SPECTRAX."""

from ._model import MODES, PARENT_COMMIT, Model, hermite_index, inner, rhs
from ._diagnostics import charge_density, energies, fit_modes, gauss_residuals, moments
from ._simulation import consistent_fields, maxwellian, proca_mode, run, save_record

__version__ = "0.1.0"
__all__ = ["MODES", "PARENT_COMMIT", "Model", "hermite_index", "inner", "rhs", "charge_density",
           "energies", "fit_modes", "gauss_residuals", "moments", "consistent_fields",
           "maxwellian", "proca_mode", "run", "save_record", "__version__"]
