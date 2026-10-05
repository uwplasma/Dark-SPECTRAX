"""Maxwell-Proca right-hand side built on the SPECTRAX Hermite-Fourier operators.

Units are those of the parent (SPECTRAX ``ab87385``): velocities in units of
``c`` (the curl coefficient is 1), and for every species

    Omega_cs[s] = Omega_cs[0] / m_s,   n_s = alpha_x alpha_y alpha_z C_000,

so species 0 has unit reference mass.  The ordinary normalized Ampere source
is ``-J/Omega_cs[0]`` and the field-energy prefactor is ``Omega_cs[0]**2``.
The dark fields use the same layout, mask, source (times ``eta``) and energy
prefactor.  All parent calls go through the small adapter section below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Version-pinned adapter to the parent (private names at the pinned commit).
# ---------------------------------------------------------------------------
from spectrax._initialization import initialize_simulation_parameters as _parent_init
from spectrax._diagnostics import _rfft_weights as _parent_rfft_weights
from spectrax._model import Hermite_Fourier_system as _parent_kinetic
from spectrax._model import plasma_current as _parent_current
from spectrax._model import uniform_acceleration as _parent_uniform_acceleration
from spectrax._simulation import _twothirds_mask as _parent_mask
from spectrax._simulation import cross_product as _cross

jax.config.update("jax_enable_x64", True)

PARENT_COMMIT = "16400f887f0609b7699cd5df5b1188fbfb74ef5e"  # SPECTRAX integration/dark-baseline, not a release
MODES = ("ordinary", "prescribed_drive", "self_consistent")
FRAMES = ("fixed", "pump")
_FFT_AXES = (-1, -3, -2)  # parent convention: rfft along x, stored on axis -2

__all__ = ["Model", "MODES", "FRAMES", "PARENT_COMMIT", "rhs", "inner", "hermite_index", "basis_of"]


def hermite_index(n, m, p, Nn, Nm):
    """Flat parent Hermite index of (n_x, n_y, n_z) = (n, m, p); storage is (p, m, n)."""
    return n + Nn * (m + Nm * p)


@dataclass(frozen=True, eq=False)
class Model:
    """Physical and numerical configuration (parent-normalized units).

    Grid sizes are the full transform lengths ``(Nx, Ny, Nz)``; arrays are
    stored as ``(..., Ny, Nx//2+1, Nz)`` exactly as in the parent.
    """

    Nx: int = 1
    Ny: int = 1
    Nz: int = 1
    Nn: int = 4
    Nm: int = 1
    Np: int = 1
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    qs: tuple = (-1.0,)
    Omega_cs: tuple = (1.0,)
    alpha_s: tuple = (0.1, 0.1, 0.1)
    u_s: tuple = (0.0, 0.0, 0.0)
    nu: float = 0.0
    rho_background: float = 0.0
    mode: str = "ordinary"
    eta: float = 0.0
    Omega_D: float = 1.0
    E_drive: tuple = (0.0, 0.0, 0.0)
    omega_drive: float = 1.0
    phase_drive: float = 0.0
    sweep_drive: float = 0.0
    frame: str = "fixed"
    _p: dict = field(default=None, repr=False)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {self.frame!r}")
        Ns = len(self.qs)
        if len(self.Omega_cs) != Ns or len(self.alpha_s) != 3 * Ns or len(self.u_s) != 3 * Ns:
            raise ValueError("qs, Omega_cs, alpha_s (3*Ns) and u_s (3*Ns) are inconsistent")
        if self.mode != "prescribed_drive" and any(e != 0 for e in self.E_drive):
            raise ValueError("E_drive is only used in mode='prescribed_drive'")
        if self.mode == "ordinary" and self.eta != 0:
            raise ValueError("mode='ordinary' has no dark coupling; set eta=0")
        self.p  # build parent grids eagerly, never inside a JAX trace

    @property
    def Ns(self):
        return len(self.qs)

    @property
    def p(self):
        """Parent parameter dictionary (grids, ladders, mask), built once."""
        if self._p is None:
            user = {k: jnp.asarray(getattr(self, k), dtype=float)
                    for k in ("qs", "Omega_cs", "alpha_s", "u_s", "Lx", "Ly", "Lz", "nu")}
            user["D"] = 0.0
            p = _parent_init(user, self.Nx, self.Ny, self.Nz, self.Nn, self.Nm, self.Np,
                             self.Ns, 2, 0.01)
            p["mask23"] = _parent_mask(self.Ny, self.Nx, self.Nz)
            object.__setattr__(self, "_p", p)
        return self._p

    @property
    def masses(self):
        om = np.asarray(self.Omega_cs, float)
        return om[0] / om

    @property
    def shape(self):
        return (self.Ny, self.Nx // 2 + 1, self.Nz)

    def zeros(self):
        """Empty structured state."""
        c = jnp.complex128
        return {"Ck": jnp.zeros((self.Ns * self.Nn * self.Nm * self.Np, *self.shape), c),
                "Fk": jnp.zeros((6, *self.shape), c),
                "Dk": jnp.zeros((10, *self.shape), c),
                "W": jnp.zeros(3, c),
                **({"B": jnp.stack([jnp.asarray(self.u_s, float), jnp.asarray(self.alpha_s, float)]).astype(c)}
                   if self.frame == "pump" else {})}

    def drive(self, t):
        """Uniform prescribed field E0 cos(theta), theta = (omega + sweep t) t + phase.

        The instantaneous frequency is d theta/dt = omega + 2 sweep t (not omega + sweep t).
        The electron acceleration amplitude is Omega_cs[0] |E0| in parent units."""
        theta = (self.omega_drive + self.sweep_drive * t) * t + self.phase_drive
        return jnp.asarray(self.E_drive) * jnp.cos(theta)


def basis_of(model: Model, y=None):
    """Hermite basis ``(u_s, alpha_s)``, each shape ``(3*Ns,)``: the state's ``B`` in the pump frame,
    otherwise the model's constants."""
    if y is not None and "B" in y:
        return jnp.real(y["B"][0]), jnp.real(y["B"][1])
    return model.p["u_s"], model.p["alpha_s"]


def _weights(Nx, shape):
    """Parseval weights for rfft storage along x: <f g> = sum w Re(f_k g_k*) (parent's, from PR #9)."""
    return _parent_rfft_weights(Nx, shape[1])[None, :, None]


def inner(Nx, a, b):
    """Box average <a.b> of real fields from parent-layout Fourier coefficients.

    ``a`` and ``b`` have shape ``(ncomp, Ny, Nx//2+1, Nz)``; components are summed.
    """
    w = _weights(Nx, a.shape[-3:])
    return jnp.sum(w * jnp.real(a * jnp.conj(b)))


def rhs(t, y, model: Model):
    """Structured RHS: one parent kinetic call, current computed once."""
    p, m = model.p, model
    Ns, Nn, Nm, Np, Nx, Ny, Nz = m.Ns, m.Nn, m.Nm, m.Np, m.Nx, m.Ny, m.Nz
    Ck, Fk, Dk = y["Ck"], y["Fk"], y["Dk"]
    mask = p["mask23"]
    om0 = p["Omega_cs"][0]
    nabla = p["nabla"]
    dark = m.mode == "self_consistent"

    Fforce_k = Fk + m.eta * Dk[:6] if dark else Fk
    F = jnp.fft.irfftn(Fforce_k * mask, s=(Nz, Ny, Nx), axes=_FFT_AXES, norm="forward")
    Ed = m.drive(t)
    if m.mode == "prescribed_drive":
        F = F.at[:3].add(Ed[:, None, None, None])
    C = jnp.fft.irfftn(Ck * mask, s=(Nz, Ny, Nx), axes=_FFT_AXES, norm="forward")
    u_s, a_s = basis_of(m, y)
    # Pump frame: each centre follows (q/m)(E0 + u x B0) of the total uniform force field
    # (ordinary + eta*dark + drive), and the kernel drops that force (exact change of variables).
    pump = m.frame == "pump"
    u_dot, F0 = _parent_uniform_acceleration(F, u_s, p["qs"], p["Omega_cs"], Ns) if pump else (None, None)
    dCk = _parent_kinetic(
        Ck, C, F, p["kx_grid"], p["ky_grid"], p["kz_grid"], p["k2_grid"], p["collision_matrix"],
        p["sqrt_n_plus"], p["sqrt_n_minus"], p["sqrt_m_plus"], p["sqrt_m_minus"],
        p["sqrt_p_plus"], p["sqrt_p_minus"], p["Lx"], p["Ly"], p["Lz"], p["nu"], p["D"],
        a_s, u_s, p["qs"], p["Omega_cs"], Nn, Nm, Np, Ns, mask23=mask, F0=F0)
    J = _parent_current(p["qs"], a_s, u_s, Ck, Nn, Nm, Np, Ns)

    dE = 1j * _cross(nabla, Fk[3:]) - J / om0
    dB = -1j * _cross(nabla, Fk[:3])
    if dark:
        ED, BD, AD, phi = Dk[0:3], Dk[3:6], Dk[6:9], Dk[9]
        dED = 1j * _cross(nabla, BD) + m.Omega_D ** 2 * AD - m.eta * J / om0
        dBD = -1j * _cross(nabla, ED)
        dAD = -ED - 1j * nabla * phi[None]
        dphi = -1j * jnp.sum(nabla * AD, axis=0)
        dDk = jnp.concatenate([dED, dBD, dAD, dphi[None]], axis=0)
        P_D = om0 * m.eta * inner(Nx, J, ED)
    else:
        dDk = jnp.zeros_like(Dk)
        P_D = jnp.zeros(())
    P_em = om0 * inner(Nx, J, Fk[:3])
    P_ext = om0 * jnp.sum(jnp.real(J[:, 0, 0, 0]) * Ed) if m.mode == "prescribed_drive" else jnp.zeros(())
    # Ledger scalars are stored as complex with zero imaginary part (Diffrax
    # requires one dtype across the RK stage buffers).
    dW = jnp.stack([P_em, P_D, P_ext]).astype(jnp.complex128)
    out = {"Ck": dCk.reshape(Ck.shape), "Fk": jnp.concatenate([dE, dB]), "Dk": dDk, "W": dW}
    if pump:
        out["B"] = jnp.stack([u_dot, jnp.zeros_like(u_dot)]).astype(y["B"].dtype)
    return out
