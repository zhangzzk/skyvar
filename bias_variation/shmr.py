"""Moster+13 stellar-to-halo mass relation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Moster13Params:
    """Table-1 Moster et al. 2013 SHMR parameters."""

    m10: float = 11.590
    m11: float = 1.195
    n10: float = 0.0351
    n11: float = -0.0247
    beta10: float = 1.376
    beta11: float = -0.826
    gamma10: float = 0.608
    gamma11: float = 0.329


def moster13_parameter_values(
    redshift: np.ndarray | float,
    params: Moster13Params = Moster13Params(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``log10(M1)``, ``N``, ``beta``, and ``gamma`` at redshift."""

    z = np.asarray(redshift, dtype=float)
    a = 1.0 / (1.0 + z)
    one_minus_a = 1.0 - a
    log_m1 = params.m10 + params.m11 * one_minus_a
    n = params.n10 + params.n11 * one_minus_a
    beta = params.beta10 + params.beta11 * one_minus_a
    gamma = params.gamma10 + params.gamma11 * one_minus_a
    return log_m1, n, beta, gamma


def stellar_mass_from_halo_mass_moster13(
    halo_mass_msun: np.ndarray | float,
    redshift: np.ndarray | float,
    params: Moster13Params = Moster13Params(),
) -> np.ndarray:
    """Median stellar mass from halo mass using the Moster+13 form.

    The prototype keeps the mass convention explicit but simple: input and
    output are numeric solar masses, with no hidden ``h`` conversion.  The
    Colossus adapter handles its own conversion to ``Msun / h``.
    """

    mh = np.asarray(halo_mass_msun, dtype=float)
    log_m1, n, beta, gamma = moster13_parameter_values(redshift, params)
    m1 = 10.0**log_m1
    ratio = np.clip(mh / m1, 1e-300, None)
    return 2.0 * n * mh / (ratio ** (-beta) + ratio**gamma)


def _invert_one(
    stellar_mass: float,
    redshift: float,
    log_halo_min: float,
    log_halo_max: float,
    params: Moster13Params,
) -> float:
    if not np.isfinite(stellar_mass) or stellar_mass <= 0:
        return np.nan

    log_mstar = np.log10(stellar_mass)

    def residual(log_mhalo: float) -> float:
        mhalo = 10.0**log_mhalo
        mstar_model = stellar_mass_from_halo_mass_moster13(mhalo, redshift, params)
        return np.log10(mstar_model) - log_mstar

    lo = residual(log_halo_min)
    hi = residual(log_halo_max)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo * hi > 0:
        return np.nan
    return brentq(residual, log_halo_min, log_halo_max, xtol=1e-5, rtol=1e-5, maxiter=100)


def inverse_moster13_halo_mass(
    stellar_mass_msun: np.ndarray,
    redshift: np.ndarray,
    params: Moster13Params = Moster13Params(),
    log_halo_min: float = 8.0,
    log_halo_max: float = 16.5,
) -> np.ndarray:
    """Invert the median Moster+13 relation to ``log10(Mhalo / Msun)``."""

    stellar_mass_msun = np.asarray(stellar_mass_msun, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if stellar_mass_msun.shape != redshift.shape:
        raise ValueError("stellar_mass_msun and redshift must have the same shape.")
    out = np.empty(stellar_mass_msun.shape, dtype=float)
    flat_mstar = stellar_mass_msun.ravel()
    flat_z = redshift.ravel()
    flat_out = out.ravel()
    for i, (mstar, z) in enumerate(zip(flat_mstar, flat_z)):
        flat_out[i] = _invert_one(mstar, z, log_halo_min, log_halo_max, params)
    return out


def inverse_moster13_halo_mass_interpolated(
    stellar_mass_msun: np.ndarray,
    redshift: np.ndarray,
    params: Moster13Params = Moster13Params(),
    log_halo_min: float = 8.0,
    log_halo_max: float = 16.5,
    n_z_nodes: int = 80,
    n_halo_grid: int = 2048,
) -> np.ndarray:
    """Fast approximate inverse using redshift nodes and a halo-mass grid.

    Each galaxy is assigned to the nearest redshift node.  At that node, the
    median Moster+13 relation is evaluated on a log-halo-mass grid and inverted
    with one-dimensional interpolation in ``log10(Mstar)``.
    """

    stellar_mass_msun = np.asarray(stellar_mass_msun, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if stellar_mass_msun.shape != redshift.shape:
        raise ValueError("stellar_mass_msun and redshift must have the same shape.")

    out = np.full(stellar_mass_msun.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(stellar_mass_msun)
        & (stellar_mass_msun > 0)
        & np.isfinite(redshift)
        & (redshift >= 0)
    )
    if not np.any(valid):
        return out

    z_valid = redshift[valid]
    z_min = float(np.min(z_valid))
    z_max = float(np.max(z_valid))
    if z_min == z_max:
        z_nodes = np.array([z_min])
    else:
        z_nodes = np.linspace(z_min, z_max, max(2, int(n_z_nodes)))

    log_halo_grid = np.linspace(log_halo_min, log_halo_max, int(n_halo_grid))
    halo_grid = 10.0**log_halo_grid
    log_mstar_valid = np.log10(stellar_mass_msun[valid])
    z_index = np.abs(z_valid[:, None] - z_nodes[None, :]).argmin(axis=1)

    out_valid = np.full(log_mstar_valid.shape, np.nan, dtype=float)
    for i, z_node in enumerate(z_nodes):
        use = z_index == i
        if not np.any(use):
            continue
        log_mstar_grid = np.log10(
            stellar_mass_from_halo_mass_moster13(halo_grid, z_node, params)
        )
        good = np.isfinite(log_mstar_grid)
        out_valid[use] = np.interp(
            log_mstar_valid[use],
            log_mstar_grid[good],
            log_halo_grid[good],
            left=np.nan,
            right=np.nan,
        )

    out[valid] = out_valid
    return out


def apply_log_normal_scatter_to_log_mass(
    log_mass: np.ndarray,
    scatter_dex: float = 0.15,
    seed: int = 20260516,
) -> np.ndarray:
    """Apply additive Gaussian scatter in log10 mass."""

    log_mass = np.asarray(log_mass, dtype=float)
    if scatter_dex <= 0:
        return log_mass.copy()
    rng = np.random.default_rng(seed)
    scattered = log_mass + rng.normal(0.0, float(scatter_dex), size=log_mass.shape)
    scattered[~np.isfinite(log_mass)] = np.nan
    return scattered


def stellar_mass_proxy_from_absolute_magnitude(
    abs_mag: np.ndarray,
    solar_abs_mag_r: float = 4.65,
    mass_to_light: float = 1.0,
) -> np.ndarray:
    """Return a documented r-band luminosity proxy for stellar mass.

    This is only for explicitly requested diagnostic runs on catalogs without
    stellar mass.  It assumes ``M*/L_r = mass_to_light`` in solar units.
    """

    abs_mag = np.asarray(abs_mag, dtype=float)
    luminosity = 10.0 ** (-0.4 * (abs_mag - solar_abs_mag_r))
    return float(mass_to_light) * luminosity
