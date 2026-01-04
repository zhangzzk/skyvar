import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyccl as ccl
import healpy as hp


@dataclass(frozen=True)
class EnhancementResult:
    delta_w: np.ndarray
    w_model: np.ndarray
    w_true: np.ndarray
    w_selection: Optional[np.ndarray]
    xi_m: np.ndarray
    var_n: np.ndarray 
    nbar: np.ndarray
    z_mid: np.ndarray
    theta_deg: np.ndarray
    dz: np.ndarray


class ClusteringEnhancement:
    """
    Compute the clustering enhancement due to spatial variance of n(z, theta).

    Inputs are binned in redshift. For each bin i:
      n_maps[i, :] = integral of n(z, theta) over [z_i, z_{i+1}]
      nbar[i] = global mean integral over the same bin

    The enhancement is computed as:
      w_sel(theta, z_i) = <delta n_i(theta1) delta n_i(theta2)> at separation theta
      delta_w(theta) = sum_i dz_i * w_sel(theta, z_i) * xi_m(theta; z_i)
    with xi_m computed from normalized thin-shell kernels in PyCCL.

    If nbar_z is provided, w_model is computed directly from the continuous
    global dndz using PyCCL (matching analytical_calculation.ipynb).
    """

    def __init__(
        self,
        cosmo: ccl.Cosmology,
        ell_max: int = 3000,
        ell_min: int = 2,
    ) -> None:
        self.cosmo = cosmo
        self.ell_max = int(ell_max)
        self.ell_min = int(ell_min)
        if self.ell_min < 0 or self.ell_min >= self.ell_max:
            raise ValueError("Require 0 <= ell_min < ell_max.")
        self._cache_xi_m = {}  # Simple internal cache for xi_m per bin

    @staticmethod
    def _ccl_correlation_compat(
        cosmo: ccl.Cosmology,
        ell: np.ndarray,
        cls: np.ndarray,
        theta_deg: np.ndarray,
    ) -> np.ndarray:
        for kwargs in (
            {"type": "NN", "method": "fftlog"},
            {},
            {"corr_type": "gg"},
            {"correlation_type": "gg"},
            {"type": "gg"},
        ):
            for ell_key, cl_key in (("ell", "C_ell"), ("ell", "cl"), ("ells", "C_ell")):
                try:
                    return ccl.correlation(
                        cosmo=cosmo,
                        **{ell_key: ell, cl_key: cls},
                        theta=theta_deg,
                        **kwargs,
                    )
                except (TypeError, ValueError):
                    pass
        return ccl.correlation(cosmo, ell, cls, theta_deg)

    @staticmethod
    def _cl_to_wtheta_fullsky(cl: np.ndarray, theta_rad: np.ndarray, ell_min: int) -> np.ndarray:
        from numpy.polynomial.legendre import legval

        lmax = len(cl) - 1
        coeff = np.zeros(lmax + 1, dtype=float)
        ell = np.arange(lmax + 1, dtype=float)
        coeff[:] = (2.0 * ell + 1.0) * cl / (4.0 * np.pi)
        if ell_min > 0:
            coeff[:ell_min] = 0.0

        x = np.cos(theta_rad)
        return legval(x, coeff)

    def selection_wtheta_from_map(
        self,
        n_map: np.ndarray,
        theta_deg: np.ndarray,
        nside: Optional[int] = None,
        seen_idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_map = np.asarray(n_map, dtype=float)
        theta_deg = np.asarray(theta_deg, dtype=float)

        if seen_idx is not None:
            if nside is None:
                raise ValueError("nside must be provided if seen_idx is used.")
            full_map = np.zeros(hp.nside2npix(nside))
            # Subtract mean of seen pixels so that masked area (zeros) 
            # represents zero fluctuation
            mean_val = np.mean(n_map)
            full_map[seen_idx] = n_map - mean_val
            delta = full_map
        else:
            delta = n_map - np.mean(n_map)
            if nside is None:
                nside = hp.npix2nside(delta.size)
        
        lmax_map = 3 * nside - 1
        lmax = min(self.ell_max, lmax_map)
        cl = hp.anafast(delta, lmax=lmax)
        
        # Simple sky-fraction correction (fsky)
        if seen_idx is not None:
            fsky = len(seen_idx) / len(full_map)
            if fsky > 0:
                cl /= fsky

        if self.ell_min > 0:
            cl[: min(self.ell_min, lmax + 1)] = 0.0

        theta_rad = np.deg2rad(theta_deg)
        return self._cl_to_wtheta_fullsky(cl, theta_rad, min(self.ell_min, lmax))

    def _xi_matter_shell(
        self,
        z_support: np.ndarray,
        W_norm: np.ndarray,
        theta_deg: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        z_support = np.asarray(z_support, dtype=float)
        W_norm = np.asarray(W_norm, dtype=float)
        theta_deg = np.asarray(theta_deg, dtype=float)

        norm = np.trapezoid(W_norm, z_support)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("W_norm must have positive finite integral on z_support.")
        W_norm = W_norm / norm

        if bias is None:
            bias = np.ones_like(z_support, dtype=float)
        else:
            bias = np.asarray(bias, dtype=float)
            if bias.shape != z_support.shape:
                raise ValueError("bias must have the same shape as z_support.")

        tracer = ccl.NumberCountsTracer(
            self.cosmo,
            has_rsd=False,
            dndz=(z_support, W_norm),
            bias=(z_support, bias),
        )

        ell = np.arange(self.ell_max + 1, dtype=int)
        cls = ccl.angular_cl(self.cosmo, tracer, tracer, ell)
        if self.ell_min > 0:
            cls[: self.ell_min] = 0.0

        return self._ccl_correlation_compat(self.cosmo, ell, cls, theta_deg)

    def matter_xi_theta_per_bin(
        self,
        z: np.ndarray,
        theta_deg: np.ndarray,
        z_support: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        nz: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float)
        theta_deg = np.asarray(theta_deg, dtype=float)

        if nz is None:
            # We don't know the map shape yet, so we assume z is either edges or centers
            # This will be refined in compute_enhancement_from_maps
            if z.ndim != 1 or len(z) < 1:
                raise ValueError("z must be a 1D array.")
            nz_eff = len(z) if z[0] == 0 else len(z) # dummy
        
        # If nz is provided, we can disambiguate centers vs edges
        if nz is not None:
            if len(z) == nz:
                # Centers provided, compute edges
                dz_val = np.diff(z)
                z_edges = np.zeros(nz + 1)
                if nz > 1:
                    z_edges[1:-1] = z[:-1] + 0.5 * dz_val
                    z_edges[0] = z[0] - 0.5 * dz_val[0]
                    z_edges[-1] = z[-1] + 0.5 * dz_val[-1]
                else:
                    z_edges = np.array([z[0] - 0.05, z[0] + 0.05])
                
                # Clip edges to be >= 0 for PyCCL compatibility
                z_edges = np.maximum(0, z_edges)
                z_mid = z
            elif len(z) == nz + 1:
                z_edges = z
                z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
            else:
                raise ValueError(f"Redshift array length {len(z)} must be {nz} or {nz+1}")
        else:
            # Assume z_edges if nz not known
            z_edges = z
            z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

        # Final dz calculation
        dz_bins = np.diff(z_edges)
        nz_bins = len(dz_bins)
        
        # Check cache
        cache_key = (tuple(z_edges.tolist()), tuple(theta_deg.tolist()))
        if cache_key in self._cache_xi_m:
            return self._cache_xi_m[cache_key], z_mid, dz_bins

        xi_m = np.zeros((nz_bins, len(theta_deg)), dtype=float)
        print(f"Computing {nz_bins} thin-shell matter correlations (CCL)...")
        for i in range(nz_bins):
            z_lo, z_hi = float(z_edges[i]), float(z_edges[i + 1])
            dz_cur = dz_bins[i]
            if dz_cur <= 0:
                raise ValueError("Non-positive dz encountered; check redshifts.")

            if z_support is None:
                zmin, zmax = float(z_edges[0]), float(z_edges[-1])
                z_s = np.linspace(zmin, zmax, 1000, dtype=float)
            else:
                z_s = z_support

            W = np.zeros_like(z_s)
            in_bin = (z_s >= z_lo) & (z_s < z_hi) if i < nz_bins - 1 else (z_s >= z_lo) & (z_s <= z_hi)
            W[in_bin] = 1.0 / dz_cur
            xi_m[i, :] = self._xi_matter_shell(z_s, W, theta_deg, bias=bias)
        
        self._cache_xi_m[cache_key] = xi_m
        return xi_m, z_mid, dz_bins

    def compute_enhancement_from_maps(
        self,
        n_maps: np.ndarray,
        nbar: np.ndarray,
        z: np.ndarray,
        theta_deg: np.ndarray,
        z_support: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        nbar_z: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        selection_mode: str = "wtheta",
        nside: Optional[int] = None,
        seen_idx: Optional[np.ndarray] = None,
    ) -> EnhancementResult:
        n_maps = np.asarray(n_maps, dtype=float)
        nbar = np.asarray(nbar, dtype=float)
        z = np.asarray(z, dtype=float)
        theta_deg = np.asarray(theta_deg, dtype=float)

        if n_maps.ndim != 2:
            raise ValueError("n_maps must be 2D with shape (nz, npix).")
        nz, npix = n_maps.shape
        if nbar.shape != (nz,):
            raise ValueError("nbar must have shape (nz,).")

        if seen_idx is None:
            if nside is None:
                nside = hp.npix2nside(npix)
            ell_max_eff = min(self.ell_max, 3 * nside - 1)
        else:
            if nside is None:
                raise ValueError("nside must be provided if seen_idx is used.")
            if len(seen_idx) != npix:
                 raise ValueError(f"len(seen_idx)={len(seen_idx)} must match n_maps.shape[1]={npix}")
            ell_max_eff = min(self.ell_max, 3 * nside - 1)

        var_n = np.mean((n_maps - nbar[:, None]) ** 2, axis=1)
        ell_max_orig = self.ell_max
        self.ell_max = ell_max_eff
        try:
            xi_m, z_mid, dz = self.matter_xi_theta_per_bin(
                z,
                theta_deg,
                z_support=z_support,
                bias=bias,
                nz=nz,
            )
        finally:
            self.ell_max = ell_max_orig

        selection_mode = selection_mode.lower()
        if selection_mode not in ("wtheta", "variance"):
            raise ValueError("selection_mode must be 'wtheta' or 'variance'.")

        w_selection: Optional[np.ndarray]
        if selection_mode == "wtheta":
            print(f"Computing {nz} angular variations (anafast)...")
            w_selection_density = np.zeros((nz, len(theta_deg)), dtype=float)
            for i in range(nz):
                w_selection_density[i, :] = self.selection_wtheta_from_map(
                    n_maps[i], theta_deg, nside=nside, seen_idx=seen_idx
                )
            # Scale by dz^2 to convert selection-density variation to projected-count variation
            w_selection = w_selection_density * (dz[:, None] ** 2)
            delta_w = np.sum(w_selection * xi_m, axis=0)
        else:
            # var_n is variance of density n_i. Scale to bin integral variance.
            delta_w = np.sum(var_n[:, None] * (dz[:, None] ** 2) * xi_m, axis=0)
            w_selection = None
        if nbar_z is not None:
            z_model, nbar_model = nbar_z
            z_model = np.asarray(z_model, dtype=float)
            nbar_model = np.asarray(nbar_model, dtype=float)
            if z_model.shape != nbar_model.shape:
                raise ValueError("nbar_z must provide (z, nbar) arrays with the same shape.")
            nbar_norm = np.trapezoid(nbar_model, z_model)
            if not np.isfinite(nbar_norm) or nbar_norm <= 0:
                raise ValueError("nbar_z must have positive finite integral.")
            nbar_model = nbar_model / nbar_norm

            ell = np.arange(ell_max_eff + 1, dtype=int)
            tracer = ccl.NumberCountsTracer(
                self.cosmo,
                has_rsd=False,
                dndz=(z_model, nbar_model),
                bias=(z_model, np.ones_like(z_model)),
            )
            cls = ccl.angular_cl(self.cosmo, tracer, tracer, ell)
            if self.ell_min > 0:
                cls[: min(self.ell_min, len(cls))] = 0.0
            w_model = self._ccl_correlation_compat(self.cosmo, ell, cls, theta_deg)
        else:
            # nbar is density.
            # Total correlation = Sum (nbar_i * dz_i)^2 * xi_m_i
            weight_model = (nbar * dz) ** 2
            w_model = np.sum(weight_model[:, None] * xi_m, axis=0)
        w_true = w_model + delta_w

        return EnhancementResult(
            delta_w=delta_w,
            w_model=w_model,
            w_true=w_true,
            w_selection=w_selection,
            xi_m=xi_m,
            var_n=var_n,
            nbar=nbar,
            z_mid=z_mid,
            theta_deg=theta_deg,
            dz=dz,
        )
