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
    w_mat: np.ndarray  # Full (Nz, Nz, Ntheta) matter correlation matrix
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
      delta_w(theta) = sum_i dz_i * w_sel(theta, z_i) * w_mat(theta; z_i, z_i)
    
    The binned model w_model is computed directly using shell summation (user version):
      w_model = Sum_{i,j} (nbar_i*dz_i) * (nbar_j*dz_j) * w_mat(theta; z_i, z_j)
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
        self._cache_w_mat = {}

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
        n_samples: Optional[float] = None,
        dz: Optional[float] = None,
    ) -> np.ndarray:
        n_map = np.asarray(n_map, dtype=float)
        theta_deg = np.asarray(theta_deg, dtype=float)

        if seen_idx is not None:
            if nside is None:
                raise ValueError("nside must be provided if seen_idx is used.")
            full_map = np.zeros(hp.nside2npix(nside))
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
        
        if seen_idx is not None:
            fsky = len(seen_idx) / len(full_map)
            if fsky > 0:
                cl /= fsky
        
        # Shot Noise Subtraction
        if n_samples is not None and dz is not None:
            # Theoretical Poisson noise floor in C_ell: sigma^2 * Omega_pix
            # sigma^2 = nbar / (n_samples * dz)
            omega_pix = 4.0 * np.pi / len(full_map)
            cl_shot = (mean_val / (n_samples * dz)) * omega_pix
            cl = np.maximum(cl - cl_shot, 0.0)

        if self.ell_min > 0:
            cl[: min(self.ell_min, lmax + 1)] = 0.0

        theta_rad = np.deg2rad(theta_deg)
        return self._cl_to_wtheta_fullsky(cl, theta_rad, min(self.ell_min, lmax))

    def _get_bin_tracer(self, z_support, z_lo, z_hi, dz, bias=None):
        W = np.zeros_like(z_support)
        in_bin = (z_support >= z_lo) & (z_support <= z_hi)
        W[in_bin] = 1.0 / dz
        
        if bias is None:
            b = np.ones_like(z_support)
        else:
            b = bias
            
        return ccl.NumberCountsTracer(
            self.cosmo,
            has_rsd=False,
            dndz=(z_support, W),
            bias=(z_support, b),
        )

    def matter_correlation_matrix(
        self,
        z: np.ndarray,
        theta_deg: np.ndarray,
        bias: Optional[np.ndarray] = None,
        nz: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute full [Nz, Nz, Ntheta] matter correlation matrix."""
        z = np.asarray(z, dtype=float)
        theta_deg = np.asarray(theta_deg, dtype=float)

        if nz is not None:
            if len(z) == nz:
                dz_val = np.diff(z)
                z_edges = np.zeros(nz + 1)
                if nz > 1:
                    z_edges[1:-1] = z[:-1] + 0.5 * dz_val
                    z_edges[0] = z[0] - 0.5 * dz_val[0]
                    z_edges[-1] = z[-1] + 0.5 * dz_val[-1]
                else:
                    z_edges = np.array([z[0] - 0.05, z[0] + 0.05])
                z_edges = np.maximum(0, z_edges)
                z_mid = z
            elif len(z) == nz + 1:
                z_edges = z
                z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
            else:
                raise ValueError(f"Redshift array length {len(z)} must be {nz} or {nz+1}")
        else:
            z_edges = z
            z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

        dz_bins = np.diff(z_edges)
        nz_bins = len(dz_bins)
        
        cache_key = (tuple(z_edges.tolist()), tuple(theta_deg.tolist()))
        if cache_key in self._cache_w_mat:
            return self._cache_w_mat[cache_key], z_mid, dz_bins

        print(f"Computing {nz_bins}x{nz_bins} shell correlations (CCL matrix)...")
        w_mat = np.zeros((nz_bins, nz_bins, len(theta_deg)), dtype=float)
        
        # High resolution support for resolving shell kernels accurately
        zmin, zmax = float(z_edges[0]), float(z_edges[-1])
        z_s = np.linspace(zmin, zmax, 10000, dtype=float)

        tracers = [
            self._get_bin_tracer(z_s, z_edges[i], z_edges[i+1], dz_bins[i], bias=bias)
            for i in range(nz_bins)
        ]

        ell = np.arange(self.ell_max + 1, dtype=int)
        for i in range(nz_bins):
            for j in range(i, nz_bins):
                cls = ccl.angular_cl(self.cosmo, tracers[i], tracers[j], ell)
                if self.ell_min > 0:
                    cls[: self.ell_min] = 0.0
                
                w_ij = self._ccl_correlation_compat(self.cosmo, ell, cls, theta_deg)
                w_mat[i, j, :] = w_ij
                w_mat[j, i, :] = w_ij
        
        self._cache_w_mat[cache_key] = w_mat
        return w_mat, z_mid, dz_bins

    def compute_enhancement_from_maps(
        self,
        n_maps: np.ndarray,
        nbar: np.ndarray,
        z: np.ndarray,
        theta_deg: np.ndarray,
        bias: Optional[np.ndarray] = None,
        selection_mode: str = "wtheta",
        nside: Optional[int] = None,
        seen_idx: Optional[np.ndarray] = None,
        n_samples: Optional[float] = None,
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

        if nside is None:
            if seen_idx is None:
                nside = hp.npix2nside(npix)
            else:
                raise ValueError("nside must be provided if seen_idx is used.")

        var_n = np.mean((n_maps - nbar[:, None]) ** 2, axis=1)
        
        # Calculate matter correlation matrix at full resolution (not capped by nside)
        w_mat, z_mid, dz = self.matter_correlation_matrix(
            z,
            theta_deg,
            bias=bias,
            nz=nz,
        )

        selection_mode = selection_mode.lower()
        if selection_mode not in ("wtheta", "variance"):
            raise ValueError("selection_mode must be 'wtheta' or 'variance'.")

        w_selection: Optional[np.ndarray]
        if selection_mode == "wtheta":
            print(f"Computing {nz} angular variations (anafast + shot-noise sub)...")
            w_selection_density = np.zeros((nz, len(theta_deg)), dtype=float)
            for i in range(nz):
                w_selection_density[i, :] = self.selection_wtheta_from_map(
                    n_maps[i], theta_deg, nside=nside, seen_idx=seen_idx,
                    n_samples=n_samples, dz=dz[i]
                )
            w_selection = w_selection_density * (dz[:, None] ** 2)
            
            xi_diagonal = np.diagonal(w_mat, axis1=0, axis2=1).T  # (nz, ntheta)
            delta_w = np.sum(w_selection * xi_diagonal, axis=0)
        else:
            xi_diagonal = np.diagonal(w_mat, axis1=0, axis2=1).T
            delta_w = np.sum(var_n[:, None] * (dz[:, None] ** 2) * xi_diagonal, axis=0)
            w_selection = None
        
        # Calculate w_model using full matrix shell summation (User version)
        weights = nbar * dz
        norm = np.sum(weights)
        if norm > 0:
            w_model = np.einsum("i,j,ijk->k", weights, weights, w_mat) / (norm**2)
        else:
            w_model = np.zeros_like(theta_deg)
            
        w_true = w_model + delta_w

        return EnhancementResult(
            delta_w=delta_w,
            w_model=w_model,
            w_true=w_true,
            w_selection=w_selection,
            w_mat=w_mat,
            var_n=var_n,
            nbar=nbar,
            z_mid=z_mid,
            theta_deg=theta_deg,
            dz=dz,
        )

