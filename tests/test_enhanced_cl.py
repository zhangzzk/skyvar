"""
Checks for the harmonic-space clustering enhancement (src.clustering).

1. couple_cl_products reproduces the harmonic transform of a product of two
   Legendre series (Gauss-Legendre quadrature, exact for band-limited inputs).
2. compute_enhanced_cl_from_maps, transformed back to theta, matches the
   theta-space product sum_ij S_ij(theta) w_mat_ij(theta) built from the same
   spectra on a small toy sky.

Run from the repo root:  python tests/test_enhanced_cl.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import healpy as hp
import numpy as np
from numpy.polynomial.legendre import leggauss, legval
from scipy.special import eval_legendre

from src.clustering import ClusteringEnhancement, build_pyccl_cosmology, couple_cl_products


def legendre_w(cl, x):
    ell = np.arange(len(cl))
    return legval(x, (2 * ell + 1) * cl / (4 * np.pi))


def cl_from_w(w_nodes, x, wts, lmax):
    ell = np.arange(lmax + 1)
    p = eval_legendre(ell[:, None], x[None, :])
    return 2 * np.pi * (p * (wts * w_nodes)[None, :]).sum(axis=1)


def test_coupling_matches_quadrature():
    rng = np.random.default_rng(1)
    la, lb, n_pairs = 40, 70, 3
    cl_a = rng.random((n_pairs, la + 1))
    cl_b = rng.random((n_pairs, lb + 1)) / (1 + np.arange(lb + 1)) ** 2

    lmax_out = la + lb
    x, wts = leggauss(lmax_out + 2)
    w_prod = sum(legendre_w(cl_a[p], x) * legendre_w(cl_b[p], x) for p in range(n_pairs))
    expected = cl_from_w(w_prod, x, wts, lmax_out)

    got = couple_cl_products(cl_a.T @ cl_b, lmax_out)
    err = np.max(np.abs(got - expected)) / np.max(np.abs(expected))
    print(f"[coupling vs quadrature] max rel err = {err:.2e}")
    assert err < 1e-10


def test_enhanced_cl_matches_theta_space():
    rng = np.random.default_rng(2)
    np.random.seed(2)  # hp.synfast draws from the global numpy RNG.
    nside, nz, ell_max = 16, 4, 120
    npix = hp.nside2npix(nside)
    seen_idx = np.where(hp.pix2ang(nside, np.arange(npix))[0] < 2.0)[0]

    z = np.linspace(0.3, 1.2, nz)
    nbar = np.exp(-0.5 * ((z - 0.7) / 0.3) ** 2)
    nbar /= nbar.sum() * (z[1] - z[0])
    # Smooth fluctuating n(z, pixel) maps with a footprint mean offset.
    n_maps = np.empty((nz, len(seen_idx)))
    for i in range(nz):
        field = hp.synfast(1e-3 / (1 + np.arange(3 * nside)) ** 1.5, nside)
        n_maps[i] = nbar[i] * (1.02 + field[seen_idx] + 0.05 * rng.standard_normal())

    ce = ClusteringEnhancement(build_pyccl_cosmology(), ell_max=ell_max, ell_min=2)
    lmax_sel = min(ell_max, 3 * nside - 1)
    lmax_out = ell_max + lmax_sel

    for auto_only in (True, False):
        res = ce.compute_enhanced_cl_from_maps(
            n_maps, nbar, z, nside=nside, seen_idx=seen_idx, auto_only=auto_only
        )

        # Theta-space reference on Gauss-Legendre nodes, exact full-sky Legendre sums.
        x, wts = leggauss(lmax_out + 2)
        cl_mat, _, dz = ce.matter_cl_matrix(z, nz=nz, auto_only=auto_only)
        mean_shift, mean_product, cl_sel = ce.selection_cls_from_map(
            n_maps, nbar, nside=nside, seen_idx=seen_idx, auto_only=auto_only
        )
        cl_sel[..., :2] = 0.0
        w_nodes = np.zeros_like(x)
        w_model_nodes = np.zeros_like(x)
        weights = nbar * dz
        for i in range(nz):
            for j in range(nz):
                w_mat_ij = legendre_w(cl_mat[i, j], x)
                s_ij = mean_shift[i, j] + mean_product[i, j] + legendre_w(cl_sel[i, j], x)
                w_nodes += dz[i] * dz[j] * s_ij * w_mat_ij
                w_model_nodes += weights[i] * weights[j] * w_mat_ij
        w_model_nodes /= weights.sum() ** 2

        expected_dcl = cl_from_w(w_nodes, x, wts, lmax_out)
        expected_model = cl_from_w(w_model_nodes, x, wts, lmax_out)

        err_d = np.max(np.abs(res.delta_cl - expected_dcl)) / np.max(np.abs(expected_dcl))
        err_m = np.max(np.abs(res.cl_model - expected_model)) / np.max(np.abs(expected_model))
        ratio = res.cl_true[2:ell_max] / res.cl_model[2:ell_max]
        print(
            f"[enhanced C_ell vs theta space, auto_only={auto_only}] "
            f"delta_cl rel err = {err_d:.2e}, cl_model rel err = {err_m:.2e}, "
            f"cl_true/cl_model in [{ratio.min():.4f}, {ratio.max():.4f}]"
        )
        assert err_d < 1e-8 and err_m < 1e-8


if __name__ == "__main__":
    test_coupling_matches_quadrature()
    test_enhanced_cl_matches_theta_space()
    print("All enhanced-C_ell checks passed.")
