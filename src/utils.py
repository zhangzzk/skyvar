import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib.colors import Normalize
from scipy.spatial import KDTree
import os

try:
    from .config import PHOTOZ_PARAMS, STATS_PARAMS
except ImportError:
    from config import PHOTOZ_PARAMS, STATS_PARAMS

def calculate_geometric_stats(z, dndzs, dndz_glob, frac_pix=None):
    """
    Calculate geometric widths and the geometric enhancement factor.
    Returns: (geo_width_pix, geo_width_global, enhancement_factor)
    """
    if dndzs.ndim == 1:
        dndzs = dndzs[None, :]
        
    # L2 norms (inverse geometric widths)
    l2_pix = np.trapezoid(dndzs**2, z, axis=1)
    l2_glob = np.trapezoid(dndz_glob**2, z)

    geo_width_pix = np.where(l2_pix > 0, 1.0 / l2_pix, 0.0)
    geo_width_global = 1.0 / l2_glob if l2_glob > 0 else 0.0

    # correction for resolution bias
    dz = np.mean(np.diff(z))
    geo_width_pix = np.sqrt(geo_width_pix**2 + dz**2/12)
    geo_width_global = np.sqrt(geo_width_global**2 + dz**2/12)
    
    if frac_pix is None:
        mean_l2 = np.mean(l2_pix)
    else:
        mean_l2 = np.average(l2_pix, weights=frac_pix)
    
    enhancement = mean_l2 / l2_glob if l2_glob > 0 else 1.0
    
    # If input was 1D, return 1D array for geo_width_pix
    if geo_width_pix.size == 1:
        geo_width_pix = geo_width_pix[0]
        
    return geo_width_pix, geo_width_global, enhancement

def calculate_geometric_enhancement(z, dndzs, dndz_glob, frac_pix=None):
    """Deprecated: use calculate_geometric_stats instead."""
    _, _, enhancement = calculate_geometric_stats(z, dndzs, dndz_glob, frac_pix)
    return enhancement


def e1e2_to_q_phi(e1, e2):
    """Convert distortion ellipticity to axis ratio q and position angle phi."""
    e = np.hypot(e1, e2)
    e = np.clip(e, 0, 0.999999999)
    q = np.sqrt((1 - e) / (1 + e))
    phi = 0.5 * np.arctan2(e2, e1)
    return q, phi

def kdt_neighbor_finder(pos1, pos2, r_min=0, r_max=10, k=30):
    """Find neighbors using KDTree."""
    kdt_in = KDTree(pos2)
    dst, ind = kdt_in.query(pos1, k=k, distance_upper_bound=r_max, workers=-1)

    found_idx = np.where((ind.reshape(-1) != pos2.shape[0]) & (dst.reshape(-1) > r_min))[0]
    
    idx1 = np.array([np.arange(0, pos1.shape[0]),] * k).T.reshape(-1)[found_idx]
    idx2 = ind.reshape(-1)[found_idx]
    
    return idx1, idx2, dst.reshape(-1)[found_idx]

def weighted_quantile(x_in, w_in, q):
    """Compute the weighted quantile of a dataset."""
    x = np.asarray(x_in)
    if w_in is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w_in)
    
    # Sort by x
    s = np.argsort(x)
    x_s = x[s]
    w_s = w[s]
    
    cw = np.cumsum(w_s)
    if cw[-1] <= 0:
        return np.nan
    cw = cw / cw[-1]
    
    return np.interp(q, cw, x_s)

def get_redshift_bins(z_vals, weights=None, n_bins=None, q_lo=0.001, q_hi=0.999):
    """
    Generate unified redshift bins based on data quantiles.
    Returns: (z_centers, z_edges)
    """
    if n_bins is None:
        try:
            from .config import ANALYSIS_SETTINGS
        except ImportError:
            from config import ANALYSIS_SETTINGS
        n_bins = ANALYSIS_SETTINGS['z_bins']
        z_min = ANALYSIS_SETTINGS.get('z_min')
        z_max = ANALYSIS_SETTINGS.get('z_max')
    else:
        z_min = None
        z_max = None
        
    if z_min is None:
        z_min = weighted_quantile(z_vals, weights, q_lo)
    if z_max is None:
        z_max = weighted_quantile(z_vals, weights, q_hi)
    
    edges = np.linspace(z_min, z_max, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    
    return centers, edges


def cal_sigz(dndz,z):
    
    if dndz.ndim == 1:
        I  = np.trapezoid(dndz, z)
        mu = np.trapezoid(z * dndz, z) / I
        var = np.trapezoid((z - mu)**2 * dndz, z) / I

    if dndz.ndim == 2:
        I  = np.trapezoid(dndz, z, axis=1)
        mu = np.trapezoid(z * dndz, z, axis=1) / I
        var = np.trapezoid((z - mu[:,None])**2 * dndz, z, axis=1) / I

    return mu, np.sqrt(var)

def compute_pixel_histograms(pix_idx, vals, weights, edges, n_pix=None):
    """
    Compute per-pixel normalized histograms.
    Returns: (pixel_counts, normalized_hists)
    """
    vals = np.asarray(vals)
    pix_idx = np.asarray(pix_idx)
    weights = np.asarray(weights)
    edges = np.asarray(edges)
    
    if n_pix is None:
        n_pix = int(np.max(pix_idx) + 1)
    
    n_bins = len(edges) - 1
    dz = np.diff(edges)
    
    bins = np.digitize(vals, edges) - 1
    mask = (bins >= 0) & (bins < n_bins) & (pix_idx >= 0) & (pix_idx < n_pix)
    
    flat_idx = pix_idx[mask] * n_bins + bins[mask]
    counts_flat = np.bincount(flat_idx, weights=weights[mask], minlength=n_pix * n_bins)
    pixel_counts = counts_flat.reshape(n_pix, n_bins)
    
    sum_num = pixel_counts.sum(axis=1)
    hists = np.zeros_like(pixel_counts)
    active = sum_num > 0
    hists[active] = pixel_counts[active] / (sum_num[active][:, None] * dz)
    
    return pixel_counts, hists

def compute_redshift_stats(pix_idx, z_vals, weights, n_pix):
    """
    Compute global and per-pixel redshift standard deviation.
    Returns: (std_z_all, std_z_pix, z_std_ratio)
    """
    z_vals = np.asarray(z_vals)
    pix_idx = np.asarray(pix_idx)
    weights = np.asarray(weights)
    
    sum_w = np.sum(weights)
    if sum_w <= 0:
        return 0.0, np.zeros(n_pix), 1.0
        
    mean_z_all = np.average(z_vals, weights=weights)
    std_z_all = np.sqrt(np.average((z_vals - mean_z_all)**2, weights=weights))
    
    # Per-pixel sums
    w_sum = np.bincount(pix_idx, weights=weights, minlength=n_pix)
    wz_sum = np.bincount(pix_idx, weights=weights * z_vals, minlength=n_pix)
    wz2_sum = np.bincount(pix_idx, weights=weights * z_vals**2, minlength=n_pix)
    
    min_count = STATS_PARAMS['min_count']
    mask_v = w_sum > min_count
    std_z_pix = np.zeros(n_pix)
    
    mean_z_pix = wz_sum[mask_v] / w_sum[mask_v]
    var_z_pix = (wz2_sum[mask_v] / w_sum[mask_v]) - mean_z_pix**2
    std_z_pix[mask_v] = np.sqrt(np.maximum(var_z_pix, 0))
    
    z_std_ratio = (1/std_z_pix[mask_v]).mean()/(1/std_z_all)
    
    return std_z_all, std_z_pix, z_std_ratio

def compute_redshift_stats_from_sums(w_sum_pix, wz_sum_pix, wz2_sum_pix, 
                                     total_w, total_wz, total_wz2):
    """
    Compute global and per-pixel redshift standard deviation using pre-calculated sums.
    Returns: (std_z_all, std_z_pix, z_std_ratio)
    """
    n_pix = len(w_sum_pix)
    if total_w <= 0:
        return 0.0, np.zeros(n_pix), 1.0
        
    mean_z_all = total_wz / total_w
    std_z_all = np.sqrt(np.maximum(total_wz2 / total_w - mean_z_all**2, 0))
    
    mask_v = w_sum_pix > 0
    std_z_pix = np.zeros(n_pix)
    
    mean_z_pix = wz_sum_pix[mask_v] / w_sum_pix[mask_v]
    var_z_pix = (wz2_sum_pix[mask_v] / w_sum_pix[mask_v]) - mean_z_pix**2
    std_z_pix[mask_v] = np.sqrt(np.maximum(var_z_pix, 0))
    
    active_pix_std = std_z_pix[mask_v]
    
    # Use the inverse-std ratio formula: mean(1/sigma_i) / (1/sigma_global)
    # Filter out zero std to avoid division by zero
    valid_std = active_pix_std > 0
    if np.any(valid_std):
        z_std_ratio = np.mean(1.0 / active_pix_std[valid_std]) * std_z_all
    else:
        z_std_ratio = 1.0
    
    return std_z_all, std_z_pix, z_std_ratio
