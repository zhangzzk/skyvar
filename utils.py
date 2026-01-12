import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib.colors import Normalize
from scipy.stats import norm
from scipy.spatial import KDTree
import os

try:
    from .config import PHOTOZ_PARAMS
except ImportError:
    from config import PHOTOZ_PARAMS

def calculate_geometric_enhancement(z, dndzs, dndz_glob, frac_pix=None):
    """
    Calculate the geometric enhancement factor.
    """
    local_inv_widths = np.array([np.trapezoid(nz**2, z) for nz in dndzs])
    
    if frac_pix is None:
        mean_local = np.mean(local_inv_widths)
    else:
        mean_local = np.average(local_inv_widths, weights=frac_pix)
        
    global_inv_width = np.trapezoid(dndz_glob**2, z)
    
    if global_inv_width == 0:
        return 1.0
        
    return mean_local / global_inv_width

def plot_geo_factor_z(z_mid, n_maps, nbar, output_dir, filename="geo_factor_z_toy.png", frac_pix=None):
    """Plot geometric enhancement factor per redshift bin."""
    plt.figure(figsize=(8, 6))
    
    if frac_pix is None:
        mean_local_z = np.mean(n_maps**2, axis=1)
    else:
        # n_maps is (nz, npix)
        mean_local_z = np.average(n_maps**2, weights=frac_pix, axis=1)
    
    global_z = nbar**2
    total_ratio = np.trapezoid(mean_local_z, z_mid) / np.trapezoid(global_z, z_mid)
    
    plt.plot(z_mid, mean_local_z, 'b-o', markersize=4, label='$<n^2>$')
    plt.plot(z_mid, global_z, 'r-o', markersize=4, label='$\\bar{n}^2$')
    
    plt.xlabel("Redshift z")
    plt.ylabel("Geometric Factor")
    plt.title("Geometric Enhancement: "+str(total_ratio))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def get_photoz_weights(df, bin_edges):
    """
    Calculate tomographic bin membership probabilities for each galaxy.
    """
    z_g = df['redshift_input_p'].values
    m_g = df['r_input_p'].values
    delta_m = 2.5 * np.log10(df['pixel_rms_input_p'] / PHOTOZ_PARAMS['pixel_rms_ref'])
    
    mu_g = z_g + PHOTOZ_PARAMS['b0'] + PHOTOZ_PARAMS['b1'] * z_g + \
           PHOTOZ_PARAMS['bm'] * (m_g - PHOTOZ_PARAMS['m_ref'])
    
    sigma_g = PHOTOZ_PARAMS['sigma0'] * (1 + z_g) * \
              (1 + PHOTOZ_PARAMS['alpha'] * (m_g + delta_m - PHOTOZ_PARAMS['m_ref']))
    sigma_g = np.maximum(sigma_g, 0.01)
    
    weights = []
    for i in range(len(bin_edges) - 1):
        z_min, z_max = bin_edges[i], bin_edges[i+1]
        p_i = norm.cdf(z_max, loc=mu_g, scale=sigma_g) - \
              norm.cdf(z_min, loc=mu_g, scale=sigma_g)
        weights.append(p_i)
    
    return np.array(weights).T

def plt_map(map_data, sys_nside, mask, label='value', s=None, save_path=None, ax=None):
    """Plot HEALPix map for seen pixels."""
    if s is None:
        # Heuristic for adaptive dot size based on nside
        s = 12*(256.0 / sys_nside)**2
        s = np.clip(s, 0.1, 100)

    n_pix = hp.nside2npix(sys_nside)
    lon, lat = hp.pix2ang(sys_nside, np.arange(n_pix), lonlat=True)

    vmin, vmax = np.percentile(map_data[mask], [2, 98])
    norm_scale = Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        plt.figure(figsize=(16, 2))
        ax = plt.gca()
        show_plot = True
    else:
        show_plot = False

    sc = ax.scatter(lon[mask], lat[mask], c=map_data[mask], s=s, cmap=plt.cm.coolwarm, norm=norm_scale, edgecolors='none')
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_xlim(240, 140)
    ax.set_ylim(-5, 5)
    
    if save_path:
        print(f"Saving map plot to {save_path}")
        plt.savefig(save_path)
    
    if show_plot:
        plt.close()

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