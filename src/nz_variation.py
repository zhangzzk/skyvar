import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyccl as ccl
from astropy.io import fits

try:
    from . import selection as sel
    from . import utils
    from . import config
    from . import plotting as plt_nz
    from .clustering import ClusteringEnhancement
except ImportError:
    import selection as sel
    import utils
    import config
    import plotting as plt_nz
    from clustering import ClusteringEnhancement

# Add enhance directory to path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def save_clustering_results_to_fits(all_clustering_results, output_dir, filename="w_true_w_model.fits"):
    """Save theta, w_model, and w_true for each bin to a multi-HDU FITS file."""
    output_path = os.path.join(output_dir, filename)
    hdus = [fits.PrimaryHDU()]

    for key, res in all_clustering_results.items():
        ext = key.upper()[:40]
        hdus.append(fits.ImageHDU(np.asarray(res.theta_deg, dtype=np.float64), name=f"{ext}_THETA"))
        hdus.append(fits.ImageHDU(np.asarray(res.w_model, dtype=np.float64), name=f"{ext}_WMODEL"))
        hdus.append(fits.ImageHDU(np.asarray(res.w_true, dtype=np.float64), name=f"{ext}_WTRUE"))

    fits.HDUList(hdus).writeto(output_path, overwrite=True)
    print(f"Saved w_model/w_true curves to {output_path}")




def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Pre-calculated Statistics from FITS
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    results_stats = {}
    
    print("Loading pre-calculated statistics from FITS files...")
    # Load full sample (Using bin_idx=99 for full)
    full_stats = sel.load_fits_output(bin_idx=99)
    if full_stats is not None:
        results_stats['full'] = full_stats
        
    # Load tomographic bins
    for i in range(len(tomo_bin_edges) - 1):
        tomo_stats = sel.load_fits_output(bin_idx=i)
        if tomo_stats is not None:
            results_stats[f'tomo_{i}'] = tomo_stats

    if not results_stats:
        print("Error: No statistics could be loaded. Please run selection.py first.")
        return

    # 2. Setup Clustering Enhancement
    cosmo = ccl.Cosmology(
        Omega_c=config.COSMO_PARAMS['Omega_c'], 
        Omega_b=config.COSMO_PARAMS['Omega_b'], 
        h=config.COSMO_PARAMS['h'], 
        sigma8=config.COSMO_PARAMS['sigma8'], 
        n_s=config.COSMO_PARAMS['n_s']
    )
    
    ce = ClusteringEnhancement(
        cosmo, 
        ell_max=config.CLUSTERING_SETTINGS['ell_max'], 
        ell_min=config.CLUSTERING_SETTINGS['ell_min']
    )
    
    theta_deg = np.logspace(
        np.log10(config.CLUSTERING_SETTINGS['theta_min_deg']), 
        np.log10(config.CLUSTERING_SETTINGS['theta_max_deg']), 
        config.CLUSTERING_SETTINGS['theta_bins']
    )

    all_clustering_results = {}
    all_geo_factors = {}

    print("\n--- Clustering Enhancement Results ---")
    for key, stats in results_stats.items():
        print(f"Computing clustering for {key}...")
        
        # Geometric Factor (for comparison)
        geo_enhancement = utils.calculate_geometric_enhancement(
            stats['z'], stats['dndzs'], stats['dndz_det'], frac_pix=None,
        )
        all_geo_factors[key] = geo_enhancement

        # Clustering Result
        # n_maps expects (nz, npix) - stats['dndzs'] is (npix, nz)
        n_maps = stats['dndzs'].T 
        nbar = stats['dndz_det'] 
        
        res = ce.compute_enhancement_from_maps(
            n_maps=n_maps,
            nbar=nbar,
            z=stats['z'],
            theta_deg=theta_deg,
            seen_idx=stats['SEEN_idx'],
            nside=config.SIM_SETTINGS['sys_nside_stats'],
            weights=stats['frac_pix']
        )
        
        all_clustering_results[key] = res
        
        clustering_enh_factor = res.w_true[0] / res.w_model[0]
        print(f"[{key:10s}] Geometric: {geo_enhancement:.6f}, Clustering (theta_min): {clustering_enh_factor:.6f}")

    # 3. Consolidated Plotting
    plt_nz.plot_all_comparisons(all_clustering_results, all_geo_factors, output_dir)
    plt_nz.plot_model_diagnostics(all_clustering_results, results_stats, output_dir)
    save_clustering_results_to_fits(all_clustering_results, output_dir)


if __name__ == "__main__":
    main()
