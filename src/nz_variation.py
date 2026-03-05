import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits

logger = logging.getLogger(__name__)

try:
    from . import selection as sel
    from . import utils
    from . import config
    from . import plotting as plt_nz
    from .clustering import ClusteringEnhancement, build_pyccl_cosmology
except ImportError:
    import selection as sel
    import utils
    import config
    import plotting as plt_nz
    from clustering import ClusteringEnhancement, build_pyccl_cosmology

# Add enhance directory to path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def save_clustering_results_to_fits(all_clustering_results, output_path):
    """Save theta, w_model, and w_true for each bin to a multi-HDU FITS file."""
    hdus = [fits.PrimaryHDU()]

    for key, res in all_clustering_results.items():
        ext = key.upper()[:40]
        hdus.append(fits.ImageHDU(np.asarray(res.theta_deg, dtype=np.float64), name=f"{ext}_THETA"))
        hdus.append(fits.ImageHDU(np.asarray(res.w_model, dtype=np.float64), name=f"{ext}_WMODEL"))
        hdus.append(fits.ImageHDU(np.asarray(res.w_true, dtype=np.float64), name=f"{ext}_WTRUE"))
        hdus.append(fits.ImageHDU(np.asarray(res.delta_w, dtype=np.float64), name=f"{ext}_DW_TOT"))
        hdus.append(fits.ImageHDU(np.asarray(res.delta_w_1, dtype=np.float64), name=f"{ext}_DW_TERM1"))
        hdus.append(fits.ImageHDU(np.asarray(res.delta_w_2, dtype=np.float64), name=f"{ext}_DW_TERM2"))

    fits.HDUList(hdus).writeto(output_path, overwrite=True)
    logger.info("Saved w_theta curves to %s", output_path)




def main():
    logging.basicConfig(level=getattr(logging, config.SIM_SETTINGS.get('log_level', 'INFO')),
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    output_dir = 'output/'
    w_theta_path = utils.get_output_path("w_theta_fits")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Pre-calculated Statistics from FITS
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    results_stats = {}
    
    logger.info("Loading pre-calculated statistics from FITS files...")
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
        logger.error("No statistics could be loaded. Please run selection.py first.")
        return

    # 2. Setup Clustering Enhancement
    cosmo = build_pyccl_cosmology()
    
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

    nbar_label = 'flat' if config.CLUSTERING_SETTINGS.get('flat_global', False) else 'det'
    auto_label = 'auto' if config.CLUSTERING_SETTINGS.get('auto_only', False) else 'full'
    logger.info("\n--- Clustering Enhancement Results (nbar=%s, corr=%s) ---", nbar_label, auto_label)
    for key, stats in results_stats.items():
        
        # Determine which reference n(z) to use
        if config.CLUSTERING_SETTINGS.get('flat_global', False):
            nbar = stats['dndz_det_flat']
        else:
            nbar = stats['dndz_det']
            
        # Geometric Factor (for comparison)
        geo_enh = utils.calculate_geometric_enhancement(
            stats['z'], stats['dndzs'], nbar, frac_pix=None,
        )
        all_geo_factors[key] = geo_enh

        # Clustering Result
        # n_maps expects (nz, npix) - stats['dndzs'] is (npix, nz)
        n_maps = stats['dndzs'].T 
        
        res = ce.compute_enhancement_from_maps(
            n_maps=n_maps,
            nbar=nbar,
            z=stats['z'],
            theta_deg=theta_deg,
            seen_idx=stats['SEEN_idx'],
            nside=config.SIM_SETTINGS['sys_nside_stats'],
            weights=stats['frac_pix'],
            auto_only=config.CLUSTERING_SETTINGS.get('auto_only', False)
        )
        
        all_clustering_results[key] = res
        
        clustering_enh_factor = res.w_true[0] / res.w_model[0]
        logger.info("[%10s] Geometric: %.6f, Clustering (theta_min): %.6f", key, geo_enh, clustering_enh_factor)

    # 3. Consolidated Plotting
    plt_nz.plot_all_comparisons(all_clustering_results, all_geo_factors, output_dir)
    plt_nz.plot_all_delta_w_components(all_clustering_results, output_dir)
    plt_nz.plot_model_diagnostics(all_clustering_results, results_stats, output_dir)
    save_clustering_results_to_fits(all_clustering_results, w_theta_path)


if __name__ == "__main__":
    main()
