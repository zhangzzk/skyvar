import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyccl as ccl

try:
    from . import selection as sel
    from . import utils
    from . import config
    from .clustering import ClusteringEnhancement
except ImportError:
    import selection as sel
    import utils
    import config
    from clustering import ClusteringEnhancement

# Add enhance directory to path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def plot_all_comparisons(all_results, geometric_factors, output_dir):
    """
    Plot w_model vs w_true and their fractional difference for all bins in a multi-panel figure.
    """
    n_bins = len(all_results)
    keys = list(all_results.keys())
    
    fig, axes = plt.subplots(2, n_bins, figsize=(5 * n_bins, 8), sharex=True)
    if n_bins == 1:
        axes = axes[:, np.newaxis]
        
    for i, key in enumerate(keys):
        res = all_results[key]
        geo_factor = geometric_factors[key]
        
        # Upper panel: w(theta)
        ax0 = axes[0, i]
        ax0.loglog(res.theta_deg, res.w_model, 'k--', label='Model (global)')
        ax0.loglog(res.theta_deg, res.w_true, 'r-', label='True (local var)')
        ax0.set_title(f"Bin: {key}\nGeo Enhancement: {geo_factor:.4f}")
        ax0.set_ylabel(r"$w(\theta)$")
        if i == 0:
            ax0.legend()
        
        # Lower panel: Fractional difference
        ax1 = axes[1, i]
        clustering_enhancement = res.w_true / res.w_model
        ax1.semilogx(res.theta_deg, clustering_enhancement, 'b-')
        ax1.axhline(geo_factor, color='g', linestyle='--', label='Geometric')
        ax1.set_ylabel(r"$w_{true} / w_{model}$")
        ax1.set_xlabel(r"$\theta$ [deg]")
        if i == 0:
            ax1.legend()
            
    plt.tight_layout()
    save_path = os.path.join(output_dir, "w_comparison_all_bins.png")
    plt.savefig(save_path)
    print(f"Consolidated comparison plot saved to {save_path}")
    plt.close()


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data/Setup using selection logic and config
    print(f"Loading predictions from {config.PATHS['output_preds']}...")
    cla_cat = pd.read_feather(config.PATHS['output_preds'])
    maps, SEEN_idx = sel.load_system_maps()
    psf_hp_map = maps[0]

    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    results_stats = sel.generate_summary_statistics(
        cla_cat, psf_hp_map, SEEN_idx, output_dir, tomo_bin_edges=tomo_bin_edges
    )

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
            stats['z'], stats['dndzs'], stats['dndz_det'], frac_pix=stats['frac_pix']
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
            nside=config.SIM_SETTINGS['sys_nside']
        )
        
        all_clustering_results[key] = res
        
        clustering_enh_factor = res.w_true[0] / res.w_model[0]
        print(f"[{key:10s}] Geometric: {geo_enhancement:.6f}, Clustering (theta_min): {clustering_enh_factor:.6f}")

    # 3. Consolidated Plotting
    plot_all_comparisons(all_clustering_results, all_geo_factors, output_dir)


if __name__ == "__main__":
    main()
