import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import pyccl as ccl

try:
    from . import utils
    from . import config
    from . import selection as sel
    from . import plotting as plt_nz
    from .clustering import ClusteringEnhancement
except ImportError:
    import utils
    import config
    import selection as sel
    import plotting as plt_nz
    from clustering import ClusteringEnhancement


# Constants from config
SYS_NSIDE = config.SIM_SETTINGS['sys_nside']
OUTPUT_PREDS = config.PATHS['output_preds']
N_POP_SAMPLE = config.SIM_SETTINGS['n_pop_sample']



def main() -> None:
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Regenerate Statistics from Predictions (consistent with new binning)
    print("Loading predictions from catalog to regenerate statistics...")
    preds_path = config.PATHS['output_preds']
    if not os.path.exists(preds_path):
        print(f"Error: Predictions file {preds_path} not found. Run selection.py first.")
        return

    print(f"Loading existing predictions from {OUTPUT_PREDS}...")
    cla_cat = pd.read_feather(OUTPUT_PREDS)
    
    # print("Processing loaded catalog (photo-z, cuts)...")
    # cla_cat = sel.process_classified_catalog(cla_cat)
    
    # Determine redshift bins from data
    print("Determining consistent redshift bins from loaded catalog...")
    z_mid, edges = utils.get_redshift_bins(cla_cat['redshift_input_p'])
    
    maps, seen_idx = sel.load_system_maps()
    psf_hp_map = maps[0]
    stats_dict = sel.generate_summary_statistics_from_cat(cla_cat, psf_hp_map, seen_idx, output_dir, z=z_mid, edges=edges)
    full_stats = stats_dict['full']

    n_maps = full_stats['dndzs'].T
    nbar = full_stats['dndz_det']
    # nbar = n_maps.mean(axis=1) # Use map mean for consistency with variance calculation
    
    # Redshift std ratio from stats
    z_std_ratio = full_stats.get('z_std_ratio', 1.0)

    # 2. Setup Cosmology and Clustering
    cosmo = ccl.Cosmology(
        Omega_c=config.COSMO_PARAMS['Omega_c'], 
        Omega_b=config.COSMO_PARAMS['Omega_b'], 
        h=config.COSMO_PARAMS['h'], 
        sigma8=config.COSMO_PARAMS['sigma8'], 
        n_s=config.COSMO_PARAMS['n_s']
    )

    ell_max = config.CLUSTERING_SETTINGS['ell_max']
    theta_deg = np.logspace(
        np.log10(config.CLUSTERING_SETTINGS['theta_min_deg']), 
        np.log10(config.CLUSTERING_SETTINGS['theta_max_deg']), 
        config.CLUSTERING_SETTINGS['theta_bins']
    )

    enhancer = ClusteringEnhancement(cosmo, ell_max=ell_max, ell_min=2)
    
    # 3. Compute Clustering Enhancement
    print("Computing clustering enhancement (wtheta mode)...")
    # Note: For real data, we must enable shot noise subtraction if not already handled.
    # variation.py passes n_samples to compute_enhancement_from_maps.
    result = enhancer.compute_enhancement_from_maps(
        n_maps=n_maps,
        nbar=nbar,
        z=z_mid,
        theta_deg=theta_deg,
        selection_mode="wtheta",
        nside=SYS_NSIDE,
        seen_idx=seen_idx,
        weights=full_stats['frac_pix'],
    )

    # 4. Compute Variance Mode Calculation (for comparison)
    # We need band-limited maps for fair variance comparison
    # to match the effective filtering occurring in wtheta mode (anafast).
    print("Computing clustering enhancement (variance mode)...")
    lmax_map = 3 * SYS_NSIDE - 1
    n_maps_bl = np.zeros_like(n_maps)
    npix = hp.nside2npix(SYS_NSIDE)
    
    for i in range(len(n_maps)):
        # Reconstruct full map for SHT
        m_full = np.zeros(npix)
        m_full[seen_idx] = n_maps[i]
        
        # Band-limit (smooth)
        alms = hp.map2alm(m_full, lmax=lmax_map)
        m_bl = hp.alm2map(alms, nside=SYS_NSIDE)
        
        # Extract seen pixels back
        n_maps_bl[i] = m_bl[seen_idx]
    
    result_var = enhancer.compute_enhancement_from_maps(
        n_maps=n_maps_bl, 
        nbar=nbar,
        z=z_mid,
        theta_deg=theta_deg,
        selection_mode="variance",
        nside=SYS_NSIDE,
        seen_idx=seen_idx,
        weights=full_stats['frac_pix'],
    )

    # 5. Calculate Geometric Enhancement
    geo_enhancement = utils.calculate_geometric_enhancement(
        z_mid, n_maps.T, nbar, frac_pix=full_stats['frac_pix']
    )

    # 6. Print Summary
    print(f"Geometric Enhancement Factor: {geo_enhancement:.6f}")
    print(f"Redshift-based std Ratio:   {z_std_ratio:.6f}")
    print(f"Measured Clust. Enhanc (theta_min): {1.0 + result.delta_w[0] / result.w_model[0]:.6f}")

    # Diagnostic: matrix weighted enhancement
    weights = nbar * result.dz
    w_mat0 = result.w_mat[:, :, 0]
    w_estim = np.einsum("i,j,ij->", weights, weights, w_mat0) / (np.sum(weights)**2)
    
    var_n_arr = np.average((n_maps - nbar[:, None]) ** 2, weights=full_stats['frac_pix'], axis=1)
    w_sel_density = var_n_arr * (result.dz**2)
    xi_diag0 = np.diagonal(w_mat0)
    dw_estim = np.sum(w_sel_density * xi_diag0)
    print(f"Binned Predicted Factor (theta_min): {1.0 + dw_estim / w_estim:.6f}")

    # Diagnostic: Check decoherence at theta_min
    peak_idx = np.argmax(result.nbar)
    if result.w_selection is not None:
        w_sel_0 = result.w_selection[peak_idx, 0] / (result.dz[peak_idx]**2)
        var_peak = np.average((n_maps[peak_idx] - nbar[peak_idx])**2, weights=full_stats['frac_pix'])
        print(f"Decoherence Factor (z={z_mid[peak_idx]:.2f}): w_sys(theta_min) / var_sys = {w_sel_0 / var_peak:.4f}")

    # 7. Plotting
    plt_nz.plot_nz_variations(z_mid, n_maps, nbar, output_dir, title="n(z) Variations (Data)", filename="nz_distribution_data.png")
    plt_nz.plot_model_vs_ccl(result, cosmo, nbar, z_mid, output_dir, filename="w_model_vs_ccl_data.png")
    plt_nz.plot_selection_wtheta(result, 60.0 * result.theta_deg, output_dir, filename="selection_wtheta_z_data.png")
    plt_nz.plot_clustering_comparison(result, result_var, geo_enhancement, z_std_ratio, output_dir, filename="w_comparison_data.png")
    plt_nz.plot_geo_factor_z(z_mid, n_maps, nbar, output_dir, "geo_factor_z_data.png", frac_pix=full_stats['frac_pix'])

if __name__ == "__main__":
    main()
