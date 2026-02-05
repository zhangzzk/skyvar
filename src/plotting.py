import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pyccl as ccl
from matplotlib.colors import Normalize

try:
    from . import config
    from . import utils
except ImportError:
    import config
    import utils

from scipy.stats import norm

def plot_selection_wtheta(result, theta_arcmin, output_dir, filename="selection_wtheta_z.png"):
    """Plot spatial variation of n(z) density correlation."""
    plt.figure(figsize=(8, 6))
    
    # Choose representative z-bins where global n(z) is significant
    nbar_peak = np.max(result.nbar)
    significant_indices = np.where(result.nbar > 0.05 * nbar_peak)[0]
    
    if len(significant_indices) > 5:
        # Pick 5 equidistant indices from significant ones
        indices = np.linspace(significant_indices[0], significant_indices[-1], 5, dtype=int)
        # Ensure peak is included if possible
        peak_idx = np.argmax(result.nbar)
        if peak_idx not in indices:
             closest = np.argmin(np.abs(indices - peak_idx))
             indices[closest] = peak_idx
        indices = np.sort(np.unique(indices))
    else:
        indices = significant_indices

    for i in indices:
        z_val = result.z_mid[i]
        if result.w_selection is not None:
             # w_selection can be (nz, ntheta) or (nz, nz, ntheta)
             if result.w_selection.ndim == 3:
                 w_density = result.w_selection[i, i] / (result.dz[i] ** 2)
             else:
                 w_density = result.w_selection[i] / (result.dz[i] ** 2)
             plt.plot(theta_arcmin, w_density, linewidth=2, label=f"z={z_val:.2f}")

    plt.xlabel(r"$\theta$ [arcmin]")
    plt.ylabel(r"$\langle \delta n(z, \theta) \delta n(z, 0) \rangle$")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.title(r"Spatial Variation of $n(z)$ Density")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_nz_variations(z, n_maps, nbar, output_dir, title="n(z) Variations", filename="nz_distribution.png", z_edges=None):
    """Plot n(z) variations for multiple pixels."""
    plt.figure(figsize=(8, 6))
    
    # n_maps is (nz, npix), we want to plot profiles for a subset of pixels
    # Transpose to (npix, nz) for easier iteration
    profiles = n_maps.T
    
    # Subsample pixels if there are too many
    n_plot = 50
    if profiles.shape[0] > n_plot:
        idx = np.random.choice(profiles.shape[0], n_plot, replace=False)
        profiles_subset = profiles[idx]
    else:
        profiles_subset = profiles
        
    for prof in profiles_subset:
        plt.plot(z, prof, color="gray", alpha=0.1)
        
    plt.plot(z, nbar, "k-", linewidth=2, label=r"Global $\bar{n}(z)$")
    
    if z_edges is not None:
        plt.axvline(z_edges[0], color="r", linestyle=":", alpha=0.2)
        plt.axvline(z_edges[-1], color="r", linestyle=":", alpha=0.2)

    plt.xlabel("Redshift z")
    plt.ylabel("n(z)")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_model_vs_ccl(result, cosmo, nbar, z, output_dir, filename="w_model_vs_ccl.png"):
    """Plot Model w(theta) vs direct CCL calculation."""
    theta_arcmin = 60.0 * result.theta_deg
    plt.figure(figsize=(7, 5))
    plt.plot(theta_arcmin, theta_arcmin * result.w_model, "k--", linewidth=2, label=r"$w_{\rm model}$ (binned)")
    
    dndz_global = nbar / np.trapezoid(nbar, z)
    gtracer = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z, dndz_global),
        bias=(z, np.ones_like(z)),
    )
    # Estimate lmax from nside or config if possible, else default
    lmax = 3 * config.SIM_SETTINGS['sys_nside'] - 1
    ell = np.arange(lmax + 1, dtype=int)
    cell = ccl.angular_cl(cosmo, gtracer, gtracer, ell)
    w_direct = ccl.correlation(
        cosmo,
        ell=ell,
        C_ell=cell,
        theta=result.theta_deg,
        type="NN",
        method="fftlog",
    )
    plt.plot(theta_arcmin, theta_arcmin * w_direct, "g-", linewidth=2, label=r"$w_{\rm total}$ (direct CCL)")
    
    plt.xlabel(r"$\theta$ [arcmin]")
    plt.ylabel(r"$\theta \cdot w(\theta)$ [arcmin]")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(r"Model vs Direct CCL")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_clustering_comparison(result, result_var, enhancement_factor, z_std_ratio, output_dir, filename="w_comparison.png"):
    """Plot comparison of clustering enhancement between different methods."""
    theta_arcmin = 60.0 * result.theta_deg
    fig_comp, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.plot(theta_arcmin, theta_arcmin * result.w_model, "k--", linewidth=2, label=r"$w_{\rm model}$")
    ax_top.plot(theta_arcmin, theta_arcmin * result.w_true, "r-", linewidth=2, label=r"$w_{\rm true}$ (wtheta)")
    ax_top.plot(theta_arcmin, theta_arcmin * result_var.w_true, "g:", linewidth=2, label=r"$w_{\rm true}$ (variance)")
    ax_top.set_ylabel(r"$\theta \cdot w(\theta)$ [arcmin]")
    ax_top.set_xscale("log")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=12)
    header_text = f"Geo Enhanc: {enhancement_factor:.4f} | Ratio(std_z): {z_std_ratio:.4f}\nClust. Enhanc: {result.delta_w[0]/result.w_model[0]:.4f}"
    ax_top.set_title(header_text)

    w_abs = np.abs(result.w_model)
    thresh = 0.05 * np.nanmax(w_abs)
    mask = w_abs > thresh
    frac_diff = np.full_like(result.w_model, np.nan)
    frac_diff[mask] = (result.w_true[mask] - result.w_model[mask]) / result.w_model[mask]
    
    frac_diff_var = np.full_like(result.w_model, np.nan)
    frac_diff_var[mask] = (result_var.w_true[mask] - result.w_model[mask]) / result.w_model[mask]

    ax_bot.plot(theta_arcmin, frac_diff, "r-", linewidth=1.5, label="wtheta")
    ax_bot.plot(theta_arcmin, frac_diff_var, "g:", linewidth=1.5, label="variance")
    ax_bot.set_xlim(theta_arcmin.min(), theta_arcmin.max())
    ax_bot.set_ylabel(r"$\Delta w / w_{\rm model}$")
    ax_bot.set_xlabel(r"$\theta$ [arcmin]")
    ax_bot.set_xscale("log")
    ax_bot.grid(True, alpha=0.3, which="both")
    ax_bot.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_model_diagnostics(all_results, all_stats, output_dir):
    """
    Diagnostic plot comparing binned shell summation (User Version) vs direct CCL integral for w_model.
    """
    n_bins = len(all_results)
    keys = list(all_results.keys())
    
    fig, axes = plt.subplots(1, n_bins, figsize=(5 * n_bins, 4), squeeze=False)
        
    for i, key in enumerate(keys):
        res = all_results[key]
        ax = axes[0, i]
        theta_arcmin = res.theta_deg * 60.0
        
        y_binned = res.w_model * theta_arcmin
        ax.plot(theta_arcmin, y_binned, 'k--', lw=2, label=r'$w_{\rm model}$ (binned maps)')
        
        cosmo = ccl.Cosmology(
            Omega_c=config.COSMO_PARAMS['Omega_c'], 
            Omega_b=config.COSMO_PARAMS['Omega_b'], 
            h=config.COSMO_PARAMS['h'], 
            sigma8=config.COSMO_PARAMS['sigma8'], 
            n_s=config.COSMO_PARAMS['n_s']
        )
        ell = np.arange(config.CLUSTERING_SETTINGS['ell_max'] + 1, dtype=int)
        tracer = ccl.NumberCountsTracer(
            cosmo,
            has_rsd=False,
            dndz=(res.z_mid, res.nbar),
            bias=(res.z_mid, np.ones_like(res.z_mid)),
        )
        cls = ccl.angular_cl(cosmo, tracer, tracer, ell)
        if config.CLUSTERING_SETTINGS['ell_min'] > 0:
            cls[: config.CLUSTERING_SETTINGS['ell_min']] = 0.0
        w_direct = ccl.correlation(cosmo, ell=ell, C_ell=cls, theta=res.theta_deg)
        
        ax.plot(theta_arcmin, w_direct * theta_arcmin, 'g-', lw=2, label=r'$w_{\rm total}$ (direct CCL)')

        ax.set_xscale('log')
        ax.set_title(f"Model Diagnostic: {key}")
        ax.set_xlabel(r"$\theta$ [arcmin]")
        ax.set_ylabel(r"$\theta \cdot w(\theta)$")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    save_path = os.path.join(output_dir, "w_model_diagnostic.png")
    plt.savefig(save_path)
    plt.close()

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
        theta_arcmin = res.theta_deg * 60.0
        
        ax0 = axes[0, i]
        ax0.plot(theta_arcmin, theta_arcmin * res.w_model, 'k--', label='Model (global)')
        ax0.plot(theta_arcmin, theta_arcmin * res.w_true, 'r-', label='True (local var)')
        ax0.set_title(f"Bin: {key}\nGeo Enhancement: {geo_factor:.4f}")
        ax0.set_ylabel(r"$\theta \cdot w(\theta)$ [arcmin]")
        ax0.set_xscale('log')
        if i == 0:
            ax0.legend()
        
        ax1 = axes[1, i]
        clustering_enhancement = res.w_true / res.w_model
        ax1.semilogx(theta_arcmin, clustering_enhancement, 'b-')
        ax1.axhline(geo_factor, color='g', linestyle='--', label='Geometric')
        ax1.set_ylabel(r"$w_{true} / w_{model}$")
        ax1.set_xlabel(r"$\theta$ [arcmin]")
        if i == 0:
            ax1.legend()
            
    plt.tight_layout()
    save_path = os.path.join(output_dir, "w_comparison_all_bins.png")
    plt.savefig(save_path)
    plt.close()

def plt_map(map_data, sys_nside, mask, label='value', s=None, save_path=None, ax=None):
    """Plot HEALPix map for seen pixels."""
    if s is None:
        s = 3*(256.0 / sys_nside)**3
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
        plt.savefig(save_path)
    
    if show_plot:
        plt.close()

def plot_geo_factor_z(z_mid, n_maps, nbar, output_dir, filename="geo_factor_z.png", frac_pix=None):
    """Plot geometric enhancement factor per redshift bin."""
    plt.figure(figsize=(8, 6))
    
    if frac_pix is None:
        mean_local_z = np.mean(n_maps**2, axis=1)
    else:
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

def plot_tomographic_bins(results, output_dir):
    """Plot the global n(z) for all tomographic bins on one plot with variations."""
    plt.figure(figsize=(10, 6))
    tomo_keys = sorted([k for k in results.keys() if k.startswith('tomo_')], 
                       key=lambda x: int(x.split('_')[1]))
    
    standard_colors = ['#3f51b5', '#e91e63', '#4caf50', '#ff9800', '#9c27b0', '#607d8b']
    if len(tomo_keys) <= len(standard_colors):
        colors = standard_colors[:len(tomo_keys)]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(tomo_keys)))
    
    for i, key in enumerate(tomo_keys):
        stats = results[key]
        z = stats['z']
        dndzs = stats['dndzs']
        color = colors[i]
        
        n_pixels = dndzs.shape[0]
        n_plot = min(n_pixels, 150)
        step = max(1, n_pixels // n_plot)
        for j in range(0, n_pixels, step):
            plt.plot(z, dndzs[j], color=color, alpha=0.2, lw=0.5)

        plt.plot(z, stats['dndz_det'], color=color, lw=2., label=f"Bin {i}")

    plt.xlim(0, 2)
    plt.xlabel('Redshift $z$')
    plt.ylabel('$n(z)$')
    plt.title('Tomographic Bin Redshift Distributions (with spatial variations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "tomographic_bins_nz.png"))
    plt.close()

def plot_snr_fractions(cla_cat, output_dir, z=None, edges=None):
    """Plot SNR fractions of detected galaxies."""
    z_col = 'redshift_input_p' if 'redshift_input_p' in cla_cat.columns else 'redshift'
    snr_col = 'snr_input_p' if 'snr_input_p' in cla_cat.columns else 'snr'
    
    if z_col not in cla_cat.columns or snr_col not in cla_cat.columns:
        z_col = next((c for c in cla_cat.columns if 'redshift' in c), None)
        snr_col = next((c for c in cla_cat.columns if 'snr' in c), None)

    if z_col is None or snr_col is None:
        return

    if z is None or edges is None:
        z, edges = utils.get_redshift_bins(cla_cat[z_col], weights=cla_cat['detection'].values if 'detection' in cla_cat.columns else None)
    
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    n_tomo = len(tomo_bin_edges) - 1
    
    n_panels = 1 + n_tomo
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=False, sharey=True)
    axes = axes.flatten()
    
    thresholds = [5, 10, 15]
    colors = ['#3f51b5', '#e91e63', '#4caf50']
    
    def calculate_fractions(df, edges, weights=None):
        if weights is None:
            weights = np.ones(len(df))
        counts_total, _ = np.histogram(df[z_col], bins=edges, weights=weights)
        fractions = {}
        for thr in thresholds:
            mask = df[snr_col] > thr
            counts_thr, _ = np.histogram(df[z_col][mask], bins=edges, weights=weights[mask])
            with np.errstate(divide='ignore', invalid='ignore'):
                fractions[thr] = np.where(counts_total > 0, counts_thr / counts_total, 0)
        return fractions

    fr_full = calculate_fractions(cla_cat, edges, weights=cla_cat['detection'].values if 'detection' in cla_cat.columns else None)
    ax = axes[0]
    for i, thr in enumerate(thresholds):
        ax.plot(z, fr_full[thr], label=f'SNR > {thr}', color=colors[i], lw=2)
    ax.set_title("Full Sample")
    ax.set_ylabel("Fraction of Detections")
    ax.set_xlabel("Redshift $z$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    for i in range(n_tomo):
        ax = axes[i+1]
        tomo_weights = cla_cat[f'tomo_p_{i}'].values * (cla_cat['detection'].values if 'detection' in cla_cat.columns else 1.0)
        fr_tomo = calculate_fractions(cla_cat, edges, weights=tomo_weights)
        for j, thr in enumerate(thresholds):
            ax.plot(z, fr_tomo[thr], label=f'SNR > {thr}', color=colors[j], lw=2)
        ax.set_title(f"Tomo Bin {i} ($z_p \in [{tomo_bin_edges[i]}, {tomo_bin_edges[i+1]}]$)")
        ax.set_xlabel("Redshift $z$")
        ax.grid(True, alpha=0.3)

    for j in range(n_panels, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Fraction of Detected Galaxies Above SNR Thresholds", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "snr_fractions.png"), dpi=200)
    plt.close()

def plot_pixel_std_histograms(results, output_dir):
    """Plot histograms of n(z) redshift std and geometric width per pixel."""
    tomo_keys = sorted([k for k in results.keys() if k.startswith('tomo_')], 
                       key=lambda x: int(x.split('_')[1]))
    keys = ['full'] + tomo_keys
    
    n_keys = len(keys)
    n_cols = 2
    n_rows = n_keys
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
    sys_nside = config.SIM_SETTINGS['sys_nside']
    
    for i, k in enumerate(keys):
        stats = results[k]
        
        std_vals = stats['std_z_pix']
        active_std = std_vals[std_vals > 0]
        ax_std = axes[i, 0]
        
        if len(active_std) > 0:
            ax_std.hist(active_std, bins=80, density=True, color='skyblue', alpha=0.7, 
                        label=f'pixel values (NSIDE={sys_nside})')
            global_std = stats.get('std_z_global', 0.0)
            ax_std.axvline(global_std, color='red', lw=1.5, ls='--', label=f'global: {global_std:.5f}')
            ratio = stats.get('z_std_ratio', 1.0)
            ax_std.set_title(f"{k}: Redshift Std ($\sigma_z$)\nStd Ratio: {ratio:.5f}")
            ax_std.set_xlabel("Redshift std $\sigma_z$")
            ax_std.set_ylabel("Probability Density")
            ax_std.legend(fontsize=9)
            ax_std.grid(True, alpha=0.3)
        else:
            ax_std.text(0.5, 0.5, f"No data for {k}", ha='center', va='center')

        geo_vals = stats['geo_width_pix']
        active_geo = geo_vals[geo_vals > 0]
        ax_geo = axes[i, 1]
        
        if len(active_geo) > 0:
            ax_geo.hist(active_geo, bins=80, density=True, color='salmon', alpha=0.7,
                         label=f'pixel values (NSIDE={sys_nside})')
            global_geo = stats.get('geo_width_global', 0.0)
            ax_geo.axvline(global_geo, color='red', lw=1.5, ls='--', label=f'global: {global_geo:.5f}')
            mean_geo_pix = np.mean(active_geo)
            geo_ratio = global_geo / mean_geo_pix if mean_geo_pix > 0 else 1.0
            ax_geo.set_title(f"{k}: Geometric Width ($w_{{geo}}$)\nGlobal/Mean: {geo_ratio:.5f}")
            ax_geo.set_xlabel("Geometric width $w_{geo} = 1/\\int n(z)^2 dz$")
            ax_geo.set_ylabel("Probability Density")
            ax_geo.legend(fontsize=9)
            ax_geo.grid(True, alpha=0.3)
        else:
            ax_geo.text(0.5, 0.5, f"No data for {k}", ha='center', va='center')

    plt.suptitle("Distribution of Per-Pixel Redshift Variations (Std and Geometric Width)", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "pixel_variations_histograms.png"), dpi=200)
    plt.close()

def plot_geo_vs_std_scatter(results, output_dir):
    """Plot pixel-by-pixel scatter comparison of geometric width vs redshift std."""
    tomo_keys = sorted([k for k in results.keys() if k.startswith('tomo_')], 
                       key=lambda x: int(x.split('_')[1]))
    keys = ['full'] + tomo_keys
    
    n_keys = len(keys)
    n_cols = min(n_keys, 3)
    n_rows = (n_keys + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    for i, k in enumerate(keys):
        stats = results[k]
        std_vals = stats.get('std_z_pix')
        geo_vals = stats.get('geo_width_pix') / (2*np.pi**0.5) # the factor assumes n(z) is Gaussian
        
        if std_vals is None or geo_vals is None:
            axes[i].text(0.5, 0.5, f"Missing data for {k}", ha='center', va='center')
            continue

        mask = (std_vals > 0) & (geo_vals > 0)
        ax = axes[i]
        
        if np.any(mask):
            x = std_vals[mask]
            y = geo_vals[mask]
            
            # Use hexbin or scatter with alpha for many points
            if len(x) > 2000:
                hb = ax.hexbin(x, y, gridsize=40, cmap='YlOrRd', mincnt=1)
                fig.colorbar(hb, ax=ax, label='Counts')
            else:
                ax.scatter(x, y, color='blue', alpha=0.3, s=15, edgecolors='none')
            
            ax.set_title(f"Bin: {k}")
            ax.set_xlabel(r"Redshift Std $\sigma_z$")
            ax.set_ylabel(r"Geometric Width $w_{geo}$")
            
            # Add y=x reference line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='y=x')
            
            # Add correlation coefficient
            if len(x) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                ax.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)
        else:
            ax.text(0.5, 0.5, f"No active data for {k}", ha='center', va='center')

    for j in range(n_keys, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle("Pixel-by-Pixel: Redshift Std vs Geometric Width", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "geo_vs_std_scatter.png"), dpi=200)
    plt.close()

def plot_photoz_weight_histograms(cla_cat, output_dir):
    """Plot histograms of photo-z weights for each tomographic bin."""
    tomo_cols = sorted([c for c in cla_cat.columns if c.startswith('tomo_p_')],
                        key=lambda x: int(x.split('_')[2]))
    
    if not tomo_cols:
        return

    n_panels = len(tomo_cols)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_panels == 1:
        axes = [axes]
    elif n_panels > 1:
        axes = axes.flatten()
    
    for i, col in enumerate(tomo_cols):
        weights = cla_cat[col].values
        ax = axes[i]
        ax.hist(weights, bins=100, density=True, color='skyblue', alpha=0.7)
        ax.set_title(f"Photo-z weight distribution: Bin {i}")
        ax.set_xlabel("Weight $w_i$")
        ax.set_ylabel("Probability Density")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        mean_w = np.mean(weights)
        ax.axvline(mean_w, color='red', linestyle='--', label=f'Mean: {mean_w:.3f}')
        ax.legend(fontsize=9)

    for j in range(n_panels, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Distribution of Tomographic Bin Photo-z Weights (Diagnostic)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "photoz_weight_histograms.png"), dpi=200)
    plt.close()

def plot_systematics_consolidated(maps, labels, nside, mask, output_path):
    """Plot multiple systematic maps in a single consolidated figure."""
    n_maps = len(maps)
    fig, axes = plt.subplots(n_maps, 1, figsize=(16, 2.5 * n_maps))
    if n_maps == 1:
        axes = [axes]

    for i, (map_data, label) in enumerate(zip(maps, labels)):
        plt_map(map_data, nside, mask, label=label, ax=axes[i])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_systematics_histograms(maps, labels, mask, output_path):
    """Plot histograms of multiple systematic maps."""
    n_maps = len(maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(5 * n_maps, 4))
    if n_maps == 1:
        axes = [axes]

    colors = ['royalblue', 'coral', 'mediumseagreen']
    
    for i, (map_data, label) in enumerate(zip(maps, labels)):
        data = map_data[mask]
        data = data[~np.isnan(data)] 
        
        axes[i].hist(data, bins=50, color=colors[i % len(colors)], alpha=0.7, edgecolor='black', density=True)
        axes[i].set_title(label)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_systematics_overview(maps, labels, nside, mask, output_path):
    """Plot maps and histograms for multiple systematics side-by-side (panel by panel)."""
    n_maps = len(maps)
    fig, axes = plt.subplots(n_maps, 2, figsize=(14, 3 * n_maps), 
                             gridspec_kw={'width_ratios': [3.5, 1]})
    
    if n_maps == 1:
        axes = axes[np.newaxis, :]
        
    colors = ['royalblue', 'coral', 'mediumseagreen']
    
    for i, (map_data, label) in enumerate(zip(maps, labels)):
        # Map panel (Column 0)
        plt_map(map_data, nside, mask, label=label, ax=axes[i, 0])
        
        # Histogram panel (Column 1)
        data = map_data[mask]
        data = data[~np.isnan(data)]
        axes[i, 1].hist(data, bins=50, color=colors[i % len(colors)], 
                          alpha=0.7, edgecolor='black', density=True)
        axes[i, 1].set_title(f"{label} Distribution")
        axes[i, 1].set_xlabel("Value")
        axes[i, 1].set_ylabel("Density")
        axes[i, 1].grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_systematics_vs_metrics(sys_maps, sys_labels, metrics, metric_labels, seen_idx, output_dir, prefix="sys_vs"):
    """Plot per-pixel systematics vs summary metrics with binned trends."""
    if len(sys_maps) != len(sys_labels):
        raise ValueError("sys_maps and sys_labels must have the same length.")
    if len(metrics) != len(metric_labels):
        raise ValueError("metrics and metric_labels must have the same length.")

    seen_idx = np.asarray(seen_idx)
    n_metrics = len(metrics)

    for sys_map, sys_label in zip(sys_maps, sys_labels):
        x_full = np.asarray(sys_map)[seen_idx]

        fig, axes = plt.subplots(1, n_metrics, figsize=(5.5 * n_metrics, 4.2), squeeze=False)
        axes = axes[0]

        for ax, metric, metric_label in zip(axes, metrics, metric_labels):
            y_full = np.asarray(metric)

            finite = np.isfinite(x_full) & np.isfinite(y_full)
            x = x_full[finite]
            y = y_full[finite]

            if x.size == 0:
                ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
                ax.set_title(f"{sys_label} vs {metric_label}")
                continue

            ax.scatter(x, y, s=4, alpha=0.08, color="tab:blue", edgecolors="none")

            lo, hi = np.percentile(x, [1, 99])
            if hi > lo:
                bins = np.linspace(lo, hi, 16)
                bin_idx = np.digitize(x, bins) - 1
                centers = 0.5 * (bins[:-1] + bins[1:])
                med = np.full_like(centers, np.nan, dtype=float)
                p16 = np.full_like(centers, np.nan, dtype=float)
                p84 = np.full_like(centers, np.nan, dtype=float)

                for i in range(len(centers)):
                    sel = bin_idx == i
                    if np.any(sel):
                        med[i] = np.median(y[sel])
                        p16[i] = np.percentile(y[sel], 16)
                        p84[i] = np.percentile(y[sel], 84)

                ax.plot(centers, med, color="black", lw=2, label="median")
                ax.fill_between(centers, p16, p84, color="black", alpha=0.2, label="16-84%")

            if x.size > 1:
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = np.nan

            ax.set_title(f"{sys_label} vs {metric_label}\nPearson r={corr:.3f}")
            ax.set_xlabel(sys_label)
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{prefix}_{sys_label.replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

def save_diagnostic_plots(results, output_dir, key='full'):
    """Generate and save distributions plots with pixel variations."""
    stats = results[key]
    z = stats['z']
    dndzs = stats['dndzs']
    
    plt.figure(figsize=(10, 6))
    n_pixels = dndzs.shape[0]
    n_plot = min(n_pixels, 150)
    step = max(1, n_pixels // n_plot)
    for i in range(0, n_pixels, step):
        plt.plot(z, dndzs[i], color='gray', alpha=0.2, lw=0.5)

    plt.plot(z, stats['dndz_det'], 'r-', lw=2, label='Mean Detected')
    plt.plot(z, stats['dndz_in'], 'k--', lw=1.5, label='Input (Truth)')
    
    plt.xlabel('Redshift $z$')
    plt.ylabel('$n(z)$')
    plt.legend()
    plt.title(f"Distribution: {key} (including pixel variations)")
    plt.savefig(os.path.join(output_dir, f"pixel_nz_variations_{key}.png"))
    plt.close()

def plot_dm_c_comparison_objects(cla_cat, output_dir):
    """Plot distribution and comparison of dm_c for individual objects."""
    if 'pixel_rms_input_p' not in cla_cat.columns or 'psf_fwhm_input_p' not in cla_cat.columns:
        print("Warning: Systematic columns not found in catalog for dm_c plot.")
        return

    rms = cla_cat['pixel_rms_input_p']
    psf = cla_cat['psf_fwhm_input_p']
    
    ref_rms = config.PHOTOZ_PARAMS['pixel_rms_ref']
    ref_psf = config.PHOTOZ_PARAMS['psf_fwhm_ref']
    
    # Formula 1 calculation
    mu1, sigma1, dm1 = utils.get_photoz_params(cla_cat, dm_c_type='rms')
    # Formula 2 calculation (default)
    mu2, sigma2, dm2 = utils.get_photoz_params(cla_cat, dm_c_type='psf')
    
    # Get bin weights using formula 2 (standard)
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    weights = utils.get_photoz_weights(cla_cat, tomo_bin_edges)
    n_tomo = len(tomo_bin_edges) - 1
    true_z = cla_cat['redshift_input_p'].values
    
    # Sample for plotting if too large
    if len(cla_cat) > 10000:
        sample_idx = np.random.choice(len(cla_cat), 10000, replace=False)
        dm1_s, dm2_s = dm1[sample_idx], dm2[sample_idx]
        sigma1_s, sigma2_s = sigma1[sample_idx], sigma2[sample_idx]
        true_z_s = true_z[sample_idx]
    else:
        dm1_s, dm2_s = dm1, dm2
        sigma1_s, sigma2_s = sigma1, sigma2
        true_z_s = true_z

    # Create a grid of plots: 2 rows (dm and sigma_g) x (n_tomo + 1) columns (Full + bins)
    # Note: sharing axes might be tricky if ranges vary wildly, but let's try
    fig, axes = plt.subplots(2, n_tomo + 1, figsize=(4 * (n_tomo + 1), 8))
    
    # --- Row 0: dm1 vs dm2 ---
    # Full Sample
    axes[0, 0].scatter(dm1_s, dm2_s, s=1, alpha=0.2, color='purple')
    lims = [max(axes[0, 0].get_xlim()[0], axes[0, 0].get_ylim()[0]), 
            min(axes[0, 0].get_xlim()[1], axes[0, 0].get_ylim()[1])]
    axes[0, 0].plot(lims, lims, 'r--', alpha=0.5, label='1:1')
    axes[0, 0].set_title('dm Comparison: Full Sample')
    axes[0, 0].grid(alpha=0.3)
    
    # Tomo Bins
    for i in range(n_tomo):
        ax = axes[0, i + 1]
        z_min, z_max = tomo_bin_edges[i], tomo_bin_edges[i+1]
        mask = (true_z_s >= z_min) & (true_z_s < z_max)
        
        if np.any(mask):
            ax.scatter(dm1_s[mask], dm2_s[mask], s=1, alpha=0.2, color='purple')
            lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
            ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_title(f'dm: Bin {i}')
        ax.grid(alpha=0.3)

    # --- Row 1: sigma1 vs sigma2 ---
    # Full Sample
    axes[1, 0].scatter(sigma1_s, sigma2_s, s=1, alpha=0.2, color='teal')
    lims = [axes[1, 0].get_xlim()[0], axes[1, 0].get_xlim()[1]]
    axes[1, 0].plot(lims, lims, 'r--', alpha=0.5, label='1:1')
    axes[1, 0].set_title('sigma_g Comparison: Full Sample')
    axes[1, 0].grid(alpha=0.3)
    
    # Tomo Bins
    for i in range(n_tomo):
        ax = axes[1, i + 1]
        z_min, z_max = tomo_bin_edges[i], tomo_bin_edges[i+1]
        mask = (true_z_s >= z_min) & (true_z_s < z_max)
        
        if np.any(mask):
            ax.scatter(sigma1_s[mask], sigma2_s[mask], s=1, alpha=0.2, color='teal')
            lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
            ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_title(f'sigma_g: Bin {i}')
        ax.grid(alpha=0.3)

    # Global Labels
    for ax in axes[0, :]: 
        ax.set_xlabel('dm (RMS only)')
        ax.set_ylabel('dm (RMS + PSF)')
    for ax in axes[1, :]: 
        ax.set_xlabel('sigma_g (RMS only)')
        ax.set_ylabel('sigma_g (RMS + PSF)')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dm_c_comparison_objects.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Object-level comparison plot saved to {plot_path}")

def plot_z_distribution_comparison(cla_cat, output_dir, z=None, edges=None):
    """Plot histograms of true redshift and photo-z (mu_g) together."""
    if z is None or edges is None:
        z, edges = utils.get_redshift_bins(None)

    true_z = cla_cat['redshift_input_p'].values
    weights = cla_cat['detection'].values

    mu_g, sigma_g, _ = utils.get_photoz_params(cla_cat, dm_c_type='psf')
    
    z_hat_g = np.random.normal(mu_g, sigma_g, len(true_z))
    
    plt.figure(figsize=(10, 6))
    
    # True z histogram
    plt.hist(true_z, alpha=1, bins=100, label='True Redshift', weights=weights, density=False, color='blue', histtype='step')
    
    # Photo-z histogram
    plt.hist(mu_g, alpha=1, bins=100, label='Photo-z', weights=weights, density=False, color='red', histtype='step')
    
    plt.title('Global Redshift Distribution: True vs Photo-z')
    plt.xlabel('Redshift z')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(output_dir, "z_distribution_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Z-distribution comparison plot saved to {plot_path}")
