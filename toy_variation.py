import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pyccl as ccl

try:
    from . import utils
    from .clustering import ClusteringEnhancement
except ImportError:
    import utils
    from clustering import ClusteringEnhancement


def make_uncorrelated_nz_maps(
    z_edges: np.ndarray,
    nside: int,
    mean_z: float,
    sigma0: float,
    mean_scatter: float = 0.05,
    width_scatter: float = 0.0,
    noise_level: float = 0.0,
    seed: int = 42,
    n_plot: int = 50,
    chunk_size: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    np.random.seed(seed)
    nz = len(z_edges) - 1
    npix = hp.nside2npix(nside)

    z_fine = np.linspace(z_edges[0], z_edges[-1], 400)
    dz_fine = z_fine[1] - z_fine[0]

    # Mean variation
    mean_fluct = mean_scatter * np.random.randn(npix)
    mean_pix = mean_z + mean_fluct
    
    # Width variation
    width_fluct = width_scatter * np.random.randn(npix)
    width_scale = np.clip(1 + width_fluct, 0.05, None)
    sigma_pix = sigma0 * width_scale

    bin_masks = []
    for i in range(nz):
        z_lo, z_hi = z_edges[i], z_edges[i + 1]
        mask = (z_fine >= z_lo) & (z_fine < z_hi) if i < nz - 1 else (z_fine >= z_lo) & (z_fine <= z_hi)
        bin_masks.append(mask)

    n_maps = np.zeros((nz, npix), dtype=float)
    nbar_fine_sum = np.zeros_like(z_fine)
    sigma_inv_local_sum = 0.0

    z_2d = z_fine[None, :]
    for start in range(0, npix, chunk_size):
        end = min(start + chunk_size, npix)
        sigma_chunk = sigma_pix[start:end][:, None]
        mean_chunk = mean_pix[start:end][:, None]
        # Smail distribution: n(z) ~ z^2 * exp(-z/z0)
        # Peak location ~ 2*z0.
        # Width scales with z0. So varying z0 effectively varies both mean and width.
        # To vary width independently, we can perturb z0 further.
        
        z0_chunk = np.clip(mean_chunk / 2.0, 0.05, None)
        # Apply width scatter as a multiplicative perturbation
        if width_scatter > 0:
             z0_chunk *= np.clip(1 + width_scatter * np.random.randn(end-start)[:, None], 0.5, 2.0)
        
        profiles = (z_2d**2) * np.exp(-z_2d / z0_chunk)
        norms = np.trapezoid(profiles, z_fine, axis=1)
        norms[norms == 0] = 1e-9 # avoid division by zero
        profiles /= norms[:, None]

        # Add Shot Noise / Uncorrelated Noise
        # This adds noise to each pixel's N(z) independently
        if noise_level > 0:
             noise = noise_level * np.random.randn(*profiles.shape)
             profiles += noise
             # Ensure positivity? Or allow fluctuations?
             # Real histograms are positive, but delta_n can be negative. 
             # Let's clip to be safe for n(z) interpretation.
             profiles = np.maximum(0, profiles)

        nbar_fine_sum += np.sum(profiles, axis=0)
        sigma_inv_local_sum += np.sum(np.trapezoid(profiles ** 2, z_fine, axis=1))

        for i in range(nz):
            n_maps[i, start:end] = np.sum(profiles[:, bin_masks[i]], axis=1) * dz_fine

    z_widths = np.diff(z_edges)
    n_maps = n_maps / z_widths[:, None]
    nbar = np.mean(n_maps, axis=1)
    nbar_fine = nbar_fine_sum / npix
    sigma_inv_local_mean = sigma_inv_local_sum / npix

    subset_idx = np.random.choice(npix, n_plot, replace=False)
    sigma_subset = sigma_pix[subset_idx][:, None]
    mean_subset = mean_pix[subset_idx][:, None]
    z0_subset = mean_subset / 2.0
    if width_scatter > 0:
         z0_subset *= np.clip(1 + width_scatter * np.random.randn(n_plot)[:, None], 0.5, 2.0)
    profiles_subset = (z_2d**2) * np.exp(-z_2d / z0_subset)
    profiles_subset /= np.trapezoid(profiles_subset, z_fine, axis=1)[:, None]

    return n_maps, nbar, z_fine, profiles_subset, nbar_fine, sigma_inv_local_mean


def plot_selection_wtheta(result, theta_arcmin, output_dir):
    plt.figure(figsize=(8, 6))
    
    # Choose representative z-bins where global n(z) is significant
    # Indices where nbar is above a threshold
    nbar_peak = np.max(result.nbar)
    significant_indices = np.where(result.nbar > 0.05 * nbar_peak)[0]
    
    if len(significant_indices) > 5:
        # Pick 5 equidistant indices from significant ones
        indices = np.linspace(significant_indices[0], significant_indices[-1], 5, dtype=int)
        # Ensure peak is included if possible
        peak_idx = np.argmax(result.nbar)
        if peak_idx not in indices:
             # Find closest in indices and replace
             closest = np.argmin(np.abs(indices - peak_idx))
             indices[closest] = peak_idx
        indices = np.sort(np.unique(indices))
    else:
        indices = significant_indices

    for i in indices:
        z_val = result.z_mid[i]
        # result.w_selection is <delta N^2> (count fluctuation correlation)
        # Divide by dz^2 to get density correlation <delta n^2>
        w_density = result.w_selection[i] / (result.dz[i] ** 2)
        
        plt.plot(theta_arcmin, w_density, linewidth=2, label=f"z={z_val:.2f}")

    plt.xlabel(r"$\theta$ [arcmin]")
    plt.ylabel(r"$\langle \delta n(z, \theta) \delta n(z, 0) \rangle$")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.title(r"Spatial Variation of $n(z)$ Density")
    plt.savefig(os.path.join(output_dir, "selection_wtheta_z.png"))
    plt.close()

def main() -> None:
    nside = 128
    ell_max = 2048
    theta_deg = np.logspace(-2, 1.0, 30)

    # Match real data peaks
    z_edges = np.linspace(0.0, 3.0, 30) 
    mean_z = 0.5
    sigma0 = 0.2
    mean_scatter = 0.04 
    width_scatter = 0.15 
    noise_level = 0.5 # Add DOMINANT shot noise

    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.8,
        n_s=0.965,
    )

    n_maps, nbar, z_fine, profiles, nbar_fine, sigma_inv_local_mean = make_uncorrelated_nz_maps(
        z_edges,
        nside,
        mean_z,
        sigma0,
        mean_scatter=mean_scatter,
        width_scatter=width_scatter,
        noise_level=noise_level,
        seed=42,
    )

    # Create a footprint similar to real data (~2.5% of sky)
    npix = hp.nside2npix(nside)
    center_vec = hp.ang2vec(0, 0, lonlat=True)
    radius_rad = np.deg2rad(10) # 5 degree radius is ~2.4% fsky at nside 128
    seen_idx = hp.query_disc(nside, center_vec, radius_rad)
    n_maps_partial = n_maps[:, seen_idx]

    enhancer = ClusteringEnhancement(cosmo, ell_max=ell_max, ell_min=2)
    result = enhancer.compute_enhancement_from_maps(
        n_maps=n_maps_partial,
        nbar=nbar,
        z=z_edges,
        theta_deg=theta_deg,
        nbar_z=(z_fine, nbar_fine),
        selection_mode="wtheta",
        nside=nside,
        seen_idx=seen_idx,
    )

    # For fair comparison in variance mode, we need to use band-limited maps
    # to match the effective filtering occurring in wtheta mode (anafast).
    lmax_map = min(ell_max, 3 * nside - 1)
    n_maps_bl = np.zeros_like(n_maps)
    for i in range(len(n_maps)):
        alms = hp.map2alm(n_maps[i], lmax=lmax_map)
        n_maps_bl[i] = hp.alm2map(alms, nside=nside)

    result_var = enhancer.compute_enhancement_from_maps(
        n_maps=n_maps_bl[:, seen_idx],
        nbar=nbar,
        z=z_edges,
        theta_deg=theta_deg,
        nbar_z=(z_fine, nbar_fine),
        selection_mode="variance",
        nside=nside,
        seen_idx=seen_idx,
    )
    
    # Compare Delta w at smallest theta
    dw_wtheta = result.delta_w[0]
    dw_var = result_var.delta_w[0]
    print(f"Sanity Check (Delta w at theta={theta_deg[0]:.4f} deg):")
    print(f"  Via w(theta) integration: {dw_wtheta:.6e}")
    print(f"  Via variance integration: {dw_var:.6e}")
    print(f"  Ratio (wtheta/variance):  {dw_wtheta / dw_var:.4f}")
    
    # Calculate Geometric factor using binned results matches selection.py/variation.py logic
    z_mid = result.z_mid
    enhancement_factor = utils.calculate_geometric_enhancement(
        z_mid, n_maps.T, nbar
    )

    # Diagnostic: xi_m weighted enhancement
    xi0 = result.xi_m[:, 0]
    var_n_arr = np.mean((n_maps - nbar[:, None]) ** 2, axis=1)
    dz_bins = result.dz
    dw_estim = np.sum(var_n_arr * (dz_bins**2) * xi0)
    w_estim = np.sum((nbar**2) * (dz_bins**2) * xi0)
    
    print(f"Geometric Enhancement Factor: {enhancement_factor:.6f}")
    print(f"Measured Clust. Enhanc (theta_min): {1.0 + result.delta_w[0] / result.w_model[0]:.6f}")
    print(f"xi_m-weighted Predicted Factor: {1.0 + dw_estim / w_estim:.6f}")

    # Diagnostic: Check decoherence at theta_min
    peak_idx = np.argmax(result.nbar)
    w_sel_0 = result.w_selection[peak_idx, 0] / (result.dz[peak_idx]**2)
    # Re-calculate var_peak properly from maps
    var_peak = np.mean((n_maps[peak_idx] - nbar[peak_idx])**2)
    print(f"Decoherence Factor (z={z_edges[peak_idx]:.2f}): w_sys(theta_min) / var_sys = {w_sel_0 / var_peak:.4f}")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for prof in profiles:
        plt.plot(z_fine, prof, color="gray", alpha=0.1)
    plt.plot(z_fine, nbar_fine, "k-", linewidth=2, label=r"Global $\bar{n}(z)$")
    for z_edge in z_edges[::20]: # Only show some edges to avoid clutter
        plt.axvline(z_edge, color="r", linestyle=":", alpha=0.2)
    plt.xlabel("Redshift z")
    plt.ylabel("n(z)")
    plt.title(f"n(z) Variations (mean_scatter={mean_scatter})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "nz_distribution_uncorrelated.png"))
    plt.close()

    # plt.figure(figsize=(7, 5))
    # theta_arcmin = 60.0 * result.theta_deg
    # plt.plot(theta_arcmin, theta_arcmin * result.delta_w, "k-", linewidth=2, label="wtheta integration")
    # plt.plot(theta_arcmin, theta_arcmin * result_var.delta_w, "r--", linewidth=2, label="variance integration")
    # plt.xlabel(r"$\theta$ [arcmin]")
    # plt.ylabel(r"$\theta \cdot \Delta w(\theta)$")
    # plt.xscale("log")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.title("Clustering Enhancement")
    # plt.savefig(os.path.join(output_dir, "delta_w.png"))
    # plt.close()

    ell = np.arange(lmax_map + 1, dtype=int)
    dndz_global = nbar_fine / np.trapezoid(nbar_fine, z_fine)
    gtracer = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z_fine, dndz_global),
        bias=(z_fine, np.ones_like(z_fine)),
    )
    cell = ccl.angular_cl(cosmo, gtracer, gtracer, ell)
    w_total = ccl.correlation(
        cosmo,
        ell=ell,
        C_ell=cell,
        theta=result.theta_deg,
        type="NN",
        method="fftlog",
    )

    theta_arcmin = 60.0 * result.theta_deg
    plt.figure(figsize=(7, 5))
    plt.plot(theta_arcmin, theta_arcmin * result.w_model, "k--", linewidth=2, label=r"$w_{\rm model}$ (binned)")
    plt.plot(theta_arcmin, theta_arcmin * w_total, "g-", linewidth=2, label=r"$w_{\rm total}$ (direct CCL)")
    plt.xlabel(r"$\theta$ [arcmin]")
    plt.ylabel(r"$\theta \cdot w(\theta)$ [arcmin]")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(r"Model vs Direct CCL")
    plt.savefig(os.path.join(output_dir, "w_model_vs_ccl.png"))
    plt.close()

    plot_selection_wtheta(result, theta_arcmin, output_dir)

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
    header_text = f"Enhanc. Factor: {enhancement_factor:.4f}\nClust. Enhanc: {result.delta_w[0]/result.w_model[0]:.4f}"
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
    plt.savefig(os.path.join(output_dir, "w_comparison.png"))
    plt.close()


if __name__ == "__main__":
    main()
