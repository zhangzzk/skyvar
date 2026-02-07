import os
os.environ["NUMEXPR_MAX_THREADS"] = "128"
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from scipy.stats import norm
import xgboost as xgb
import gc
import psutil


def get_memory_usage():
    """Return memory usage of current process in GB."""
    return psutil.Process().memory_info().rss / 1e9

def apply_post_detection_cuts(df):
    """Apply all post-detection filters (e.g., SNR, magnitude, etc.)."""
    if df is None or df.empty:
        return df
        
    snr_thresh = config.ANALYSIS_SETTINGS.get('post_det_snr_thresh', 0.0)
    if snr_thresh > 0:
        # Check for both possible column names
        snr_col = 'snr_input_p' if 'snr_input_p' in df.columns else 'snr'
        if snr_col in df.columns:
            df = df[df[snr_col] > snr_thresh]
            
    # Add future cuts here
    # Example: df = df[df['redshift'] < 2.5]
    
    return df

try:
    from . import utils
    from . import config
    from . import plotting as plt_nz
except ImportError:
    import utils
    import config
    import plotting as plt_nz

# Add paths to sys.path
BLENDING_EMULATOR_DIR = config.BLENDING_EMULATOR_DIR
if BLENDING_EMULATOR_DIR not in sys.path:
    sys.path.append(BLENDING_EMULATOR_DIR)

# Custom imports after sys.path update
try:
    import nz_utils
    from cosmic_toolbox import arraytools as at
except ImportError as e:
    print(f"Warning: Could not import some custom modules: {e}")

# Constants from config
SYS_NSIDE = config.SIM_SETTINGS['sys_nside']
N_POP_SAMPLE = config.SIM_SETTINGS['n_pop_sample']
CHUNK_SIZE = config.SIM_SETTINGS['chunk_size']
N_JOBS = config.SIM_SETTINGS['n_jobs']
GAL_CAT_PATH = config.PATHS['gal_cat']
MOCK_SYS_MAP_PATH = config.PATHS['mock_sys_map']
MODEL_JSON = config.PATHS['model_json']
OUTPUT_PREDS = config.PATHS['output_preds']
OUTPUT_FITS_TEMPLATE = config.PATHS['output_fits_template']
DETECTION_THRESHOLD = config.SIM_SETTINGS['detection_threshold']
NPIX = hp.nside2npix(SYS_NSIDE)
PHOTOZ_PARAMS = config.PHOTOZ_PARAMS

def groupby_dndz(sys_cat, z, edges, post_cut=None, weight_col=None, sim_truth=None):
    """Compute per-pixel normalized n(z) and sum_num using vectorized operations."""
    z = np.asarray(z)
    edges = np.asarray(edges)
    dz = np.diff(edges)

    n_z = len(z)

    # Filter and Prepare Weights
    if post_cut is not None:
        df_cut = sys_cat.loc[post_cut(sys_cat)].copy()
    else:
        df_cut = sys_cat.copy()

    if weight_col is None:
        df_cut["_w"] = df_cut["detection"]
    else:
        df_cut["_w"] = df_cut["detection"] * df_cut[weight_col]

    # Vectorized 2D Histogram
    pix_idx = df_cut["pix_idx_input_p"].values

    pixel_counts, hist_raw = utils.compute_pixel_histograms(
        pix_idx=pix_idx,
        vals=df_cut["redshift_input_p"].values,
        weights=df_cut["_w"].values,
        edges=edges,
        n_pix=NPIX
    )
    sum_num = pixel_counts.sum(axis=1)
    
    # Calculate global and per-pixel standard deviation
    std_z_all, std_z_pix, z_std_ratio = utils.compute_redshift_stats(
        pix_idx=pix_idx,
        z_vals=df_cut["redshift_input_p"].values,
        weights=df_cut["_w"].values,
        n_pix=NPIX
    )

    label = weight_col if weight_col else "full"
    # Inverse-std ratio (Weighted): average(1/sigma_i, weights=w_i) / (1/sigma_global)
    mask_v = (sum_num > 0) & (std_z_pix > 0)
    mean_std_z_pix_unweighted = np.mean(std_z_pix[sum_num > 0]) if np.any(sum_num > 0) else 0.0
    
    if np.any(mask_v):
        z_std_ratio_weighted = np.average(1.0 / std_z_pix[mask_v], weights=sum_num[mask_v]) * std_z_all
    else:
        z_std_ratio_weighted = 1.0

    print(
        f"[{label:10s}] Redshift-based std ratio: {z_std_ratio:.6f} "
        f"(pix-wtd inverse: {z_std_ratio_weighted:.6f}; "
        f"mean_std_pix={mean_std_z_pix_unweighted:.6f})"
    )

    out = pd.DataFrame(hist_raw, index=np.arange(NPIX))
    out.columns = np.arange(n_z)
    out["sum_num"] = sum_num
    out["std_z_pix"] = std_z_pix
    out.attrs['z_std_ratio'] = z_std_ratio
    out.attrs['z_std_ratio_pix_weighted'] = z_std_ratio_weighted

    # Global Truth; If sim_truth is provided use it, otherwise fallback to sys_cat
    if sim_truth is not None:
        dndz_in = sim_truth['dndz_in']
        num_in = sim_truth['num_in']
    else:
        # This assumes sys_cat contains all input galaxies (fallback)
        dndz_in = np.histogram(sys_cat["redshift_input_p"], bins=edges, density=True)[0]
        num_in = sys_cat.shape[0] 

    dndz_det = np.histogram(df_cut["redshift_input_p"], bins=edges, density=True, weights=df_cut["_w"])[0]
    num_det = df_cut["_w"].sum()

    out.loc["total_input"] = list(dndz_in) + [num_in, 0.0]
    out.loc["total_detected"] = list(dndz_det) + [num_det, std_z_all]
    
    return out


def smooth_nz_preserve_moments(
    z,
    nz,
    sigma_dz=0.05,
    preserve_norm=True,
    boundary_taper=True,
    taper_scale_factor=2.0,
    taper_power=2.0,
    outer_iter=8,
    mean_bracket=50.0,
    p_bracket=(0.15, 6.0),
    tol_mean=1e-10,
    tol_l2=1e-10,
    eps=1e-300,
):
    """
    Smooth n(z) and preserve:
      - mean μ
      - L2 norm S = ∫ n(z)^2 dz
      - (optionally) normalization I = ∫ n(z) dz

    Improvements over quadratic-exp tilt:
      - Mean is enforced by a 1D exponential tilt (stable).
      - L2 is enforced by a 1D power transform (stable, no tail explosion).
      - Optional taper enforces n(0)=0 by construction.
      - Gaussian smoothing uses zero-padding at boundaries.

    Parameters
    ----------
    z : (Nz,) array, strictly increasing
    nz : (Nz,) or (N, Nz) array
    sigma_dz : float
        Gaussian smoothing sigma in z-units.
    preserve_norm : bool
        If True, enforce ∫g = ∫n. Recommended for PDFs.
        If False, still enforces mean(g)=mean(n) and ∫g^2=∫n^2 but leaves ∫g free.
    boundary_taper : bool
        If True, multiply by w(z) with w(0)=0.
    taper_scale_factor : float
        z0 = taper_scale_factor * sigma_dz
    taper_power : float
        w(z) = 1 - exp(-((z-0)/z0)^p), z>=0
    outer_iter : int
        Alternating-projection iterations (usually 4–8 is enough).
    mean_bracket : float
        Search bracket for the mean-tilt parameter alpha in [-mean_bracket, +mean_bracket].
    p_bracket : (float, float)
        Search bracket for power p.
    tol_mean, tol_l2 : float
        Tolerances for constraints.
    eps : float
        Small floor for numerical stability.

    Returns
    -------
    g : same shape as nz
    """
    z = np.asarray(z, dtype=float)
    nz = np.asarray(nz, dtype=float)
    is_1d = (nz.ndim == 1)
    if is_1d:
        nz = nz[None, :]

    if z.size < 2:
        return nz[0] if is_1d else nz

    dz = np.diff(z).mean()
    if not np.isfinite(dz) or dz <= 0:
        raise ValueError("z must be strictly increasing and finite.")

    sigma_bins = float(sigma_dz / dz)

    # Smooth with zero padding (reduces boundary artifacts)
    f = gaussian_filter1d(nz, sigma=sigma_bins, axis=1, mode="constant", cval=0.0)
    f = np.clip(f, 0.0, None)

    # Optional taper enforcing n(0)=0 (physical boundary)
    if boundary_taper:
        z0 = max(taper_scale_factor * sigma_dz, 1e-12)
        u = np.clip((z - 0.0) / z0, 0.0, None)
        w = 1.0 - np.exp(-(u ** taper_power))  # w(0)=0, ->1 smoothly
        f = f * w[None, :]

    # Targets from original
    I0 = np.trapezoid(nz, z, axis=1)                   # ∫ n
    S0 = np.trapezoid(nz * nz, z, axis=1)              # ∫ n^2
    good = (I0 > 0) & (S0 > 0)

    mu0 = np.zeros(nz.shape[0])
    mu0[good] = np.trapezoid(z * nz[good], z, axis=1) / I0[good]

    g_out = f.copy()

    for i in range(nz.shape[0]):
        if not good[i]:
            # fallback: just return smoothed (optionally renormalized)
            if preserve_norm:
                If = np.trapezoid(g_out[i], z)
                if If > 0:
                    g_out[i] *= I0[i] / If
            continue

        gi = np.maximum(f[i], eps)

        # initial normalization (if requested)
        if preserve_norm:
            If = np.trapezoid(gi, z)
            if If > 0:
                gi *= I0[i] / If

        # Helpers
        def norm_to_I(x):
            """Scale x so that ∫x = I0 (if preserve_norm), else no-op."""
            if not preserve_norm:
                return x
            Ix = np.trapezoid(x, z)
            if Ix <= 0:
                return x
            return x * (I0[i] / Ix)

        def mean_of(x):
            Ix = np.trapezoid(x, z)
            if Ix <= 0:
                return np.nan
            return np.trapezoid(z * x, z) / Ix

        # For L2 constraint we always use the raw integral ∫g^2
        def l2_of(x):
            return np.trapezoid(x * x, z)

        # Alternate enforcing mean and L2
        for _ in range(outer_iter):
            # ---- (1) Enforce mean via exponential tilt: x -> x * exp(alpha*(z - zref))
            zref = mu0[i]  # centering improves conditioning

            def mean_residual(alpha):
                x = gi * np.exp(alpha * (z - zref))
                x = norm_to_I(x)
                return mean_of(x) - mu0[i]

            # If already close, skip solve
            m_now = mean_of(gi)
            if np.isfinite(m_now) and abs(m_now - mu0[i]) > tol_mean:
                # Bracket alpha; mean_residual is monotone in alpha if gi>=0
                aL, aR = -mean_bracket, mean_bracket
                fL, fR = mean_residual(aL), mean_residual(aR)
                if np.isfinite(fL) and np.isfinite(fR) and fL * fR < 0:
                    alpha = brentq(mean_residual, aL, aR, maxiter=200)
                    gi = gi * np.exp(alpha * (z - zref))
                    gi = norm_to_I(gi)

            # ---- (2) Enforce L2 via power transform: x -> x^p (controls peakiness)
            S_target = S0[i]
            S_now = l2_of(gi)

            if S_now > 0 and abs(S_now - S_target) / S_target > tol_l2:
                # Define S(p) after normalization (if enabled)
                # If preserve_norm: scaling changes S, so include it.
                def l2_residual(p):
                    x = np.maximum(gi, eps) ** p
                    x = norm_to_I(x)
                    return l2_of(x) - S_target

                pL, pR = p_bracket
                rL, rR = l2_residual(pL), l2_residual(pR)

                # If not bracketed, do nothing (means target is incompatible with current gi under this transform)
                if np.isfinite(rL) and np.isfinite(rR) and rL * rR < 0:
                    p_star = brentq(l2_residual, pL, pR, maxiter=200)
                    gi = np.maximum(gi, eps) ** p_star
                    gi = norm_to_I(gi)

        g_out[i] = gi

    return g_out[0] if is_1d else g_out



def load_and_filter_catalog():
    """Load and process the input galaxy catalog."""
    print(f"Loading catalog from {GAL_CAT_PATH}...")
    gal_cat = at.rec2pd(at.load_hdf(GAL_CAT_PATH))
    gal_cat = gal_cat[['sersic_n', 'int_mag', 'int_r50_arcsec', 'z', 'e1', 'e2']].rename(columns={
        'int_mag': 'r',
        'int_r50_arcsec': 'Re',
        'z': 'redshift',
    })

    BA, angle = utils.e1e2_to_q_phi(gal_cat['e1'], gal_cat['e2'])
    gal_cat['BA'] = BA
    gal_cat['angle'] = angle / np.pi * 180 + 90
    gal_cat['Re'] /= np.sqrt(gal_cat['BA'])

    cs = config.CATALOG_SETTINGS
    gal_cat = gal_cat.loc[
        (gal_cat['BA'] > cs['ba_min']) & (gal_cat['BA'] < cs['ba_max']) &
        (gal_cat['r'] < cs['mag_max']) & (gal_cat['r'] > cs['mag_min']) &
        (gal_cat['Re'] < cs['re_max']) & (gal_cat['Re'] > cs['re_min']) &
        (gal_cat['sersic_n'] < cs['sersic_max']) & (gal_cat['sersic_n'] > cs['sersic_min'])
    ].reset_index(drop=True).astype(np.float64)
    return gal_cat


def load_system_maps():
    """Load system maps and return maps and SEEN_idx."""
    print(f"Loading system maps from {MOCK_SYS_MAP_PATH}...")
    maps = hp.read_map(MOCK_SYS_MAP_PATH, field=None)
    SEEN_idx = np.where(~np.isnan(maps[0]))[0]
    return maps, SEEN_idx


def galaxy_snr_from_mag_size(mag, r_half, seeing_fwhm, sigma_pix, zeropoint=30.0, pixscale=0.2):
    """Approximate galaxy SNR."""

    # TODO: double check the factors 1.678 and 2.355
    flux = 10.0 ** (-0.4 * (mag - zeropoint))
    sigma_gal = r_half / 1.678
    sigma_psf = seeing_fwhm / 2.355
    sigma_eff2 = sigma_gal**2 + sigma_psf**2
    n_eff = 4.0 * np.pi * sigma_eff2 / pixscale**2
    snr = flux / (sigma_pix * np.sqrt(n_eff))
    return snr


def process_one(i, idx, icat, conditions, gal_num, psf_hp_map, noise_hp_map, galactic_hp_map, detec_mag_bound):
    """Worker function for parallel processing."""
    rng_local = np.random.default_rng(i)
    randi = rng_local.integers(0, icat.shape[0], size=gal_num)
    subset = icat.iloc[randi].copy()
    subset['pix_idx'] = idx

    # Compute RA/DEC for this pixel correctly
    nside = hp.npix2nside(len(psf_hp_map))
    ra, dec = hp.pix2ang(nside, idx, lonlat=True)
    # Add small jitter within the pixel (approx pixel size)
    pix_size_deg = np.sqrt(4 * np.pi * (180/np.pi)**2 / hp.nside2npix(nside))
    rng_jitter = np.random.default_rng(i + 12345) 
    subset['RA'] = ra + rng_jitter.uniform(-0.5 * pix_size_deg, 0.5 * pix_size_deg, size=gal_num)
    subset['DEC'] = dec + rng_jitter.uniform(-0.5 * pix_size_deg, 0.5 * pix_size_deg, size=gal_num)

    conds = conditions.copy()
    conds['psf_fwhm'] = psf_hp_map[idx]
    conds['pixel_rms'] = noise_hp_map[idx]
    
    subset.loc[:, 'r'] += galactic_hp_map[idx]
    subset.loc[subset['r'] > detec_mag_bound, 'r'] = detec_mag_bound

    for key, value in conds.items():
        subset[key] = value

    subset['snr_input_p'] = galaxy_snr_from_mag_size(
        subset['r'],
        subset['Re'],
        subset['psf_fwhm'],
        subset['pixel_rms'],
        zeropoint=subset['zero_point'],
        pixscale=subset['pixel_size'],
    )
    
    return subset

def compute_obs_stats(subset, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    # Use names with _input_p suffix as they appear in the classified catalog
    subset['snr_input_p'] = galaxy_snr_from_mag_size(
        subset['r_input_p'],
        subset['Re_input_p'],
        subset['psf_fwhm_input_p'],
        subset['pixel_rms_input_p'],
        zeropoint=subset['zero_point_input_p'],
        pixscale=subset['pixel_size_input_p'],
    )

    sigma_m = 2.5 / np.log(10) / subset['snr_input_p']
    subset['r_obs_input_p'] = subset['r_input_p'] + rng.normal(0, sigma_m, size=subset.shape[0])
    # subset['sigma_m_input_p'] = sigma_m

    # --- New photo-z model (from snippet) ---
    m = subset['r_input_p']
    rms = subset['pixel_rms_input_p']
    psf_fwhm = subset['psf_fwhm_input_p']

    alpha = PHOTOZ_PARAMS['alpha']
    sigma0 = PHOTOZ_PARAMS['sigma0']
    m_ref = PHOTOZ_PARAMS['m_ref']
    rms_ref = PHOTOZ_PARAMS['rms_ref']
    psf_fwhm_ref = PHOTOZ_PARAMS['psf_fwhm_ref']

    dm = 2.5 * np.log10((rms / rms_ref) * (psf_fwhm / psf_fwhm_ref)**2)
    k = 10**(alpha * (m - m_ref + dm))

    sigma_z = sigma0 * k
    sigma_z = np.maximum(sigma_z, PHOTOZ_PARAMS['sigma_min'])
    mu_z = subset['redshift_input_p']

    # Compute photo-z with Gaussian scatter
    z_pho = mu_z + rng.normal(0, sigma_z, size=subset.shape[0])
    
    subset['z_pho_input_p'] = z_pho
    # subset['sigma_z_input_p'] = sigma_z

    return subset

def apply_maglim_selection(subset, rng=None):
    """
    Apply MagLim-like selection using input properties.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    faint_cut = subset['r_obs_input_p'] < (PHOTOZ_PARAMS['maglim0'] * subset['z_pho_input_p'] + PHOTOZ_PARAMS['maglim1'])
    bright_cut = subset['r_obs_input_p'] > PHOTOZ_PARAMS['maglim2']
    snr_cut = subset['snr_input_p'] > PHOTOZ_PARAMS['snr_min']

    lens_keep = faint_cut & bright_cut & snr_cut
    # lens_keep &= (subset['redshift_input_p'] < config.ANALYSIS_SETTINGS['z_max']) # save memory
    subset = subset[lens_keep].reset_index(drop=True)
    return subset

def process_classified_catalog(df, rng=None):
    """
    Perform any actions between having a classified catalog and summary statistics.
    Currently: photo-z assignment and MagLim selection.
    """

    # compute observed magnitude and redshift based on input properties
    df = compute_obs_stats(df, rng=rng)

    # Apply MagLim selection
    df = apply_maglim_selection(df, rng=rng)

    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    bin_mask = get_binning_weights(df, tomo_bin_edges)
    for i in range(len(tomo_bin_edges) - 1):
        df[f"tomo_p_{i}"] = bin_mask[:, i]
    
    # Final cleanup of unused columns to save memory
    keep_cols = ['pix_idx_input_p', 'redshift_input_p', 'detection', 'snr_input_p',
                 'r_input_p', 'Re_input_p', 'sigma_m_input_p', 'sigma_z_input_p',
                 'pixel_rms_input_p', 'psf_fwhm_input_p']
    tomo_cols = [c for c in df.columns if c.startswith('tomo_p_')]
    final_cols = list(dict.fromkeys([c for c in keep_cols + tomo_cols if c in df.columns]))
    df = df[final_cols].copy()
    
    return df

def get_binning_weights(df, bin_edges):
    """
    Calculate tomographic bin membership probabilities for each galaxy.
    Using "hard" binning (binary weights) based on simulated photo-z.
    """
    z_hat_g = df['z_pho_input_p'].values
    weights = np.zeros((len(z_hat_g), len(bin_edges) - 1))
    
    for i in range(len(bin_edges) - 1):
        z_min, z_max = bin_edges[i], bin_edges[i+1]
        mask = (z_hat_g >= z_min) & (z_hat_g < z_max)
        weights[mask, i] = 1.0

    return weights


def simulate_and_classify_chunked(gal_cat, z, edges):
    """
    Memory-efficient simulation and classification. 
    Filters non-detections immediately to save 90%+ memory.
    """
    import gc
    maps, SEEN_idx = load_system_maps()
    psf_hp_map, noise_hp_map, galactic_hp_map = maps
    
    conditions = {
        "pixel_size": config.OBS_CONDITIONS['pixel_size'],
        "zero_point": config.OBS_CONDITIONS['zero_point'],
        "psf_fwhm": config.OBS_CONDITIONS['psf_fwhm_nominal'],
        "moffat_beta": config.OBS_CONDITIONS['moffat_beta'],
        "pixel_rms": config.OBS_CONDITIONS['pixel_rms_nominal'],
    }
    detec_mag_bound = config.OBS_CONDITIONS['detec_mag_bound']
    z_bins_n = len(z)

    print("Loading XGBoost model...")
    bst_cla = xgb.Booster({'device': 'cuda', 'n_jobs': -1})
    bst_cla.load_model(MODEL_JSON)
    
    temp_dir = os.path.join(os.path.dirname(OUTPUT_PREDS), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Global Input Statistics Accumulators
    global_hist_in = np.zeros(z_bins_n)
    global_num_in = 0
    
    pixels_per_chunk = max(1, CHUNK_SIZE // N_POP_SAMPLE)
    n_pixels = len(SEEN_idx)
    chunk_files = []
    
    print(f"Starting chunked processing: {n_pixels} pixels in groups of {pixels_per_chunk}")
    
    for start_p in range(0, n_pixels, pixels_per_chunk):
        end_p = min(start_p + pixels_per_chunk, n_pixels)
        block_indices = SEEN_idx[start_p:end_p]
        
        print(f"  Chunk {start_p//pixels_per_chunk + 1}/{(n_pixels-1)//pixels_per_chunk + 1} ({len(block_indices)} pixels)")
        
        # 1. Simulate this block
        results = Parallel(n_jobs=N_JOBS, backend="threading")(
            delayed(process_one)(start_p + i, idx, gal_cat, conditions, N_POP_SAMPLE,
                                 psf_hp_map, noise_hp_map, galactic_hp_map, detec_mag_bound)
            for i, idx in enumerate(block_indices)
        )
        block_fullset = pd.concat(results, ignore_index=True)
        results = None 
        
        # Accumulate input statistics before classification/filtering
        h_in = np.histogram(block_fullset["redshift"], bins=edges)[0]
        global_hist_in += h_in
        global_num_in += len(block_fullset)
        
        # 2. Coordinates are now handled in process_one.

        # 3. Classify and Filter
        try:
            block_cla = nz_utils.icat2cla_v2(block_fullset, block_fullset, bst_cla, predict=True)
            # CRITICAL: Keep only detections to save memory
            block_cla = block_cla[block_cla['detection'] > DETECTION_THRESHOLD].copy()
            
            # Keep all columns needed by downstream photo-z, MagLim, and stats.
            # !CAREFUL: only part of columns are kept to save memory, while it can lead to bug if some columns are missing
            keep_cols = [
                'pix_idx_input_p', 'redshift_input_p', 'detection',
                'r_input_p', 'Re_input_p', 'sersic_n_input_p', 'axis_ratio_input_p',
                'psf_fwhm_input_p', 'pixel_rms_input_p', 'zero_point_input_p',
                'pixel_size_input_p', 'moffat_beta_input_p'
            ]
            block_cla = block_cla[[c for c in keep_cols if c in block_cla.columns]].copy()
            
            if not block_cla.empty:
                temp_path = os.path.join(temp_dir, f"cla_chunk_det_{start_p}.feather")
                block_cla.to_feather(temp_path)
                chunk_files.append(temp_path)
        except Exception as e:
            print(f"CRITICAL ERROR processing block {start_p}: {e}")
            raise RuntimeError(f"Simulation failed at block {start_p}. Terminating for robustness.") from e
        
        block_fullset = None
        block_cla = None
        gc.collect()
        print(f"    Memory usage: {get_memory_usage():.2f} GB")

    # Pre-calculated density and num
    dz = np.diff(edges)
    dndz_in_total = global_hist_in / (global_num_in * dz) if global_num_in > 0 else global_hist_in
    
    # Pack global simulation truth
    sim_truth = {
        'dndz_in': dndz_in_total,
        'num_in': global_num_in,
        'edges': edges,
        'dz': dz,
        'z': z
    }
        
    return chunk_files, psf_hp_map, SEEN_idx, sim_truth


def generate_summary_statistics_from_cat(cla_cat, SEEN_idx, output_dir, z, edges, sim_truth=None):
    """Compute detection maps and dN/dz distributions from a catalog."""
    mean_p = np.full(NPIX, hp.UNSEEN)
    
    # Map pixels to active indices to avoid out-of-bounds
    pixel_counts = np.bincount(cla_cat['pix_idx_input_p'], weights=cla_cat['detection'], minlength=NPIX)
    mean_p[SEEN_idx] = pixel_counts[SEEN_idx] / N_POP_SAMPLE
    
    p_valid = mean_p[SEEN_idx]
    print(f"Detection Rate Stats: min={np.min(p_valid):.4f}, max={np.max(p_valid):.4f}, mean={np.mean(p_valid):.4f}")

    plt_nz.plt_map(mean_p, SYS_NSIDE, SEEN_idx, 
            save_path=os.path.join(output_dir, "detection_rate_map.png"))
    
    results = {}
    dz = np.diff(edges)

    # Full Sample
    sys_res_full = groupby_dndz(cla_cat, z, edges, sim_truth=sim_truth)
    metadata_rows_full = sys_res_full.loc[["total_input", "total_detected"]].copy()
    sys_res_data_full = sys_res_full.reindex(SEEN_idx).fillna(0)
    sys_res_final_full = pd.concat([sys_res_data_full, metadata_rows_full])
    
    results['full'] = process_stats(sys_res_final_full, z, SEEN_idx, smooth=config.ANALYSIS_SETTINGS['smooth_nz'])
    
    # Tomographic Bins
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    for i in range(len(tomo_bin_edges)-1):
        tomo_col = f"tomo_p_{i}"
        if tomo_col in cla_cat.columns:
            sys_res_i = groupby_dndz(cla_cat, z, edges, weight_col=tomo_col, sim_truth=sim_truth)
            meta_i = sys_res_i.loc[["total_input", "total_detected"]].copy()
            sys_res_i_data = sys_res_i.reindex(SEEN_idx).fillna(0)
            sys_res_i_final = pd.concat([sys_res_i_data, meta_i])
            
            results[f'tomo_{i}'] = process_stats(sys_res_i_final, z, SEEN_idx, smooth=config.ANALYSIS_SETTINGS['smooth_nz'])
            
    return results


def process_stats(sys_res, z, SEEN_idx, smooth=False):
    """Auxiliary to package dndz results."""
    dndzs = sys_res.drop(["sum_num", "std_z_pix"], axis=1)
    if "total_input" in dndzs.index:
        dndzs = dndzs.drop(["total_input", "total_detected"])
    
    dndzs = dndzs.to_numpy()
    sum_num = sys_res["sum_num"].drop(["total_input", "total_detected"]).values
    frac_pix = sum_num / N_POP_SAMPLE
    std_z_pix = sys_res["std_z_pix"].drop(["total_input", "total_detected"]).values
    
    dndz_in = sys_res.drop(["sum_num", "std_z_pix"], axis=1).loc["total_input"].to_numpy().astype(float)
    dndz_det = sys_res.drop(["sum_num", "std_z_pix"], axis=1).loc["total_detected"].to_numpy().astype(float)
    frac = sys_res.loc["total_detected", "sum_num"] / sys_res.loc["total_input", "sum_num"]
    std_z_global = sys_res.loc["total_detected", "std_z_pix"]

    z_std_ratio = sys_res.attrs.get('z_std_ratio', 1.0)

    # Calculate unsmoothed stats (baseline)
    geo_w_pix, geo_w_glob, geo_enhancement = utils.calculate_geometric_stats(z, dndzs, dndz_det, frac_pix=None)
    z_std_ratio_binned = utils.calculate_binned_std_ratio(z, dndzs, dndz_det, frac_pix=None)
    z_std_ratio_weighted = sys_res.attrs.get('z_std_ratio_pix_weighted', 1.0)

    if smooth:
        print(f"Smoothing {dndzs.shape[0]} distributions...")
        sigma_dz = config.ANALYSIS_SETTINGS['smoothing_sigma_dz']
        sm_dndzs = smooth_nz_preserve_moments(z, dndzs, sigma_dz=sigma_dz)
        sm_dndz_in = smooth_nz_preserve_moments(z, dndz_in, sigma_dz=sigma_dz)
        sm_dndz_det = smooth_nz_preserve_moments(z, dndz_det, sigma_dz=sigma_dz)
        
        # Recalculate widths for smoothed distributions
        sm_geo_w_pix, sm_geo_w_glob, sm_geo_enhancement = utils.calculate_geometric_stats(z, sm_dndzs, sm_dndz_det, frac_pix=None)
        sm_z_std_ratio_binned = utils.calculate_binned_std_ratio(z, sm_dndzs, sm_dndz_det, frac_pix=None)

        return {
            'z': z, 'dndzs': sm_dndzs, 'dndz_in': sm_dndz_in, 'dndz_det': sm_dndz_det,
            'frac': frac, 'frac_pix': frac_pix, 'SEEN_idx': SEEN_idx,
            'z_std_ratio': z_std_ratio, 
            'z_std_ratio_weighted': z_std_ratio_weighted,
            'z_std_ratio_binned': sm_z_std_ratio_binned,
            'z_std_ratio_binned_unsmoothed': z_std_ratio_binned,
            'std_z_pix': std_z_pix, 'std_z_global': std_z_global,
            'geo_width_pix': sm_geo_w_pix, 'geo_width_global': sm_geo_w_glob, 
            'geo_enhancement': sm_geo_enhancement,
            'geo_enhancement_unsmoothed': geo_enhancement
        }
    else:
        return {
            'z': z, 'dndzs': dndzs, 'dndz_in': dndz_in, 'dndz_det': dndz_det,
            'frac': frac, 'frac_pix': frac_pix, 'SEEN_idx': SEEN_idx,
            'z_std_ratio': z_std_ratio, 
            'z_std_ratio_weighted': z_std_ratio_weighted,
            'z_std_ratio_binned': z_std_ratio_binned,
            'z_std_ratio_binned_unsmoothed': z_std_ratio_binned,
            'std_z_pix': std_z_pix, 'std_z_global': std_z_global,
            'geo_width_pix': geo_w_pix, 'geo_width_global': geo_w_glob, 
            'geo_enhancement': geo_enhancement,
            'geo_enhancement_unsmoothed': geo_enhancement
        }


# def generate_summary_statistics_incremental(chunk_files, SEEN_idx, output_dir, z, edges):
#     """Memory-efficient incremental statistics generation."""
#     import gc
#     tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
#     keys = ['full'] + [f'tomo_{i}' for i in range(len(tomo_bin_edges)-1)]
#     dz = np.diff(edges)
#     n_z = len(z)

#     # Accumulators
#     det_counts_map = np.zeros(NPIX)
    
#     accumulators = {k: np.zeros((NPIX, n_z)) for k in keys}
#     total_det_nums = {k: 0.0 for k in keys}
#     total_det_hists = {k: np.zeros(n_z) for k in keys}
    
#     # Track the global input population distribution
#     global_input_hist = np.zeros(n_z)
#     total_input_num = 0
    
#     # Accumulators for z_std_ratio
#     w_sum_pix = {k: np.zeros(NPIX) for k in keys}
#     wz_sum_pix = {k: np.zeros(NPIX) for k in keys}
#     wz2_sum_pix = {k: np.zeros(NPIX) for k in keys}
    
#     total_z_w_sum = {k: 0.0 for k in keys}
#     total_z_wz_sum = {k: 0.0 for k in keys}
#     total_z_wz2_sum = {k: 0.0 for k in keys}

#     print(f"Incremental accumulation from chunks (with post-detection cuts)...")
#     for f in chunk_files:
#         df = pd.read_feather(f)
#         if df.empty: continue
        
#         df = process_classified_catalog(df)
#         if df.empty: continue

#         pixel_indices = df["pix_idx_input_p"].values
#         z_vals = df["redshift_input_p"].values
#         weights_det = df["detection"].values
        
#         # Accumulate input truth (unweighted by detection)
#         global_input_hist += np.histogram(z_vals, bins=edges)[0]
#         total_input_num += len(df)
        
#         # Detection rate map (soft count)
#         pixel_counts = np.bincount(pixel_indices, weights=weights_det, minlength=NPIX)
#         det_counts_map += pixel_counts
        

#         for k in keys:
#             if k == 'full':
#                 weights = weights_det
#             else:
#                 idx = int(k.split('_')[1])
#                 weights = weights_det * df[f'tomo_p_{idx}'].values
                
#             total_det_nums[k] += weights.sum()
#             total_det_hists[k] += np.histogram(z_vals, bins=edges, weights=weights)[0]
            
#             # Per-pixel histogram accumulation
#             pixel_counts_k, _ = utils.compute_pixel_histograms(
#                 pix_idx=pixel_indices,
#                 vals=z_vals,
#                 weights=weights,
#                 edges=edges,
#                 n_pix=NPIX
#             )
#             accumulators[k] += pixel_counts_k

#             # Redshift-based accumulation
#             w_sum_pix[k] += np.bincount(pixel_indices, weights=weights, minlength=NPIX)
#             wz_sum_pix[k] += np.bincount(pixel_indices, weights=weights * z_vals, minlength=NPIX)
#             wz2_sum_pix[k] += np.bincount(pixel_indices, weights=weights * z_vals**2, minlength=NPIX)
            
#             total_z_w_sum[k] += weights.sum()
#             total_z_wz_sum[k] += (weights * z_vals).sum()
#             total_z_wz2_sum[k] += (weights * z_vals**2).sum()
            
#         df = None
#         gc.collect()

#     # Post-process and packaging
#     final_results = {}
    
#     # 1. Detection Rate Map
#     mean_p = np.full(NPIX, hp.UNSEEN)
#     mean_p[SEEN_idx] = det_counts_map[SEEN_idx] / N_POP_SAMPLE
    
#     p_valid = mean_p[SEEN_idx]
#     print(f"Detection Rate Stats: min={np.min(p_valid):.4f}, max={np.max(p_valid):.4f}, mean={np.mean(p_valid):.4f}")
    
#     plt_nz.plt_map(mean_p, SYS_NSIDE, SEEN_idx, 
#             save_path=os.path.join(output_dir, "detection_rate_map.png"))

#     for k in keys:
#         pixel_counts = accumulators[k]
#         sum_num = pixel_counts.sum(axis=1)
        
#         # Filter to SEEN_idx and normalize
#         active_counts = pixel_counts[SEEN_idx]
#         active_sum_num = sum_num[SEEN_idx]
        
#         hist_raw = np.zeros_like(active_counts)
#         mask_v = active_sum_num > 0
#         hist_raw[mask_v] = active_counts[mask_v] / (active_sum_num[mask_v][:, None] * dz)
        
#         df_stats = pd.DataFrame(hist_raw)
#         df_stats["sum_num"] = active_sum_num
#         # Calculate redshift stats for this bin using accumulated sums
#         std_z_all, std_z_pix, z_std_ratio = utils.compute_redshift_stats_from_sums(
#             w_sum_pix=w_sum_pix[k][SEEN_idx],
#             wz_sum_pix=wz_sum_pix[k][SEEN_idx],
#             wz2_sum_pix=wz2_sum_pix[k][SEEN_idx],
#             total_w=total_z_w_sum[k],
#             total_wz=total_z_wz_sum[k],
#             total_wz2=total_z_wz2_sum[k]
#         )
#         df_stats["std_z_pix"] = std_z_pix
#         df_stats.attrs['z_std_ratio'] = z_std_ratio

#         w_sum_seen = w_sum_pix[k][SEEN_idx]
#         mask_v = (w_sum_seen > 0) & (std_z_pix > 0)
#         mean_std_z_pix_unweighted = np.mean(std_z_pix[w_sum_seen > 0]) if np.any(w_sum_seen > 0) else 0.0
        
#         if np.any(mask_v):
#             z_std_ratio_weighted = np.average(1.0 / std_z_pix[mask_v], weights=w_sum_seen[mask_v]) * std_z_all
#         else:
#             z_std_ratio_weighted = 1.0

#         df_stats.attrs['z_std_ratio_pix_weighted'] = z_std_ratio_weighted
#         print(
#             f"[{k:10s}] Redshift-based std ratio: {z_std_ratio:.6f} "
#             f"(pix-wtd inverse: {z_std_ratio_weighted:.6f}; "
#             f"mean_std_pix={mean_std_z_pix_unweighted:.6f})"
#         )

#         # dndz totals
#         d_det = total_det_hists[k] / (total_det_nums[k] * dz) if total_det_nums[k] > 0 else total_det_hists[k]
        
#         # Reference Baseline (Input)
#         d_in = global_input_hist / (total_input_num * dz) if total_input_num > 0 else d_det

#         df_stats.loc["total_input"] = list(d_in) + [total_input_num, 0.0]
#         df_stats.loc["total_detected"] = list(d_det) + [total_det_nums[k], std_z_all]
        
#         final_results[k] = process_stats(df_stats, z, SEEN_idx, smooth=config.ANALYSIS_SETTINGS['smooth_nz'])
    
#     return final_results






def save_fits_output(stats, bin_idx=4):
    """Store final results in a multi-HDU FITS file."""
    # Using magbin=1 as default for consistency with notebook template
    output_fits_path = OUTPUT_FITS_TEMPLATE.format(SYS_NSIDE, N_POP_SAMPLE, bin_idx)
    os.makedirs(os.path.dirname(output_fits_path), exist_ok=True)
    
    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(stats['z'], name='Z'),
        fits.ImageHDU(stats['dndzs'], name='DNDZS'),
        fits.ImageHDU(stats['dndz_in'], name='DNDZ_IN'),
        fits.ImageHDU(stats['dndz_det'], name='DNDZ_DET'),
        fits.ImageHDU([stats['frac']], name='FRAC'),
        fits.ImageHDU(stats['frac_pix'], name='FRAC_PIX'),
        fits.ImageHDU(stats['SEEN_idx'], name='SEEN_IDX'),
        fits.ImageHDU([stats.get('z_std_ratio', 1.0)], name='Z_STD_RATIO'),
        fits.ImageHDU(stats['std_z_pix'], name='STD_Z_PIX'),
        fits.ImageHDU(stats.get('geo_width_pix', np.zeros_like(stats['std_z_pix'])), name='GEO_WIDTH_PIX'),
        fits.ImageHDU([stats.get('geo_width_global', 0.0)], name='GEO_WIDTH_GLOBAL'),
        fits.ImageHDU([stats.get('geo_enhancement_unsmoothed', 1.0)], name='GEO_ENH_UNSM')
    ]
    fits.HDUList(hdus).writeto(output_fits_path, overwrite=True)
    print(f"Results saved to {output_fits_path}")


def load_fits_output(bin_idx=4):
    """Load results from a multi-HDU FITS file."""
    output_fits_path = OUTPUT_FITS_TEMPLATE.format(SYS_NSIDE, N_POP_SAMPLE, bin_idx)
    if not os.path.exists(output_fits_path):
        print(f"Warning: FITS file {output_fits_path} not found.")
        return None
    
    with fits.open(output_fits_path) as hdul:
        stats = {
            'z': hdul['Z'].data,
            'dndzs': hdul['DNDZS'].data,
            'dndz_in': hdul['DNDZ_IN'].data,
            'dndz_det': hdul['DNDZ_DET'].data,
            'frac': hdul['FRAC'].data[0],
            'frac_pix': hdul['FRAC_PIX'].data,
            'SEEN_idx': hdul['SEEN_IDX'].data,
            'z_std_ratio': hdul['Z_STD_RATIO'].data[0] if 'Z_STD_RATIO' in hdul else 1.0,
            'std_z_pix': hdul['STD_Z_PIX'].data if 'STD_Z_PIX' in hdul else np.zeros_like(hdul['FRAC_PIX'].data),
            'geo_width_pix': hdul['GEO_WIDTH_PIX'].data if 'GEO_WIDTH_PIX' in hdul else np.zeros_like(hdul['FRAC_PIX'].data),
            'geo_width_global': hdul['GEO_WIDTH_GLOBAL'].data[0] if 'GEO_WIDTH_GLOBAL' in hdul else 0.0,
            'geo_enhancement_unsmoothed': hdul['GEO_ENH_UNSM'].data[0] if 'GEO_ENH_UNSM' in hdul else 1.0
        }
    return stats




def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Use fixed redshift bins from config
    z, edges = utils.get_redshift_bins(None)
    print(f"[Binning] Fixed Grid: z=[{edges[0]:.4f}, {edges[-1]:.4f}], dz={(edges[1]-edges[0]):.4f}, n_bins={len(z)}")

    maps, SEEN_idx = load_system_maps()

    if config.ANALYSIS_SETTINGS.get('load_preds', True) and os.path.exists(OUTPUT_PREDS):
        print(f"Loading existing predictions from {OUTPUT_PREDS}...")
        cla_cat = pd.read_feather(OUTPUT_PREDS)
        
        print("Processing loaded catalog (photo-z, cuts)...")
        cla_cat = process_classified_catalog(cla_cat)
        
        results = generate_summary_statistics_from_cat(cla_cat, SEEN_idx, output_dir, z, edges)
    else:
        if not os.path.exists(OUTPUT_PREDS):
             print(f"Predictions file {OUTPUT_PREDS} not found. Running simulation...")
        gal_cat = load_and_filter_catalog()
        
        chunk_files, psf_hp_map, SEEN_idx, sim_truth = simulate_and_classify_chunked(gal_cat, z=z, edges=edges)
        
        # Consolidation of detected galaxies (ONLY detected, so much smaller)
        print(f"Re-assembling {len(chunk_files)} detected-only chunks...")
        cla_cat = pd.concat([pd.read_feather(f) for f in chunk_files], ignore_index=True)

        print(f"Saving raw detected catalog to {OUTPUT_PREDS}...")
        os.makedirs(os.path.dirname(OUTPUT_PREDS), exist_ok=True)
        cla_cat.to_feather(OUTPUT_PREDS)
        
        print("Final processing of consolidated catalog...")
        cla_cat = process_classified_catalog(cla_cat)
        results = generate_summary_statistics_from_cat(cla_cat, SEEN_idx, output_dir, z, edges, sim_truth=sim_truth)
            
        print(f"    Memory usage after re-assembly and processing: {get_memory_usage():.2f} GB")
        
        # Cleanup temporary chunks
        for f in chunk_files:
            os.remove(f)
        temp_dir = os.path.join(os.path.dirname(OUTPUT_PREDS), "temp_chunks")
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
        print(f"    Memory usage at end of simulation: {get_memory_usage():.2f} GB")
    
    plt_nz.save_diagnostic_plots(results, output_dir)
    plt_nz.plot_tomographic_bins(results, output_dir)
    plt_nz.plot_snr_fractions(cla_cat, output_dir, z=z, edges=edges)
    plt_nz.plot_pixel_std_histograms(results, output_dir)
    plt_nz.plot_geo_vs_std_scatter(results, output_dir)
    # plt_nz.plot_photoz_weight_histograms(cla_cat, output_dir)
    # plt_nz.plot_dm_c_comparison_objects(cla_cat, output_dir)
    # plt_nz.plot_z_distribution_comparison(cla_cat, output_dir, z=z, edges=edges)
    
    # Save full sample and tomographic bins
    # Using bin_idx=99 for full sample to avoid overlap with tomo bins
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    save_fits_output(results['full'], bin_idx=99) 
    for i in range(len(tomo_bin_edges) - 1):
        if f'tomo_{i}' in results:
            save_fits_output(results[f'tomo_{i}'], bin_idx=i)

    print("\n--- Enhancement Factor Results ---")
    print(f"{'Bin':<12} | {'GeoEnh(Sm)':<10} | {'GeoEnh(Unsm)':<12} | {'zStd(Unw)':<10} | {'zStd(Wtd)':<10} | {'Binned(Sm)':<10} | {'Binned(Unsm)':<12}")
    print("-" * 100)
    for key, stats in results.items():
        print(f"{key:<12} | "
              f"{stats.get('geo_enhancement', 1.0):.4f}     | "
              f"{stats.get('geo_enhancement_unsmoothed', 1.0):.4f}       | "
              f"{stats.get('z_std_ratio', 1.0):.4f}     | "
              f"{stats.get('z_std_ratio_weighted', 1.0):.4f}     | "
              f"{stats.get('z_std_ratio_binned', 1.0):.4f}     | "
              f"{stats.get('z_std_ratio_binned_unsmoothed', 1.0):.4f}")


if __name__ == "__main__":
    main()
