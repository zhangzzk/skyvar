import os
os.environ["NUMEXPR_MAX_THREADS"] = "128"
import sys
import logging
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
import time

logger = logging.getLogger(__name__)


def get_memory_usage():
    """Return the current process memory usage in GB."""
    return psutil.Process().memory_info().rss / 1e9

def apply_post_detection_cuts(df):
    """Apply optional cuts after detection."""
    if df is None or df.empty:
        return df
        
    snr_thresh = config.ANALYSIS_SETTINGS.get('post_det_snr_thresh', 0.0)
    if snr_thresh > 0:
        # Support both legacy and current column names.
        snr_col = 'snr_input_p' if 'snr_input_p' in df.columns else 'snr'
        if snr_col in df.columns:
            df = df[df[snr_col] > snr_thresh]
            
    # Add extra post-detection cuts here if needed.
    
    return df

try:
    from . import utils
    from . import config
    from . import plotting as plt_nz
except ImportError:
    import utils
    import config
    import plotting as plt_nz

# Make external emulator modules importable.
BLENDING_EMULATOR_DIR = config.BLENDING_EMULATOR_DIR
if BLENDING_EMULATOR_DIR not in sys.path:
    sys.path.append(BLENDING_EMULATOR_DIR)

# Import external modules after updating sys.path.
try:
    import nz_utils
    from cosmic_toolbox import arraytools as at
except ImportError as e:
    logger.warning("Could not import some custom modules: %s", e)

# Constants loaded from config.
SYS_NSIDE_SIM = config.SIM_SETTINGS['sys_nside_sim']
SYS_NSIDE_STATS = config.SIM_SETTINGS['sys_nside_stats']
N_POP_SAMPLE = config.SIM_SETTINGS['n_pop_sample']
CHUNK_SIZE = config.SIM_SETTINGS['chunk_size']
N_JOBS = config.SIM_SETTINGS['n_jobs']
GAL_CAT_PATH = config.PATHS['gal_cat']
MOCK_SYS_MAP_PATH = utils.get_output_path("mock_sys_map")
MODEL_JSON = config.PATHS['model_json']
OUTPUT_PREDS = utils.get_output_path("output_preds")
DETECTION_THRESHOLD = config.SIM_SETTINGS['detection_threshold']
NPIX = hp.nside2npix(SYS_NSIDE_STATS)
PHOTOZ_PARAMS = config.PHOTOZ_PARAMS


def downcast_float64_to32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to reduce memory use."""
    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        df.loc[:, float64_cols] = df.loc[:, float64_cols].astype(np.float32)
    return df

def map_pix_sim_to_stats(pix_idx_sim, nside_sim, nside_stats):
    """Map HEALPix pixel IDs from the simulation grid to the stats grid."""
    pix_idx_sim = np.asarray(pix_idx_sim, dtype=np.int64)
    if int(nside_sim) == int(nside_stats):
        return pix_idx_sim.copy()

    unique_sim, inv = np.unique(pix_idx_sim, return_inverse=True)
    ra_u, dec_u = hp.pix2ang(int(nside_sim), unique_sim, lonlat=True)
    unique_stats = hp.ang2pix(int(nside_stats), ra_u, dec_u, lonlat=True).astype(np.int64)
    return unique_stats[inv]


def get_input_counts_per_stats_pixel(seen_idx_sim, nside_sim, nside_stats, n_pop_sample):
    """Return total simulated input counts per stats pixel."""
    n_pix_stats = hp.nside2npix(int(nside_stats))
    pix_stats = map_pix_sim_to_stats(seen_idx_sim, nside_sim, nside_stats)
    n_sim_per_stats = np.bincount(pix_stats, minlength=n_pix_stats).astype(np.int64)
    return n_sim_per_stats * int(n_pop_sample)


def groupby_dndz(sys_cat, z, edges, post_cut=None, weight_col=None, pix_col="pix_idx_input_p", n_pix=None):
    """Compute per-pixel normalized n(z) and effective counts with vectorized ops."""
    z = np.asarray(z)
    edges = np.asarray(edges)
    dz = np.diff(edges)

    n_z = len(z)

    # Apply optional row filter and build weights.
    if post_cut is not None:
        df_cut = sys_cat.loc[post_cut(sys_cat)].copy()
    else:
        df_cut = sys_cat.copy()

    if weight_col is None:
        df_cut["_w"] = df_cut["detection"]
    else:
        df_cut["_w"] = df_cut["detection"] * df_cut[weight_col]

    # Build per-pixel redshift histograms.
    if n_pix is None:
        n_pix = NPIX
    pix_idx = df_cut[pix_col].values

    pixel_counts, hist_raw = utils.compute_pixel_histograms(
        pix_idx=pix_idx,
        vals=df_cut["redshift_input_p"].values,
        weights=df_cut["_w"].values,
        edges=edges,
        n_pix=n_pix
    )
    sum_num = pixel_counts.sum(axis=1)
    
    # Compute global and per-pixel redshift scatter.
    std_z_all, std_z_pix, z_std_ratio = utils.compute_redshift_stats(
        pix_idx=pix_idx,
        z_vals=df_cut["redshift_input_p"].values,
        weights=df_cut["_w"].values,
        n_pix=n_pix
    )

    label = weight_col if weight_col else "full"
    # Weighted inverse-std ratio: <1/sigma_i>_w / (1/sigma_global).
    mask_v = (sum_num > 0) & (std_z_pix > 0)
    mean_std_z_pix_unweighted = np.mean(std_z_pix[sum_num > 0]) if np.any(sum_num > 0) else 0.0
    
    if np.any(mask_v):
        z_std_ratio_weighted = np.average(1.0 / std_z_pix[mask_v], weights=sum_num[mask_v]) * std_z_all
    else:
        z_std_ratio_weighted = 1.0

    logger.info(
        "[%10s] Redshift-based std ratio: %.6f "
        "(pix-wtd inverse: %.6f; mean_std_pix=%.6f)",
        label, z_std_ratio, z_std_ratio_weighted, mean_std_z_pix_unweighted
    )

    out = pd.DataFrame(hist_raw, index=np.arange(n_pix))
    out.columns = np.arange(n_z)
    out["sum_num"] = sum_num
    out["std_z_pix"] = std_z_pix
    out.attrs['z_std_ratio'] = z_std_ratio
    out.attrs['z_std_ratio_pix_weighted'] = z_std_ratio_weighted

    dndz_det = np.histogram(df_cut["redshift_input_p"], bins=edges, density=True, weights=df_cut["_w"])[0]
    num_det = df_cut["_w"].sum()

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

    # Smooth with zero padding to reduce edge artifacts.
    f = gaussian_filter1d(nz, sigma=sigma_bins, axis=1, mode="constant", cval=0.0)
    f = np.clip(f, 0.0, None)

    # Optional taper so n(0)=0 (physical boundary condition).
    if boundary_taper:
        z0 = max(taper_scale_factor * sigma_dz, 1e-12)
        u = np.clip((z - 0.0) / z0, 0.0, None)
        w = 1.0 - np.exp(-(u ** taper_power))  # w(0)=0 and approaches 1 smoothly.
        f = f * w[None, :]

    # Moment targets from the original profile.
    I0 = np.trapezoid(nz, z, axis=1)                   # ∫ n
    S0 = np.trapezoid(nz * nz, z, axis=1)              # ∫ n^2
    good = (I0 > 0) & (S0 > 0)

    mu0 = np.zeros(nz.shape[0])
    mu0[good] = np.trapezoid(z * nz[good], z, axis=1) / I0[good]

    g_out = f.copy()

    for i in range(nz.shape[0]):
        if not good[i]:
            # Fallback: return the smoothed profile (optionally renormalized).
            if preserve_norm:
                If = np.trapezoid(g_out[i], z)
                if If > 0:
                    g_out[i] *= I0[i] / If
            continue

        gi = np.maximum(f[i], eps)

        # Initial normalization (if requested).
        if preserve_norm:
            If = np.trapezoid(gi, z)
            if If > 0:
                gi *= I0[i] / If

        # Helper functions.
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

        # For the L2 constraint, use the raw integral ∫g^2.
        def l2_of(x):
            return np.trapezoid(x * x, z)

        # Alternate mean and L2 enforcement.
        for _ in range(outer_iter):
            # (1) Enforce mean via exponential tilt: x -> x * exp(alpha*(z - zref)).
            zref = mu0[i]  # Centering improves conditioning.

            def mean_residual(alpha):
                x = gi * np.exp(alpha * (z - zref))
                x = norm_to_I(x)
                return mean_of(x) - mu0[i]

            # Skip solve when already close.
            m_now = mean_of(gi)
            if np.isfinite(m_now) and abs(m_now - mu0[i]) > tol_mean:
                # Bracket alpha; mean_residual is monotone if gi >= 0.
                aL, aR = -mean_bracket, mean_bracket
                fL, fR = mean_residual(aL), mean_residual(aR)
                if np.isfinite(fL) and np.isfinite(fR) and fL * fR < 0:
                    alpha = brentq(mean_residual, aL, aR, maxiter=200)
                    gi = gi * np.exp(alpha * (z - zref))
                    gi = norm_to_I(gi)

            # (2) Enforce L2 via power transform: x -> x^p.
            S_target = S0[i]
            S_now = l2_of(gi)

            if S_now > 0 and abs(S_now - S_target) / S_target > tol_l2:
                # Define S(p) after normalization (if enabled).
                # If preserve_norm is True, scaling changes S and must be included.
                def l2_residual(p):
                    x = np.maximum(gi, eps) ** p
                    x = norm_to_I(x)
                    return l2_of(x) - S_target

                pL, pR = p_bracket
                rL, rR = l2_residual(pL), l2_residual(pR)

                # If not bracketed, keep the current profile for this iteration.
                if np.isfinite(rL) and np.isfinite(rR) and rL * rR < 0:
                    p_star = brentq(l2_residual, pL, pR, maxiter=200)
                    gi = np.maximum(gi, eps) ** p_star
                    gi = norm_to_I(gi)

        g_out[i] = gi

    return g_out[0] if is_1d else g_out



def load_and_filter_catalog():
    """Load the input catalog and apply baseline quality cuts."""
    logger.info("Loading catalog from %s...", GAL_CAT_PATH)
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
    ].reset_index(drop=True)

    # Keep only fields used downstream.
    gal_cat = gal_cat[['sersic_n', 'r', 'Re', 'redshift', 'BA', 'angle']]
    gal_cat = gal_cat.astype(np.float32)
    return gal_cat


def load_system_maps(return_sim_idx: bool = False):
    """Load system maps and return footprint indices on the stats grid.
    """
    logger.info("Loading system maps from %s...", MOCK_SYS_MAP_PATH)
    maps = hp.read_map(MOCK_SYS_MAP_PATH, field=None)
    seen_idx_sim = np.where(~np.isnan(maps[0]))[0]

    if SYS_NSIDE_SIM == SYS_NSIDE_STATS:
        seen_idx_stats = seen_idx_sim
    else:
        mask_sim = np.zeros(hp.nside2npix(SYS_NSIDE_SIM), dtype=float)
        mask_sim[seen_idx_sim] = 1.0
        mask_stats = hp.ud_grade(mask_sim, nside_out=SYS_NSIDE_STATS, power=0) > 0
        seen_idx_stats = np.where(mask_stats)[0]

    if return_sim_idx:
        return maps, seen_idx_stats, seen_idx_sim
    return maps, seen_idx_stats


def galaxy_snr_from_mag_size(mag, r_half, seeing_fwhm, sigma_pix, zeropoint=30.0, pixscale=0.2):
    """Approximate galaxy SNR from magnitude, size, and observing conditions."""

    # TODO: Recheck the factors 1.678 and 2.355.
    flux = 10.0 ** (-0.4 * (mag - zeropoint))
    sigma_gal = r_half / 1.678
    sigma_psf = seeing_fwhm / 2.355
    sigma_eff2 = sigma_gal**2 + sigma_psf**2
    n_eff = 4.0 * np.pi * sigma_eff2 / pixscale**2
    snr = flux / (sigma_pix * np.sqrt(n_eff))
    return snr


def process_one(
    i,
    idx_sim,
    icat,
    conditions,
    gal_num,
    psf_hp_map,
    noise_hp_map,
    galactic_hp_map,
    detec_mag_bound,
    nside_sim,
):
    """Simulate one HEALPix pixel's worth of galaxies."""
    rng_local = np.random.default_rng(i)
    randi = rng_local.integers(0, icat.shape[0], size=gal_num)
    subset = icat.iloc[randi].copy()

    # RA/DEC are required by nz_utils.icat2cla_v2 for neighbor features.
    ra_c, dec_c = hp.pix2ang(nside_sim, idx_sim, lonlat=True)
    pix_size_deg = np.sqrt(4 * np.pi * (180 / np.pi) ** 2 / hp.nside2npix(nside_sim))
    rng_jitter = np.random.default_rng(i + 12345)
    subset['RA'] = (
        ra_c + rng_jitter.uniform(-0.5 * pix_size_deg, 0.5 * pix_size_deg, size=gal_num)
    ) % 360.0
    subset['DEC'] = np.clip(
        dec_c + rng_jitter.uniform(-0.5 * pix_size_deg, 0.5 * pix_size_deg, size=gal_num),
        -90.0,
        90.0,
    )

    # Keep simulation-grid pixel IDs; remap later for stats.
    subset['pix_idx'] = np.int64(idx_sim)

    conds = conditions.copy()
    conds['psf_fwhm'] = psf_hp_map[idx_sim]
    conds['pixel_rms'] = noise_hp_map[idx_sim]
    
    subset.loc[:, 'r'] += galactic_hp_map[idx_sim]
    # Flag objects beyond the detectability bound so they can be skipped as primaries.
    subset['beyond_detec_bound'] = subset['r'] > detec_mag_bound

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

    subset = downcast_float64_to32(subset)
    return subset

def compute_obs_stats(subset, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    # Use _input_p column names to match the classified catalog schema.
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

    # Photo-z scatter model.
    m = subset['r_input_p']
    rms = subset['pixel_rms_input_p']
    psf_fwhm = subset['psf_fwhm_input_p']

    alpha = PHOTOZ_PARAMS['alpha']
    sigma_pho = PHOTOZ_PARAMS['sigma_pho']
    sigma_int = PHOTOZ_PARAMS['sigma_int']
    m_ref = PHOTOZ_PARAMS['m_ref']
    rms_ref = PHOTOZ_PARAMS['rms_ref']
    psf_fwhm_ref = PHOTOZ_PARAMS['psf_fwhm_ref']

    dm = 2.5 * np.log10((rms / rms_ref) * (psf_fwhm / psf_fwhm_ref)**2)
    k = 10**(alpha * (m - m_ref + dm))

    sigma_pho *= k
    # sigma_z = np.maximum(sigma_z, PHOTOZ_PARAMS['sigma_min'])
    sigma_z = np.sqrt(sigma_pho**2 + sigma_int**2)
    mu_z = subset['redshift_input_p']

    # Draw photo-z with Gaussian scatter.
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
    Apply post-classification processing before summary statistics.
    Currently this includes photo-z assignment and MagLim selection.
    """

    # Keep this idempotent for already-processed catalogs.
    if 'z_pho_input_p' not in df.columns:
        # Compute observed magnitude and photo-z from input properties.
        df = compute_obs_stats(df, rng=rng)

        # Apply MagLim selection.
        df = apply_maglim_selection(df, rng=rng)

    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    bin_mask = get_binning_weights(df, tomo_bin_edges)
    for i in range(len(tomo_bin_edges) - 1):
        df[f"tomo_p_{i}"] = bin_mask[:, i]
    
    # Final cleanup of unused columns to save memory.
    keep_cols = ['pix_idx_input_p', 'redshift_input_p', 'detection', 'snr_input_p',
                 'r_input_p', 'Re_input_p',
                 'pixel_rms_input_p', 'psf_fwhm_input_p', 'z_pho_input_p']
    tomo_cols = [c for c in df.columns if c.startswith('tomo_p_')]
    final_cols = list(dict.fromkeys([c for c in keep_cols + tomo_cols if c in df.columns]))
    df = df[final_cols].copy()
    df = downcast_float64_to32(df)
    
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


def simulate_and_classify_chunked(gal_cat, z, edges, output_dir=None):
    """
    Run chunked simulation and classification with memory-aware filtering.
    """
    import gc
    maps, SEEN_idx, SEEN_idx_SIM = load_system_maps(return_sim_idx=True)
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

    t_sim_total_start = time.perf_counter()
    logger.info("Loading XGBoost model...")
    bst_cla = xgb.Booster({'device': 'cuda', 'n_jobs': -1})
    bst_cla.load_model(MODEL_JSON)
    
    temp_dir = os.path.join(os.path.dirname(OUTPUT_PREDS), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Global input-statistics accumulators.
    global_hist_in = np.zeros(z_bins_n)
    global_num_in = 0
    
    pixels_per_chunk = max(1, CHUNK_SIZE // N_POP_SAMPLE)
    n_pixels = len(SEEN_idx_SIM)
    chunk_files = []
    
    logger.info("Starting chunked processing: %d pixels in groups of %d", n_pixels, pixels_per_chunk)
    
    for start_p in range(0, n_pixels, pixels_per_chunk):
        end_p = min(start_p + pixels_per_chunk, n_pixels)
        block_indices = SEEN_idx_SIM[start_p:end_p]
        
        logger.info("  Chunk %d/%d (%d pixels)", start_p//pixels_per_chunk + 1, (n_pixels-1)//pixels_per_chunk + 1, len(block_indices))
        
        # 1) Simulate this block.
        results = Parallel(n_jobs=N_JOBS, backend="threading")(
            delayed(process_one)(
                start_p + i,
                idx,
                gal_cat,
                conditions,
                N_POP_SAMPLE,
                psf_hp_map,
                noise_hp_map,
                galactic_hp_map,
                detec_mag_bound,
                SYS_NSIDE_SIM,
            )
            for i, idx in enumerate(block_indices)
        )
        block_fullset = pd.concat(results, ignore_index=True)
        block_fullset = downcast_float64_to32(block_fullset)
        results = None 
        
        # Accumulate input statistics before classification/filtering.
        h_in = np.histogram(block_fullset["redshift"], bins=edges)[0]
        global_hist_in += h_in
        global_num_in += len(block_fullset)
        
        # 2) Coordinates are handled in process_one.

        # 3) Classify and filter.
        try:
            predictable = ~(block_fullset['beyond_detec_bound'].astype(bool).to_numpy())
            block_fullset_pred = block_fullset.loc[predictable].copy()

            # Predict only for primaries within the detectability bound.
            if not block_fullset_pred.empty:
                block_cla_pred = nz_utils.icat2cla_v2(
                    block_fullset_pred,
                    # Keep full neighbor context for feature construction.
                    block_fullset,
                    bst_cla,
                    predict=True,
                )
            else:
                block_cla_pred = pd.DataFrame(columns=['detection'])

            # Faint primaries are skipped as primaries but still act as neighbors.
            block_cla = block_cla_pred

            # Keep only downstream columns before filtering to avoid dtype issues.
            # Important: dropping too many columns can break later steps.
            keep_cols = [
                'pix_idx_input_p', 'redshift_input_p', 'detection',
                'r_input_p', 'Re_input_p',
                'psf_fwhm_input_p', 'pixel_rms_input_p', 'zero_point_input_p',
                'pixel_size_input_p'
            ]
            block_cla = block_cla[[c for c in keep_cols if c in block_cla.columns]].copy()

            # Keep detections only to save memory.
            # Coerce to numeric for robustness against unexpected dtypes.
            det_vals = pd.to_numeric(block_cla['detection'], errors='coerce').fillna(0.0).to_numpy()
            block_cla = block_cla.loc[det_vals > DETECTION_THRESHOLD].copy()

            block_cla = downcast_float64_to32(block_cla)
            if 'pix_idx_input_p' in block_cla.columns:
                block_cla['pix_idx_input_p'] = block_cla['pix_idx_input_p'].astype(np.int32)
            
            if not block_cla.empty:
                temp_path = os.path.join(temp_dir, f"cla_chunk_det_{start_p}.feather")
                block_cla.to_feather(temp_path)
                chunk_files.append(temp_path)
        except Exception as e:
            logger.error("CRITICAL ERROR processing block %d: %s", start_p, e)
            raise RuntimeError(f"Simulation failed at block {start_p}. Terminating for robustness.") from e
        
        block_fullset = None
        block_cla = None
        gc.collect()
        logger.info("    Memory usage: %.2f GB", get_memory_usage())

    # Precompute global input density and counts.
    dz = np.diff(edges)
    dndz_in_total = global_hist_in / (global_num_in * dz) if global_num_in > 0 else global_hist_in
    
    # Plot input dN/dz if output directory provided.
    if output_dir:
        plt_nz.plot_input_dndz(z, dndz_in_total, output_dir)
        
    logger.info("Simulation+classification stage completed in %.1fs", time.perf_counter() - t_sim_total_start)
    return chunk_files, SEEN_idx, SEEN_idx_SIM


def generate_summary_statistics_from_cat(cla_cat, SEEN_idx, seen_idx_sim, output_dir, z, edges):
    """Compute detection maps and dN/dz summaries from a catalog."""
    n_input_pix = get_input_counts_per_stats_pixel(
        seen_idx_sim,
        nside_sim=SYS_NSIDE_SIM,
        nside_stats=SYS_NSIDE_STATS,
        n_pop_sample=N_POP_SAMPLE,
    )
    # Total input count derived from the survey footprint geometry.
    # This is the authoritative denominator for all detection fractions.
    n_total_input = int(n_input_pix.sum())
    logger.info("Total input galaxies (from footprint): %s", f"{n_total_input:,}")

    cla_cat = cla_cat.copy()
    cla_cat["pix_idx_stats"] = map_pix_sim_to_stats(
        cla_cat["pix_idx_input_p"].to_numpy(dtype=np.int64),
        nside_sim=SYS_NSIDE_SIM,
        nside_stats=SYS_NSIDE_STATS,
    )

    # Detection map (full sample, before tomo binning).
    n_det_pix_full = np.bincount(cla_cat['pix_idx_stats'], weights=cla_cat['detection'], minlength=NPIX)
    frac_det, frac_det_pix = compute_detection_fractions(
        n_det_pix_full[SEEN_idx], n_input_pix[SEEN_idx]
    )

    mean_p = np.full(NPIX, hp.UNSEEN)
    mean_p[SEEN_idx] = frac_det_pix
    logger.info("Detection Rate Stats: frac=%.4f, min=%.4f, max=%.4f, mean=%.4f",
                frac_det, np.min(frac_det_pix), np.max(frac_det_pix), np.mean(frac_det_pix))

    plt_nz.plt_map(mean_p, SYS_NSIDE_STATS, SEEN_idx, 
            save_path=os.path.join(output_dir, "detection_rate_map.png"))
    
    results = {}
    dz = np.diff(edges)

    # Full sample.
    sys_res_full = groupby_dndz(
        cla_cat,
        z,
        edges,
        pix_col="pix_idx_stats",
        n_pix=NPIX,
    )
    metadata_rows_full = sys_res_full.loc[["total_detected"]].copy()
    sys_res_data_full = sys_res_full.reindex(SEEN_idx).fillna(0)
    sys_res_final_full = pd.concat([sys_res_data_full, metadata_rows_full])
    
    results['full'] = process_stats(
        sys_res_final_full,
        z,
        SEEN_idx,
        smooth=config.ANALYSIS_SETTINGS['smooth_nz'],
        n_input_pix=n_input_pix[SEEN_idx],
    )
    
    # Tomographic bins.
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    for i in range(len(tomo_bin_edges)-1):
        tomo_col = f"tomo_p_{i}"
        if tomo_col in cla_cat.columns:
            sys_res_i = groupby_dndz(
                cla_cat,
                z,
                edges,
                weight_col=tomo_col,
                pix_col="pix_idx_stats",
                n_pix=NPIX,
            )
            meta_i = sys_res_i.loc[["total_detected"]].copy()
            sys_res_i_data = sys_res_i.reindex(SEEN_idx).fillna(0)
            sys_res_i_final = pd.concat([sys_res_i_data, meta_i])
            
            results[f'tomo_{i}'] = process_stats(
                sys_res_i_final,
                z,
                SEEN_idx,
                smooth=config.ANALYSIS_SETTINGS['smooth_nz'],
                n_input_pix=n_input_pix[SEEN_idx],
            )
            
    return results


def compute_detection_fractions(n_det_pix, n_input_pix):
    """Compute global and per-pixel detection fractions.

    frac      = sum(n_det_pix) / sum(n_input_pix)   (global)
    frac_pix  = n_det_pix / n_input_pix               (per-pixel)
    """
    n_det_pix = np.asarray(n_det_pix, dtype=float)
    n_input_pix = np.asarray(n_input_pix, dtype=float)

    n_total = n_input_pix.sum()
    frac = n_det_pix.sum() / n_total if n_total > 0 else 0.0

    frac_pix = np.divide(
        n_det_pix, n_input_pix,
        out=np.zeros_like(n_det_pix, dtype=float),
        where=n_input_pix > 0,
    )

    return frac, frac_pix


def process_stats(sys_res, z, SEEN_idx, smooth=False, n_input_pix=None):
    """Normalize and package dN/dz statistics (trapezoidal integration)."""
    dndzs_raw = sys_res.drop(["sum_num", "std_z_pix"], axis=1)
    if "total_detected" in dndzs_raw.index:
        dndzs_raw = dndzs_raw.drop(["total_detected"])
    
    dndzs_raw = dndzs_raw.to_numpy().astype(float)
    # Ensure consistent trapezoidal normalization across all pixels
    norms = np.trapezoid(dndzs_raw, z, axis=1)
    dndzs = np.divide(dndzs_raw, norms[:, None], out=np.zeros_like(dndzs_raw), where=norms[:, None] > 0)

    sum_num = sys_res["sum_num"].drop(["total_detected"]).to_numpy(dtype=float)
    if n_input_pix is None:
        n_input_pix = np.full_like(sum_num, N_POP_SAMPLE)
    frac, frac_pix = compute_detection_fractions(sum_num, n_input_pix)
    std_z_pix = sys_res["std_z_pix"].drop(["total_detected"]).values
    
    dndz_det_raw = sys_res.drop(["sum_num", "std_z_pix"], axis=1).loc["total_detected"].to_numpy().astype(float)
    dndz_det = dndz_det_raw / np.trapezoid(dndz_det_raw, z)
    
    # dndz_det_flat: Each galaxy is weighted by inverse frac_pix to remove selection-induced density variations.
    # This simplifies to an area-weighted (n_input_pix) average of per-pixel normalized shapes.
    dndz_det_flat_raw = np.sum(dndzs * n_input_pix[:, None], axis=0)
    dndz_det_flat = dndz_det_flat_raw / np.trapezoid(dndz_det_flat_raw, z)

    std_z_global = sys_res.loc["total_detected", "std_z_pix"]

    z_std_ratio = sys_res.attrs.get('z_std_ratio', 1.0)
    min_count = int(config.STATS_PARAMS.get('min_count', 0))
    valid_pix = sum_num > min_count
    n_valid_pix = int(np.sum(valid_pix))
    n_total_pix = len(sum_num)

    # Unsmoothed baseline stats using the active min_count mask.
    if np.any(valid_pix):
        dndzs_valid = dndzs[valid_pix]
        frac_pix_valid = frac_pix[valid_pix]
        geo_w_pix_valid, geo_w_glob, geo_enh_det = utils.calculate_geometric_stats(
            z, dndzs_valid, dndz_det, frac_pix=frac_pix_valid
        )
        z_std_ratio_binned = utils.calculate_binned_std_ratio(
            z, dndzs_valid, dndz_det, frac_pix=frac_pix_valid
        )

        geo_w_pix = np.zeros_like(frac_pix, dtype=float)
        geo_w_pix[valid_pix] = geo_w_pix_valid
    else:
        geo_w_pix = np.zeros_like(frac_pix, dtype=float)
        geo_w_glob = 0.0
        geo_enh_det = 1.0
        z_std_ratio_binned = 1.0
    z_std_ratio_weighted = sys_res.attrs.get('z_std_ratio_pix_weighted', 1.0)

    if smooth:
        logger.info("Smoothing %d distributions...", dndzs.shape[0])
        sigma_dz = config.ANALYSIS_SETTINGS['smoothing_sigma_dz']
        sm_dndzs = smooth_nz_preserve_moments(z, dndzs, sigma_dz=sigma_dz)
        sm_dndz_det = smooth_nz_preserve_moments(z, dndz_det, sigma_dz=sigma_dz)
        sm_dndz_det_flat = smooth_nz_preserve_moments(z, dndz_det_flat, sigma_dz=sigma_dz)
        
        # Recompute widths for smoothed distributions with the same min_count mask.
        if np.any(valid_pix):
            sm_dndzs_valid = sm_dndzs[valid_pix]
            frac_pix_valid = frac_pix[valid_pix]
            sm_geo_w_pix_valid, sm_geo_w_glob, sm_geo_enh_det = utils.calculate_geometric_stats(
                z, sm_dndzs_valid, sm_dndz_det, frac_pix=frac_pix_valid
            )
            sm_z_std_ratio_binned = utils.calculate_binned_std_ratio(
                z, sm_dndzs_valid, sm_dndz_det, frac_pix=frac_pix_valid
            )
            sm_geo_w_pix = np.zeros_like(frac_pix, dtype=float)
            sm_geo_w_pix[valid_pix] = sm_geo_w_pix_valid
        else:
            sm_geo_w_pix = np.zeros_like(frac_pix, dtype=float)
            sm_geo_w_glob = 0.0
            sm_geo_enh_det = 1.0
            sm_z_std_ratio_binned = 1.0

        return {
            'z': z, 'dndzs': sm_dndzs, 'dndz_det': sm_dndz_det, 'dndz_det_flat': sm_dndz_det_flat,
            'frac': frac, 'frac_pix': frac_pix, 'SEEN_idx': SEEN_idx,
            'z_std_ratio': z_std_ratio, 
            'z_std_ratio_weighted': z_std_ratio_weighted,
            'z_std_ratio_binned': sm_z_std_ratio_binned,
            'z_std_ratio_binned_unsmoothed': z_std_ratio_binned,
            'std_z_pix': std_z_pix, 'std_z_global': std_z_global,
            'geo_width_pix': sm_geo_w_pix, 'geo_width_global': sm_geo_w_glob, 
            'geo_enh_det': sm_geo_enh_det,
            'geo_enh_det_unsmoothed': geo_enh_det,
            'n_valid_pix': n_valid_pix, 'n_total_pix': n_total_pix,
        }
    else:
        return {
            'z': z, 'dndzs': dndzs, 'dndz_det': dndz_det, 'dndz_det_flat': dndz_det_flat,
            'frac': frac, 'frac_pix': frac_pix, 'SEEN_idx': SEEN_idx,
            'z_std_ratio': z_std_ratio, 
            'z_std_ratio_weighted': z_std_ratio_weighted,
            'z_std_ratio_binned': z_std_ratio_binned,
            'z_std_ratio_binned_unsmoothed': z_std_ratio_binned,
            'std_z_pix': std_z_pix, 'std_z_global': std_z_global,
            'geo_width_pix': geo_w_pix, 'geo_width_global': geo_w_glob, 
            'geo_enh_det': geo_enh_det,
            'geo_enh_det_unsmoothed': geo_enh_det,
            'n_valid_pix': n_valid_pix, 'n_total_pix': n_total_pix,
        }

def save_fits_output(stats, bin_idx=4):
    """Store final results in a multi-HDU FITS file."""
    output_fits_path = utils.get_output_path("nz_bin_fits", bin_idx=bin_idx)
    os.makedirs(os.path.dirname(output_fits_path), exist_ok=True)
    
    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(stats['z'], name='Z'),
        fits.ImageHDU(stats['dndzs'], name='DNDZS'),
        fits.ImageHDU(stats['dndz_det'], name='DNDZ_DET'),
        fits.ImageHDU(stats.get('dndz_det_flat', stats['dndz_det']), name='DNDZ_DET_FLAT'),
        fits.ImageHDU([stats['frac']], name='FRAC'),
        fits.ImageHDU(stats['frac_pix'], name='FRAC_PIX'),
        fits.ImageHDU(stats['SEEN_idx'], name='SEEN_IDX'),
        fits.ImageHDU([stats.get('z_std_ratio', 1.0)], name='Z_STD_RATIO'),
        fits.ImageHDU(stats['std_z_pix'], name='STD_Z_PIX'),
        fits.ImageHDU(stats.get('geo_width_pix', np.zeros_like(stats['std_z_pix'])), name='GEO_WIDTH_PIX'),
        fits.ImageHDU([stats.get('geo_width_global', 0.0)], name='GEO_WIDTH_GLOBAL'),
        fits.ImageHDU([stats.get('geo_enh_det_unsmoothed', 1.0)], name='GEO_ENH_UNSM')
    ]
    fits.HDUList(hdus).writeto(output_fits_path, overwrite=True)
    logger.info("Results saved to %s", output_fits_path)


def load_fits_output(bin_idx=4):
    """Load results from a multi-HDU FITS file."""
    output_fits_path = utils.get_output_path("nz_bin_fits", bin_idx=bin_idx)
    if not os.path.exists(output_fits_path):
        logger.warning("FITS file %s not found.", output_fits_path)
        return None
    
    with fits.open(output_fits_path) as hdul:
        if 'DNDZ_DET_FLAT' in hdul:
            dndz_det_flat = hdul['DNDZ_DET_FLAT'].data
        else:
            logger.warning("DNDZ_DET_FLAT not found in %s; falling back to DNDZ_DET. "
                           "Re-run selection.py to generate it.", output_fits_path)
            dndz_det_flat = hdul['DNDZ_DET'].data

        stats = {
            'z': hdul['Z'].data,
            'dndzs': hdul['DNDZS'].data,
            'dndz_det': hdul['DNDZ_DET'].data,
            'dndz_det_flat': dndz_det_flat,
            'frac': hdul['FRAC'].data[0],
            'frac_pix': hdul['FRAC_PIX'].data,
            'SEEN_idx': hdul['SEEN_IDX'].data,
            'z_std_ratio': hdul['Z_STD_RATIO'].data[0] if 'Z_STD_RATIO' in hdul else 1.0,
            'std_z_pix': hdul['STD_Z_PIX'].data if 'STD_Z_PIX' in hdul else np.zeros_like(hdul['FRAC_PIX'].data),
            'geo_width_pix': hdul['GEO_WIDTH_PIX'].data if 'GEO_WIDTH_PIX' in hdul else np.zeros_like(hdul['FRAC_PIX'].data),
            'geo_width_global': hdul['GEO_WIDTH_GLOBAL'].data[0] if 'GEO_WIDTH_GLOBAL' in hdul else 0.0,
            'geo_enh_det_unsmoothed': hdul['GEO_ENH_UNSM'].data[0] if 'GEO_ENH_UNSM' in hdul else 1.0
        }
    return stats




def main():
    logging.basicConfig(level=getattr(logging, config.SIM_SETTINGS.get('log_level', 'INFO')),
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Use fixed redshift bins from config.
    z, edges = utils.get_redshift_bins(None)
    logger.info("[Binning] Fixed Grid: z=[%.4f, %.4f], dz=%.4f, n_bins=%d",
                edges[0], edges[-1], edges[1]-edges[0], len(z))

    maps, SEEN_idx, SEEN_idx_SIM = load_system_maps(return_sim_idx=True)

    if config.ANALYSIS_SETTINGS.get('load_preds', True) and os.path.exists(OUTPUT_PREDS):
        logger.info("Loading existing predictions from %s...", OUTPUT_PREDS)
        cla_cat = pd.read_feather(OUTPUT_PREDS)
        
        logger.info("Processing loaded catalog if needed (photo-z, cuts)...")
        cla_cat = process_classified_catalog(cla_cat)
        
        results = generate_summary_statistics_from_cat(cla_cat, SEEN_idx, SEEN_idx_SIM, output_dir, z, edges)
    else:
        if not os.path.exists(OUTPUT_PREDS):
             logger.info("Predictions file %s not found. Running simulation...", OUTPUT_PREDS)
        gal_cat = load_and_filter_catalog()
        
        chunk_files, SEEN_idx, SEEN_idx_SIM = simulate_and_classify_chunked(gal_cat, z=z, edges=edges, output_dir=output_dir)
        
        # Consolidate detected galaxies (detected-only catalog is much smaller).
        logger.info("Re-assembling %d detected-only chunks...", len(chunk_files))
        cla_cat = pd.concat([pd.read_feather(f) for f in chunk_files], ignore_index=True)

        logger.info("Saving raw detected catalog to %s...", OUTPUT_PREDS)
        os.makedirs(os.path.dirname(OUTPUT_PREDS), exist_ok=True)
        cla_cat.to_feather(OUTPUT_PREDS)
        
        logger.info("Final processing of consolidated catalog...")
        cla_cat = process_classified_catalog(cla_cat)

        results = generate_summary_statistics_from_cat(cla_cat, SEEN_idx, SEEN_idx_SIM, output_dir, z, edges)
            
        logger.info("    Memory usage after re-assembly and processing: %.2f GB", get_memory_usage())
        
        # Clean up temporary chunks.
        for f in chunk_files:
            os.remove(f)
        temp_dir = os.path.join(os.path.dirname(OUTPUT_PREDS), "temp_chunks")
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
        logger.info("    Memory usage at end of simulation: %.2f GB", get_memory_usage())
    
    # plt_nz.save_diagnostic_plots(results, output_dir)
    plt_nz.plot_tomographic_bins(results, output_dir)
    # plt_nz.plot_snr_fractions(cla_cat, output_dir, z=z, edges=edges)
    plt_nz.plot_pixel_std_histograms(results, output_dir)
    # plt_nz.plot_geo_vs_std_scatter(results, output_dir)
    # plt_nz.plot_photoz_weight_histograms(cla_cat, output_dir)
    # plt_nz.plot_dm_c_comparison_objects(cla_cat, output_dir)
    # plt_nz.plot_z_distribution_comparison(cla_cat, output_dir, z=z, edges=edges)
    
    # Save full sample and tomographic bins.
    # Use bin_idx=99 for the full sample to avoid overlap with tomo bins.
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    save_fits_output(results['full'], bin_idx=99) 
    for i in range(len(tomo_bin_edges) - 1):
        if f'tomo_{i}' in results:
            save_fits_output(results[f'tomo_{i}'], bin_idx=i)

    logger.info("\n--- Enhancement Factor Results (nbar=det, smooth_nz=%s) ---", config.ANALYSIS_SETTINGS['smooth_nz'])
    logger.info("(min_count=%d; fractions use ALL footprint pixels; enhancement factors use only valid pixels)",
                config.STATS_PARAMS.get('min_count', 0))
    logger.info("%-12s | %-12s | %-8s | %-10s | %-12s | %-10s | %-10s | %-10s | %-12s",
                "Bin", "Pix(v/t)", "Frac", "GeoEnh(Sm)", "GeoEnh(Unsm)", "zStd(Unw)", "zStd(Wtd)", "Binned(Sm)", "Binned(Unsm)")
    logger.info("-" * 130)
    for key, stats in results.items():
        pix_str = f"{stats.get('n_valid_pix', '?')}/{stats.get('n_total_pix', '?')}"
        logger.info("%-12s | %-12s | %.4f   | %.4f     | %.4f       | %.4f     | %.4f     | %.4f     | %.4f",
                     key, pix_str,
                     stats.get('frac', 0.0),
                     stats.get('geo_enh_det', 1.0),
                     stats.get('geo_enh_det_unsmoothed', 1.0),
                     stats.get('z_std_ratio', 1.0),
                     stats.get('z_std_ratio_weighted', 1.0),
                     stats.get('z_std_ratio_binned', 1.0),
                     stats.get('z_std_ratio_binned_unsmoothed', 1.0))


if __name__ == "__main__":
    main()
