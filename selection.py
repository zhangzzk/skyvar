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
except ImportError:
    import utils
    import config

# Add paths to sys.path
CODE_SRC = config.CODE_SRC
BLENDING_EMULATOR_DIR = config.BLENDING_EMULATOR_DIR
for path in [CODE_SRC, BLENDING_EMULATOR_DIR]:
    if path not in sys.path:
        sys.path.append(path)

# Custom imports after sys.path update
try:
    import glass_mock
    import generate_mocksys
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


def groupby_dndz(sys_cat, z, edges, post_cut=None, weight_col=None):
    """Compute per-pixel normalized n(z) and sum_num using vectorized operations."""
    z = np.asarray(z)
    edges = np.asarray(edges)
    dz = np.diff(edges)

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
    z_vals = df_cut["redshift_input_p"].values
    z_bins = np.digitize(z_vals, edges) - 1
    
    n_pix = int(sys_cat["pix_idx_input_p"].max() + 1)
    n_z = len(z)
    mask_valid = (z_bins >= 0) & (z_bins < n_z) & (pix_idx >= 0) & (pix_idx < n_pix)
    
    flat_idx = pix_idx[mask_valid] * n_z + z_bins[mask_valid]
    counts_flat = np.bincount(flat_idx, weights=df_cut["_w"].values[mask_valid], minlength=n_pix * n_z)
    pixel_counts = counts_flat.reshape(n_pix, n_z)
    
    sum_num = pixel_counts.sum(axis=1)
    hist_raw = np.zeros_like(pixel_counts)
    active_pix = sum_num > 0
    hist_raw[active_pix] = pixel_counts[active_pix] / (sum_num[active_pix][:, None] * dz)
    
    # Calculate std(dndz_det) / mean(std(dndzs)) from redshifts directly
    z_all = df_cut["redshift_input_p"].values
    w_all = df_cut["_w"].values
    
    if w_all.sum() > 0:
        mean_z_all = np.average(z_all, weights=w_all)
        std_z_all = np.sqrt(np.average((z_all - mean_z_all)**2, weights=w_all))
        
        # Per-pixel std calculation using vectorized bincount
        w_sum = np.bincount(pix_idx, weights=w_all, minlength=n_pix)
        wz_sum = np.bincount(pix_idx, weights=w_all * z_all, minlength=n_pix)
        wz2_sum = np.bincount(pix_idx, weights=w_all * z_all**2, minlength=n_pix)
        
        # We only care about pixels that are in the SEEN footprint AND have detections
        # In groupby_dndz, sys_cat might contain all pixels, but df_cut only has detections.
        mask_v = w_sum > 0
        mean_z_pix = wz_sum[mask_v] / w_sum[mask_v]
        var_z_pix = (wz2_sum[mask_v] / w_sum[mask_v]) - mean_z_pix**2
        std_z_pix = np.sqrt(np.maximum(var_z_pix, 0))
        mean_std_z_pix = np.mean(std_z_pix)
        
        z_std_ratio = std_z_all / mean_std_z_pix if mean_std_z_pix > 0 else 1.0
        label = weight_col if weight_col else "full"
        print(f"[{label:10s}] Redshift-based std ratio: {z_std_ratio:.6f} "
              f"(global std: {std_z_all:.4f}, mean local std: {mean_std_z_pix:.4f})")
    else:
        z_std_ratio = 1.0

    out = pd.DataFrame(hist_raw, index=np.arange(n_pix))
    out.columns = np.arange(n_z)
    out["sum_num"] = sum_num
    out["std_z_pix"] = 0.0
    if w_all.sum() > 0:
        out.loc[np.where(mask_v)[0], "std_z_pix"] = std_z_pix
    out.attrs['z_std_ratio'] = z_std_ratio

    dndz_in = np.histogram(sys_cat["redshift_input_p"], bins=edges, density=True)[0]
    num_in = sys_cat.shape[0]

    dndz_det = np.histogram(df_cut["redshift_input_p"], bins=edges, density=True, weights=df_cut["_w"])[0]
    num_det = df_cut["_w"].sum()

    out.loc["total_input"] = list(dndz_in) + [num_in, 0.0]
    out.loc["total_detected"] = list(dndz_det) + [num_det, std_z_all if w_all.sum() > 0 else 0.0]
    
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

    gal_cat = gal_cat.loc[
        (gal_cat['BA'] > 0.05) & (gal_cat['BA'] < 1.0) &
        (gal_cat['r'] < 28) & (gal_cat['r'] > 0) &
        (gal_cat['Re'] < 5.) & (gal_cat['Re'] > 0.01) &
        (gal_cat['sersic_n'] < 6.) & (gal_cat['sersic_n'] > 0.5)
    ].reset_index(drop=True).astype(np.float64)

    cat_area = 5.968310365946076
    n_degree2 = gal_cat.shape[0] / cat_area
    print(f"Number density: {n_degree2 / 60**2:.2f} gal/arcmin^2")
    return gal_cat, n_degree2


def load_system_maps():
    """Load system maps and return maps and SEEN_idx."""
    print(f"Loading system maps from {MOCK_SYS_MAP_PATH}...")
    maps = hp.read_map(MOCK_SYS_MAP_PATH, field=None)
    psf_hp_map = maps[0]
    SEEN_idx = np.where(~np.isnan(psf_hp_map))[0]
    return maps, SEEN_idx


def galaxy_snr_from_mag_size(mag, r_half, seeing_fwhm, sigma_pix, zeropoint=30.0, pixscale=0.2):
    """Approximate galaxy SNR."""
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

    subset['snr'] = galaxy_snr_from_mag_size(
        subset['r'],
        subset['Re'],
        subset['psf_fwhm'],
        subset['pixel_rms'],
        zeropoint=subset['zero_point'],
        pixscale=subset['pixel_size'],
    )
    return subset


def process_classified_catalog(df):
    """
    Perform any actions between having a classified catalog and summary statistics.
    Currently: photo-z assignment and post-detection cuts.
    """
    if df is None or df.empty:
        return df
    
    # 1. Assign Photo-z weights
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    p_weights = utils.get_photoz_weights(df, tomo_bin_edges)
    for i in range(len(tomo_bin_edges) - 1):
        df[f"tomo_p_{i}"] = p_weights[:, i]
        
    # 2. Apply post-detection cuts
    # df = apply_post_detection_cuts(df)
    
    return df


def simulate_and_classify_chunked(gal_cat, n_degree2, z, edges):
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
            
            if not block_cla.empty:
                temp_path = os.path.join(temp_dir, f"cla_chunk_det_{start_p}.feather")
                block_cla.to_feather(temp_path)
                chunk_files.append(temp_path)
        except Exception as e:
            print(f"Error processing block {start_p}: {e}")
        
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


def generate_summary_statistics_from_cat(cla_cat, psf_hp_map, SEEN_idx, output_dir, z, edges):
    """Compute detection maps and dN/dz distributions from a catalog."""
    mean_p = np.full(psf_hp_map.shape, hp.UNSEEN)
    
    # Map pixels to active indices to avoid out-of-bounds
    pixel_counts = np.bincount(cla_cat['pix_idx_input_p'], minlength=len(psf_hp_map))
    mean_p[SEEN_idx] = pixel_counts[SEEN_idx] / N_POP_SAMPLE
    
    utils.plt_map(mean_p, SYS_NSIDE, np.where(~np.isnan(psf_hp_map)), 
            save_path=os.path.join(output_dir, "detection_rate_map.png"))
    
    results = {}
    
    # 1. Full Sample
    print("Calculating dN/dz for full sample...")
    mask_in_footprint = (cla_cat['pix_idx_input_p'] < len(psf_hp_map))
    df_filtered = cla_cat[mask_in_footprint].copy()
    
    z_full, edges_full = utils.get_redshift_bins(df_filtered['redshift_input_p'], weights=df_filtered['detection'].values)
    z_bins_n_full = len(z_full)
    dz_full = np.diff(edges_full)
    
    global_hist_in_full = np.histogram(df_filtered["redshift_input_p"], bins=edges_full)[0] # Note: this is an approximation for 'in'
    dndz_in_full = global_hist_in_full / (global_hist_in_full.sum() * dz_full)
    
    sys_res_full = groupby_dndz(df_filtered, z_full, edges_full, post_cut=None)
    metadata_rows_full = sys_res_full.loc[["total_input", "total_detected"]].copy()
    sys_res_data_full = sys_res_full.reindex(SEEN_idx).fillna(0)
    sys_res_final_full = pd.concat([sys_res_data_full, metadata_rows_full])
    
    sys_res_final_full.loc["total_input", sys_res_final_full.columns[:z_bins_n_full]] = dndz_in_full
    sys_res_final_full.loc["total_input", "sum_num"] = df_filtered.shape[0] / 0.1
    sys_res_final_full.loc["total_input", "std_z_pix"] = 0.0
    
    results['full'] = process_stats(sys_res_final_full, z_full, SEEN_idx, smooth=config.ANALYSIS_SETTINGS['smooth_nz'])
    
    # 2. Tomographic Bins
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    for i in range(len(tomo_bin_edges)-1):
        tomo_col = f"tomo_p_{i}"
        if tomo_col in cla_cat.columns:
            print(f"Calculating dN/dz for tomo_{i}...")
            # Bin-specific redshift bins
            z_i, edges_i = utils.get_redshift_bins(df_filtered['redshift_input_p'], weights=df_filtered['detection'] * df_filtered[tomo_col])
            z_bins_n_i = len(z_i)
            dz_i = np.diff(edges_i)
            
            sys_res_i = groupby_dndz(df_filtered, z_i, edges_i, post_cut=None, weight_col=tomo_col)
            meta_i = sys_res_i.loc[["total_input", "total_detected"]].copy()
            sys_res_i_data = sys_res_i.reindex(SEEN_idx).fillna(0)
            sys_res_i_final = pd.concat([sys_res_i_data, meta_i])
            
            # Input distribution for this bin (approximated from detected sample range)
            global_hist_in_i = np.histogram(df_filtered["redshift_input_p"], bins=edges_i, weights=df_filtered['detection'])[0] 
            dndz_in_i = global_hist_in_i / (global_hist_in_i.sum() * dz_i) if global_hist_in_i.sum() > 0 else global_hist_in_i
            
            sys_res_i_final.loc["total_input", sys_res_i_final.columns[:z_bins_n_i]] = dndz_in_i
            sys_res_i_final.loc["total_input", "sum_num"] = df_filtered.shape[0] / 0.1 # Placeholder
            sys_res_i_final.loc["total_input", "std_z_pix"] = 0.0
            
            results[f'tomo_{i}'] = process_stats(sys_res_i_final, z_i, SEEN_idx, smooth=config.ANALYSIS_SETTINGS['smooth_nz'])
            
    return results


def process_stats(sys_res, z, SEEN_idx, smooth=False):
    """Auxiliary to package dndz results."""
    dndzs = sys_res.drop(["sum_num", "std_z_pix"], axis=1)
    if "total_input" in dndzs.index:
        dndzs = dndzs.drop(["total_input", "total_detected"])
    
    dndzs = dndzs.to_numpy()
    sum_num = sys_res["sum_num"].drop(["total_input", "total_detected"]).values
    std_z_pix = sys_res["std_z_pix"].drop(["total_input", "total_detected"]).values
    
    mean_num = sys_res.loc["total_detected", "sum_num"] / len(SEEN_idx)
    frac_pix = (sum_num / mean_num) if mean_num > 0 else np.ones_like(sum_num)
    frac = sys_res.loc["total_detected", "sum_num"] / sys_res.loc["total_input", "sum_num"]
    std_z_global = sys_res.loc["total_detected", "std_z_pix"]
    
    dndz_in = sys_res.drop(["sum_num", "std_z_pix"], axis=1).loc["total_input"].to_numpy().astype(float)
    dndz_det = sys_res.drop(["sum_num", "std_z_pix"], axis=1).loc["total_detected"].to_numpy().astype(float)

    z_std_ratio = sys_res.attrs.get('z_std_ratio', 1.0)

    if smooth:
        print(f"Smoothing {dndzs.shape[0]} distributions...")
        sigma_dz = config.ANALYSIS_SETTINGS['smoothing_sigma_dz']
        sm_dndzs = smooth_nz_preserve_moments(z, dndzs, sigma_dz=sigma_dz)
        sm_dndz_in = smooth_nz_preserve_moments(z, dndz_in, sigma_dz=sigma_dz)
        sm_dndz_det = smooth_nz_preserve_moments(z, dndz_det, sigma_dz=sigma_dz)
    
        return {
            'z': z, 'dndzs': sm_dndzs, 'dndz_in': sm_dndz_in, 'dndz_det': sm_dndz_det,
            'frac': frac, 'frac_pix': frac_pix, 'SEEN_idx': SEEN_idx,
            'z_std_ratio': z_std_ratio, 'std_z_pix': std_z_pix, 'std_z_global': std_z_global
        }
    else:
        return {
            'z': z, 'dndzs': dndzs, 'dndz_in': dndz_in, 'dndz_det': dndz_det,
            'frac': frac, 'frac_pix': frac_pix, 'SEEN_idx': SEEN_idx,
            'z_std_ratio': z_std_ratio, 'std_z_pix': std_z_pix, 'std_z_global': std_z_global
        }


def generate_summary_statistics_incremental(chunk_files, bin_details, psf_hp_map, SEEN_idx, output_dir):
    """Memory-efficient incremental statistics generation."""
    import gc
    n_pix_total = len(psf_hp_map)
    keys = list(bin_details.keys())

    # Accumulators
    det_counts_map = np.zeros(n_pix_total)
    
    # Each key k might have different n_z
    accumulators = {k: np.zeros((n_pix_total, len(bin_details[k][0]))) for k in keys}
    total_det_nums = {k: 0.0 for k in keys}
    total_det_hists = {k: np.zeros(len(bin_details[k][0])) for k in keys}
    
    # Accumulators for z_std_ratio
    w_sum_pix = {k: np.zeros(n_pix_total) for k in keys}
    wz_sum_pix = {k: np.zeros(n_pix_total) for k in keys}
    wz2_sum_pix = {k: np.zeros(n_pix_total) for k in keys}
    
    total_z_w_sum = {k: 0.0 for k in keys}
    total_z_wz_sum = {k: 0.0 for k in keys}
    total_z_wz2_sum = {k: 0.0 for k in keys}

    print(f"Incremental accumulation from chunks (with post-detection cuts)...")
    for f in chunk_files:
        df = pd.read_feather(f)
        if df.empty: continue
        
        df = process_classified_catalog(df)
        if df.empty: continue

        pixel_indices = df["pix_idx_input_p"].values
        z_vals = df["redshift_input_p"].values
        weights_det = df["detection"].values
        
        # Detection rate map (soft count)
        pixel_counts = np.bincount(pixel_indices, weights=weights_det, minlength=n_pix_total)
        det_counts_map += pixel_counts
        
        for k in keys:
            z_k, edges_k = bin_details[k]
            n_z_k = len(z_k)
            z_bins_k = np.digitize(z_vals, edges_k) - 1
            mask_z_k = (z_bins_k >= 0) & (z_bins_k < n_z_k)
            
            if k == 'full':
                weights = weights_det
            else:
                idx = int(k.split('_')[1])
                weights = weights_det * df[f'tomo_p_{idx}'].values
                
            total_det_nums[k] += weights.sum()
            total_det_hists[k] += np.histogram(z_vals, bins=edges_k, weights=weights)[0]
            
            # Per-pixel histogram accumulation
            flat_idx = pixel_indices[mask_z_k] * n_z_k + z_bins_k[mask_z_k]
            counts_flat = np.bincount(flat_idx, weights=weights[mask_z_k], minlength=n_pix_total * n_z_k)
            accumulators[k] += counts_flat.reshape(n_pix_total, n_z_k)

            # Redshift-based accumulation
            w_sum_pix[k] += np.bincount(pixel_indices, weights=weights, minlength=n_pix_total)
            wz_sum_pix[k] += np.bincount(pixel_indices, weights=weights * z_vals, minlength=n_pix_total)
            wz2_sum_pix[k] += np.bincount(pixel_indices, weights=weights * z_vals**2, minlength=n_pix_total)
            
            total_z_w_sum[k] += weights.sum()
            total_z_wz_sum[k] += (weights * z_vals).sum()
            total_z_wz2_sum[k] += (weights * z_vals**2).sum()
            
        df = None
        gc.collect()

    # Post-process and packaging
    final_results = {}
    
    # 1. Detection Rate Map
    mean_p = np.full(psf_hp_map.shape, hp.UNSEEN)
    # Correcting mean_p: since only detections are in chunks, count/N_POP is the rate
    mean_p[SEEN_idx] = det_counts_map[SEEN_idx] / N_POP_SAMPLE
    
    p_valid = mean_p[SEEN_idx]
    print(f"Detection Rate Stats: min={np.min(p_valid):.4f}, max={np.max(p_valid):.4f}, mean={np.mean(p_valid):.4f}")
    
    utils.plt_map(mean_p, SYS_NSIDE, SEEN_idx, 
            save_path=os.path.join(output_dir, "detection_rate_map.png"))

    for k in keys:
        z_k, edges_k = bin_details[k]
        dz_k = np.diff(edges_k)
        pixel_counts = accumulators[k]
        sum_num = pixel_counts.sum(axis=1)
        
        # Filter to SEEN_idx and normalize
        active_counts = pixel_counts[SEEN_idx]
        active_sum_num = sum_num[SEEN_idx]
        
        hist_raw = np.zeros_like(active_counts)
        mask_v = active_sum_num > 0
        hist_raw[mask_v] = active_counts[mask_v] / (active_sum_num[mask_v][:, None] * dz_k)
        
        df_stats = pd.DataFrame(hist_raw)
        df_stats["sum_num"] = active_sum_num
        df_stats["std_z_pix"] = 0.0
        
        # Calculate z_std_ratio for this bin
        if total_z_w_sum[k] > 0:
            mean_z_all = total_z_wz_sum[k] / total_z_w_sum[k]
            std_z_all = np.sqrt(np.maximum(total_z_wz2_sum[k] / total_z_w_sum[k] - mean_z_all**2, 0))
            
            # Per-pixel std
            w = w_sum_pix[k][SEEN_idx]
            wz = wz_sum_pix[k][SEEN_idx]
            wz2 = wz2_sum_pix[k][SEEN_idx]
            
            mask_active = w > 0
            mean_z_pix = wz[mask_active] / w[mask_active]
            var_z_pix = (wz2[mask_active] / w[mask_active]) - mean_z_pix**2
            std_z_pix = np.sqrt(np.maximum(var_z_pix, 0))
            mean_std_z_pix = np.mean(std_z_pix)
            
            df_stats.loc[mask_active, "std_z_pix"] = std_z_pix
            z_std_ratio = std_z_all / mean_std_z_pix if mean_std_z_pix > 0 else 1.0
        else:
            z_std_ratio = 1.0
        
        df_stats.attrs['z_std_ratio'] = z_std_ratio
        print(f"[{k:10s}] Redshift-based std ratio: {z_std_ratio:.6f}")

        # dndz totals
        d_det = total_det_hists[k] / (total_det_nums[k] * dz_k) if total_det_nums[k] > 0 else total_det_hists[k]
        
        # Mocking the groupby_dndz structure for process_stats
        # Since we don't have global sim_truth for each bin easily here, 
        # we placeholder total_input for now.
        df_stats.loc["total_input"] = list(d_det) + [total_det_nums[k] / 0.1, 0.0]
        df_stats.loc["total_detected"] = list(d_det) + [total_det_nums[k], std_z_all if total_z_w_sum[k] > 0 else 0.0]
        
        final_results[k] = process_stats(df_stats, z_k, SEEN_idx, smooth=config.ANALYSIS_SETTINGS['smooth_nz'])
    
    return final_results


def save_diagnostic_plots(results, output_dir, key='full'):
    """Generate and save distributions plots with pixel variations."""
    stats = results[key]
    z = stats['z']
    dndzs = stats['dndzs']
    
    plt.figure(figsize=(10, 6))
    
    # Plot individual pixel variations as thin light lines
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
    plt.savefig(os.path.join(output_dir, f"pixel_nz_variations.png"))
    plt.close()


def plot_tomographic_bins(results, output_dir):
    """Plot the global n(z) for all tomographic bins on one plot with variations."""
    plt.figure(figsize=(10, 6))
    tomo_keys = sorted([k for k in results.keys() if k.startswith('tomo_')], 
                       key=lambda x: int(x.split('_')[1]))
    
    # Colors for the bins (matching the reference image's vibrant aesthetic)
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
        
        # Plot pixel variations as extremely thin, light lines
        n_pixels = dndzs.shape[0]
        n_plot = min(n_pixels, 150)
        step = max(1, n_pixels // n_plot)
        for j in range(0, n_pixels, step):
            plt.plot(z, dndzs[j], color=color, alpha=0.2, lw=0.5)

        # Plot mean
        plt.plot(z, stats['dndz_det'], color=color, lw=2., label=f"Bin {i}")
        # Add the 'Truth' (input) for each bin as a dashed line
        # plt.plot(z, stats['dndz_in'], color=color, ls='--', lw=1.2, alpha=0.7)

    plt.xlim(0, 2)
    plt.xlabel('Redshift $z$')
    plt.ylabel('$n(z)$')
    plt.title('Tomographic Bin Redshift Distributions (with spatial variations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "tomographic_bins_nz.png"))
    plt.close()


def plot_snr_fractions(cla_cat, output_dir, bin_details=None):
    """
    Plot the fraction of detected galaxies whose SNR > 5 (and 10, 15) 
    as a function of redshift for full samples and tomographic bins.
    """
    import matplotlib.pyplot as plt
    
    # Identify redshift and snr column names
    z_col = 'redshift_input_p' if 'redshift_input_p' in cla_cat.columns else 'redshift'
    snr_col = 'snr_input_p' if 'snr_input_p' in cla_cat.columns else 'snr'
    
    if z_col not in cla_cat.columns or snr_col not in cla_cat.columns:
        z_col = next((c for c in cla_cat.columns if 'redshift' in c), None)
        snr_col = next((c for c in cla_cat.columns if 'snr' in c), None)

    if z_col is None or snr_col is None:
        print(f"Warning: Redshift or SNR column missing. Available: {list(cla_cat.columns)}")
        return

    if bin_details is None:
        # Fallback to global bins if details not provided
        z_g, edges_g = utils.get_redshift_bins(cla_cat[z_col], weights=cla_cat['detection'].values if 'detection' in cla_cat.columns else None)
        bin_details = {'full': (z_g, edges_g)}
    
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    n_tomo = len(tomo_bin_edges) - 1
    
    # Panels: Full sample + individual tomo bins
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

    # 1. Full sample panel
    print("Calculating SNR fractions for full sample...")
    z_f, edges_f = bin_details.get('full', bin_details[list(bin_details.keys())[0]])
    fr_full = calculate_fractions(cla_cat, edges_f, weights=cla_cat['detection'].values)
    ax = axes[0]
    for i, thr in enumerate(thresholds):
        ax.plot(z_f, fr_full[thr], label=f'SNR > {thr}', color=colors[i], lw=2)
    ax.set_title("Full Sample")
    ax.set_ylabel("Fraction of Detections")
    ax.set_xlabel("Redshift $z$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 2. Tomo bin panels
    for i in range(n_tomo):
        tomo_key = f'tomo_{i}'
        ax = axes[i+1]
        if tomo_key in bin_details:
            print(f"Calculating SNR fractions for tomo bin {i}...")
            z_i, edges_i = bin_details[tomo_key]
            # Use the tomo_p_i column as weights, multiplied by detection weights
            tomo_weights = cla_cat[f'tomo_p_{i}'].values * cla_cat['detection'].values
            fr_tomo = calculate_fractions(cla_cat, edges_i, weights=tomo_weights)
            for j, thr in enumerate(thresholds):
                ax.plot(z_i, fr_tomo[thr], label=f'SNR > {thr}', color=colors[j], lw=2)
            ax.set_title(f"Tomo Bin {i} ($z_p \in [{tomo_bin_edges[i]}, {tomo_bin_edges[i+1]}]$)")
            ax.set_xlabel("Redshift $z$")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"Bin {i} missing", ha='center')

    # Cleanup unused axes
    for j in range(n_panels, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Fraction of Detected Galaxies Above SNR Thresholds", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, "snr_fractions.png")
    plt.savefig(save_path, dpi=200)
    print(f"SNR fractions plot saved to {save_path}")
    plt.close()


def plot_pixel_std_histograms(results, output_dir):
    """
    Plot histograms of n(z) std in each pixel for full and bins.
    """
    import matplotlib.pyplot as plt
    
    tomo_keys = sorted([k for k in results.keys() if k.startswith('tomo_')], 
                       key=lambda x: int(x.split('_')[1]))
    keys = ['full'] + tomo_keys
    
    n_panels = len(keys)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    
    for i, k in enumerate(keys):
        stats = results[k]
        std_vals = stats['std_z_pix']
        # Filter out zero or negative values (mostly zero for unseen pixels)
        active_std = std_vals[std_vals > 0]
        
        ax = axes[i]
        if len(active_std) > 0:
            ax.hist(active_std, bins=80, density=True, label=f'value in NSIDE={SYS_NSIDE} pixel')
            
            # Use global standard deviation calculated from redshifts directly
            global_std = stats.get('std_z_global', 0.0)
            
            ax.axvline(global_std, color='red', label=f'global average {global_std:.6f}')
            
            ratio = stats.get('z_std_ratio', 1.0)
            ax.set_title(f"{k} redshift std distribution\n(Std Ratio: {ratio:.6f})")
            ax.set_xlabel("Redshift standard deviation $\sigma_z$")
            ax.set_ylabel("Probability Density")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No data for {k}", ha='center', va='center')

    # Cleanup unused axes
    for j in range(n_panels, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Distribution of Per-Pixel Redshift Standard Deviation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, "pixel_std_histograms.png")
    plt.savefig(save_path, dpi=200)
    print(f"Pixel std histograms saved to {save_path}")
    plt.close()


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
        # fits.ImageHDU(stats['sm_dndzs'], name='SM_DNDZS'),
        # fits.ImageHDU(stats['sm_dndz_in'], name='SM_DNDZ_IN'),
        # fits.ImageHDU(stats['sm_dndz_det'], name='SM_DNDZ_DET'),
        fits.ImageHDU(stats['SEEN_idx'], name='SEEN_IDX'),
        fits.ImageHDU([stats.get('z_std_ratio', 1.0)], name='Z_STD_RATIO'),
        fits.ImageHDU(stats['std_z_pix'], name='STD_Z_PIX')
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
            # 'sm_dndzs': hdul['SM_DNDZS'].data,
            # 'sm_dndz_in': hdul['SM_DNDZ_IN'].data,
            # 'sm_dndz_det': hdul['SM_DNDZ_DET'].data,
            'SEEN_idx': hdul['SEEN_IDX'].data,
            'z_std_ratio': hdul['Z_STD_RATIO'].data[0] if 'Z_STD_RATIO' in hdul else 1.0,
            'std_z_pix': hdul['STD_Z_PIX'].data if 'STD_Z_PIX' in hdul else np.zeros_like(hdul['FRAC_PIX'].data)
        }
    return stats


def get_bin_details_from_cat(df, is_input_cat=False):
    """
    Determine bin-specific redshift distributions (z and edges) for a given catalog.
    If is_input_cat is True, estimates tomographic weights using nominal photo-z.
    """
    bin_details = {}
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    
    if is_input_cat:
        z_col = 'redshift'
        weights_full = None
        # Estimate tomo weights
        df_temp = df.rename(columns={'redshift':'redshift_input_p', 'r':'r_input_p'})
        df_temp['pixel_rms_input_p'] = config.OBS_CONDITIONS['pixel_rms_nominal']
        tomo_weights = utils.get_photoz_weights(df_temp, tomo_bin_edges)
    else:
        z_col = 'redshift_input_p'
        weights_full = df['detection'].values
        # Membership weights (including detection probability)
        tomo_weights = np.stack([df[f'tomo_p_{i}'].values * weights_full for i in range(len(tomo_bin_edges)-1)], axis=1)

    # 1. Full Sample Bins
    z_f, edges_f = utils.get_redshift_bins(df[z_col], weights=weights_full)
    bin_details['full'] = (z_f, edges_f)
    print(f"[Binning] Full: z=[{edges_f[0]:.4f}, {edges_f[-1]:.4f}], dz={(edges_f[1]-edges_f[0]):.4f}, n_bins={len(z_f)}")
    
    # 2. Tomographic Bins
    for i in range(len(tomo_bin_edges) - 1):
        w_i = tomo_weights[:, i]
        if np.sum(w_i) > 0:
            z_i, edges_i = utils.get_redshift_bins(df[z_col], weights=w_i)
            bin_details[f'tomo_{i}'] = (z_i, edges_i)
            print(f"[Binning] Tomo_{i}: z=[{edges_i[0]:.4f}, {edges_i[-1]:.4f}], dz={(edges_i[1]-edges_i[0]):.4f}, n_bins={len(z_i)}")
        else:
            bin_details[f'tomo_{i}'] = (z_f, edges_f)
            print(f"[Binning] Tomo_{i}: EMPTY (using full sample bins)")
            
    return bin_details


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if config.ANALYSIS_SETTINGS.get('load_preds', True) and os.path.exists(OUTPUT_PREDS):
        print(f"Loading existing predictions from {OUTPUT_PREDS}...")
        cla_cat = pd.read_feather(OUTPUT_PREDS)
        
        print("Processing loaded catalog (photo-z, cuts)...")
        cla_cat = process_classified_catalog(cla_cat)
        
        # Determine redshift bins for each bin separately
        bin_details = get_bin_details_from_cat(cla_cat, is_input_cat=False)

        maps, SEEN_idx = load_system_maps()
        psf_hp_map = maps[0]
        results = generate_summary_statistics_from_cat(cla_cat, psf_hp_map, SEEN_idx, output_dir, bin_details=bin_details)
    else:
        if not os.path.exists(OUTPUT_PREDS):
             print(f"Predictions file {OUTPUT_PREDS} not found. Running simulation...")
        gal_cat, n_degree2 = load_and_filter_catalog()
        
        # Determine redshift bins for each bin separately (from input catalog)
        bin_details = get_bin_details_from_cat(gal_cat, is_input_cat=True)
        z_full, edges_full = bin_details['full']
        
        # Pick one set of edges for the simulation phase histograms (the 'global truth' in sim_truth)
        chunk_files, psf_hp_map, SEEN_idx, sim_truth = simulate_and_classify_chunked(gal_cat, n_degree2, z=z_full, edges=edges_full)
        
        results = generate_summary_statistics_incremental(chunk_files, bin_details, psf_hp_map, SEEN_idx, output_dir)
        
        # Consolidation of detected galaxies (ONLY detected, so much smaller)
        print(f"Re-assembling {len(chunk_files)} detected-only chunks...")
        cla_cat = pd.concat([pd.read_feather(f) for f in chunk_files], ignore_index=True)
        
        print("Final processing of consolidated catalog...")
        cla_cat = process_classified_catalog(cla_cat)
            
        print(f"    Memory usage after re-assembly and processing: {get_memory_usage():.2f} GB")
        
        # Cleanup temporary chunks
        for f in chunk_files:
            os.remove(f)
        temp_dir = os.path.join(os.path.dirname(OUTPUT_PREDS), "temp_chunks")
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
        print(f"Saving detected catalog (full) to {OUTPUT_PREDS}...")
        os.makedirs(os.path.dirname(OUTPUT_PREDS), exist_ok=True)
        cla_cat.to_feather(OUTPUT_PREDS)
        print(f"    Memory usage after saving: {get_memory_usage():.2f} GB")
    
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    # results already generated incrementally in the else block
    
    save_diagnostic_plots(results, output_dir)
    plot_tomographic_bins(results, output_dir)
    plot_snr_fractions(cla_cat, output_dir, bin_details=bin_details)
    plot_pixel_std_histograms(results, output_dir)
    
    # Save full sample and tomographic bins
    # Using bin_idx=99 for full sample to avoid overlap with tomo bins
    save_fits_output(results['full'], bin_idx=99) 
    for i in range(len(tomo_bin_edges) - 1):
        if f'tomo_{i}' in results:
            save_fits_output(results[f'tomo_{i}'], bin_idx=i)

    print("\n--- Enhancement Factor Results ---")
    for key, stats in results.items():
        enhancement = utils.calculate_geometric_enhancement(
            stats['z'], stats['dndzs'], stats['dndz_det'], frac_pix=None,
        )
        print(f"Geometric Enhancement Factor ({key:10s}): {enhancement:.6f}")
        print(f"Redshift-based std Ratio   ({key:10s}): {stats.get('z_std_ratio', 1.0):.6f}")


if __name__ == "__main__":
    main()
